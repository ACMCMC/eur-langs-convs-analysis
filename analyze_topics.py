# %%
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
import polars as pl
import matplotlib.pyplot as plt
import datasets
from sklearn.manifold import TSNE
import os
import transformers


model_short_name = "minilm"
model_short_name = "mpnet"
# %%

DEVICE = torch.device("cuda")

# Load quantized model
bitsandbytes_config = transformers.BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

torch.set_default_device(DEVICE)


# Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[
        0
    ]  # First element of model_output contains all token embeddings
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    )
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )


try:
    from google.colab import userdata

    hf_token = userdata.get("HF_TOKEN")

    dataset = datasets.load_dataset(
        "acmc/chatbot_conversations_in_european_languages",
        split="train",
        token=hf_token,
    )
except:
    dataset = datasets.load_dataset(
        "acmc/chatbot_conversations_in_european_languages", split="train"
    )

# Change the column name "language" to "lang"
# There's no particular reason to do this, other than when exporting the CSV files and using them in LaTeX, with csvsimple, the column name "lang" was causing problems when it's called exactly like that, adding an extra backtick - probably some sort of conflict with another package (super weird bug)
dataset = dataset.rename_column("language", "lang")

# Load model from HuggingFace Hub
if model_short_name == "mpnet":
    tokenizer = AutoTokenizer.from_pretrained(
        "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    )
    model = AutoModel.from_pretrained(
        "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        device_map=DEVICE,
        quantization_config=bitsandbytes_config,
    )
elif model_short_name == "minilm":
    tokenizer = AutoTokenizer.from_pretrained(
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    model = AutoModel.from_pretrained(
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        device_map=DEVICE,
        quantization_config=bitsandbytes_config,
    )


def process_batch(batch):
    sentences = [x[0]["content"] for x in batch["conversation"]]
    # Tokenize sentences
    encoded_input = tokenizer(
        sentences,
        # padding=True,
        truncation=True,
        return_tensors="pt",
        padding="max_length",
    )

    # Compute token embeddings
    with torch.no_grad():
        model_output = model(**encoded_input)

    # Perform pooling. In this case, average pooling
    sentence_embeddings = mean_pooling(model_output, encoded_input["attention_mask"])

    return {
        "sentence_embeddings": sentence_embeddings.cpu(),
    }


# Convert the dataset to Polars
pl_dataset: pl.DataFrame = dataset.to_polars()

# # Select only 50 conversations per language - this is for testing purposes
# pl_dataset = (
#     pl_dataset.group_by("lang")
#     .head(50)
#     .with_columns(pl.col("*").shuffle(seed=42))
# )

dataset = datasets.Dataset.from_polars(pl_dataset)

# Process the dataset in batches
dataset_with_embeddings = dataset.map(
    process_batch,
    batched=True,
    batch_size=1500,
)

print("Finished processing the dataset")

# Save the dataset locally
dataset_with_embeddings.save_to_disk(f"topic_analysis_{model_short_name}")

print("Saved the dataset to disk")

# # Push the dataset to the hub
# dataset_with_embeddings.push_to_hub(
#     "chatbot_conversations_in_european_languages", private=True
# )

# %%
# Load the dataset from disk
dataset_with_embeddings = datasets.load_from_disk(
    f"topic_analysis_{model_short_name}"
)

print(f"Loaded the dataset from disk: {dataset_with_embeddings}")

# %%

# Reduce the dimensionality of the embeddings to 2D with TSNE
dataset_with_embeddings.set_format(type="torch", columns=["sentence_embeddings"])

embeddings = (
    dataset_with_embeddings.select_columns("sentence_embeddings")["sentence_embeddings"]
    # .cpu()
    .numpy()
)
print("Obtained the embeddings")

languages = dataset_with_embeddings.select_columns("lang")["lang"]
print("Obtained the languages")

# Create a color map for the languages
unique_languages = dataset_with_embeddings.unique("lang")
color_map = {lang: idx for idx, lang in enumerate(unique_languages)}
colors = np.array([color_map[lang] for lang in languages]) / len(
    unique_languages
)  # All colors should be in the range [0, 1]


# %%
def plot_embeddings(embeddings, colors, model_short_name):
    """Plot the embeddings. Color them by the language of the conversation."""
    embeddings_2d = TSNE(n_components=2).fit_transform(embeddings)

    plt.figure(figsize=(10, 10))
    scatter = plt.scatter(
        embeddings_2d[:, 0], embeddings_2d[:, 1], c=colors, cmap="hsv", alpha=0.25
    )
    plt.axis("off")

    # Create a legend
    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label=lang,
            markerfacecolor=plt.cm.hsv(color_map[lang] / len(unique_languages)),
        )
        for lang in unique_languages
    ]
    plt.legend(handles, unique_languages, loc="best")

    # Save the plot
    plt.savefig(f"figures/topics/embeddings_{model_short_name}.pdf")
    plt.clf()


if os.environ.get("PLOT_EMBEDDINGS", "False") == "True":
    print("Plotting the embeddings")
    plot_embeddings(embeddings, colors, model_short_name)
else:
    print("Not plotting the embeddings")
# %%
# Do a silhouette analysis to determine whether the embeddings are well-separated. Do this on the original embeddings, not the 2D ones.

# As the calculation of the silhouette score is computationally expensive, we'll randomly sample 5% of the embeddings and calculate the silhouette score on them. We'll do this 20 times and average the results.

means = []
stds = []
    
df_avg_silhouette_scores = pl.DataFrame(schema={"lang": str, "avg_silhouette_score": float})
df_std_silhouette_scores = pl.DataFrame(schema={"lang": str, "std_silhouette_score": float})

languages = np.array(languages)

# Set the random seed
np.random.seed(42)

NUMBER_OF_SUBSETS_TO_CALCULATE = 20

for _ in range(NUMBER_OF_SUBSETS_TO_CALCULATE):
    print("Calculating silhouette scores")
    import sklearn.metrics
    import pandas as pd

    indices = np.random.choice(embeddings.shape[0], int(embeddings.shape[0] * 0.05))
    embeddings_sample = embeddings[indices]
    colors_sample = colors[indices]
    languages_sample = languages[indices]
    print(f"Calculating silhouette scores on {len(embeddings_sample)} samples")

    silhouette_samples = sklearn.metrics.silhouette_samples(
        embeddings_sample, colors_sample, metric="euclidean", n_jobs=32
    )

    # Get the avg and std silhouette score regardless of the language
    avg_silhouette_score = np.mean(silhouette_samples)
    std_silhouette_score = np.std(silhouette_samples)

    means.append(avg_silhouette_score)
    stds.append(std_silhouette_score)

    # Get the average silhouette score for each language
    avg_silhouette_scores = {
        lang: np.mean(silhouette_samples[np.array(languages_sample) == lang])
        for lang in unique_languages
    }
    # Remove nans
    avg_silhouette_scores = {
        lang: score for lang, score in avg_silhouette_scores.items() if not np.isnan(score)
    }
    std_silhouette_scores = {
        lang: np.std(silhouette_samples[np.array(languages_sample) == lang])
        for lang in unique_languages
    }
    # Remove nans
    std_silhouette_scores = {
        lang: score for lang, score in std_silhouette_scores.items() if not np.isnan(score)
    }

    print(f"Average of silhouette scores: {avg_silhouette_scores}")
    print(f"Standard deviation of silhouette scores: {std_silhouette_scores}")
    # Save the silhouette scores to a CSV file
    import polars as pl

    df_avg_silhouette_scores = df_avg_silhouette_scores.vstack(
        pl.from_records(
            list(iter(avg_silhouette_scores.items())),
            schema=["lang", "avg_silhouette_score"],
        )
    )
    df_std_silhouette_scores = df_std_silhouette_scores.vstack(
        pl.from_records(
            list(iter(std_silhouette_scores.items())),
            schema=["lang", "std_silhouette_score"],
        )
    )
# %%

df_avg_silhouette_scores = df_avg_silhouette_scores.group_by("lang").agg(pl.col("avg_silhouette_score").mean().alias("avg_silhouette_score"))
df_std_silhouette_scores = df_std_silhouette_scores.group_by("lang").agg(pl.col("std_silhouette_score").mean().alias("std_silhouette_score"))
    
df_avg_std_silhouette_scores = df_avg_silhouette_scores.join(
    df_std_silhouette_scores, on="lang"
)
df_avg_std_silhouette_scores.write_csv(
    f"results/topics/avg_std_silhouette_scores_{model_short_name}.csv"
)

# Save the silhouette score to a text file
with open(f"results/topics/silhouette_score_{model_short_name}.txt", "w") as f:
    f.write(
        f"Avg silhouette score: {avg_silhouette_score}\nStd silhouette score: {std_silhouette_score}\n"
    )
# %%
