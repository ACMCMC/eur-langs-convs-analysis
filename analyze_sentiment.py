# %% [markdown]
# # Tests for sentiment analysis
# We run the first user message and the first assistant message through a sentiment analysis model to check for the sentiment of the conversation.
# %%
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import datasets
import tqdm
import polars as pl

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
# There's no particular reason to do this, other than when exporting the CSV files and using them in LaTeX, with csvsimple, the column name "language" was causing problems when it's called exactly like that, adding an extra backtick - probably some sort of conflict with another package (super weird bug)
dataset = dataset.rename_column("language", "lang")

# %%

# Generate the response
# Use a pipeline as a high-level helper
from transformers import pipeline
from transformers.pipelines.pt_utils import KeyDataset
import transformers

pipe = pipeline(
    "text-classification",
    model="cardiffnlp/twitter-xlm-roberta-base-sentiment",
    device="cuda",
)

# # Convert the dataset to Polars
# pl_dataset: pl.DataFrame = dataset.to_polars()

# # Select only some conversations per language
# pl_dataset = pl_dataset.group_by("language").head(1000)

# dataset = datasets.Dataset.from_polars(pl_dataset)

ds_first_messages = dataset.map(
    lambda x: {"first_user_message": x["conversation"][0]["content"]}, num_proc=32
)
ds_first_messages = ds_first_messages.map(
    lambda x: {"first_assistant_message": x["conversation"][1]["content"]}, num_proc=32
)

# Only do this is the column doesn't exist
if "user_sentiment" not in ds_first_messages.column_names:
    output_user = list(
        tqdm.tqdm(
            pipe(
                KeyDataset(ds_first_messages, "first_user_message"),
                batch_size=1024,
                truncation=True,
                max_length=512,
                return_all_scores=True,
            ),
            total=len(ds_first_messages),
        )
    )

    dataset_with_user_sentiment = dataset.add_column("user_sentiment", output_user)
    # Push the dataset to the hub
    dataset_with_user_sentiment.rename_column("lang", "language").push_to_hub(
        "chatbot_conversations_in_european_languages", private=True
    )
else:
    dataset_with_user_sentiment = dataset

# Only do this is the column doesn't exist
if "assistant_sentiment" not in dataset_with_user_sentiment.column_names:
    output_assistant = list(
        tqdm.tqdm(
            pipe(
                KeyDataset(ds_first_messages, "first_assistant_message"),
                batch_size=1024,
                truncation=True,
                max_length=512,
                return_all_scores=True,
            ),
            total=len(ds_first_messages),
        )
    )

    dataset_with_all_sentiment = dataset_with_user_sentiment.add_column(
        "assistant_sentiment", output_assistant
    )
    # Push the dataset to the hub
    dataset_with_all_sentiment.rename_column("lang", "language").push_to_hub(
        "chatbot_conversations_in_european_languages", private=True
    )

# %%
import datasets
# Load the dataset from the hub
dataset_with_all_sentiment = datasets.load_dataset(
    "acmc/chatbot_conversations_in_european_languages", split="train"
)

# Rename the column "language" to "lang"
dataset_with_all_sentiment = dataset_with_all_sentiment.rename_column("language", "lang")

# %%
# Plot the sentiment distribution
import matplotlib.pyplot as plt
import seaborn as sns

# Now, the column user_sentiment contains the sentiment of the first user message, with this format:
# [{'label': 'negative', 'score': 0.2781553566455841},
#  {'label': 'neutral', 'score': 0.5391951203346252},
#  {'label': 'positive', 'score': 0.18264950811862946}]

# sns.histplot(
#     dataset_with_all_sentiment.to_pandas(),
#     x="user_sentiment",
#     hue="language",
#     multiple="stack",
#     bins=20,
# )

# plt.show()

# %%
import polars as pl
# Dataframe with the percentage of each sentiment per language
df_mean_sentiment = dataset_with_all_sentiment.to_polars()
# Turn the user_sentiment column, which is a list of dictionaries, into three columns
df_mean_sentiment = df_mean_sentiment.select(
    pl.col("user_sentiment")
    .list.get(0)
    .struct.field("score")
    .alias("negative_user_sentiment"),
    pl.col("user_sentiment")
    .list.get(1)
    .struct.field("score")
    .alias("neutral_user_sentiment"),
    pl.col("user_sentiment")
    .list.get(2)
    .struct.field("score")
    .alias("positive_user_sentiment"),
    pl.col("assistant_sentiment")
    .list.get(0)
    .struct.field("score")
    .alias("negative_assistant_sentiment"),
    pl.col("assistant_sentiment")
    .list.get(1)
    .struct.field("score")
    .alias("neutral_assistant_sentiment"),
    pl.col("assistant_sentiment")
    .list.get(2)
    .struct.field("score")
    .alias("positive_assistant_sentiment"),
    pl.exclude("user_sentiment"),
)
df_mean_sentiment_per_language = df_mean_sentiment.group_by(["lang"]).agg(
    pl.mean("negative_user_sentiment").alias("negative_user_sentiment_avg"),
    pl.mean("neutral_user_sentiment").alias("neutral_user_sentiment_avg"),
    pl.mean("positive_user_sentiment").alias("positive_user_sentiment_avg"),
    pl.std("negative_user_sentiment").alias("negative_user_sentiment_std"),
    pl.std("neutral_user_sentiment").alias("neutral_user_sentiment_std"),
    pl.std("positive_user_sentiment").alias("positive_user_sentiment_std"),
    pl.mean("negative_assistant_sentiment").alias("negative_assistant_sentiment_avg"),
    pl.mean("neutral_assistant_sentiment").alias("neutral_assistant_sentiment_avg"),
    pl.mean("positive_assistant_sentiment").alias("positive_assistant_sentiment_avg"),
    pl.std("negative_assistant_sentiment").alias("negative_assistant_sentiment_std"),
    pl.std("neutral_assistant_sentiment").alias("neutral_assistant_sentiment_std"),
    pl.std("positive_assistant_sentiment").alias("positive_assistant_sentiment_std"),
    # Also add the count of each sentiment
    pl.count("negative_user_sentiment").alias("negative_user_sentiment_count"),
    pl.count("neutral_user_sentiment").alias("neutral_user_sentiment_count"),
    pl.count("positive_user_sentiment").alias("positive_user_sentiment_count"),
    pl.count("negative_assistant_sentiment").alias(
        "negative_assistant_sentiment_count"
    ),
    pl.count("neutral_assistant_sentiment").alias("neutral_assistant_sentiment_count"),
    pl.count("positive_assistant_sentiment").alias(
        "positive_assistant_sentiment_count"
    ),
    # Difference between the assistant and the user sentiment
    pl.mean("negative_assistant_sentiment").sub(
        pl.mean("negative_user_sentiment")
    ).alias("negative_diff"),
    pl.mean("neutral_assistant_sentiment").sub(
        pl.mean("neutral_user_sentiment")
    ).alias("neutral_diff"),
    pl.mean("positive_assistant_sentiment").sub(
        pl.mean("positive_user_sentiment")
    ).alias("positive_diff"),
)
# Sort by language
df_mean_sentiment_per_language = df_mean_sentiment_per_language.sort("lang")
# Save to a CSV file
df_mean_sentiment_per_language.write_csv("results/sentiment/sentiment_analysis.csv")

df_mean_sentiment_not_grouped_by_language = df_mean_sentiment.select(
    pl.mean("negative_user_sentiment").alias("negative_user_sentiment_avg"),
    pl.mean("neutral_user_sentiment").alias("neutral_user_sentiment_avg"),
    pl.mean("positive_user_sentiment").alias("positive_user_sentiment_avg"),
    pl.std("negative_user_sentiment").alias("negative_user_sentiment_std"),
    pl.std("neutral_user_sentiment").alias("neutral_user_sentiment_std"),
    pl.std("positive_user_sentiment").alias("positive_user_sentiment_std"),
    pl.mean("negative_assistant_sentiment").alias("negative_assistant_sentiment_avg"),
    pl.mean("neutral_assistant_sentiment").alias("neutral_assistant_sentiment_avg"),
    pl.mean("positive_assistant_sentiment").alias("positive_assistant_sentiment_avg"),
    pl.std("negative_assistant_sentiment").alias("negative_assistant_sentiment_std"),
    pl.std("neutral_assistant_sentiment").alias("neutral_assistant_sentiment_std"),
    pl.std("positive_assistant_sentiment").alias("positive_assistant_sentiment_std"),
    # Difference between the assistant and the user sentiment
    pl.mean("negative_assistant_sentiment").sub(
        pl.mean("negative_user_sentiment")
    ).alias("negative_diff"),
    pl.mean("neutral_assistant_sentiment").sub(
        pl.mean("neutral_user_sentiment")
    ).alias("neutral_diff"),
    pl.mean("positive_assistant_sentiment").sub(
        pl.mean("positive_user_sentiment")
    ).alias("positive_diff"),
)

# Save to a CSV file
df_mean_sentiment_not_grouped_by_language.write_csv(
    "results/sentiment/sentiment_analysis_not_grouped_by_language.csv"
)

# %%
# Plot the results as a bar plot, with the percentage of sentiment labels per language stacked (3 bars per language)
import matplotlib.pyplot as plt

# Read the CSV file
df_mean_sentiment = pl.read_csv("results/sentiment/sentiment_analysis.csv")

# Colors: F600FF, 00376E, 00FBFF
import matplotlib.colors as mcolors

colors = [
    "#F600FF",
    "#00376E",
    "#00FBFF",
]

# Plot the results
fig, ax = plt.subplots(figsize=(15, 4))
pandas_dataset_percentages = df_mean_sentiment.sort(
    "positive_user_sentiment_avg", descending=True
).to_pandas()
pandas_dataset_percentages.plot(
    x="lang",
    y=[
        "positive_user_sentiment_avg",
        "neutral_user_sentiment_avg",
        "negative_user_sentiment_avg",
    ],
    kind="bar",
    stacked=True,
    ax=ax,
    color=colors,
)

# Add counts on top of the bars
for i, col_name in enumerate(
    [
        "positive_user_sentiment_count",
        "neutral_user_sentiment_count",
        "negative_user_sentiment_count",
    ]
):
    for j in range(len(pandas_dataset_percentages)):
        value = pandas_dataset_percentages[col_name][j]
        # If it's 0, don't add the text
        if value == 0:
            continue
        language = pandas_dataset_percentages["lang"][j]
        p = ax.patches[i * len(pandas_dataset_percentages) + j]
        width, height = p.get_width(), p.get_height()
        x, y = p.get_xy()
        ax.text(
            x + width / 2,
            y + height / 2,
            f"{value}",
            ha="center",
            va="center",
            # White background
            bbox=dict(facecolor="white", alpha=0.8),
        )

plt.ylabel("")
plt.xlabel("")
plt.legend(
    [
        "Positive",
        "Neutral",
        "Negative",
    ],
    loc="upper right",
)
plt.tight_layout()
plt.savefig("figures/sentiment/sentiment_analysis_user.pdf")
plt.show()
plt.clf()

# %%
# Do the same for the assistant sentiment
