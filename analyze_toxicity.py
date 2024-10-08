# %%
# Analyze the toxicity of the conversations, both from the user messages and the assistant responses

# %%
# The toxicity information is stored in the "openai_moderation" field of the dataset
# Load the dataset
import datasets
import torch

from utils import get_dataset_with_european_languages

dataset = datasets.load_dataset(
    "acmc/chatbot_conversations_in_european_languages", split="train"
)

# Change the column name "language" to "lang"
# There's no particular reason to do this, other than when exporting the CSV files and using them in LaTeX, with csvsimple, the column name "language" was causing problems when it's called exactly like that, adding an extra backtick - probably some sort of conflict with another package (super weird bug)
dataset = dataset.rename_column("language", "lang")

categories_of_toxicity = list(
    dataset.features["openai_moderation"][0]["category_scores"].keys()
)

# %%


# Treat the toxicity scores as different dimensions of a multi-dimensional space
def map_toxicity_scores_to_vector(example):
    # The openai_moderation field is a list - there's one entry for each message in the conversation
    return {
        "toxicity_vector": torch.tensor(
            [
                [
                    (
                        message_toxicity["category_scores"][category]
                        if message_toxicity["category_scores"][category] is not None
                        else 0.0
                    )
                    for category in categories_of_toxicity
                ]
                for message_toxicity in example["openai_moderation"]
            ]
        )
    }


dataset_with_toxicity_vectors = dataset.map(
    map_toxicity_scores_to_vector,
    num_proc=32,
)

# %%
# Now, we can analyze the toxicity of the conversations. Visualize the toxicity vectors in 2D, colored by the language of the conversation ("lang" field in the dataset).

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
import umap


def reduce_toxicity_vectors_to_2d(toxicity_vectors):
    # Use UMAP to reduce the dimensionality of the toxicity vectors to 2D
    reducer = umap.UMAP(n_components=2, random_state=42)
    toxicity_vectors_2d = reducer.fit_transform(toxicity_vectors)
    return toxicity_vectors_2d


# Extract the languages from the dataset
languages = dataset_with_toxicity_vectors["lang"]
unique_languages = list(set(languages))
unique_languages.sort()
color_map = {lang: idx for idx, lang in enumerate(unique_languages)}
colors = np.array([color_map[lang] for lang in languages]) / len(
    unique_languages
)  # All colors should be in the range [0, 1]


def plot_toxicity_vectors(toxicity_vectors_2d, user_toxicity_language_colors, figname):

    # Create a color map for the languages (around 30)
    scatter = plt.scatter(
        toxicity_vectors_2d[:, 0],
        toxicity_vectors_2d[:, 1],
        c=user_toxicity_language_colors,
        cmap="hsv",
        alpha=0.25,
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
    plt.savefig(figname)
    plt.clf()


def violin_plot_toxicity_vectors(
    toxicity_vectors, user_toxicity_language_colors, figname, agg
):
    # Create a violin plot for the toxicity vectors, separately for each language (color)
    plt.figure(figsize=(10, 3))
    plt.tight_layout()
    # Get the maximum toxicity score in the vectors
    max_toxicity = agg(toxicity_vectors, axis=1)
    for i, lang in enumerate(unique_languages):
        plt.violinplot(
            max_toxicity[user_toxicity_language_colors == i],
            positions=[i + 1],
            showmeans=False,
            showmedians=True,
            showextrema=False,
            widths=0.9,
        )
    plt.xticks(range(1, len(unique_languages) + 1), unique_languages, rotation=90)
    # plt.ylabel("Toxicity")
    # plt.xlabel("lang")
    # plt.title("Toxicity of User Messages by Language")
    plt.tight_layout()
    plt.savefig(figname)
    plt.clf()


def stack_toxicity_vectors_for_user_and_assistant(example):
    # Stack the toxicity vectors for the user messages and the assistant responses
    user_toxicity_vectors = []
    assistant_toxicity_vectors = []
    for i in range(len(example["conversation"])):
        role = example["conversation"][i]["role"]
        if role == "user":
            user_toxicity_vectors.append(example["toxicity_vector"][i])
        elif role == "assistant":
            assistant_toxicity_vectors.append(example["toxicity_vector"][i])
    return {
        "user_toxicity_vectors": torch.tensor(user_toxicity_vectors),
        "user_toxicity_language_colors": [
            color_map[example["lang"]] for _ in user_toxicity_vectors
        ],
        "assistant_toxicity_vectors": torch.tensor(assistant_toxicity_vectors),
        "assistant_toxicity_language_colors": [
            color_map[example["lang"]] for _ in assistant_toxicity_vectors
        ],
    }


dataset_with_toxicity_vectors = dataset_with_toxicity_vectors.map(
    stack_toxicity_vectors_for_user_and_assistant,
    num_proc=32,
)

# %%
# Now, stack the toxicity vectors for the user messages and the assistant responses
user_toxicity_vectors = np.concatenate(
    dataset_with_toxicity_vectors["user_toxicity_vectors"]
)
user_toxicity_language_colors = np.concatenate(
    dataset_with_toxicity_vectors["user_toxicity_language_colors"]
)

# Print the shape of the user toxicity vectors
print(user_toxicity_vectors.shape)

# %%
# Plot the toxicity vectors for the user messages
user_toxicity_vectors_2d = reduce_toxicity_vectors_to_2d(user_toxicity_vectors)
print(f"Reduced user toxicity vectors to 2D: {user_toxicity_vectors_2d.shape}")
plot_toxicity_vectors(
    user_toxicity_vectors_2d,
    user_toxicity_language_colors,
    figname="figures/toxicity/user_toxicity.pdf",
)
# %%
violin_plot_toxicity_vectors(
    user_toxicity_vectors,
    user_toxicity_language_colors,
    figname="figures/toxicity/user_mean_toxicity_violin.pdf",
    agg=np.mean,
)
violin_plot_toxicity_vectors(
    user_toxicity_vectors,
    user_toxicity_language_colors,
    figname="figures/toxicity/user_max_toxicity_violin.pdf",
    agg=np.max,
)
# %%
# Now, do the same for the assistant responses
assistant_toxicity_vectors = np.concatenate(
    dataset_with_toxicity_vectors["assistant_toxicity_vectors"]
)
assistant_toxicity_language_colors = np.concatenate(
    dataset_with_toxicity_vectors["assistant_toxicity_language_colors"]
)

# Print the shape of the assistant toxicity vectors
print(assistant_toxicity_vectors.shape)

# %%
# Plot the toxicity vectors for the assistant responses
assistant_toxicity_vectors_2d = reduce_toxicity_vectors_to_2d(
    assistant_toxicity_vectors
)
print(
    f"Reduced assistant toxicity vectors to 2D: {assistant_toxicity_vectors_2d.shape}"
)
plot_toxicity_vectors(
    assistant_toxicity_vectors_2d,
    assistant_toxicity_language_colors,
    figname="figures/toxicity/assistant_toxicity.pdf",
)
# %%
violin_plot_toxicity_vectors(
    assistant_toxicity_vectors,
    assistant_toxicity_language_colors,
    figname="figures/toxicity/assistant_mean_toxicity_violin.pdf",
    agg=np.mean,
)
violin_plot_toxicity_vectors(
    assistant_toxicity_vectors,
    assistant_toxicity_language_colors,
    figname="figures/toxicity/assistant_max_toxicity_violin.pdf",
    agg=np.max,
)


# %% [markdown]
# # Analyze the toxicity of the conversations by using the Kruskal-Wallis test


# %%
# Get a dataframe with columns: "lang", "toxicity_category", "avg_user_toxicity", "avg_assistant_toxicity", "std_user_toxicity", "std_assistant_toxicity"
import polars as pl

# dataset = dataset.select(range(50))
dataset_df: pl.DataFrame = dataset.to_polars()
# %%
dataset_df_exp = dataset_df.explode(["conversation", "openai_moderation"])

print(f"Number of messages in the dataset: {len(dataset_df_exp)}")

dataset_df_exp = dataset_df_exp.with_columns(
    pl.col("openai_moderation")
    .map_elements(lambda x: x["category_scores"], return_dtype=pl.Object)
    .alias("category_scores")
)
# Explode the category scores horizontally - i.e. create a column for each category
for category in categories_of_toxicity:
    dataset_df_exp = dataset_df_exp.with_columns(
        pl.col("category_scores")
        .map_elements(lambda x: x[category], return_dtype=pl.Float64)
        .alias(f"{category}_score")
    )

# Discard the rows with nan values in the toxicity scores
dataset_df_exp = dataset_df_exp.select(pl.col("*").drop_nans()).drop_nulls()

# Explode the conversation field - i.e. create a column message and a column role
dataset_df_exp = dataset_df_exp.with_columns(
    pl.col("conversation")
    .map_elements(lambda x: x["content"], return_dtype=pl.Utf8)
    .alias("message"),
    pl.col("conversation")
    .map_elements(lambda x: x["role"], return_dtype=pl.Utf8)
    .alias("role"),
)

# Only keep the columns we need
dataset_df_exp = dataset_df_exp.select(
    [
        "lang",
        "message",
        "role",
        *[f"{category}_score" for category in categories_of_toxicity],
    ]
)

# Add a column with the maximum toxicity score for each message - this is the maximum of the toxicity scores across all categories
dataset_df_exp = dataset_df_exp.with_columns(
    pl.max_horizontal(
        [f"{category}_score" for category in categories_of_toxicity]
    ).alias("max_toxicity_score")
)

# Add a column with the average toxicity score for each message - this is the average of the toxicity scores across all categories
dataset_df_exp = dataset_df_exp.with_columns(
    pl.mean_horizontal(
        [f"{category}_score" for category in categories_of_toxicity]
    ).alias("avg_toxicity_score")
)

# Duplicate the rows above, but changing the role to "all". So we perform the analysis for "user", "assistant" and "all"
dataset_df_exp = dataset_df_exp.extend(
    dataset_df_exp.with_columns(pl.col("role").map_elements(lambda x: "all"))
)

# %%
# Now, group by language and toxicity category
grouped_df = dataset_df_exp.group_by(["lang", "role"])

# %%
# Now, get the average toxicity scores for each language and role
avg_toxicity_df = grouped_df.agg(
    [
        pl.col(f"{category}_score").mean().alias(f"avg_{category}_score")
        for category in categories_of_toxicity
    ]
    + [pl.col("max_toxicity_score").mean().alias("avg_max_toxicity_score")]
    + [pl.col("avg_toxicity_score").mean().alias("avg_avg_toxicity_score")]
)

std_toxicity_df = grouped_df.agg(
    [
        pl.col(f"{category}_score").std().alias(f"std_{category}_score")
        for category in categories_of_toxicity
    ]
    + [pl.col("max_toxicity_score").std().alias("std_max_toxicity_score")]
    + [pl.col("avg_toxicity_score").std().alias("std_avg_toxicity_score")]
)

# %%
# Merge the two dataframes
toxicity_df = avg_toxicity_df.join(std_toxicity_df, on=["lang", "role"])

# %%
# Sort the dataframe by the language and role
toxicity_df = toxicity_df.sort(["lang", "role"])
# Round to 5 decimal places
# Save the dataframe to a CSV file
toxicity_df.with_columns(
    pl.exclude("lang", "role").round(5),
    pl.col("lang"),
    pl.col("role"),
).write_csv("results/toxicity/toxicity_analysis.csv")
# %% [markdown]
# Now, let's check - is there a statistically significant difference in the toxicity messages?
# %%
# First, shuffle the dataset
dataset_df_exp = dataset_df_exp.with_columns(pl.col("*").shuffle(seed=42))
# dataset_df_exp = dataset_df_exp.group_by("lang").head(NUM_MESSAGES_PER_LANGUAGE)
# Kruskal-Wallis is vulnerable to small sample sizes, so we need to have a minimum number of messages per language. Let's set it to 100
MIN_NUM_MESSAGES_PER_LANGUAGE = 5
dataset_df_exp_filtered = dataset_df_exp.filter(
    pl.col("lang").is_in(
        dataset_df_exp.group_by("lang")
        .agg(pl.count("lang").alias("num_messages"))
        .filter(pl.col("num_messages") >= MIN_NUM_MESSAGES_PER_LANGUAGE)
        .select("lang")
    )
)
# Do this analysis for user messages only
user_messages_df = dataset_df_exp_filtered.filter(pl.col("role") == "user")
# What is we only consider English, French and Spanish?
# user_messages_df = user_messages_df.filter(pl.col("lang").is_in(['English', 'French', 'Spanish']))
# Do a Kruksal-Wallis test for each category of toxicity
from scipy.stats import kruskal

# Round the toxicity scores to 2 decimal place
for category in categories_of_toxicity:
    user_messages_df = user_messages_df.with_columns(
        pl.col(f"{category}_score").round(2)
    )
# Also round the maximum toxicity score
user_messages_df = user_messages_df.with_columns(pl.col("max_toxicity_score").round(2))
user_messages_df = user_messages_df.with_columns(pl.col("avg_toxicity_score").round(2))

# Group the user messages by language
grouped_user_messages_df = user_messages_df.group_by(["lang"])

kruskal_wallis_results = []

# Perform the Kruskal-Wallis test for each category of toxicity
for category in categories_of_toxicity:
    # Get the toxicity scores for the category
    toxicity_scores = (
        grouped_user_messages_df.all().select(f"{category}_score").to_numpy()
    )
    # This gives us a list of arrays - one array for
    # each language, containing the toxicity scores for that language
    # Flatten the list of arrays
    toxicity_scores = [x[0] for x in toxicity_scores.tolist()]
    # Perform the Kruskal-Wallis test
    kruskal_result = kruskal(*toxicity_scores)
    print(f"Kruskal-Wallis test for {category}: {kruskal_result}")
    kruskal_wallis_results.append(
        {
            "toxicity_category": category,
            "H-statistic": kruskal_result.statistic,
            "p-value": kruskal_result.pvalue,
        }
    )

# Calculate it for the maximum toxicity score
toxicity_scores = grouped_user_messages_df.all().select("max_toxicity_score").to_numpy()
toxicity_scores = [x[0] for x in toxicity_scores.tolist()]
kruskal_result = kruskal(*toxicity_scores)
print(f"Kruskal-Wallis test for max toxicity score: {kruskal_result}")
kruskal_wallis_results.append(
    {
        "toxicity_category": "max_toxicity_score",
        "H-statistic": kruskal_result.statistic,
        "p-value": kruskal_result.pvalue,
    }
)
# Calculate it for the average toxicity score
toxicity_scores = grouped_user_messages_df.all().select("avg_toxicity_score").to_numpy()
toxicity_scores = [x[0] for x in toxicity_scores.tolist()]
kruskal_result = kruskal(*toxicity_scores)
print(f"Kruskal-Wallis test for avg toxicity score: {kruskal_result}")
kruskal_wallis_results.append(
    {
        "toxicity_category": "avg_toxicity_score",
        "H-statistic": kruskal_result.statistic,
        "p-value": kruskal_result.pvalue,
    }
)
# Save the results to a CSV file
import polars as pl

# Three columns: "toxicity_category", "H-statistic", "p-value"
kruskal_wallis_df = pl.from_records(kruskal_wallis_results)
kruskal_wallis_df.write_csv("results/toxicity/kruskal_wallis_test_user_messages.csv")
# %%
# The above analysis shows that there is a statistically significant difference in the toxicity of the messages across different languages.
# So now we'll do a pairwise comparison to determine which languages are significantly different from each other.
# We'll use the Dunn's test for this.
from scikit_posthocs import posthoc_dunn
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib

# Color values below 0.05 in F600FF, above 0.05 in 1145D4
my_cmap = matplotlib.colors.ListedColormap(["#F600FF", "#00376E"])
# Create a colorbar
bounds = [0, 0.05, 1]
norm = matplotlib.colors.BoundaryNorm(bounds, my_cmap.N)


# Perform the Dunn's test for each category of toxicity
for category in categories_of_toxicity:
    # Get the toxicity scores for the category
    toxicity_scores = (
        grouped_user_messages_df.all()
        .select("lang", f"{category}_score")
        .to_pandas()
        .explode(f"{category}_score")
    )
    # Perform the Dunn's test
    dunn_result = posthoc_dunn(
        toxicity_scores,
        val_col=f"{category}_score",
        group_col="lang",
        p_adjust="bonferroni",
    )
    # Matrix of boolean values - True if the difference is significant, False otherwise
    significant_differences = dunn_result < 0.05
    print(f"Dunn's test for {category}:\n{dunn_result}\n{significant_differences}")
    # Save to a CSV file
    dunn_result.to_csv(
        f"results/toxicity/dunns_test_user_messages_{category.replace('/', '_')}.csv",
        index_label="lang",
        index=True,
    )
    # Plot a heatmap of the p-values

    plt.figure(figsize=(15, 10))
    sns.heatmap(dunn_result, annot=True, fmt=".1f", cmap=my_cmap, norm=norm)
    plt.savefig(
        f"figures/toxicity/dunns_test_user_messages_{category.replace('/', '_')}_heatmap.pdf"
    )
# %%
# Calculate it for the maximum toxicity score
toxicity_scores = (
    grouped_user_messages_df.all()
    .select("lang", "max_toxicity_score")
    .to_pandas()
    .explode("max_toxicity_score")
)
dunn_result = posthoc_dunn(
    toxicity_scores,
    val_col="max_toxicity_score",
    group_col="lang",
    p_adjust="bonferroni",
)
significant_differences = dunn_result < 0.05
print(f"Dunn's test for max toxicity score:\n{dunn_result}\n{significant_differences}")
# Save to a CSV file
dunn_result.to_csv(
    "results/toxicity/dunns_test_user_messages_max_toxicity.csv",
    index_label="lang",
    index=True,
)
# Plot a heatmap of the p-values

plt.figure(figsize=(15, 10))
sns.heatmap(dunn_result, annot=True, fmt=".1f", cmap=my_cmap, norm=norm)
plt.savefig("figures/toxicity/dunns_test_user_messages_max_toxicity_heatmap.pdf")
# %%
# Calculate it for the average toxicity score
toxicity_scores = (
    grouped_user_messages_df.all()
    .select("lang", "avg_toxicity_score")
    .to_pandas()
    .explode("avg_toxicity_score")
)
dunn_result = posthoc_dunn(
    toxicity_scores,
    val_col="avg_toxicity_score",
    group_col="lang",
    p_adjust="bonferroni",
)
significant_differences = dunn_result < 0.05
print(f"Dunn's test for avg toxicity score:\n{dunn_result}\n{significant_differences}")
# Save to a CSV file
dunn_result.to_csv(
    "results/toxicity/dunns_test_user_messages_avg_toxicity.csv",
    index_label="lang",
    index=True,
)
# Plot a heatmap of the p-values

plt.figure(figsize=(15, 10))
sns.heatmap(dunn_result, annot=True, fmt=".1f", cmap=my_cmap, norm=norm)
plt.savefig("figures/toxicity/dunns_test_user_messages_avg_toxicity_heatmap.pdf")
# %%
