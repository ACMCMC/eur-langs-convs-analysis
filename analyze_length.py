# %% [markdown]
# # Analysis of length
# In this file, we'll take the conversations dataset and analyze the length of the conversations across the different languages, as well as the number of messages per conversation.
# %%
import datasets
import polars as pl
import matplotlib.pyplot as plt

dataset = datasets.load_dataset(
    "acmc/chatbot_conversations_in_european_languages", split="train"
)

# Change the column name "language" to "lang"
# There's no particular reason to do this, other than when exporting the CSV files and using them in LaTeX, with csvsimple, the column name "lang" was causing problems when it's called exactly like that, adding an extra backtick - probably some sort of conflict with another package (super weird bug)
dataset = dataset.rename_column("language", "lang")


pl_dataset = dataset.to_polars()
examples_per_language = pl_dataset.group_by("lang")
# Write a CSV file with the number of conversations per language
num_conversations_per_language_df = examples_per_language.agg(
    pl.col("lang").count().alias("num_examples_count")
).sort("num_examples_count", descending=True)
num_conversations_per_language_df.write_csv(
    "results/length/num_conversations_per_language.csv"
)

# Plot the number of examples per language as a bar plot
plt.figure(figsize=(15, 5))
plt.bar(
    num_conversations_per_language_df["lang"],
    num_conversations_per_language_df["num_examples_count"],
    color="#F600FF",
    alpha=0.7,
)
# Log scale for the number of examples
plt.yscale("log")
# Rotate the x-axis labels
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig("figures/length/num_conversations_per_language.pdf")
# Show the plot
plt.show()
# Clear the plot
plt.clf()

# There is already a column with the number of messages (num_messages), another with the number of user words (num_user_words), and another with the number of assistant words (num_assistant_words)
pl_avg_dataset = examples_per_language.agg(
    pl.col("num_messages").mean().alias("avg_num_messages"),
    pl.col("num_messages").std().alias("std_num_messages"),
    pl.col("num_user_words").mean().alias("avg_num_user_words"),
    pl.col("num_user_words").std().alias("std_num_user_words"),
    pl.col("num_assistant_words").mean().alias("avg_num_assistant_words"),
    pl.col("num_assistant_words").std().alias("std_num_assistant_words"),
).sort("lang")

# Write to a CSV file
pl_avg_dataset.write_csv("results/length/length_analysis.csv")

# Plot the average number of messages per language
plt.figure(figsize=(15, 5))
pl_dataset.plot.bar(x="lang", y="num_messages")
plt.tight_layout()
# Rotate the x-axis labels
plt.xticks(rotation=90)
plt.savefig("figures/length/num_messages_per_language.pdf")
# Show the plot
plt.show()
# Clear the plot
plt.clf()

# Scatter plot: number of user words vs number of examples per language
number_of_user_words_per_language = examples_per_language.agg(
    pl.col("num_user_words").mean()
)
num_examples_per_language = examples_per_language.count().rename(
    {"count": "num_examples"}
)
user_words_vs_examples = number_of_user_words_per_language.join(
    num_examples_per_language, on="lang"
)
plt.scatter(
    user_words_vs_examples["num_user_words"],
    user_words_vs_examples["num_examples"],
    # Color: F600FF, alpha = 0.5
    c="#F600FF",
    alpha=0.7,
    # Shape: x
    marker="x",
)
# Log scale for the number of examples
plt.yscale("log")
# Show the language name for each point. (optionally) Only use the XX% top languages (quantile 0.XX)
langs_to_show = user_words_vs_examples.filter(
    pl.col("num_examples") >= user_words_vs_examples["num_examples"].quantile(0.00)
)
for i, txt in enumerate(user_words_vs_examples["lang"]):
    if txt in langs_to_show["lang"]:
        plt.annotate(
            txt,
            (
                user_words_vs_examples["num_user_words"][i],
                user_words_vs_examples["num_examples"][i],
            ),
            # Small text size
            fontsize=6,
            # Center the text
            ha="center",
            # Also vertically center the text
            va="center",
            # Background color: white
            # bbox=dict(facecolor="white", alpha=0.5),
        )
plt.xlabel("Number of user words")
plt.ylabel("Number of examples")
plt.savefig("figures/length/user_words_vs_examples.pdf")
plt.show()
plt.clf()

# %% [markdown]
# What is the correlation between the number of user words and the number of examples?
# We can calculate the Pearson correlation coefficient and the Spearman Rank correlation coefficient between the number of user words and the number of examples to see if there is a correlation between the two variables.
# %%
# Calculate the Pearson correlation coefficient and the Spearman Rank correlation coefficient
import scipy.stats

# We shouldn't use Pearson's correlation coefficient because the data is not normally distributed
pearson_corr = scipy.stats.pearsonr(
    user_words_vs_examples["num_user_words"], user_words_vs_examples["num_examples"]
)
print(f"Pearson correlation coefficient: {pearson_corr}")

spearman_corr = scipy.stats.spearmanr(
    user_words_vs_examples["num_user_words"], user_words_vs_examples["num_examples"]
)
print(f"Spearman Rank correlation coefficient: {spearman_corr}")

# Total number of data points
print(f"Total number of data points: {len(user_words_vs_examples)}")
# %%
# Now, do a scatter plot of the number of assistant words vs the number of user words per language
user_words_vs_assistant_words = examples_per_language.agg(
    pl.col("num_user_words").mean(), pl.col("num_assistant_words").mean()
)
# Color of the points: number of examples
num_examples_per_language = examples_per_language.count().rename(
    {"count": "num_examples"}
)
user_words_vs_assistant_words = user_words_vs_assistant_words.join(
    num_examples_per_language, on="lang"
)
plt.scatter(
    user_words_vs_assistant_words["num_user_words"],
    user_words_vs_assistant_words["num_assistant_words"],
    c=user_words_vs_assistant_words["num_examples"],
)
plt.colorbar()
plt.xlabel("Number of user words")
plt.ylabel("Number of assistant words")
# Show the language name for each point. Only use the 90% top languages (percentile 0.90)
langs_to_show = user_words_vs_assistant_words.filter(
    pl.col("num_examples")
    >= user_words_vs_assistant_words["num_examples"].quantile(0.90)
)
for i, txt in enumerate(user_words_vs_assistant_words["lang"]):
    if txt in langs_to_show["lang"]:
        plt.annotate(
            txt,
            (
                user_words_vs_assistant_words["num_user_words"][i],
                user_words_vs_assistant_words["num_assistant_words"][i],
            ),
        )
plt.savefig("figures/length/user_words_vs_assistant_words.pdf")
plt.show()
plt.clf()
# %%