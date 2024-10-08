# %% [markdown]
# # Analysis of satisfaction
# In this file, we'll take the conversations dataset, having been tagged already with a "satisfaction_label" and analyze the satisfaction of the users across the different languages.
# The "satisfaction_label" is a categorical variable that can be one of the following:
# - "YES"
# - "NO"
# - "NA"
# %%
# First, generate CSV files with the satisfaction label classes per language
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

# Filter out examples where the "satisfaction_label" is None
dataset = dataset.filter(lambda x: x["satisfaction_label"] is not None, num_proc=32)

# Convert the dataset to Polars
pl_dataset: pl.DataFrame = dataset.to_polars()

# Group by language, add three columns: "count_yes", "count_no", "count_na"
pl_dataset_grouped = pl_dataset.pivot(
    on="satisfaction_label",
    index="lang",
    values="satisfaction_label",
    aggregate_function=pl.element().count(),
).with_columns(total=pl.sum_horizontal(["YES", "NO", "NA"]))

# Replace null values with 0
pl_dataset_grouped = pl_dataset_grouped.fill_null(0)

pl_dataset_percentages = pl_dataset_grouped.select(
    lang=pl.col("lang"),
    count_yes=pl.col("YES"),
    count_no=pl.col("NO"),
    count_na=pl.col("NA"),
    total=pl.col("total"),
).with_columns(
    percentage_yes=pl.col("count_yes") / pl.col("total"),
    percentage_no=pl.col("count_no") / pl.col("total"),
    percentage_na=pl.col("count_na") / pl.col("total"),
)

# Order by "lang" descending
pl_dataset_percentages = pl_dataset_percentages.sort("lang", descending=True)

# Export the dataset to a CSV file
pl_dataset_percentages.write_csv("results/satisfaction/satisfaction_label_per_language.csv")
# %%
# Plot the results as a bar plot, with the percentage of satisfaction labels per language stacked (3 bars per language)
import matplotlib.pyplot as plt

# Read the CSV file
pl_dataset_percentages = pl.read_csv("results/satisfaction/satisfaction_label_per_language.csv")

# Colors: F600FF, 00376E, 00FBFF
import matplotlib.colors as mcolors

colors = [
    "#F600FF",
    "#00376E",
    "#00FBFF",
]

# Plot the results
fig, ax = plt.subplots(figsize=(15, 4))
pandas_dataset_percentages = pl_dataset_percentages.sort(
    "percentage_yes", descending=True
).to_pandas()
pandas_dataset_percentages.plot(
    x="lang",
    y=["percentage_yes", "percentage_no", "percentage_na"],
    kind="bar",
    stacked=True,
    ax=ax,
    color=colors,
)

# Add counts on top of the bars
for i, col_name in enumerate(["count_yes", "count_no", "count_na"]):
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
# Legend
plt.legend(title="Satisfaction label", labels=["YES", "NO", "N/A"])
# Save the plot as pdf
plt.tight_layout()
fig.savefig("figures/satisfaction/satisfaction_label_per_language.pdf")
plt.show()
plt.clf()

# %% [markdown]
# # Chi-squared test for independence
# We'll use the chi-squared test for independence to check whether there are significant differences between the languages in terms of satisfaction.
# The null hypothesis is that the satisfaction is independent of the language.
# %%
# Perform the chi-squared test for independence
from scipy.stats import chi2_contingency

# Read the CSV file
pl_dataset_percentages = pl.read_csv("results/satisfaction/satisfaction_label_per_language.csv").sort(
    "percentage_yes", descending=True
)

# First, calculate the expectations of each cell
# This is done by multiplying the row total by the column total and dividing by the grand total
grand_total = pl_dataset_percentages["total"].sum()
total_yes = pl_dataset_percentages["count_yes"].sum()
total_no = pl_dataset_percentages["count_no"].sum()
total_na = pl_dataset_percentages["count_na"].sum()
pl_dataset_percentages = pl_dataset_percentages.with_columns(
    expected_yes=pl.col("total") * total_yes / grand_total,
    expected_no=pl.col("total") * total_no / grand_total,
    expected_na=pl.col("total") * total_na / grand_total,
)

# Drop the rows where any of the expectations is less than 10
pl_dataset_percentages = pl_dataset_percentages.filter(
    pl.col("expected_yes").gt(10)
    & pl.col("expected_no").gt(10)
    & pl.col("expected_na").gt(10)
)

# Contingency table: "YES", "NO", "NA" per language
contingency_table = pl_dataset_percentages.select(
    yes=pl.col("count_yes"),
    no=pl.col("count_no"),
    na=pl.col("count_na"),
    # "lang" is not needed for the chi-squared test. We'll drop it
)

# Perform the chi-squared test for independence
res_chi2 = chi2_contingency(contingency_table.to_numpy(), correction=True)

# Print the results
print(f"Chi-squared test statistic: {res_chi2}")

# Write the results to a file
with open("results/satisfaction/satisfaction_label_per_language_chi2.txt", "w") as f:
    f.write(f"Chi-squared test statistic:\n")
    f.write(str(res_chi2))

# Chi-2 is very small because of the large sample size. We'll also calculate the Cramer's V statistic to get a better idea of the effect size.
from scipy.stats.contingency import association

cramer_v = association(observed=contingency_table.to_numpy(), method="cramer", correction=True)

# Print the results
print(f"Cramer's V statistic: {cramer_v}")

# Heatmap of the residuals
import seaborn as sns
import numpy as np

# My colormap: from 1145D4 to F600FF, interpolating between the two colors
import matplotlib.colors as mcolors

colors = [
    "#F600FF",
    "#00376E",
    "#F600FF",
]

cmap = mcolors.LinearSegmentedColormap.from_list("my_colormap", colors)

expected = res_chi2[3]
observed = contingency_table.to_numpy()
residuals = (observed - expected) / np.sqrt(expected)
# Transpose the residuals
residuals = residuals.T

fig, ax = plt.subplots(figsize=(15, 3))
sns.heatmap(
    residuals, annot=True, fmt=".2f", ax=ax, cmap=cmap, cbar=True, vmin=-2, vmax=2
)
# plt.ylabel("Satisfaction label")
# plt.xlabel("Language")
plt.xticks(
    ticks=np.arange(len(pl_dataset_percentages)) + 0.5,
    labels=pl_dataset_percentages["lang"],
    rotation=90,
)
plt.yticks(ticks=[0.5, 1.5, 2.5], labels=["YES", "NO", "NA"], rotation=0)
# plt.title("Residuals of the chi-squared test for independence")
plt.tight_layout()
fig.savefig("figures/satisfaction/satisfaction_label_per_language_residuals.pdf")
plt.show()
plt.clf()
# %%
