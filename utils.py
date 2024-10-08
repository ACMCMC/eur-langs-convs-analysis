EUROLLM_MODEL_LANGUAGES = [
    "Bulgarian",
    "Croatian",
    "Czech",
    "Danish",
    "Dutch",
    "English",
    "Estonian",
    "Finnish",
    "French",
    "German",
    "Greek",
    "Hungarian",
    "Irish",
    "Italian",
    "Latvian",
    "Lithuanian",
    "Maltese",
    "Polish",
    "Portuguese",
    "Romanian",
    "Slovak",
    "Slovenian",
    "Spanish",
    "Swedish",
    "Arabic",
    "Catalan",
    "Chinese",
    "Galician",
    "Hindi",
    "Japanese",
    "Korean",
    "Norwegian",
    "Russian",
    "Turkish",
    "Ukrainian",
]

EUROPEAN_LANGUAGES_TO_REGIONS = {
    "Bulgarian": "Eastern",
    "Catalan": "Southern",
    "Croatian": "Central",
    "Czech": "Central",
    "Danish": "Northern",
    "Dutch": "Northern",
    "English": "Western",
    "Estonian": "Eastern",
    "Finnish": "Northern",
    "French": "Western",
    "Galician": "Southern",
    "German": "Western",
    "Greek": "Southern",
    "Hungarian": "Central",
    "Irish": "Western",
    "Italian": "Southern",
    "Latvian": "Eastern",
    "Lithuanian": "Eastern",
    "Maltese": "Southern",
    "Norwegian": "Northern",
    "Polish": "Central",
    "Portuguese": "Southern",
    "Romanian": "Eastern",
    "Slovak": "Central",
    "Slovenian": "Central",
    "Spanish": "Southern",
    "Swedish": "Northern",
    "Ukrainian": "Eastern",
}


EUROPEAN_LANGUAGES = EUROPEAN_LANGUAGES_TO_REGIONS.keys()


def get_dataset_with_european_languages(dataset):
    return dataset.filter(lambda x: x["language"] in EUROPEAN_LANGUAGES, num_proc=32)


import numpy as np
import matplotlib.pyplot as plt


def violin_plot_distributions(
    data, classes, classes_assignments, figname, xlabel, ylabel, title
):
    # Create a violin plot for the vectors, separately for each language (color)
    plt.figure(figsize=(10, 3))
    for i, lang in enumerate(classes):
        plt.violinplot(
            data[classes_assignments == lang],
            positions=[i + 1],
            showmeans=False,
            showmedians=True,
            showextrema=False,
            widths=0.9,
        )
    plt.xticks(range(1, len(classes) + 1), classes, rotation=90)
    if xlabel:
        plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.savefig(figname)
    plt.clf()
