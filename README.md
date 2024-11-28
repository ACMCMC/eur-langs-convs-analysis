# Differences in user interaction and assistant responses across languages of European origin in large-scale conversational datasets – Supplementary Material
Author: Aldan Creo

This repository contains the supplementary material for the paper "An bhfuil Gaeilge agat?": Differences in user interaction and assistant responses across languages of European origin in large-scale conversational datasets.

# Setup instructions

1. Clone the repository
2. Run `pip install -r requirements.txt`
3. The different analyses can be run by executing the corresponding `analyze_[research_question].py` script in the base directory.

> [!NOTE]
> The analyses are Jupyter notebooks where the cells are separated using the `# %%` delimiter. The scripts can be run in a Jupyter notebook environment or in a Python script.

> [!NOTE]  
> _For the satisfaction analysis, it is necessary to train the classifier first by running `train_satisfaction_model.py` and tag the examples by running `add_satisfaction_label_to_dataset.py`._

# Description of the supplementary material

The supplementary material includes the raw data and the results of the analyses conducted for the research questions presented in the paper. The results are presented in the form of tables and figures, under `results/[research_question]` and `figures/[research_question]`, respectively.

The following sections describe the raw data and the figures for each research question.

## RQ1: Topics
### Raw data
- `results/topics/avg_std_silhouette_scores_minilm.csv`: The average and standard deviation of the 20 subsets (we needed to reduce the number of examples to obtain the silhouette scores due to the quadratic complexity of the algorithm) of silhouette scores for each language, obtained with the MiniLM embeddings.
- `results/topics/avg_std_silhouette_scores_mpnet.csv`: The average and standard deviation of the 20 subsets of silhouette scores for each language, obtained with the mPnet embeddings.
- `results/topics/silhouette_score_minilm.txt`: The average of all the silhouette scores, obtained with the MiniLM embeddings, regardless of the language.
- `results/topics/silhouette_score_mpnet.txt`: The average of all the silhouette scores, obtained with the mPnet embeddings, regardless of the language.

### Figures
No figures are generated for this research question.

## RQ2: Length
### Raw data
- `results/length/length_analysis.csv`: The results of the length analysis, per language, including the average and standard deviation of the number of messages per example (for user or assistant messages, it's half of that value), the average and standard deviation of the number of user words per example, and the average and standard deviation of the number of assistant words per example.
- `results/length/num_conversations_per_language.csv`: The number of conversations (examples) per language.

### Figures
- `figures/length/num_conversations_per_language.pdf`: A bar plot showing the number of conversations (examples) per language.
- `figures/length/num_messages_per_example.pdf`: A box plot showing the distribution of the number of messages per example.
- `figures/length/user_words_vs_assistant_words.pdf`: A scatter plot showing the average number of user words versus the average number of assistant words per example, for each language.
- `figures/length/user_words_vs_examples.pdf`: A scatter plot showing the average number of words per user versus the number of examples available per language.

## RQ3: Sentiment
### Raw data
- `results/sentiment/sentiment_analysis_not_grouped_by_language.csv`: The results of the sentiment analysis, without grouping the examples by language. The table includes the average and standard deviation of the sentiment scores for each sentiment label (positive, negative, neutral), both for user and assistant messages. It also includes the difference between the average sentiment scores of user and assistant.
- `results/sentiment/sentiment_analysis.csv`: The same as the previous table, but grouped by language. Additionally, this table includes the counts of each sentiment label for user and assistant messages.

### Figures
- `figures/sentiment/sentiment_analysis_user.pdf`: A bar plot showing the average sentiment scores for the first user message in each conversation, grouped by language, the sentiment labels being stacked and adding up to 1 (100%). It is important to warn the reader that there exist some languages (e.g. Lithuanian, Maltese, Slovenian, Latvian, Estonian, Croatian, Irish, and Slovak) with a very low number of examples, so caution should be taken when interpreting the results for these languages, and we advise against generalizing the results for them.

## RQ4: Toxicity
### Raw data
- `results/toxicity/dunns_test_user_messages_avg_toxicity.csv`: The results of the Dunn's test for the average toxicity of user messages, comparing all pairs of languages.
- `results/toxicity/dunns_test_user_messages_max_toxicity.csv`: The results of the Dunn's test for the maximum toxicity of user messages, comparing all pairs of languages.
- `results/toxicity/dunns_test_user_messages_[toxicity_label].csv`: The results of the Dunn's test for a specific toxicity label (e.g. `harassment`) of user messages, comparing all pairs of languages.
- `results/toxicity/toxicity_analysis.csv`: The results of the toxicity analysis, including the average and standard deviation of the toxicity scores for each toxicity label and the average and standard deviation of the toxicity scores for aggregations of toxicity labels using average and max functions. The table is grouped by language and role (user, assistant or all).
- `results/toxicity/kruskal_wallis_test_user_messages.csv`: The results of the Kruskal-Wallis test for the different toxicity labels of user messages, and their aggregations using average and max functions.

### Figures
- `figures/toxicity/dunns_test_user_messages_avg_toxicity_heatmap.pdf`: A heatmap showing the p-values of the Dunn's test for the average toxicity of user messages, comparing all pairs of languages.
- `figures/toxicity/dunns_test_user_messages_max_toxicity_heatmap.pdf`: A heatmap showing the p-values of the Dunn's test for the maximum toxicity of user messages, comparing all pairs of languages.
- `figures/toxicity/dunns_test_user_messages_[toxicity_label]_heatmap.pdf`: A heatmap showing the p-values of the Dunn's test for a specific toxicity label (e.g. `harassment`) of user messages, comparing all pairs of languages.
- `figures/toxicity/dunns_test_user_messages_[toxicity_label]_heatmap.pdf`: A heatmap showing the p-values of the Dunn's test for a specific toxicity label (e.g. `harassment`) of user messages, comparing all pairs of languages.
- `figures/toxicity/assistant_max_toxicity_violin.pdf`: A violin plot showing the distribution of the maximum toxicity scores of assistant messages, grouped by language.
- `figures/toxicity/assistant_avg_toxicity_violin.pdf`: A violin plot showing the distribution of the average toxicity scores of assistant messages, grouped by language.
- `figures/toxicity/user_max_toxicity_violin.pdf`: A violin plot showing the distribution of the maximum toxicity scores of user messages, grouped by language.
- `figures/toxicity/user_avg_toxicity_violin.pdf`: A violin plot showing the distribution of the average toxicity scores of user messages, grouped by language.

## RQ5: Satisfaction
### Raw data
- `results/satisfaction/satisfaction_model_language_counts.csv`: The number of annotated examples for each language used in the training of the satisfaction classifier.
- `results/satisfaction/satisfaction_label_per_language.csv`: The aggregated statistics (mean, standard deviation) of the results of running the satisfaction classifier on the entire dataset.

### Figures
- `figures/satisfaction/satisfaction_label_per_language.pdf`: A bar plot showing the aggregated statistics of the results of running the satisfaction classifier on the entire dataset.
- `figures/satisfaction/satisfaction_label_per_language_residuals.pdf`: A heatmap showing the chi-squared residuals of the satisfaction classifier results, comparing the observed and expected values for each language and satisfaction label.

# Annotator guidelines

For the annotation process part of **RQ5**, we provided clear and specific guidelines, which can be accessed in the [`argilla_instance.py` file](https://github.com/ACMCMC/eur-langs-convs-analysis/blob/2f8f7da02833393017fd04a0012e07fa4d3613d1/argilla_instance.py#L19).

# AI Disclaimer
We used AI code generation assitance from GitHub Copilot for this project. Nonetheless, the coding process has been essentially manual, with the AI code generator exclusively helping us to speed up the process.

# Reproducibility statement
We confirm that the results obtained were identical, and thus expect no variation in the results when running the code again. We manually set random seeds where necessary to ensure reproducibility. We have also provided the necessary data and code to reproduce the results in this repository.
