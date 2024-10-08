# %% [markdown]
# # Satisfaction model
# In this notebook, we'll take the sentence-transformers/paraphrase-multilingual-mpnet-base-v2 and add a classification head to it to predict whether a user is satisfied with a chatbot's response.
#
# The model takes three inputs (tokenized texts):
# - u1, the first user message
# - a, the chatbot response
# - u2, the second user message
# Then it runs them through the sentence-transformers/paraphrase-multilingual-mpnet-base-v2
# and generates this vector: (u1, a, u2, |a-u1|, |u2-a|, |u2-u1|) where |x| is the absolute value of x.
# We then run this vector through a classification head.
# The labels are YES, NO, or NA (Not applicable).

# %%
import transformers
import transformers.modeling_outputs

# Set random seeds for reproducibility
import torch
import numpy as np
import random

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

original_model = transformers.AutoModel.from_pretrained(
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    # device_map="cuda:5",
)

from satisfaction_model import SatisfactionModel, SatisfactionModelConfig

# %%
# Now, train the model
# First, load the dataset of tagged chatbot conversations
import datasets

try:
    from google.colab import userdata

    hf_token = userdata.get("HF_TOKEN")

    dataset = datasets.load_dataset(
        "acmc/annotated_conversation_satisfaction_triples",
        split="train",
        token=hf_token,
    )
except:
    dataset = datasets.load_dataset(
        "acmc/annotated_conversation_satisfaction_triples", split="train"
    )

# Change the column name "language" to "lang"
# There's no particular reason to do this, other than when exporting the CSV files and using them in LaTeX, with csvsimple, the column name "language" was causing problems when it's called exactly like that, adding an extra backtick - probably some sort of conflict with another package (super weird bug)
dataset = dataset.rename_column("language", "lang")

# %%
# Now, we need to tokenize the sentences
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
)


ANSWERS_TO_LABELS = {"YES": 1, "NO": 0, "NA": 2}


# %%
# Tokenize the sentences
def tokenize_sentences(batch):
    num_examples = len(batch["chat_triples"])

    # Tokenize the sentences
    u1 = tokenizer(
        [batch["chat_triples"][i][0]["content"] for i in range(num_examples)],
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    a = tokenizer(
        [batch["chat_triples"][i][1]["content"] for i in range(num_examples)],
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    u2 = tokenizer(
        [batch["chat_triples"][i][2]["content"] for i in range(num_examples)],
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    # Stack u1, a, u2 into input_ids and attention_mask
    batch_encoding = transformers.BatchEncoding(
        data={
            "input_ids": torch.stack(
                [u1["input_ids"], a["input_ids"], u2["input_ids"]]
            ).transpose(0, 1),
            "attention_mask": torch.stack(
                [u1["attention_mask"], a["attention_mask"], u2["attention_mask"]]
            ).transpose(
                0, 1
            ),  # Transpose to have the batch dimension first
        }
    )

    return {
        "label": torch.tensor(
            [
                # We only have one annotator, so we can just take the first one
                ANSWERS_TO_LABELS[x[0]]
                for x in batch["response-satisfaction.responses"]
            ]
        ),
        **batch_encoding,
    }


# Only use examples where there's an answer
dataset = dataset.filter(lambda x: x["status"] == "completed")

# Convert to Polars and group by language, then write a CSV with the count of examples per language
import polars as pl

pl_df = dataset.to_polars()
pl_df = pl_df.group_by("lang").agg(pl.count("lang").alias("count"))
pl_df = pl_df.sort("count", descending=True)
pl_df.write_csv("results/satisfaction_model_language_counts.csv")

tokenized_dataset = dataset.map(
    tokenize_sentences,
    batched=True,
    batch_size=8,
    # num_proc=32,
    # remove_columns=dataset.column_names,
)

tokenized_dataset.set_format(
    type="torch", columns=["input_ids", "attention_mask", "label"]
)

# %%

# 80% of the dataset is used for training
# 10% for validation
# 10% for testing
# Shuffle the dataset
tokenized_dataset = tokenized_dataset.shuffle(seed=42)
train_size = int(0.8 * len(tokenized_dataset))
test_size = int(0.1 * len(tokenized_dataset))
eval_size = len(tokenized_dataset) - train_size - test_size
train_dataset = tokenized_dataset.select(range(train_size))
test_dataset = tokenized_dataset.select(range(train_size, train_size + eval_size))
eval_dataset = tokenized_dataset.select(
    range(train_size + eval_size, len(tokenized_dataset))
)
# Dataset for cross-validation is the train + test datasets
cross_validation_dataset = tokenized_dataset.select(range(train_size + test_size))

# Now, we can train the model

training_args = transformers.TrainingArguments(
    # per_device_train_batch_size=32,
    num_train_epochs=10,
    output_dir="satisfaction_model",
    logging_dir="satisfaction_model_logs",
    logging_steps=50,
    # save_steps=32,
    # eval_steps=6,
    # eval_strategy="epoch",
    # save_total_limit=1,
    # overwrite_output_dir=True,
    # learning_rate=1e-5,
    # gradient_accumulation_steps=2,
    save_strategy="no",
)

import evaluate

metric = evaluate.load("accuracy", module_type="metric")


import numpy as np


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    # This is a multiclass classification problem
    return metric.compute(predictions=predictions, references=labels)


# # Use torchsummary to print the model summary
# from torchsummaryX import summary

# summary(
#     model,
#     x=torch.randint(0, tokenizer.vocab_size, (8, 3, tokenizer.model_max_length)),
# )


def model_init(trial):
    if trial is None:
        config = SatisfactionModelConfig()
    else:
        print(f"Trial {trial.number}: {trial.params}")
        config = SatisfactionModelConfig(
            hidden_size=trial.params["hidden_size"],
            num_layers=trial.params["num_layers"],
            use_bias=trial.params["use_bias"],
            use_weighted_loss=trial.params["use_weighted_loss"],
            loss_label_smoothing=trial.params["loss_label_smoothing"],
            activation_function=trial.params["activation_function"],
        )
    model = SatisfactionModel(config=config)
    print(f"Generated model: {model}")
    if torch.cuda.is_available():
        model.to("cuda")

    # Print the trainable parameters
    print("Trainable parameters:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)

    return model


# Here, we use the validation dataset to tune the hyperparameters
trainer = transformers.Trainer(
    model=None,
    model_init=model_init,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
)


# %%
import optuna
import os


if os.getenv("HYPERPARAMETER_TUNING", "True") == "True":
    # Hyperparameter search
    def optuna_hp_space(trial: optuna.Trial):
        return {
            "learning_rate": trial.suggest_float("learning_rate", 1e-6, 1e-3, log=True),
            "per_device_train_batch_size": trial.suggest_categorical(
                "per_device_train_batch_size", [1, 4, 8, 16, 32, 64, 128]
            ),
            "hidden_size": trial.suggest_int("hidden_size", 10, 3000, log=True),
            "num_layers": trial.suggest_int("num_layers", 1, 5),
            "use_bias": trial.suggest_categorical("use_bias", [True, False]),
            "use_weighted_loss": trial.suggest_categorical(
                "use_weighted_loss", [True, False]
            ),
            "loss_label_smoothing": trial.suggest_float(
                "loss_label_smoothing", 0.0, 0.3, step=0.1
            ),
            "activation_function": trial.suggest_categorical(
                "activation_function",
                [
                    "Tanh",
                    "ReLU",
                    "LeakyReLU",
                    "Sigmoid",
                    "GELU",
                    "SiLU",
                    "Mish",
                ],
            ),
        }

    best_trial = trainer.hyperparameter_search(
        direction="maximize",
        backend="optuna",
        hp_space=optuna_hp_space,
        n_trials=160,
        # compute_objective=compute_objective,
    )

    print(f"Best trial: {best_trial}, best params: {best_trial.hyperparameters}")

# %%
if "best_trial" not in locals():
    best_trial = transformers.trainer_utils.BestRun(
        hyperparameters={
            "learning_rate": 0.00046878097862004293,
            "per_device_train_batch_size": 1,
            "hidden_size": 884,
            "num_layers": 4,
            "use_bias": True,
            "use_weighted_loss": False,
            "loss_label_smoothing": 0.2,
            "activation_function": "Mish",
        },
        run_id="0",
        objective=0.0,
    )
config = SatisfactionModelConfig(
    hidden_size=best_trial.hyperparameters["hidden_size"],
    num_layers=best_trial.hyperparameters["num_layers"],
    use_bias=best_trial.hyperparameters["use_bias"],
    use_weighted_loss=best_trial.hyperparameters["use_weighted_loss"],
    loss_label_smoothing=best_trial.hyperparameters["loss_label_smoothing"],
    activation_function=best_trial.hyperparameters["activation_function"],
)

model = SatisfactionModel(config=config)

# Print the trainable parameters
print("Trainable parameters:")
for name, param in model.named_parameters():
    if param.requires_grad:
        print(name)


training_args.learning_rate = best_trial.hyperparameters["learning_rate"]
training_args.per_device_train_batch_size = best_trial.hyperparameters[
    "per_device_train_batch_size"
]

# Now, we use the test dataset to evaluate the model - not the validation dataset (this was used for hyperparameter tuning)

trainer = transformers.Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
)

# %%
# Now, train the model
trainer.train()
# Evaluate the model
trainer.evaluate()

# Push the model to the hub
trainer.push_to_hub("satisfaction_model")
# Save the model
trainer.save_model("satisfaction_model")

# %%
# Run the model on a few examples
import torch

model.eval()

# Get the first few examples
batch = test_dataset[:5]

# Run the model
output = model(
    input_ids=batch["input_ids"].to(model.device),
    attention_mask=batch["attention_mask"].to(model.device),
    labels=batch["label"].to(model.device),
)

# Get the predictions
predictions = torch.argmax(output.logits, dim=-1)

# Print the predictions
for i, prediction in enumerate(predictions):
    # Decode the input text
    u1_text = tokenizer.decode(batch["input_ids"][i][0])
    a_text = tokenizer.decode(batch["input_ids"][i][1])
    u2_text = tokenizer.decode(batch["input_ids"][i][2])
    # Print the texts
    print(f"u1: {u1_text}")
    print(f"a: {a_text}")
    print(f"u2: {u2_text}")
    print(
        f"Prediction: {'YES' if prediction == 1 else 'NO'}, Actual: {batch['label'][i]}"
    )

# Plot a confusion matrix
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Get the predictions for the entire dataset, in batches of 8
from transformers import DataCollatorWithPadding
from torch.utils.data import DataLoader
from tqdm import tqdm

data_collator = DataCollatorWithPadding(tokenizer)

dataset_to_use_for_confusion_matrix = test_dataset

dataloader = DataLoader(
    dataset_to_use_for_confusion_matrix,
    batch_size=32,
    collate_fn=data_collator,
    shuffle=False,
)

predictions = []

for batch in tqdm(dataloader):
    output = model(
        input_ids=batch["input_ids"].to(model.device),
        attention_mask=batch["attention_mask"].to(model.device),
    )

    predictions.extend(torch.argmax(output.logits, dim=-1).cpu().numpy())

# Get the labels
labels = dataset_to_use_for_confusion_matrix["label"]

# Compute the confusion matrix
cm = confusion_matrix(labels, predictions)

# Plot the confusion matrix
sns.heatmap(cm, annot=True, fmt="d")
plt.xlabel("Predicted")
plt.ylabel("Actual")
# xticks
plt.xticks([0.5, 1.5, 2.5], ["NO", "YES", "NA"])
# yticks
plt.yticks([0.5, 1.5, 2.5], ["NO", "YES", "NA"])
plt.show()
# Save as a pdf
plt.savefig("figures/satisfaction_model_confusion_matrix.pdf")
plt.clf()

# %%
# Perform a 10-fold cross-validation
from sklearn.model_selection import KFold
import datasets

kf = KFold(n_splits=10, shuffle=True, random_state=42)

# Get the dataset as a numpy array
input_ids = cross_validation_dataset["input_ids"].numpy()
attention_mask = cross_validation_dataset["attention_mask"].numpy()
labels = cross_validation_dataset["label"].numpy()

# Store the metrics
all_metrics = []

for train_index, test_index in kf.split(input_ids):
    # Split the dataset
    train_input_ids = input_ids[train_index]
    train_attention_mask = attention_mask[train_index]
    train_labels = labels[train_index]

    test_input_ids = input_ids[test_index]
    test_attention_mask = attention_mask[test_index]
    test_labels = labels[test_index]

    # Train the model
    model = SatisfactionModel(config=config)
    trainer = transformers.Trainer(
        model=model,
        args=training_args,
        train_dataset=datasets.Dataset.from_dict(
            {
                "input_ids": train_input_ids,
                "attention_mask": train_attention_mask,
                "label": train_labels,
            }
        ),
        eval_dataset=datasets.Dataset.from_dict(
            {
                "input_ids": test_input_ids,
                "attention_mask": test_attention_mask,
                "label": test_labels,
            }
        ),
        compute_metrics=compute_metrics,
    )

    trainer.train()
    metrics = trainer.evaluate()
    # Returns a dict like:
    # {'eval_loss': 0.8681525588035583,
    #  'eval_accuracy': 0.7818181818181819,
    #  'eval_runtime': 0.9232,
    #  'eval_samples_per_second': 59.578,
    #  'eval_steps_per_second': 2.166,
    #  'epoch': 10.0}

    print(f"Metrics: {metrics}")
    all_metrics.append(metrics)

# Compute the average metrics
import numpy as np

average_metrics = {
    key: np.mean([x[key] for x in all_metrics]) for key in all_metrics[0].keys()
}

print(f"Average metrics: {average_metrics}")

# Standard deviation
std_metrics = {
    key: np.std([x[key] for x in all_metrics]) for key in all_metrics[0].keys()
}

print(f"Standard deviation: {std_metrics}")
# %% [markdown]
# # Calculation of a baseline
# We can calculate a baseline by predicting the most common class.
# %%
# Calculate the baseline as a 10-fold cross-validation
from sklearn.model_selection import KFold

kf = KFold(n_splits=10, shuffle=True, random_state=42)

# Get the dataset as a numpy array
input_ids = cross_validation_dataset["input_ids"].numpy()
attention_mask = cross_validation_dataset["attention_mask"].numpy()
labels = cross_validation_dataset["label"].numpy()

# Store the metrics
all_metrics = []

for train_index, test_index in kf.split(input_ids):
    # Split the dataset
    train_input_ids = input_ids[train_index]
    train_attention_mask = attention_mask[train_index]
    train_labels = labels[train_index]

    test_input_ids = input_ids[test_index]
    test_attention_mask = attention_mask[test_index]
    test_labels = labels[test_index]

    # Calculate the most common class
    most_common_class = np.bincount(train_labels).argmax()

    # Calculate the accuracy
    metrics = metric.compute(
        predictions=np.full_like(test_labels, most_common_class), references=test_labels
    )

    print(f"Metrics: {metrics}")

    all_metrics.append(metrics)

# Compute the average metrics
import numpy as np

average_metrics = {
    key: np.mean([x[key] for x in all_metrics]) for key in all_metrics[0].keys()
}

print(f"Average metrics: {average_metrics}")

# Standard deviation
std_metrics = {
    key: np.std([x[key] for x in all_metrics]) for key in all_metrics[0].keys()
}

print(f"Standard deviation: {std_metrics}")
# %% [markdown]
# # Calculation of a baseline
# We can calculate a baseline by predicting a random class.
# %%
# Calculate the baseline as a 10-fold cross-validation
from sklearn.model_selection import KFold

kf = KFold(n_splits=10, shuffle=True, random_state=42)

# Get the dataset as a numpy array
input_ids = cross_validation_dataset["input_ids"].numpy()
attention_mask = cross_validation_dataset["attention_mask"].numpy()
labels = cross_validation_dataset["label"].numpy()

# Store the metrics
all_metrics = []

for train_index, test_index in kf.split(input_ids):
    # Split the dataset
    train_input_ids = input_ids[train_index]
    train_attention_mask = attention_mask[train_index]
    train_labels = labels[train_index]

    test_input_ids = input_ids[test_index]
    test_attention_mask = attention_mask[test_index]
    test_labels = labels[test_index]

    # Calculate the random class
    random_class = np.random.randint(0, 3, size=len(test_labels))

    # Calculate the accuracy
    metrics = metric.compute(predictions=random_class, references=test_labels)

    print(f"Metrics: {metrics}")

    all_metrics.append(metrics)

# Compute the average metrics
import numpy as np

average_metrics = {
    key: np.mean([x[key] for x in all_metrics]) for key in all_metrics[0].keys()
}

print(f"Average metrics: {average_metrics}")

# Standard deviation
std_metrics = {
    key: np.std([x[key] for x in all_metrics]) for key in all_metrics[0].keys()
}

print(f"Standard deviation: {std_metrics}")

# %%
