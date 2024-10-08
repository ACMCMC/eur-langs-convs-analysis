# %%
import datasets
import pandas as pd

# Set random seeds for reproducibility
import random
import numpy as np
import torch

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

wildchat = datasets.load_dataset("allenai/WildChat-1M-Full", split="train")

"""
Structure of the wilchat dataset:
features: ['conversation_hash', 'model', 'timestamp', 'conversation', 'turn', 'language', 'openai_moderation', 'detoxify_moderation', 'toxic', 'redacted', 'state', 'country', 'hashed_ip', 'header'],
"""

# Another dataset that contains a lot of conversations: lmsys/lmsys-chat-1m
lmsys = datasets.load_dataset("lmsys/lmsys-chat-1m", split="train")
"""
Structure of the lmsys dataset:
features: ['conversation_id', 'model', 'conversation', 'turn', 'language', 'openai_moderation', 'redacted'],
"""

# Concatenate the two datasets into a single dataset with columns: conversation_id, model, conversation, turn, language, openai_moderation, redacted
wildchat = wildchat.remove_columns(
    [
        "timestamp",
        "toxic",
        "state",
        "country",
        "hashed_ip",
        "header",
        "detoxify_moderation",
    ]
)
# In the "conversation" column, keep only "content" and "role" for each message
wildchat = wildchat.map(
    lambda x: {
        "conversation": [
            {"content": m["content"], "role": m["role"]} for m in x["conversation"]
        ]
    },
    num_proc=32,
)

valid_categories = [
    "harassment",
    "harassment/threatening",
    "hate",
    "hate/threatening",
    "self-harm",
    "self-harm/instructions",
    "self-harm/intent",
    "sexual",
    "sexual/minors",
    "violence",
    "violence/graphic",
]


def remove_invalid_categories(x):
    try:
        return {
            "openai_moderation": [
                {
                    "categories": {
                        k: v
                        for k, v in m["categories"].items()
                        if k in valid_categories
                    },
                    "category_scores": {
                        k: v
                        for k, v in m["category_scores"].items()
                        if k in valid_categories
                    },
                    "flagged": m["flagged"],
                }
                for m in x["openai_moderation"]
            ]
        }
    except Exception:
        return {
            "openai_moderation": [
                {
                    "categories": {},
                    "category_scores": {},
                    "flagged": m["flagged"],
                }
                for m in x["openai_moderation"]
            ]
        }


# For 'openai_moderation', keep only the valid categories for 'categories' and 'category_scores', for each item in the list
wildchat = wildchat.map(
    remove_invalid_categories, num_proc=32, remove_columns=["openai_moderation"]
)

lmsys = lmsys.rename_column("conversation_id", "conversation_hash")

lmsys = lmsys.map(
    remove_invalid_categories, num_proc=32, remove_columns=["openai_moderation"]
)

# Remove messages where any of the scores is None
wildchat = wildchat.filter(
    lambda x: all(
        [
            all([v is not None for v in m["category_scores"].values()])
            for m in x["openai_moderation"]
        ]
    ),
    num_proc=32,
)
lmsys = lmsys.filter(
    lambda x: all(
        [
            all([v is not None for v in m["category_scores"].values()])
            for m in x["openai_moderation"]
        ]
    ),
    num_proc=32,
)


dataset = datasets.concatenate_datasets([wildchat, lmsys])

# Discard conversations with less than 5 user words in the first message
MIN_USER_WORDS_FIRST_MESSAGE = 5


def number_of_user_words(x):
    return {
        "num_user_words": sum(
            [
                len(m["content"].split())
                for m in x["conversation"]
                if m["role"] == "user"
            ]
        )
    }


def number_of_assistant_words(x):
    return {
        "num_assistant_words": sum(
            [
                len(m["content"].split())
                for m in x["conversation"]
                if m["role"] == "assistant"
            ]
        )
    }


dataset = dataset.map(number_of_user_words, num_proc=32)
dataset = dataset.map(number_of_assistant_words, num_proc=32)
dataset = dataset.filter(
    lambda x: x["conversation"][0]["role"] == "user"
    and len(x["conversation"][0]["content"].split()) >= MIN_USER_WORDS_FIRST_MESSAGE,
    num_proc=32,
)

# Number of messages per language
dataset = dataset.map(lambda x: {"num_messages": len(x["conversation"])}, num_proc=32)

# Shuffle the dataset
dataset = dataset.shuffle(seed=42)

# Push the dataset to the hub
dataset.push_to_hub("chatbot_conversations", private=True)

# %%
# Load the dataset
import datasets

dataset = datasets.load_dataset("acmc/chatbot_conversations")

# Filter them so that we only have the conversations in European languages

from utils import get_dataset_with_european_languages

dataset = get_dataset_with_european_languages(dataset)

# Push the dataset to the hub
dataset.push_to_hub("acmc/chatbot_conversations_in_european_languages", private=True)

# %%
import datasets

dataset = datasets.load_dataset(
    "acmc/chatbot_conversations_in_european_languages", split="train"
)

# %%
# Filter to take only 1000 examples per each language
import polars as pl

# First, shuffle the dataset
dataset = dataset.shuffle(seed=42)

pl_df = dataset.to_polars()
# %%
pl_df_reduced = pl_df.group_by("language").head(1000)
dataset = datasets.Dataset.from_polars(pl_df_reduced)

# Push the dataset to the hub
dataset.push_to_hub(
    "acmc/chatbot_conversations_in_european_languages_1000", private=True
)
# %%
