# %% [markdown]
# <a href="https://colab.research.google.com/github/ACMCMC/european-languages-conversations-analysis/blob/main/experiments_perplexity.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
# %%

import datasets
import torch
import transformers
from multiprocess import set_start_method
from transformers import AutoModel, AutoTokenizer

from satisfaction_model import SatisfactionModel, SatisfactionModelConfig

# # Set the device
# torch.set_default_device(DEVICE)

tokenizer = AutoTokenizer.from_pretrained(
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
)


def map_to_triples(x):
    # The first is an user message
    # The second is a chatbot response
    # The third is an user message
    # Assert that we have at least 3 messages
    assert len(x["conversation"]) >= 3

    # We can zip them together. User, chatbot, user
    conversations = x["conversation"]

    convo_triples = [conversations[0], conversations[1], conversations[2]]

    return {
        "chat_triples": convo_triples,
    }


def tokenize_sentences(batch) -> transformers.BatchEncoding:
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

    return batch_encoding

ANSWERS_TO_LABELS = {"YES": 1, "NO": 0, "NA": 2}
ANSWERS_TO_LABELS_REVERSE = {v: k for k, v in ANSWERS_TO_LABELS.items()}


def process_examples(batch, rank):
    device = torch.device(f"cuda:{(rank or 0) % torch.cuda.device_count()}")
    model = SatisfactionModel.from_pretrained(
        "acmc/satisfaction_model",
        # quantization_config=bitsandbytes_config,
        device_map=device,
    )
    model.eval()
    print(f"Rank {rank} using device {device}")

    batch_len = len(batch["language"])
    # If there's already results for all of the examples, skip evaluating them
    if all([s is not None for s in batch["satisfaction_label"]]):
        return batch

    # Otherwise, evaluate the examples. We can do this in parallel so we don't care about not processing specific examples
    tokenized_batch = tokenize_sentences(batch)
    input_ids = tokenized_batch.input_ids.to(device)
    attention_mask = tokenized_batch.attention_mask.to(device)

    with torch.no_grad():
        output = model(input_ids, attention_mask=attention_mask)

    for i in range(batch_len):
        answer = output["logits"][i].argmax().item()
        # Convert the answer to a human readable label
        batch["satisfaction_label"][i] = ANSWERS_TO_LABELS_REVERSE[answer]
    return batch


# %%

if __name__ == "__main__":
    set_start_method("spawn")

    # Load quantized model
    bitsandbytes_config = transformers.BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
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
    
    # If there's no "satisfaction_label" column, add it
    if "satisfaction_label" not in dataset.column_names:
        dataset = (
            dataset.add_column(
                "satisfaction_label",
                [None for _ in range(len(dataset))],
            )
        )

    # Filter out the examples that don't have at least 3 messages
    dataset_with_more_than_3_messages = dataset.filter(lambda x: len(x["conversation"]) >= 3, num_proc=32)
    # Also, keep the examples that we discarded - we'll need them later
    dataset_with_less_than_3_messages = dataset.filter(lambda x: len(x["conversation"]) < 3, num_proc=32)

    dataset_with_conversation_triples = dataset_with_more_than_3_messages.map(
        map_to_triples,
        num_proc=32,
    )


    updated_dataset = dataset_with_conversation_triples.map(
        process_examples,
        batched=True,
        batch_size=900,
        with_rank=True,
        num_proc=torch.cuda.device_count(),
    )

    # Merge the datasets back together
    joined_updated_dataset = datasets.concatenate_datasets(
        [updated_dataset, dataset_with_less_than_3_messages]
    )
    # Randomize the dataset
    joined_updated_dataset = joined_updated_dataset.shuffle(seed=42)

    print(joined_updated_dataset)

    # Push the results to the Hub
    joined_updated_dataset.push_to_hub("acmc/chatbot_conversations_in_european_languages")
