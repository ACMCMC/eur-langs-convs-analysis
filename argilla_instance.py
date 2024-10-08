# %%
import argilla as rg
import os

HF_TOKEN = os.getenv("HF_TOKEN", "<hf_token>")

# %%

import argilla as rg

client = rg.Argilla(
    api_url="https://acmc-my-argilla.hf.space",
    api_key="9JEPgcUe6aUiLiTcPk6j32GaZgCuo0du9QTnPKqRsmpGHxpjbyYFE5x9_L6efkSZ7asOQlZdNSeDvtL3Oz4W6FaiTheUjdynugRJTpWoe0E",
    headers={"Authorization": f"Bearer {HF_TOKEN}"},
)

# %%
settings = rg.Settings(
    guidelines="""You will be given a conversation containing a user message, a assistant response, and a second user message. Your task is to determine if the user is satisfied with the assistant's response.
    
To judge your answer, consider the following:
- Consider the assistant's response in the context of the conversation with respect to the user's request and how well the answer addresses the user's request.
- Whenever possible, use the user's response to the assistant's response to determine satisfaction.
- If the user asks about the same topic, because they are left wanting, they are not satisfied.
- If the system refuses to provide information, the user is not satisfied.
- If the user doesn't make a request, choose 'Not applicable'.
- If the user's message is not clear or has formatting issues, choose 'Not applicable'.
- If the user changes topics, they are satisfied.
- If the user asks the system to continue generating, or to provide more information, they are satisfied.""",
    fields=[
        rg.ChatField(
            name="chat_triples_in_english",
            title="English translation",
            use_markdown=True,
            required=False,
        ),
        rg.ChatField(
            name="chat_triples",
            title="Chat",
            use_markdown=True,
            required=True,
        ),
    ],
    questions=[
        rg.LabelQuestion(
            name="response-satisfaction",
            labels={"YES": "Yes", "NO": "No", "NA": "Not applicable"},
            title="Is the user satisfied with the assistant's response?",
            description="Select the one that applies.",
            required=True,
        )
    ],
    # Add the language field
    metadata=[
        rg.TermsMetadataProperty(
            name="language",
            # options=["group-a", "group-b", "group-c"],
            title="Language",
            visible_for_annotators=True,
        )
    ],
)

try:
    client.workspaces.add(rg.Workspace("acmc"))
except:
    print("Workspace already exists")

try:
    dataset = rg.Dataset(
        name="my_dataset",
        workspace="acmc",
        settings=settings,
    )

    dataset.create()
except:
    print("Dataset already exists")

# %%
import datasets
import polars as pl

hf_dataset = (
    datasets.load_dataset(
        "acmc/chatbot_conversations_in_european_languages", split="train"
    )
    .shuffle(seed=42)
    # .take(1000)
)

# Select only some examples per language
pl_df = hf_dataset.to_polars()

pl_df_reduced = pl_df.group_by("language").head(1000)

hf_dataset = datasets.Dataset.from_polars(pl_df_reduced)
# %%


def map_to_triples(batch):
    # The first is an user message
    # The second is a chatbot response
    # The third is an user message
    # Assert that the batch size is 1
    assert len(batch["conversation"]) == 1

    # We can zip them together. User, chatbot, user
    conversations = batch["conversation"][0]
    # If there's less than 3 messages, we can't use this example
    if len(conversations) < 3:
        return {"chat_triples": []}

    convo_triples = [[conversations[0], conversations[1], conversations[2]]]

    return {
        "chat_triples": convo_triples,
        "language": batch["language"] * len(convo_triples),
    }


dataset_with_conversation_triples = hf_dataset.map(
    map_to_triples,
    batched=True,
    batch_size=1,
    num_proc=32,
    remove_columns=hf_dataset.column_names,
)

# %% [markdown]
# # Translate the messages to English
# This is necessary for the annotators to understand the conversation and be able to judge if the user is satisfied with the chatbot's response.
# %%
# Use a HF pipeline to translate the messages to English
from transformers import pipeline
import tqdm

pipe = pipeline(
    "text-generation",
    model="utter-project/EuroLLM-1.7B",
    device_map="cuda",
)

pipe.tokenizer.pad_token_id = pipe.model.config.eos_token_id
pipe.tokenizer.padding_side = "left" # Padding should be left for causal models

# Use the prompt: "[Language]: [Message] English:"
# Slice the message to 200 characters, so that it +- fits in the model's input (we just want a general idea of the message, we don't need the full context)
prompts_user_messages_1 = dataset_with_conversation_triples.select_columns(
    ["chat_triples", "language"]
).map(
    lambda x: {
        "translation_prompt": f"{x['language']}: {x['chat_triples'][0]['content'][:200]} English:"
    },
    num_proc=32,
)[
    "translation_prompt"
]
prompts_chatbot_responses = dataset_with_conversation_triples.select_columns(
    ["chat_triples", "language"]
).map(
    lambda x: {
        "translation_prompt": f"{x['language']}: {x['chat_triples'][1]['content'][:200]} English:"
    },
    num_proc=32,
)[
    "translation_prompt"
]
prompts_user_messages_2 = dataset_with_conversation_triples.select_columns(
    ["chat_triples", "language"]
).map(
    lambda x: {
        "translation_prompt": f"{x['language']}: {x['chat_triples'][2]['content'][:200]} English:"
    },
    num_proc=32,
)[
    "translation_prompt"
]

# %%
translated_user_messages_1 = list(
    tqdm.tqdm(
        pipe(
            prompts_user_messages_1,
            # batched=True,
            batch_size=32,
            max_new_tokens=100,
            padding="longest",
            truncation=True,
            # max_length=200,
        )
    )
)

translated_chatbot_responses = list(
    tqdm.tqdm(
        pipe(
            prompts_chatbot_responses,
            # batched=True,
            batch_size=32,
            max_new_tokens=100,
            padding="longest",
            truncation=True,
            # max_length=200,
        )
    )
)

translated_user_messages_2 = list(
    tqdm.tqdm(
        pipe(
            prompts_user_messages_2,
            # batched=True,
            batch_size=32,
            max_new_tokens=100,
            padding="longest",
            truncation=True,
            # max_length=200,
        )
    )
)

# %%

# Merge them back into the dataset as a "chat_triples_in_english" field
chat_triples_in_english = [
    [
        {
            "content": translated_user_messages_1[i][0]["generated_text"],
            "role": "user",
        },
        {
            "content": translated_chatbot_responses[i][0]["generated_text"],
            "role": "assistant",
        },
        {
            "content": translated_user_messages_2[i][0]["generated_text"],
            "role": "user",
        },
    ]
    for i in range(len(translated_user_messages_1))
]

# Remove everything before 'English:'
chat_triples_in_english = [
    [
        {
            "content": chat_triple["content"].split("English:")[-1].strip(),
            "role": chat_triple["role"],
        }
        for chat_triple in chat_triples
    ]
    for chat_triples in chat_triples_in_english
]

dataset_with_conversation_triples_translated = (
    dataset_with_conversation_triples.add_column(
        "chat_triples_in_english", chat_triples_in_english
    )
)

# %% [markdown]
# # Add the records to Argilla
# %%
# Now, add this to Argilla
dataset = client.datasets(name="my_dataset", workspace="acmc")
# Delete all the previous records
try:
    status_filter = rg.Query(filter=rg.Filter(("response.status", "==", "pending")))
    records_to_delete = list(dataset.records(status_filter))
    dataset.records.delete(records_to_delete)
except:
    pass

# What are the records that are not yet annotated?
records = list(dataset.records())
chat_triples_already_annotated = [record["chat_triples"] for record in records]

records_not_annotated = dataset_with_conversation_triples_translated.filter(
    lambda x: x["chat_triples"] not in chat_triples_already_annotated,
    num_proc=32,
)

dataset.records.log(
    records=dataset_with_conversation_triples_translated,
    # mapping={"chat_triples": "chat_triples", "language": "language"},
)  #

# %%
# Push the annotations to the Hub
dataset = client.datasets(name="my_dataset", workspace="acmc")
dataset.to_hub(
    repo_id="acmc/annotated_conversation_satisfaction_triples",
    with_records=True,
    generate_card=True,
)

# %%
import datasets
# Load the dataset
annotated_dataset = datasets.load_dataset(
    "acmc/annotated_conversation_satisfaction_triples", split="train"
)

# Change the column name "language" to "lang"
# There's no particular reason to do this, other than when exporting the CSV files and using them in LaTeX, with csvsimple, the column name "language" was causing problems when it's called exactly like that, adding an extra backtick - probably some sort of conflict with another package (super weird bug)
annotated_dataset = annotated_dataset.rename_column("language", "lang")

# Convert to Polars and group by language, then write a CSV with the count of examples per language
import polars as pl

# Only use examples where there's an answer
pl_df = annotated_dataset.filter(lambda x: x["status"] == "completed").to_polars()
pl_df = pl_df.group_by("lang").agg(pl.count("lang").alias("count"))
pl_df = pl_df.sort("count", descending=True)
pl_df.write_csv("results/satisfaction_model_language_counts.csv")
# %%
