# %%
import logging
import random

import numpy
import torch
#from torch import mps  # noqa: F401
#torch.mps.device = mps
from datasets import Dataset, load_dataset

from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerModelCardData,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.losses import MultipleNegativesRankingLoss, CachedMultipleNegativesRankingLoss
from sentence_transformers.training_args import BatchSamplers

# %% [markdown]
# Beim Training gibt es viele Zufallsprozesse - z.B. Sampling der Trainingsdaten
# Um das zu vermeiden - setzen wir seeds (für python, torch und numpy).

# %%
logging.basicConfig(format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)
random.seed(12)
torch.manual_seed(12)
numpy.random.seed(12)

# %%
# Feel free to adjust these variables:
# Soll der Prompt mit reingegeben werden? (wird dann mit Question / Answer getagged)
use_prompts = False
# soll Question / Answer mit beachtet werden
include_prompts_in_pooling = True

# %%
# Bestimmtes huggingface model
# Können wir erstmal benutzen - gibt aber sehr viele moderne Modelle
# wir brauchen SentenceSimilarity Modelle
base_model_name = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"


# %%
model = SentenceTransformer(
    base_model_name,
    #tokenizer_kwargs={"max_seq_length": 512},
    model_card_data=SentenceTransformerModelCardData(
        language="de",
        license="apache-2.0",
        model_name=f"{base_model_name} trained on german Natural Questions pairs",
    ),
).to(torch.bfloat16)

# %%
model.set_pooling_include_prompt(include_prompts_in_pooling)

# %%
from peft import LoraModel, LoraConfig, TaskType

# %%
peft_config = LoraConfig(
    task_type= TaskType.FEATURE_EXTRACTION,
    inference_mode=False,
    r=64,
    lora_alpha=128,
    lora_dropout=0.1,
)
model.add_adapter(peft_config)

# 2. (Optional) Define prompts
if use_prompts:
    query_prompt = "query: "
    corpus_prompt = "document: "
    prompts = {
        "query": query_prompt,
        "answer": corpus_prompt,
    }

# %%
natural_questions_german = load_dataset("oliverguhr/natural-questions-german", split="train")

natural_questions_german = natural_questions_german.remove_columns(["answer", "query"]) # delete the english language columns
natural_questions_german = natural_questions_german.rename_column("query_de", "query").rename_column("answer_de", "answer")

# Aufteilung in Trainings und Testdaten - hier 10%
natural_questions_german = natural_questions_german.train_test_split(test_size=0.1, seed=12)

train_dataset: Dataset = natural_questions_german["train"]
eval_dataset: Dataset = natural_questions_german["test"]

# %%
revosax = load_dataset('csv', data_files="../data/training/training-data.csv", split="train")
revosax = revosax.rename_columns({"result": "query", "chunk": "answer"})
revosax = revosax.select_columns(["query", "answer"])

revosax = revosax.train_test_split(test_size=0.2, seed=12)

train_dataset: Dataset = revosax["train"]
eval_dataset: Dataset = revosax["test"]


revosax

# %%
train_dataset

# %%
# 4. Define a loss function
# Cached ist schneller aber funktioniert auf Apple nicht - bei uns müsste es aber so gehen
loss = CachedMultipleNegativesRankingLoss(model, mini_batch_size=32) # <- this does not work with mps
#loss = MultipleNegativesRankingLoss(model)

# %%
# 5. (Optional) Specify training arguments
run_name = "german-nq-" + base_model_name.split("/")[-1]
if use_prompts:
    run_name += "-prompts"
if not include_prompts_in_pooling:
    run_name += "-exclude-pooling-prompts"
args = SentenceTransformerTrainingArguments(
    # Required parameter:
    output_dir=f"models/{run_name}",
    ### Für unsere Zwecke oder mit sehr großem Datensatz spielen die Paramtere eine untergeordnete Rolle
    # Optional training parameters:
    num_train_epochs=3, # wie häufig wird über alle Samples trainiert
    per_device_train_batch_size=32, # wie viele fälle werden in ein sample gepackt - ist eine optimierung
    # batches sind wichtig damit unser aussagen etwas gemittelt werden - sonst wackelt unser model zu sehr hin und her
    per_device_eval_batch_size=32, # 
    gradient_accumulation_steps=1, # wenn nicht alles in den GPU Speicher passt - kann ich auch schrittweise berechnen und das update des netzes akkumuliert machen
    learning_rate=4e-5, # Wie schnell soll das Netz angepasst werden - also wie viele Änderungen übernehme ich
    # also übernehme ich nur 0.0004% der vorgeschlagenen Änderungen
    # Es ist aber besser die learning Rate nicht konstant zu machen
    # deswegen machen wir ein Warmup - gehen von weniger auf unser LR und fallen dann wieder ab
    warmup_ratio=0.1, # peak Learning rate nach 10% der samples
    # Je größer Batchsize -> weniger Samples -> höhere learning_rate
    fp16=False,  # Set to False if you get an error that your GPU can't run on FP16
    bf16=True,  # Set to True if you have a GPU that supports BF16
    batch_sampler=BatchSamplers.NO_DUPLICATES,  # MultipleNegativesRankingLoss benefits from no duplicate samples in a batch
    # Optional tracking/debugging parameters:
    # wann mache ich evaluationen und wie oft speichere ich die
    eval_strategy="steps",
    eval_steps=0.5,
    save_strategy="steps",
    save_steps=0.5,
    save_total_limit=2,
    logging_steps=5,
    logging_first_step=True,
    run_name=run_name,  # Will be used in W&B if `wandb` is installed
    seed=12,
    prompts=prompts if use_prompts else None,
    report_to="tensorboard",
)

# %%
# 7. Create a trainer & train
# erzeugt modell unter Training / Models
trainer = SentenceTransformerTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    loss=loss,
    #evaluator=dev_evaluator,
)
trainer.train()


