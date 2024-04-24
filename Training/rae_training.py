from huggingface_hub import notebook_login
#notebook_login()

import json
import random
import os
import psutil

import torch
from torch.cuda import OutOfMemoryError

import accelerate
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from trl import SFTTrainer
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    BitsAndBytesConfig,
    Trainer, 
    TrainingArguments, 
    TrainerCallback, 
    EarlyStoppingCallback,
    TextDataset, 
    DataCollatorForLanguageModeling, 
    IntervalStrategy,
    get_cosine_schedule_with_warmup
)
from datasets import Dataset

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
OUTPUT_DIR = "Training/results/aguila-rae/aguila-rae"

class CustomTrainer(Trainer):
    def create_optimizer_and_scheduler(self, num_training_steps: int):
        super().create_optimizer_and_scheduler(num_training_steps)
        self.lr_scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=700,
            num_training_steps=num_training_steps
        )

def load_dataset(train_path, val_path, tokenizer, max_seq_length=256):
    train_dataset = TextDataset(
        tokenizer=tokenizer,
        file_path=train_path,
        block_size=128)

    val_dataset = TextDataset(
        tokenizer=tokenizer,
        file_path=val_path,
        block_size=128)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False,
    )
    return train_dataset, val_dataset, data_collator

#You will need to add the path to the corpus
with open('', 'r', encoding='utf-8') as file:
    data = json.load(file)

for entry in data:
    entry['definitions'] = entry['definitions'][:2]

if len(data) > 60000:
    data = random.sample(data, 60000)

processed_data = []
for entry in data:
    word = entry['word']
    pos = entry['pos']
    definitions = "; ".join(entry['definitions'][:2])
    processed_entry = f"[BOS] {word} (POS: {pos}) <definition> {definitions} [EOS]"
    processed_data.append(processed_entry)


random.shuffle(processed_data)

split_index = int(0.9 * len(processed_data))
train_data = processed_data[:split_index]
val_data = processed_data[split_index:]

train_data_path = 'Training/Data\RAE/train_data.txt'
val_data_path = 'Training/Data\RAE/val_data.txt'


with open(train_data_path, 'w', encoding='utf-8') as f:
    for item in train_data:
        f.write("%s\n" % item)

with open(val_data_path, 'w', encoding='utf-8') as f:
    for item in val_data:
        f.write("%s\n" % item)

model_id = "projecte-aina/aguila-7b"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.padding_side = 'right'

model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    quantization_config=bnb_config, 
    device_map="auto",
    trust_remote_code=True,)

model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)
model.config.use_cache = False
train_dataset, val_dataset, data_collator = load_dataset(train_data_path, val_data_path, tokenizer, max_seq_length=128)

lora_config = LoraConfig(
    lora_alpha=64,
    lora_dropout=0.1,
    r=16,
    bias="none",
    target_modules=[
        "query_key_value",
        "dense",
        "dense_h_to_4h",
        "dense_4h_to_h",
    ],
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)

training_args = TrainingArguments(
    auto_find_batch_size=True,
    #per_device_train_batch_size=8,
    #per_device_eval_batch_size=8,
    evaluation_strategy=IntervalStrategy.STEPS,
    eval_steps=50,
    gradient_accumulation_steps=3,
    num_train_epochs=8,
    warmup_steps=700,
    weight_decay=0.01,
    learning_rate=1e-3,
    bf16=True,
    logging_steps=10,
    save_steps=500,
    output_dir=OUTPUT_DIR,
    report_to="all",
    run_name="oxford_gemma",
    load_best_model_at_end=True, 
    metric_for_best_model='loss',
    optim="paged_adamw_8bit"
    )


trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        #callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )

print("Starting training with the custom cosine scheduler")
print("using GPUs 1 and 2")
try:
    trainer.train()

except OutOfMemoryError:
    print("Ran out of memory during training. Try reducing the batch size or sequence length.")
    torch.cuda.empty_cache()