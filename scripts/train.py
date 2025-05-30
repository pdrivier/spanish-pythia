# Training Pythia-style models
# tools to install: 
# pip install transformers datasets deepspeed pyyaml accelerate


import yaml
from transformers import AutoTokenizer, GPT2Config, GPT2LMHeadModel, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_dataset

def load_yaml(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def main():
    # Load configs
    model_config_dict = load_yaml("../config/model-config.yaml")
    train_config = load_yaml("../config/training-config.yaml")

    # Initialize model config and model
    config = GPT2Config(**model_config_dict)
    model = GPT2LMHeadModel(config)

    # Load dataset
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
    tokenizer = AutoTokenizer.from_pretrained("gpt2") # Load or initialize tokenizer

    # Tokenize dataset
    def tokenize_function(examples):
        return tokenizer(examples["text"], return_special_tokens_mask=True)
    
    tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

    # Data collator
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=train_config["output_dir"],
        per_device_train_batch_size=train_config["train_batch_size"],
        per_device_eval_batch_size=train_config["eval_batch_size"],
        num_train_epochs=train_config["num_train_epochs"],
        gradient_accumulation_steps=train_config["gradient_accumulation_steps"],
        evaluation_strategy="steps",
        save_steps=train_config["save_steps"],
        eval_steps=train_config["eval_steps"],
        logging_steps=train_config["logging_steps"],
        save_total_limit=train_config["save_total_limit"],
        learning_rate=train_config["learning_rate"],
        weight_decay=train_config["weight_decay"],
        warmup_steps=train_config["warmup_steps"],
        fp16=train_config["fp16"],
        deepspeed=train_config.get("deepspeed_config", None)
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator
    )

    # Train
    trainer.train()

if __name__ == "__main__":
    main()
