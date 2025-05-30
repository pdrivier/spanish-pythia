import yaml
import os
import torch

from torch.utils.data import IterableDataset, DataLoader
from transformers import GPT2Config, GPT2LMHeadModel, Trainer, TrainingArguments, DataCollatorForLanguageModeling, get_scheduler
from datasets import load_dataset
from transformers import AutoTokenizer
from copy import deepcopy

## TMP CODE
# ds = load_dataset("rotten_tomatoes", split="train", streaming=True)
# next(iter(ds))
# list(ds.take(3))
# ds = ds.select_columns("text")

def train_model(model_config_dict, training_config, run_name):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize model config and model
    config = GPT2Config(**model_config_dict)
    model = GPT2LMHeadModel(config).to(device)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    # Add custom padtoken 
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id

    # Set up your dataset for streaming
    dataset = load_dataset("josecannete/large_spanish_corpus",split="train",streaming=True)
    dataset = dataset.shuffle(buffer_size=10_000)

    def tokenize(examples):
        return tokenizer(examples["text"], truncation=True, max_length=512)

    tokenized_dataset = dataset.map(tokenize, batched=True, remove_columns=["text"])

    # Data collator and DataLoader
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    dataloader = DataLoader(
        tokenized_dataset,
        batch_size=training_config["train_batch_size"],
        collate_fn=data_collator,
        # num_workers=4
    )

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=training_config["learning_rate"], weight_decay=training_config["weight_decay"])
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=training_config["warmup_steps"],
        num_training_steps=training_config["max_train_steps"]  # Define this in config or calculate
    )

    # Mixed precision
    use_fp16 = training_config["fp16"]
    scaler = torch.cuda.amp.GradScaler() if use_fp16 else None

    model.train()
    step = 0
    accumulation_steps = training_config["gradient_accumulation_steps"]
    output_dir = os.path.join(training_config["output_dir"], run_name)
    os.makedirs(output_dir, exist_ok=True)

    for epoch in range(training_config["num_train_epochs"]):
        for batch in dataloader:
            inputs = {k: v.to(device) for k, v in batch.items()}

            with torch.cuda.amp.autocast(enabled=use_fp16):
                outputs = model(**inputs)
                loss = outputs.loss / accumulation_steps

            if use_fp16:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if (step + 1) % accumulation_steps == 0:
                if use_fp16:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

                optimizer.zero_grad()
                lr_scheduler.step()

            if step % training_config["logging_steps"] == 0:
                print(f"Step {step}: loss = {loss.item() * accumulation_steps:.4f}")

            if step % training_config["save_steps"] == 0:
                checkpoint_path = os.path.join(output_dir, f"checkpoint-{step}")
                model.save_pretrained(checkpoint_path)
                tokenizer.save_pretrained(checkpoint_path)
                
            step += 1
            if step >= training_config["max_train_steps"]:
                print("Training complete.")
                
                return

def main():
    # Load base configs
    base_model_config = {
        "model_type": "GPT2",
        "vocab_size": 50257,
        "max_position_embeddings": 1024,
        "resid_pdrop": 0.1,
        "embd_pdrop": 0.1,
        "attn_pdrop": 0.1,
        "layer_norm_epsilon": 1e-5,
        "initializer_range": 0.02,
        "bos_token_id": 0,
        "eos_token_id": 1
    }

    with open("../config/training-config.yaml", "r") as f:
        training_config = yaml.safe_load(f)

    # Define model configurations to test
    model_variants = [
        {"n_layer": 1, "n_head": 4, "n_embd": 512},
        {"n_layer": 2, "n_head": 4, "n_embd": 512},
        {"n_layer": 1, "n_head": 8, "n_embd": 512},
    ]

    for i, variant in enumerate(model_variants):
        # Merge base config with variant
        model_config = deepcopy(base_model_config)
        model_config.update(variant)
        # Create a descriptive name for the run
        run_name = f"gpt_layers{variant['n_layer']}_heads{variant['n_head']}_embd{variant['n_embd']}"
        print(f"\n==== Training {run_name} ====\n")

        train_model(model_config, training_config, run_name)

if __name__ == "__main__":
    main()
