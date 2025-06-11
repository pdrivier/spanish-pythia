import random
import numpy as np
import torch
import yaml
from copy import deepcopy
import os

from torch.utils.data import IterableDataset, DataLoader # seems to require pip install fsspec==2023.9.2
from transformers import GPT2Config, GPT2LMHeadModel, Trainer, TrainingArguments, DataCollatorForLanguageModeling, get_scheduler
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import PreTrainedTokenizerFast


def train_model(model_config_dict, training_config, run_name, dataname):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize model config and model
    config = GPT2Config(**model_config_dict)
    model = GPT2LMHeadModel(config).to(device)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("PlanTL-GOB-ES/gpt2-base-bne") #load desired tokenizer

    # Add custom padtoken
    # tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    ### Resize based on size of tokenizer
    
    
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        print("Added pad token.")

    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id

    # Set up your dataset for streaming
    if "cannete" in dataname:
      dataset = load_dataset("josecannete/large_spanish_corpus",split="train",streaming=True)
    elif "oscar" in dataname:
      dataset = load_dataset("oscar","unshuffled_deduplicated_es", split="train",streaming=True)
    elif "bsc" in dataname:
      dataset = load_dataset("BSC-LT/open_data_26B_tokens_balanced_es_ca", split="train", streaming=True)
    
    
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

            def should_save(step):
                return (
                    step < 100 and step % 10 == 0 or
                    step <= 1000 and step % 50 == 0 or
                    step == 2000 or 
                    step % 5000 == 0
                )

            if should_save(step):
                print("Saving step", step)
                checkpoint_path = os.path.join(output_dir, f"checkpoint-{step}")
                model.save_pretrained(checkpoint_path)
                tokenizer.save_pretrained(checkpoint_path)

            if step % training_config["logging_steps"] == 0:
                print(f"Step {step}: loss = {loss.item() * accumulation_steps:.4f}")

            # if step % training_config["save_steps"] == 0:
            #    checkpoint_path = os.path.join(output_dir, f"checkpoint-{step}")
            #    model.save_pretrained(checkpoint_path)
            #    tokenizer.save_pretrained(checkpoint_path)

            step += 1
            if step >= training_config["max_train_steps"]:
                print("Training complete.")
                final_checkpoint_path = os.path.join(output_dir, "final")
                model.save_pretrained(final_checkpoint_path)
                tokenizer.save_pretrained(final_checkpoint_path)
                return

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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
        # {"n_layer": 6, "n_head": 6, "n_embd": 768},
        {"n_layer": 6, "n_head": 12, "n_embd": 768},
        {"n_layer": 12, "n_head": 6, "n_embd": 768},
        {"n_layer": 12, "n_head": 12, "n_embd": 768},
        # {"n_layer": 1, "n_head": 8, "n_embd": 512},
    ]

    # Specify number of seeds per variant to run
    num_seeds = 3

    # Specify which dataset you will train on
    dataname = "cannete" #"oscar", "bsc"

    for i, variant in enumerate(model_variants):

        for s in range(num_seeds):
          # Set a different seed per variant
          seed = s # random.randint(0, 10000) #replace with seed = s if want seed num to match iteration num
          set_seed(seed)

          # Merge base config with variant
          model_config = deepcopy(base_model_config)
          model_config.update(variant)

          # Add seed to config or run name if useful
          run_name = f"gpt_layers{variant['n_layer']}_heads{variant['n_head']}_embd{variant['n_embd']}_seed{s}_seedval{seed}_trainset{dataname}"
          print(f"\n==== Training {run_name} with seed {seed} ====\n")

          train_model(model_config, training_config, run_name, dataname)

if __name__ == "__main__":
    main()