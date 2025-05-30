"""Just test out different models."""

from transformers import GPT2LMHeadModel, AutoTokenizer
import os
import torch
import torch.nn.functional as F




### Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




#### Outputs root
outputs_root = "scripts/outputs"

### Test text
input_text = "La niña miró por la ventana y vio"


def compute_surprisal(model, tokenizer, text):
    # Tokenize input
    enc = tokenizer(text, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits[:, :-1]
        targets = input_ids[:, 1:]

        # Compute per-token cross-entropy
        log_probs = F.log_softmax(logits, dim=-1)
        token_log_probs = log_probs.gather(2, targets.unsqueeze(-1)).squeeze(-1)
        surprisals = -token_log_probs  # Negative log probs = surprisal

    return surprisals[0].tolist(), tokenizer.convert_ids_to_tokens(targets[0])


def count_parameters(model):
    """credit: https://stackoverflow.com/questions/49201236/check-the-total-number-of-parameters-in-a-pytorch-model"""
    
    total_params = 0
    for name, parameter in model.named_parameters():
        
        # if the param is not trainable, skip it
        if not parameter.requires_grad:
            continue
        
        # otherwise, count it towards your number of params
        params = parameter.numel()
        total_params += params
    # print(f"Total Trainable Params: {total_params}")
    
    return total_params



for model_dir in sorted(os.listdir(outputs_root)):
    full_model_path = os.path.join(outputs_root, model_dir)
    if not os.path.isdir(full_model_path):
        continue

    print(f"\n=== Model: {model_dir} ===")


    for checkpoint in sorted(os.listdir(full_model_path)):
        ckpt_path = os.path.join(full_model_path, checkpoint)
        if not os.path.isdir(ckpt_path) or not checkpoint.startswith("checkpoint-"):
            continue

        # Load model and tokenizer
        model = GPT2LMHeadModel.from_pretrained(ckpt_path).to(device)
        tokenizer = AutoTokenizer.from_pretrained(ckpt_path)
        model.eval()

        # Count params
        num_params = count_parameters(model)
        print("Number of parameters: " + str(num_params))

        # Do generation (optional)
        gen_inputs = tokenizer(input_text, return_tensors="pt").to(device)
        with torch.no_grad():
            gen_output = model.generate(**gen_inputs, max_length=30)
        decoded = tokenizer.decode(gen_output[0], skip_special_tokens=True)

        # Compute surprisal
        surprisals, tokens = compute_surprisal(model, tokenizer, input_text)

        mean_surprisal = sum(surprisals) / len(surprisals)

        print(f"\n-- {checkpoint} --")
        print(f"Generated: {decoded}")
        print(f"Mean surprisal: {mean_surprisal:.2f}")
