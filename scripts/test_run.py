"""Just test out different models."""

from transformers import GPT2LMHeadModel, AutoTokenizer



#### Outputs root
outputs_root = "scripts/outputs"

### Test text
input_text = "The man went for a walk."


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

        # Do generation (optional)
        gen_inputs = tokenizer(input_text, return_tensors="pt").to(device)
        with torch.no_grad():
            gen_output = model.generate(**gen_inputs, max_length=30)
        decoded = tokenizer.decode(gen_output[0], skip_special_tokens=True)

        # Compute surprisal
        surprisals, tokens = compute_surprisal(model, tokenizer, input_text)

        print(f"\n-- {checkpoint} --")
        print(f"Generated: {decoded}")
        print("Surprisal per token:")
        for tok, s in zip(tokens, surprisals):
            print(f"{tok:>10}: {s:.2f}")
