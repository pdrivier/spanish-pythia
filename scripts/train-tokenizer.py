import sentencepiece as spm
from datasets import load_dataset
import tempfile
import os

# 1. Load a streamed Spanish dataset
dataset = load_dataset("oscar", "unshuffled_deduplicated_es", split="train", streaming=True)

# 2. Extract only the text column and write to temp file
with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix=".txt") as temp_file:
    print(f"Writing streamed data to: {temp_file.name}")
    for i, example in enumerate(dataset):
        text = example.get("text", "").strip()
        if text:
            if len(text.split()) > 5:
                temp_file.write(text.replace("\n", " ") + "\n")  # normalize line breaks
        if i >= 1_000_000:  # limit examples to ~1 million lines for this example
            break
    temp_filepath = temp_file.name

# 3. Train SentencePiece on the temporary file
spm.SentencePieceTrainer.train(
    input=temp_filepath,
    model_prefix="spanish_spm",
    vocab_size=50257,
    model_type="bpe",  # or "unigram"
    character_coverage=0.9995,
    pad_id=0,
    unk_id=1,
    bos_id=2,
    eos_id=3,
    user_defined_symbols=["[CLS]", "[SEP]", "[MASK]"]
)


# 4. Clean up temp file (optional)
os.remove(temp_filepath)


# 5. Save your tokenizer in HuggingFace formatting for easy loading 
from transformers import T5Tokenizer

# Initialize from your .model file
tokenizer = T5Tokenizer("spanish_spm.model")

# Save in Hugging Face format
tokenizer.save_pretrained("spanish_tokenizer")
