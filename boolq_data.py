from datasets import load_dataset, concatenate_datasets
import json

def prepare_boolq_json(output_file="boolq_dataset.json", num_samples=12500, seed=42):
    """
    Loads the BoolQ dataset from Hugging Face, randomly samples num_samples examples,
    and saves them in a simplified JSON format for model evaluation.
    """
    print("🔹 Loading BoolQ dataset from Hugging Face...")
    ds = load_dataset("google/boolq")

    # Merge training and validation splits
    combined = concatenate_datasets([ds["train"], ds["validation"]])
    print(f"✅ Loaded total {len(combined)} examples from BoolQ.")

    # Shuffle for randomness
    combined = combined.shuffle(seed=seed)
    print("🔀 Shuffled dataset for random sampling.")

    # Select the first num_samples examples after shuffling
    num_samples = min(num_samples, len(combined))
    subset = combined.select(range(num_samples))
    print(f"📊 Selected {num_samples} examples.")

    # Format examples for your evaluation script
    formatted = []
    for i, ex in enumerate(subset):
        formatted.append({
            "id": f"boolq_{i:05d}",
            "question": ex["question"],
            "passage": ex["passage"],
            "answer": bool(ex["answer"])  # ensure native Python bool
        })

    # Save as JSON
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(formatted, f, indent=2)

    print(f"✅ Successfully saved {num_samples} examples to '{output_file}'")

if __name__ == "__main__":
    prepare_boolq_json()
