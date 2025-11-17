from datasets import load_dataset
import json

def main():
    # Load a large WinoGrande split — 'train' has ~40k examples
    dataset = load_dataset("allenai/winogrande", "winogrande_xl", split="train")

    data = []
    for i, ex in enumerate(dataset):
        item = {
            "id": f"winogrande_{i}",       # unique generated ID
            "question": ex["sentence"],    # sentence with a blank "_"
            "option1": ex["option1"],
            "option2": ex["option2"],
            "answer": ex["answer"]         # "1" or "2"
        }
        data.append(item)

    # Save as JSON
    output_path = "winogrande_train.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"✅ Saved {len(data)} examples to {output_path}")

if __name__ == "__main__":
    main()
