from datasets import load_dataset
import json

def main():
    dataset = load_dataset("lighteval/siqa", split="train")

    data = []
    for i, ex in enumerate(dataset):
        item = {
            "id": f"siqa_{i}",
            "context": ex["context"],
            "question": ex["question"],
            "answerA": ex["answerA"],
            "answerB": ex["answerB"],
            "answerC": ex["answerC"],
            "label": ex["label"]
        }
        data.append(item)

    # Save as JSON
    output_path = "siqa_dataset.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"✅ Saved {len(data)} examples to {output_path}")

if __name__ == "__main__":
    main()
    