# get_gemma_boolq_2000_each.py
import os, json, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login
from dotenv import load_dotenv

# ---------- Env & Auth ----------
load_dotenv()

def hf_login():
    token = os.getenv("HF_TOKEN")
    if token:
        login(token=token)
        print("✅ Logged into Hugging Face.")
    else:
        print("⚠️ HF_TOKEN not found. Model download may fail if not cached.")

def get_device_and_dtype():
    if torch.cuda.is_available():
        return torch.device("cuda"), torch.bfloat16
    if torch.backends.mps.is_available():
        return torch.device("mps"), torch.float16
    return torch.device("cpu"), torch.float32

# ---------- Scoring ----------
@torch.no_grad()
def evaluate_boolq_example(model, tokenizer, device, example):
    passage, question = example["passage"], example["question"]
    correct_answer = "Yes" if example["answer"] else "No"

    prompt = f"Passage: {passage}\nQuestion: {question}\nAnswer:"

    def get_ll(completion: str) -> float:
        inputs = tokenizer(prompt + " " + completion, return_tensors="pt").to(device)
        outputs = model(**inputs, labels=inputs["input_ids"])
        # total log-likelihood = - (avg CE loss per token) * (num tokens)
        return -outputs.loss.item() * inputs["input_ids"].shape[1]

    ll_yes = get_ll("Yes")
    ll_no  = get_ll("No")
    gemma_answer = "Yes" if ll_yes >= ll_no else "No"
    is_correct = (gemma_answer == correct_answer)

    return {
        "id": example["id"],
        "question": question,
        "passage": passage,
        "correct_answer": correct_answer,
        "gemma_answer": gemma_answer,
        "is_correct": is_correct
    }

# ---------- Main ----------
def main(
    input_file="boolq_dataset.json",
    output_file="gemma_boolq_answers.json", 
    target_each=2000
):
    hf_login()
    device, dtype = get_device_and_dtype()
    print(f"\nUsing device: {device} | dtype: {dtype}")

    # Try cache-only first so we don't re-download; if missing, fall back to normal load
    model_name = "google/gemma-2-2b"
    print("Loading Gemma-2-2B model & tokenizer (cache-first)...")
    try:
        model = AutoModelForCausalLM.from_pretrained(model_name, local_files_only=True, dtype=dtype).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
        print("✅ Loaded from local cache.")
    except Exception:
        print("ℹ️ Local cache not found or incomplete; loading from Hub (will cache for future runs).")
        model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("✅ Model and tokenizer ready.")

    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")

    with open(input_file, "r", encoding="utf-8") as f:
        examples = json.load(f)

    total = len(examples)
    print(f"\n📊 Scanning {total} BoolQ examples... Target: {target_each} correct + {target_each} incorrect")

    correct_bucket, incorrect_bucket = [], []
    correct_count = incorrect_count = 0

    for idx, ex in enumerate(examples, start=1):
        res = evaluate_boolq_example(model, tokenizer, device, ex)

        # Bucket and cap
        if res["is_correct"]:
            if correct_count < target_each:
                correct_bucket.append(res)
                correct_count += 1
        else:
            if incorrect_count < target_each:
                incorrect_bucket.append(res)
                incorrect_count += 1

        # Progress line with counters
        print(
            f"[{idx}/{total}] {ex['id']} → Gemma: {res['gemma_answer']} | True: {res['correct_answer']} "
            f"| {'✅' if res['is_correct'] else '❌'}  "
            f"| Totals → correct: {correct_count}/{target_each}, incorrect: {incorrect_count}/{target_each}"
        )

        # Early stop once both buckets are full
        if correct_count >= target_each and incorrect_count >= target_each:
            print("\n✅ Reached targets for both buckets. Stopping early.")
            break

    # Save exactly target_each of each (already capped)
    output_obj = {
        "correct": correct_bucket,
        "incorrect": incorrect_bucket
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_obj, f, indent=2)

    print("\n==============================")
    print("Finished collection")
    print("==============================")
    print(f"✔️ Correct saved:   {len(correct_bucket)} / {target_each}")
    print(f"✔️ Incorrect saved: {len(incorrect_bucket)} / {target_each}")
    print(f"📁 Results written to: {output_file}")

if __name__ == "__main__":
    main()
