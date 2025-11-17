import os, json, random, torch
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

# ---------- Scoring Helpers ----------
@torch.no_grad()
def get_partial_ll(model, tokenizer, device, prefix: str, option: str, suffix: str) -> float:
    """
    Compute log-likelihood of the suffix tokens conditioned on prefix + option.
    """
    context = prefix + option
    context_ids = tokenizer(context, return_tensors="pt").to(device)
    suffix_ids = tokenizer(suffix, return_tensors="pt").input_ids.to(device)

    input_ids = torch.cat([context_ids.input_ids, suffix_ids], dim=1)
    labels = input_ids.clone()
    labels[:, :context_ids.input_ids.shape[1]] = -100  # mask context

    outputs = model(input_ids, labels=labels)
    return -outputs.loss.item() * suffix_ids.shape[1]

@torch.no_grad()
def evaluate_winogrande_example(model, tokenizer, device, example):
    """
    Partial-scoring evaluation of one WinoGrande example.
    """
    q = example["question"]
    o1, o2 = example["option1"], example["option2"]
    gold = str(example["answer"]).strip()

    # Split question into prefix and suffix around '_'
    parts = q.split("_")
    prefix, suffix = (parts[0], parts[1]) if len(parts) == 2 else (q, "")

    ll1 = get_partial_ll(model, tokenizer, device, prefix, o1, suffix)
    ll2 = get_partial_ll(model, tokenizer, device, prefix, o2, suffix)

    gemma_answer = "1" if ll1 >= ll2 else "2"
    is_correct = (gemma_answer == gold)

    return {
        "id": example["id"],
        "question": q,
        "option1": o1,
        "option2": o2,
        "gold": gold,
        "gemma_answer": gemma_answer,
        "ll_option1": ll1,
        "ll_option2": ll2,
        "is_correct": is_correct,
    }

# ---------- Main ----------
def main(
    input_file="winogrande_train.json",
    output_file="gemma_winogrande_2000_each.json",
    target_each=2000
):
    hf_login()
    device, dtype = get_device_and_dtype()
    print(f"\nUsing device: {device} | dtype: {dtype}")

    model_name = "google/gemma-2-2b"
    print("Loading Gemma-2-2B model & tokenizer (cache-first)...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, local_files_only=True, dtype=dtype
        ).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
        print("✅ Loaded from local cache.")
    except Exception:
        print("ℹ️ Cache not found; loading from Hub.")
        model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("✅ Model and tokenizer ready.")

    # Load dataset
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")

    with open(input_file, "r", encoding="utf-8") as f:
        all_examples = json.load(f)

    print(f"\n📊 Loaded {len(all_examples)} total examples. Target: {target_each} correct + {target_each} incorrect\n")

    # Shuffle for randomness
    random.shuffle(all_examples)

    correct_bucket, incorrect_bucket = [], []
    correct_count = incorrect_count = 0
    total_processed = 0

    for idx, ex in enumerate(all_examples, start=1):
        res = evaluate_winogrande_example(model, tokenizer, device, ex)
        total_processed += 1

        # Add to appropriate bucket
        if res["is_correct"] and correct_count < target_each:
            correct_bucket.append(res)
            correct_count += 1
        elif not res["is_correct"] and incorrect_count < target_each:
            incorrect_bucket.append(res)
            incorrect_count += 1

        # Print progress
        running_acc = (correct_count + incorrect_count) / total_processed * 100
        print(
            f"[{idx}] → Gemma: {res['gemma_answer']} | True: {res['gold']} "
            f"| {'✅' if res['is_correct'] else '❌'} "
            f"| Totals → correct: {correct_count}/{target_each}, incorrect: {incorrect_count}/{target_each} "
            f"| Running collected acc: {running_acc:.2f}%"
        )

        # Stop when both buckets full
        if correct_count >= target_each and incorrect_count >= target_each:
            print("\n✅ Reached targets for both buckets. Stopping early.")
            break

    # Save combined results
    output_obj = {
        "correct": correct_bucket,
        "incorrect": incorrect_bucket,
        "summary": {
            "total_processed": total_processed,
            "correct_collected": correct_count,
            "incorrect_collected": incorrect_count
        }
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
