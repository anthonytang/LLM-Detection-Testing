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
def compute_ll(model, tokenizer, device, prompt: str, answer: str) -> float:
    """
    Compute total log-likelihood of answer tokens conditioned on prompt.
    Handles NaNs and long sequences safely.
    """
    inputs = tokenizer(
        prompt + answer,
        return_tensors="pt",
        truncation=True,
        max_length=4096,
    ).to(device)

    prompt_len = len(tokenizer(prompt, truncation=True, max_length=4096).input_ids)
    if inputs.input_ids.shape[1] <= prompt_len:
        return float("-inf")  # invalid continuation

    labels = inputs.input_ids.clone()
    labels[:, :prompt_len] = -100  # mask prompt tokens

    outputs = model(inputs.input_ids, labels=labels)
    loss_val = outputs.loss.detach().float().item()

    if torch.isnan(torch.tensor(loss_val)) or torch.isinf(torch.tensor(loss_val)):
        return float("-inf")

    seq_len = inputs.input_ids.shape[1] - prompt_len
    return -loss_val * seq_len  # total log-likelihood


@torch.no_grad()
def evaluate_siqa_example(model, tokenizer, device, example):
    """
    Evaluate one SIQA example using log-likelihood scoring.
    Dataset labels are 1, 2, 3 (A/B/C).
    """
    context = example["context"].strip()
    question = example["question"].strip()
    answers = [example["answerA"].strip(), example["answerB"].strip(), example["answerC"].strip()]
    gold = int(example["label"])

    prompt = (
        "Read the context and question below, then choose the most sensible answer.\n\n"
        f"Context: {context}\n"
        f"Question: {question}\n"
        "Answer: "
    )

    lls = [compute_ll(model, tokenizer, device, prompt, ans) for ans in answers]
    pred_idx = int(torch.tensor(lls).argmax().item())
    pred = pred_idx + 1  # 1-indexed (A/B/C)
    is_correct = (pred == gold)

    return {
        "context": context,
        "question": question,
        "answers": answers,
        "gold": gold,
        "pred": pred,
        "loglikelihoods": lls,
        "is_correct": is_correct,
    }


# ---------- Main ----------
def main(
    input_file="siqa_dataset.json",
    output_file="gemma_siqa_2000_each.json",
    target_each=2000
):
    hf_login()
    device, dtype = get_device_and_dtype()
    print(f"\nUsing device: {device} | dtype: {dtype}")

    model_name = "google/gemma-2-2b"
    print("Loading Gemma-2-2B model & tokenizer (cache-first)...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, local_files_only=True, torch_dtype=dtype
        ).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
        print("✅ Loaded from local cache.")
    except Exception:
        print("ℹ️ Cache not found; loading from Hub.")
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("✅ Model and tokenizer ready.")

    # ---------- Load Dataset ----------
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")

    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    random.shuffle(data)
    print(f"\n📊 Loaded {len(data)} total SIQA examples.")
    print(f"🎯 Target: {target_each} correct + {target_each} incorrect\n")

    correct_bucket, incorrect_bucket = [], []
    correct_count = incorrect_count = 0
    total_processed = 0

    for idx, ex in enumerate(data, start=1):
        res = evaluate_siqa_example(model, tokenizer, device, ex)
        total_processed += 1

        if res["is_correct"] and correct_count < target_each:
            correct_bucket.append(res)
            correct_count += 1
        elif not res["is_correct"] and incorrect_count < target_each:
            incorrect_bucket.append(res)
            incorrect_count += 1

        # Track total model accuracy, not just collected samples
        if idx == 1:
            running_correct = 0
        running_correct += int(res["is_correct"])
        running_acc = running_correct / total_processed * 100

        print(
            f"[{idx}] → Pred: {res['pred']} | True: {res['gold']} "
            f"| {'✅' if res['is_correct'] else '❌'} "
            f"| Totals → correct: {correct_count}/{target_each}, incorrect: {incorrect_count}/{target_each} "
            f"| Running model acc: {running_acc:.2f}%"
        )

        if correct_count >= target_each and incorrect_count >= target_each:
            print("\n✅ Reached targets for both buckets. Stopping early.")
            break

    # ---------- Save Combined Results ----------
    output_obj = {
        "correct": correct_bucket,
        "incorrect": incorrect_bucket,
        "summary": {
            "total_processed": total_processed,
            "correct_collected": correct_count,
            "incorrect_collected": incorrect_count,
        },
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_obj, f, indent=2)

    print("\n==============================")
    print("✅ SIQA Collection Complete")
    print("==============================")
    print(f"✔️ Correct saved:   {len(correct_bucket)} / {target_each}")
    print(f"✔️ Incorrect saved: {len(incorrect_bucket)} / {target_each}")
    print(f"📁 Results written to: {output_file}\n")


if __name__ == "__main__":
    main()
