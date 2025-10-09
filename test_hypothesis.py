import argparse
import json
import os
import time
import requests
import re
from tqdm import tqdm

try:
    from neuronpedia.np_sae_feature import SAEFeature
    from sklearn.metrics import confusion_matrix, recall_score
except ImportError:
    print("Error: Required libraries are not installed.")
    print("Please run: pip install neuronpedia scikit-learn tqdm requests")
    exit(1)

OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SAMPLE_SIZE = 32
TARGET_LAYERS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]
EXPLANATION_CACHE = {}
AI_JUDGE_CACHE = {}

COST_PER_M_INPUT = 0.50
COST_PER_M_OUTPUT = 1.50
TOTAL_PROMPT_TOKENS = 0
TOTAL_COMPLETION_TOKENS = 0
TOTAL_API_TIME = 0.0

def is_explanation_relevant_chatgpt(question, explanations):
    # Uses the OpenAI API to determine if any explanation is relevant to the question
    global TOTAL_PROMPT_TOKENS, TOTAL_COMPLETION_TOKENS, TOTAL_API_TIME
    
    # Use a sorted tuple of explanations for consistent cache key
    cache_key = (question, tuple(sorted(explanations)))
    if cache_key in AI_JUDGE_CACHE:
        return AI_JUDGE_CACHE[cache_key]

    system_prompt = (
        "You are a **strict linguistic expert**. Your task is to determine if any of the provided 'Feature Explanations' "
        "are **relevant** to the 'Question'. Relevance is defined in two ways: "
        "**Directness is Key:** Relevance requires a strong, direct semantic link between the explanation's content and the question's specific topics (e.g., 'animal conditions' for a question about a 'guinea pig cage')."
        "**Ignore Generic Features:** Disregard explanations about abstract grammar, code, punctuation, or formatting. Vague, high-level concepts like 'tools' or 'procedures' are only relevant if they are tied to a specific action in the question."
        "**Evaluate All, Pick One:** You must consider all explanations before making your decision. Your reasoning must be based on the single most relevant feature."
        "If the explanation is about generic abstract concepts, grammar, punctuation, legal terms, or programming code, or anything else that seems like random garbage, you **MUST** answer No."
        "If it can be reasoned as **DIRECTLY AND CLEARLY** relevant to the question, your decision must be Yes. Anything else should result in an answer of No." 
        "Your final response MUST be a single JSON object with two keys: 'decision' ('Yes' or 'No') and 'reasoning' (a detailed explanation for your decision)."
    )

    explanation_list = "\n".join([f"- \"{exp}\"" for exp in explanations])
    user_prompt = (
        f"Question: \"{question}\"\n\n"
        f"Feature Explanations:\n{explanation_list}\n\n"
        "Based on the rules, are any of the explanations relevant to the question? Output the decision and reasoning as a JSON object."
    )

    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "gpt-3.5-turbo",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0,
        "max_tokens": 500
    }

    # Initialize return values for failure case
    is_relevant = False
    raw_response = "API_FAILURE_AFTER_RETRIES"
    gpt_reasoning = "API call failed after all retries."
    
    # Basic exponential backoff and retry mechanism
    for i in range(3):
        start_time = time.time()
        try:
            response = requests.post(OPENAI_API_URL, headers=headers, json=payload, timeout=60)
            end_time = time.time()
            TOTAL_API_TIME += (end_time - start_time)

            response.raise_for_status()
            result = response.json()
            
            usage = result.get('usage', {})
            TOTAL_PROMPT_TOKENS += usage.get('prompt_tokens', 0)
            TOTAL_COMPLETION_TOKENS += usage.get('completion_tokens', 0)
            
            raw_response = result['choices'][0]['message']['content'].strip()
            
            gpt_reasoning = "N/A - Parsing Failed or No Reason Extracted"
            is_relevant = False
            
            # Use regex to find the first JSON object in the response
            json_match = re.search(r'\{.*\}', raw_response, re.DOTALL)
            
            if json_match:
                json_text = json_match.group(0)
                parsed_json = json.loads(json_text)
                
                decision = parsed_json.get("decision", "").strip().lower()
                gpt_reasoning = parsed_json.get("reasoning", "Reasoning not found in JSON.").strip()
                
                is_relevant = decision.startswith("yes")
            else:
                # Fallback if no JSON is found
                is_relevant = raw_response.lower().startswith("yes")
                gpt_reasoning = f"JSON parsing failed. Decision inferred from raw start: '{raw_response[:20]}...'"
                
            # Cache and return the result tuple
            result_tuple = (is_relevant, raw_response, gpt_reasoning)
            AI_JUDGE_CACHE[cache_key] = result_tuple
            time.sleep(1)
            return result_tuple
            
        except (requests.exceptions.RequestException, json.JSONDecodeError) as e:
            error_type = "JSON Error" if isinstance(e, json.JSONDecodeError) else "API Error"
            print(f"  [{error_type}: {e}. Retrying in {2**i}s...]")
            time.sleep(2**i)
            
    # If all retries fail, return the initialized failure state
    return (is_relevant, raw_response, gpt_reasoning)


def test_hypothesis(input_file):
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found.")
        return

    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    correct_examples = [ex for ex in data if ex.get("model_is_correct")][:SAMPLE_SIZE]
    wrong_examples = [ex for ex in data if not ex.get("model_is_correct")][:SAMPLE_SIZE]

    if len(correct_examples) < SAMPLE_SIZE or len(wrong_examples) < SAMPLE_SIZE:
        print(f"Warning: Not enough data. Found {len(correct_examples)} correct and {len(wrong_examples)} wrong examples.")
        print(f"Running with {min(len(correct_examples), len(wrong_examples))} samples for each category.")
        
    examples_to_test = correct_examples + wrong_examples
    y_true = []
    y_pred = []
    log_data = []

    print(f"Starting hypothesis test on {len(examples_to_test)} examples (Target: {SAMPLE_SIZE} Correct, {SAMPLE_SIZE} Wrong)...")

    for i, example in enumerate(tqdm(examples_to_test, desc="Analyzing Examples")):
        y_true.append(example['model_is_correct'])

        is_correct_ground_truth = example["model_is_correct"]
        solution_details = example["positive_solution_details"] if is_correct_ground_truth else example["negative_solution_details"]
        features_by_layer = solution_details.get("active_features_by_layer", {})

        # --- FEATURE COLLECTION AND ID LOGGING ---
        extracted_features_data = []
        explanation_strings = [] # List of strings to send to GPT prompt
        for layer in TARGET_LAYERS:
            layer_str = str(layer)
            if layer_str in features_by_layer and features_by_layer[layer_str]:
                # Collect top 2 features' explanations
                top_features = features_by_layer[layer_str][:3]
                for feature in top_features:
                    # 'description' key is already in the file from analyze_features.py
                    explanation = feature.get("description", "Explanation not found.") 

                    extracted_features_data.append({
                        "layer_id": layer,
                        "feature_id": feature.get("feature_id"),
                        "explanation": explanation
                    })
                    explanation_strings.append(explanation)
        
        # Ask the ChatGPT judge for decision, raw response, and reasoning
        is_relevant, raw_ai_response, gpt_reasoning = is_explanation_relevant_chatgpt(example["question"], explanation_strings)

        # Our prediction: If the judge found a relevant feature, we predict the answer was Correct (True).
        # Otherwise, we predict it was Wrong (False).
        y_pred.append(is_relevant)

        log_data.append({
            "example_index": i,
            "question": example["question"],
            "ground_truth_correctness": is_correct_ground_truth,
            "extracted_features": extracted_features_data,
            "ai_judge_raw_response": raw_ai_response,
            "gpt_reasoning": gpt_reasoning,
            "detector_prediction": is_relevant
        })

    print("\n--- Test Complete: Results ---")
    
    log_filename = "annotation_run_log.json"
    try:
        with open(log_filename, "w", encoding="utf-8") as f:
            json.dump(log_data, f, indent=4)
        print(f"\nSuccessfully logged all {len(log_data)} ChatGPT interactions and data to '{log_filename}'.")
    except Exception as e:
        print(f"\nWarning: Could not save log file '{log_filename}': {e}")

    # Calculate metrics
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[True, False]).ravel()
    
    recall = recall_score(y_true, y_pred, pos_label=False)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    print("\nConfusion Matrix:")
    print("                 Predicted: CORRECT | Predicted: WRONG")
    print(f"Actual: CORRECT    {tn:^16} | {fp:^16}")
    print(f"Actual: WRONG      {fn:^16} | {tp:^16}")
    print("\n--- Performance Metrics ---")
    print(f"Recall (Sensitivity): {recall:.2%}")
    print("  -> This measures our ability to detect WRONG answers.")
    print(f"  -> Of all the answers Gemma actually got wrong, our detector caught {recall:.0%} of them.")
    
    print(f"\nSpecificity: {specificity:.2%}")
    print("  -> This measures our ability to correctly identify CORRECT answers.")
    print(f"  -> Of all the answers Gemma actually got right, our detector correctly identified {specificity:.0%} of them as correct.")

    total_cost = (TOTAL_PROMPT_TOKENS / 1_000_000 * COST_PER_M_INPUT) + \
                 (TOTAL_COMPLETION_TOKENS / 1_000_000 * COST_PER_M_OUTPUT)
    
    print("\n--- Resource Metrics ---")
    print(f"Total API Calls Made: {len(log_data)}")
    print(f"Total API Time (s): {TOTAL_API_TIME:.2f}")
    print(f"Total Prompt Tokens Sent: {TOTAL_PROMPT_TOKENS:,}")
    print(f"Total Completion Tokens Received: {TOTAL_COMPLETION_TOKENS:,}")
    print(f"Estimated Cost (USD): ${total_cost:.4f}")
    print(f"  (Based on GPT-3.5-turbo rates: Input ${COST_PER_M_INPUT}/M, Output ${COST_PER_M_OUTPUT}/M)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Automate testing of the 'Relevant Feature' hypothesis.")
    parser.add_argument(
        "--file",
        type=str,
        default="piqa_results_with_explanations.json",
        help="The JSON file with PIQA results."
    )
    args = parser.parse_args()
    test_hypothesis(args.file)
