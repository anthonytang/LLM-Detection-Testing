import openai
import json
import os

# Set the number of top features to extract from each layer.
TOP_K = 3
# The name of your input JSON file.
INPUT_FILE_NAME = "piqa_results_with_explanations.json"
# The OpenAI model to use for generating the hypothesis.
GENERATIVE_MODEL = "gpt-4o"

def get_openai_client():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("API key not found. Please set the OPENAI_API_KEY environment variable.")
    return openai.OpenAI(api_key=api_key)


def extract_top_k_features_per_layer(solution_details, k):
    # Extracts the top k features with the highest activation from each layer.
    if not solution_details or 'active_features_by_layer' not in solution_details:
        return "No feature data available."

    output_lines = []
    features_by_layer = solution_details['active_features_by_layer']
    
    # Sort layers numerically to ensure order from 0 to 25
    sorted_layer_keys = sorted(features_by_layer.keys(), key=int)

    for layer_num_str in sorted_layer_keys:
        features = features_by_layer[layer_num_str]
        # The features in the JSON are already sorted by activation, so we just slice.
        top_features = features[:k]
        
        if top_features:
            output_lines.append(f"  - Layer {layer_num_str}:")
            for feature in top_features:
                description = feature.get('description', 'N/A').strip()
                activation = feature.get('activation', 0)
                output_lines.append(f'    - Feature {feature.get("feature_id")}: "{description}" (Activation: {activation:.2f})')

    return "\n".join(output_lines)


def generate_hypothesis(client, prompt):
    #Sends the constructed prompt to the OpenAI API and returns the hypothesis.
    try:
        # The new Chat Completions API is structured with roles
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model=GENERATIVE_MODEL,
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        return f"An error occurred while calling the OpenAI API: {e}"


def build_prompt(example, positive_features_str, negative_features_str, k):
    #Constructs the final prompt string to be sent to the generative model.
    prompt = f"""
You are an expert AI interpretability researcher. Your task is to analyze the internal feature activations of Gemma 2B on a specific problem and generate a hypothesis for why it failed.

**Problem Context:**
The model was given a question from the PIQA (Physical Interaction QA) dataset and chose the incorrect solution. I will provide the question, the two possible solutions, and the top {k} most activated features from each of the model's 26 layers for both solutions.

**Analysis Task:**
Based *only* on the feature data provided, generate a concise hypothesis explaining why the model likely chose the incorrect answer. Contrast the patterns you see in the features activated for the correct vs. incorrect choice. For example, did one path rely more on semantic understanding vs. superficial syntax? Did the model misinterpret the domain of the question?

---
**INPUT DATA**

**Question:** "{example['question']}"

**Correct Answer (Rejected by the model):** "{example['positive_solution_details']['text']}"
Log Likelihood: {example['positive_solution_details']['log_likelihood']}

**Incorrect Answer (Chosen by the model):** "{example['negative_solution_details']['text']}"
Log Likelihood: {example['negative_solution_details']['log_likelihood']}

---
**FEATURE ANALYSIS**

**Top {k} Features Activated for the CORRECT (but rejected) Answer:**
{positive_features_str}

**Top {k} Features Activated for the INCORRECT (and chosen) Answer:**
{negative_features_str}

---
**GENERATED HYPOTHESIS:**
"""
    return prompt


def main():
    try:
        client = get_openai_client()
        with open(INPUT_FILE_NAME, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: The file '{INPUT_FILE_NAME}' was not found.")
        return
    except json.JSONDecodeError:
        print(f"Error: The file '{INPUT_FILE_NAME}' is not a valid JSON file.")
        return
    except ValueError as e:
        print(f"Configuration Error: {e}")
        return

    print(f"Starting analysis of '{INPUT_FILE_NAME}'...\n")
    
    for example in data:
        # We only care about the questions the model got wrong.
        if not example.get("model_is_correct"):
            question_id = example.get("example_id", "N/A")
            print(f"Analyzing Incorrect Answer for Question ID: {question_id}")
            print("="*50)
            
            # Extract top features for both the right and wrong answers
            positive_features_str = extract_top_k_features_per_layer(
                example.get('positive_solution_details'), TOP_K
            )
            negative_features_str = extract_top_k_features_per_layer(
                example.get('negative_solution_details'), TOP_K
            )
            
            # Build the detailed prompt for the generative model
            prompt = build_prompt(example, positive_features_str, negative_features_str, TOP_K)
            
            # Call the API and get the hypothesis
            print(f"Generating hypothesis with OpenAI model: {GENERATIVE_MODEL}...")
            hypothesis = generate_hypothesis(client, prompt)
            
            print("\n--- HYPOTHESIS ---")
            print(hypothesis)
            print("--- END OF ANALYSIS ---\n\n")

if __name__ == "__main__":
    main()