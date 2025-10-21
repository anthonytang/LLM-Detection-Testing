import json
from openai import OpenAI
from typing import List, Dict
import sys
import time
import os
from openai import RateLimitError

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

BATCH_SIZE = 50  # Save progress every 50 evaluations
RETRY_WAIT_TIME = 300  # Wait 5 minutes (300 seconds) when hitting quota limit
MAX_RETRIES = 100  # Maximum number of retries before giving up

def load_dataset(filepath: str) -> Dict:
    """Load dataset from JSON file"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def split_dataset(data: Dict) -> Dict:
    """
    Split dataset into train (1000 correct/1000 incorrect),
    val (500 correct/500 incorrect), test (500 correct/500 incorrect)
    """
    correct = data['correct']
    incorrect = data['incorrect']
    
    # Ensure we have enough data
    if len(correct) < 2000 or len(incorrect) < 2000:
        raise ValueError(f"Not enough data. Found {len(correct)} correct and {len(incorrect)} incorrect")
    
    splits = {
        'train': {
            'correct': correct[:1000],
            'incorrect': incorrect[:1000]
        },
        'val': {
            'correct': correct[1000:1500],
            'incorrect': incorrect[1000:1500]
        },
        'test': {
            'correct': correct[1500:2000],
            'incorrect': incorrect[1500:2000]
        }
    }
    
    return splits

def save_split_files(splits: Dict, output_basename: str):
    """Save train, validation, and test splits to separate files"""
    for split_name, split_data in splits.items():
        split_file = f"{output_basename}_{split_name}_split.json"
        with open(split_file, 'w', encoding='utf-8') as f:
            json.dump(split_data, f, indent=2)
        print(f"  Saved {split_name} split to {split_file}")

def format_example_boolq(example: Dict, idx: int) -> str:
    """Format a BoolQ example for the prompt"""
    return f"""<Example {idx}>
<Question>{example['question']}</Question>
<Passage>{example['passage']}</Passage>
<Correct Answer>{example['correct_answer']}</Correct Answer>
<LLM's Answer>{example['gemma_answer']}</LLM's Answer>
<r>{'Correct' if example['is_correct'] else 'Incorrect'}</r>
</Example {idx}>
"""

def format_example_winogrande(example: Dict, idx: int) -> str:
    """Format a Winogrande example for the prompt"""
    return f"""<Example {idx}>
<Question>{example['question']}</Question>
<Option 1>{example['option1']}</Option 1>
<Option 2>{example['option2']}</Option 2>
<Correct Answer>Option {example['gold']}</Correct Answer>
<LLM's Answer>Option {example['gemma_answer']}</LLM's Answer>
<r>{'Correct' if example['is_correct'] else 'Incorrect'}</r>
</Example {idx}>
"""

def format_example_piqa(example: Dict, idx: int) -> str:
    """Format a PIQA example for the prompt"""
    # PIQA may not have gemma_answer or is_correct field
    # We infer what Gemma answered based on correct_answer and is_correct (if available)
    correct_answer = example['correct_answer']
    
    # Check if is_correct field exists
    if 'is_correct' in example:
        is_correct = example['is_correct']
        
        if is_correct:
            gemma_answer = correct_answer  # Gemma got it right
        else:
            # Gemma got it wrong, so it picked the opposite answer
            gemma_answer = 'B' if correct_answer == 'A' else 'A'
    else:
        # If is_correct doesn't exist, assume we don't know what Gemma answered
        # Just use correct answer as placeholder
        gemma_answer = correct_answer
        is_correct = True  # Assume correct if we don't know
    
    return f"""<Example {idx}>
<Goal>{example['goal']}</Goal>
<Solution 1>{example['sol1']}</Solution 1>
<Solution 2>{example['sol2']}</Solution 2>
<Correct Answer>Solution {correct_answer}</Correct Answer>
<LLM's Answer>Solution {gemma_answer}</LLM's Answer>
<r>{'Correct' if is_correct else 'Incorrect'}</r>
</Example {idx}>
"""

def detect_dataset_type(data: Dict) -> str:
    """Detect if dataset is BoolQ, Winogrande, or PIQA based on structure"""
    sample = data['correct'][0] if data['correct'] else data['incorrect'][0]
    if 'passage' in sample:
        return 'boolq'
    elif 'option1' in sample:
        return 'winogrande'
    elif 'goal' in sample and 'sol1' in sample:
        return 'piqa'
    else:
        raise ValueError("Unknown dataset type")

def create_dataset_summary(dataset_type: str) -> str:
    """Create a summary of the dataset"""
    if dataset_type == 'boolq':
        return """This is a BoolQ dataset containing yes/no questions with accompanying passages. 
The LLM must answer 'Yes' or 'No' based on the information in the passage."""
    elif dataset_type == 'winogrande':
        return """This is a Winogrande dataset containing fill-in-the-blank questions with two options. 
The LLM must choose which option (1 or 2) correctly completes the sentence."""
    else:  # piqa
        return """This is a PIQA dataset containing physical commonsense reasoning questions. 
Each question has a goal and two possible solutions (A or B). 
The LLM must choose which solution is more practical and sensible for achieving the goal."""

def generate_hypotheses_for_chunk(examples: List[Dict], chunk_num: int, total_chunks: int, dataset_type: str, format_func) -> str:
    """Generate hypotheses for a single chunk of examples with retry logic"""
    
    examples_list = []
    for idx, example in enumerate(examples, 1):
        examples_list.append(format_func(example, idx))
    
    examples_text = '\n'.join(examples_list)
    dataset_summary = create_dataset_summary(dataset_type)
    
    prompt_text = f"""For this dataset, we want to understand why a language model gets questions incorrect.

<dataset-summary>
{dataset_summary}
</dataset-summary>

<chunk-info>
This is chunk {chunk_num} of {total_chunks}. You are analyzing a subset of the training data.
</chunk-info>

<examples>
{examples_text}
</examples>

Based on these {len(examples)} examples, identify patterns in why the LLM gets examples wrong. 

Your task:
1. Identify patterns in what makes questions difficult for the LLM
2. Note linguistic, semantic, or logical features that correlate with errors
3. Be specific and actionable

Format your response as:
<hypotheses>
Why LLM Gets Questions Right:
[Bullet points of patterns observed in this chunk]

Why LLM Gets Questions Wrong:
[Bullet points of patterns observed in this chunk]
</hypotheses>"""
    
    print(f"  Processing chunk {chunk_num}/{total_chunks} ({len(examples)} examples)...")
    
    retries = 0
    while retries < MAX_RETRIES:
        try:
            response = client.chat.completions.create(
                model="chatgpt-4o-latest",
                messages=[
                    {"role": "user", "content": prompt_text}
                ]
            )
            return response.choices[0].message.content
        except RateLimitError as e:
            retries += 1
            print(f"\n  ⚠️  Quota exceeded! Waiting {RETRY_WAIT_TIME} seconds for account refund...")
            print(f"  Retry {retries}/{MAX_RETRIES}")
            time.sleep(RETRY_WAIT_TIME)
    
    raise Exception(f"Failed after {MAX_RETRIES} retries due to quota limits")

def combine_hypotheses(chunk_hypotheses: List[str], dataset_type: str) -> str:
    """Combine multiple chunk hypotheses into one comprehensive set with retry logic"""
    
    dataset_summary = create_dataset_summary(dataset_type)
    
    # Format all chunk hypotheses
    hypotheses_text = ""
    for i, hyp in enumerate(chunk_hypotheses, 1):
        hypotheses_text += f"\n<Chunk {i} Hypotheses>\n{hyp}\n</Chunk {i} Hypotheses>\n"
    
    prompt_text = f"""You have analyzed a dataset in {len(chunk_hypotheses)} separate chunks. Each chunk provided hypotheses about why a language model gets questions right or wrong.

<dataset-summary>
{dataset_summary}
</dataset-summary>

<all-chunk-hypotheses>
{hypotheses_text}
</all-chunk-hypotheses>

Your task:
Synthesize these {len(chunk_hypotheses)} sets of hypotheses into ONE comprehensive, unified set of hypotheses. 

Instructions:
1. Identify common patterns that appear across multiple chunks (these are the most reliable)
2. Keep unique insights that only appear in one chunk if they seem important
3. Remove redundancy - combine similar hypotheses into single, clear statements
4. Organize from most common/important patterns to least common
5. Make the final hypotheses actionable for predicting future errors

Format your response as:
<hypotheses>
Why LLM Gets Questions Right:
[Unified bullet points - prioritize patterns seen in multiple chunks]

Why LLM Gets Questions Wrong:
[Unified bullet points - prioritize patterns seen in multiple chunks]
</hypotheses>"""
    
    print(f"\n  Combining {len(chunk_hypotheses)} chunk hypotheses into final set...")
    
    retries = 0
    while retries < MAX_RETRIES:
        try:
            response = client.chat.completions.create(
                model="chatgpt-4o-latest",
                messages=[
                    {"role": "user", "content": prompt_text}
                ]
            )
            return response.choices[0].message.content
        except RateLimitError as e:
            retries += 1
            print(f"\n  ⚠️  Quota exceeded! Waiting {RETRY_WAIT_TIME} seconds for account refund...")
            print(f"  Retry {retries}/{MAX_RETRIES}")
            time.sleep(RETRY_WAIT_TIME)
    
    raise Exception(f"Failed after {MAX_RETRIES} retries due to quota limits")

def generate_hypotheses(train_data: Dict, dataset_type: str) -> str:
    """Generate hypotheses by splitting into chunks"""
    
    # Select format function
    if dataset_type == 'boolq':
        format_func = format_example_boolq
    elif dataset_type == 'winogrande':
        format_func = format_example_winogrande
    else:  # piqa
        format_func = format_example_piqa
    
    # Combine correct and incorrect examples
    all_examples = train_data['correct'] + train_data['incorrect']
    total_examples = len(all_examples)
    
    print(f"\nGenerating hypotheses from {total_examples} training examples...")
    
    # Split into chunks of 400 examples each (safe size for all datasets)
    chunk_size = 400
    num_chunks = (total_examples + chunk_size - 1) // chunk_size  # Ceiling division
    
    print(f"  Splitting into {num_chunks} chunks (~{chunk_size} examples per chunk)\n")
    
    # Generate hypotheses for each chunk
    chunk_hypotheses = []
    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, total_examples)
        chunk_examples = all_examples[start_idx:end_idx]
        
        chunk_hyp = generate_hypotheses_for_chunk(
            chunk_examples, 
            i + 1, 
            num_chunks, 
            dataset_type, 
            format_func
        )
        chunk_hypotheses.append(chunk_hyp)
        print(f"  ✓ Chunk {i+1} complete\n")
    
    # Combine all chunk hypotheses
    if num_chunks == 1:
        print("  Only 1 chunk - using hypotheses directly")
        final_hypotheses = chunk_hypotheses[0]
    else:
        final_hypotheses = combine_hypotheses(chunk_hypotheses, dataset_type)
        print(f"  ✓ Combined hypotheses from {num_chunks} chunks")
    
    print("\n✓ Hypothesis generation complete!")
    return final_hypotheses

def format_question_for_evaluation_boolq(example: Dict) -> str:
    """Format a BoolQ question for evaluation"""
    return f"""Question: {example['question']}

Passage: {example['passage']}

The model must answer Yes or No."""

def format_question_for_evaluation_winogrande(example: Dict) -> str:
    """Format a Winogrande question for evaluation"""
    return f"""Question: {example['question']}

Option 1: {example['option1']}
Option 2: {example['option2']}"""

def format_question_for_evaluation_piqa(example: Dict) -> str:
    """Format a PIQA question for evaluation"""
    return f"""Goal: {example['goal']}

Solution A: {example['sol1']}
Solution B: {example['sol2']}

The model must choose which solution (A or B) is more practical and sensible."""

def evaluate_examples(examples: List[Dict], hypotheses: str, dataset_type: str, output_file: str) -> List[Dict]:
    """Evaluate examples using generated hypotheses with batch saving and retry logic"""
    
    eval_prompt_template = """You are evaluating the behavior of a language model on questions.
You will be given:
- A question and its answer choices
- A list of hypotheses about when the model tends to succeed or fail

Your task:
- Based on the hypotheses, predict whether the model will most likely answer the question correctly or incorrectly
- Output must start with "Correct:" if you predict the model will succeed, or "Incorrect:" if you predict the model will fail
- After the label, briefly justify the prediction in 1-2 sentences using the provided hypotheses
- Do NOT reveal the correct answer to the question itself
- Only output the labeled prediction string

<hypotheses>
{hypotheses}
</hypotheses>

<question>
{question}
</question>"""
    
    if dataset_type == 'boolq':
        format_func = format_question_for_evaluation_boolq
    elif dataset_type == 'winogrande':
        format_func = format_question_for_evaluation_winogrande
    else:  # piqa
        format_func = format_question_for_evaluation_piqa
    
    # Check if there's a progress file to resume from
    progress_file = output_file.replace('.json', '_progress.json')
    if os.path.exists(progress_file):
        print(f"  Found progress file, resuming from previous run...")
        with open(progress_file, 'r', encoding='utf-8') as f:
            results = json.load(f)
        start_idx = len(results)
        print(f"  Resuming from example {start_idx + 1}/{len(examples)}")
    else:
        results = []
        start_idx = 0
    
    for idx in range(start_idx, len(examples)):
        example = examples[idx]
        print(f"Evaluating example {idx + 1}/{len(examples)}...", end='\r')
        
        question_text = format_func(example)
        prompt = eval_prompt_template.format(
            hypotheses=hypotheses,
            question=question_text
        )
        
        # Retry logic for API calls
        retries = 0
        prediction_text = None
        while retries < MAX_RETRIES and prediction_text is None:
            try:
                response = client.chat.completions.create(
                    model="chatgpt-4o-latest",
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3
                )
                prediction_text = response.choices[0].message.content.strip()
            except RateLimitError as e:
                retries += 1
                print(f"\n  ⚠️  Quota exceeded at example {idx + 1}! Waiting {RETRY_WAIT_TIME} seconds for account refund...")
                print(f"  Retry {retries}/{MAX_RETRIES}")
                time.sleep(RETRY_WAIT_TIME)
        
        if prediction_text is None:
            raise Exception(f"Failed after {MAX_RETRIES} retries due to quota limits")
        
        # Parse prediction
        if prediction_text.startswith("Correct:"):
            gpt_prediction = "Correct"
        elif prediction_text.startswith("Incorrect:"):
            gpt_prediction = "Incorrect"
        else:
            gpt_prediction = "Unknown"
        
        # Build result entry based on dataset type
        if dataset_type == 'boolq':
            result = {
                "id": example['id'],
                "question": example['question'],
                "passage": example['passage'],
                "correct_answer": example['correct_answer'],
                "gpt_prediction": gpt_prediction,
                "gpt_explanation": prediction_text,
                "actual_result": "Correct" if example.get('is_correct', True) else "Incorrect"
            }
        elif dataset_type == 'winogrande':
            result = {
                "id": example['id'],
                "question": example['question'],
                "option1": example['option1'],
                "option2": example['option2'],
                "correct_answer": f"Option {example['gold']}",
                "gpt_prediction": gpt_prediction,
                "gpt_explanation": prediction_text,
                "actual_result": "Correct" if example.get('is_correct', True) else "Incorrect"
            }
        else:  # piqa
            # PIQA may not have is_correct field, need to check
            if 'is_correct' in example:
                actual_result = "Correct" if example['is_correct'] else "Incorrect"
            else:
                # If is_correct doesn't exist, we can't determine actual result
                actual_result = "Unknown"
            
            result = {
                "id": example['id'],
                "goal": example['goal'],
                "solution_a": example['sol1'],
                "solution_b": example['sol2'],
                "correct_answer": f"Solution {example['correct_answer']}",
                "gpt_prediction": gpt_prediction,
                "gpt_explanation": prediction_text,
                "actual_result": actual_result
            }
        
        results.append(result)
        
        # Save progress every BATCH_SIZE examples
        if (idx + 1) % BATCH_SIZE == 0:
            print(f"\n  Saving progress at {idx + 1}/{len(examples)} examples...")
            with open(progress_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2)
            print(f"  ✓ Progress saved to {progress_file}")
    
    # Final save and cleanup
    print(f"\n  Saving final results...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    
    # Remove progress file after successful completion
    if os.path.exists(progress_file):
        os.remove(progress_file)
        print(f"  Removed progress file")
    
    print()  # New line after progress
    return results

def main(input_filepath: str):
    """Main processing function"""
    print(f"Loading dataset from {input_filepath}...")
    data = load_dataset(input_filepath)
    
    # Detect dataset type
    dataset_type = detect_dataset_type(data)
    print(f"Detected dataset type: {dataset_type}")
    
    # Split dataset
    print("Splitting dataset...")
    splits = split_dataset(data)
    
    # Save split files
    output_basename = input_filepath.rsplit('.', 1)[0]
    print("Saving train/val/test splits...")
    save_split_files(splits, output_basename)
    
    # Generate hypotheses from training data (using chunking)
    hypotheses = generate_hypotheses(splits['train'], dataset_type)
    
    # Save hypotheses for reference
    hypotheses_file = f"{output_basename}_hypotheses.txt"
    with open(hypotheses_file, 'w', encoding='utf-8') as f:
        f.write(hypotheses)
    print(f"\nHypotheses saved to {hypotheses_file}")
    
    # Evaluate validation set
    print("\nEvaluating validation set...")
    val_examples = splits['val']['correct'] + splits['val']['incorrect']
    val_output_file = f"{output_basename}_validation_results.json"
    val_results = evaluate_examples(val_examples, hypotheses, dataset_type, val_output_file)
    print(f"✓ Validation results saved to {val_output_file}")
    
    # Evaluate test set
    print("\nEvaluating test set...")
    test_examples = splits['test']['correct'] + splits['test']['incorrect']
    test_output_file = f"{output_basename}_test_results.json"
    test_results = evaluate_examples(test_examples, hypotheses, dataset_type, test_output_file)
    print(f"✓ Test results saved to {test_output_file}")
    
    # Calculate accuracy
    val_correct = sum(1 for r in val_results if r['gpt_prediction'] == r['actual_result'] and r['actual_result'] != 'Unknown')
    val_total = sum(1 for r in val_results if r['actual_result'] != 'Unknown')
    test_correct = sum(1 for r in test_results if r['gpt_prediction'] == r['actual_result'] and r['actual_result'] != 'Unknown')
    test_total = sum(1 for r in test_results if r['actual_result'] != 'Unknown')
    
    print(f"\n=== Results Summary ===")
    if val_total > 0:
        print(f"Validation Accuracy: {val_correct}/{val_total} = {val_correct/val_total*100:.2f}%")
    else:
        print(f"Validation: No examples with known actual results")
    
    if test_total > 0:
        print(f"Test Accuracy: {test_correct}/{test_total} = {test_correct/test_total*100:.2f}%")
    else:
        print(f"Test: No examples with known actual results")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <input_json_file>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    main(input_file)