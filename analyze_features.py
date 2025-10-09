import json
import argparse
import os
from tqdm import tqdm
import time
from concurrent.futures import ThreadPoolExecutor

try:
    from neuronpedia.np_sae_feature import SAEFeature
except ImportError:
    print("Error: The 'neuronpedia' library is not installed.")
    print("Please install it by running: pip install neuronpedia")
    exit(1)

def fetch_explanation_worker(layer_idx, feature_id):
    sae_name = f"{layer_idx}-gemmascope-res-16k"
    
    try:
        sae_feature = SAEFeature.get("gemma-2-2b", sae_name, str(feature_id))
        data = json.loads(sae_feature.jsonData)
        print(data['explanations'])
        
        explanation = data.get('explanations', [{}])[0].get('description', "Explanation not found.")
        
        time.sleep(0.1) 
        return (layer_idx, feature_id), explanation

    except Exception as e:
        error_message = f"Could not fetch explanation for L{layer_idx} F{feature_id} using SAE name '{sae_name}'."
        return (layer_idx, feature_id), error_message

def main(args):
    if not os.path.exists(args.input_file):
        print(f"Error: Input file not found at '{args.input_file}'")
        print("Please run the data generation script first.")
        return

    print(f"Loading feature data from '{args.input_file}'...")
    with open(args.input_file, "r", encoding="utf-8") as f:
        results = json.load(f)
    print(f"Loaded {len(results)} examples to process.")

    if args.num_test_samples is not None and args.num_test_samples > 0:
        if args.num_test_samples > len(results):
            print(f"Warning: Requested {args.num_test_samples} samples, but file only contains {len(results)}. Processing all.")
        else:
            results = results[:args.num_test_samples]
            print(f"Processing a test sample of {len(results)} examples.")
    
    all_unique_features = set()
    print("Scanning file to identify all unique features...")
    for example in results:
        for sol_type in ["positive_solution_details", "negative_solution_details"]:
            if sol_type in example and "active_features_by_layer" in example[sol_type]:
                for layer_str, features in example[sol_type]["active_features_by_layer"].items():
                    try:
                        layer_idx = int(layer_str)
                        for feature in features:
                            feature_id = feature.get("feature_id")
                            if feature_id is not None:
                                all_unique_features.add((layer_idx, feature_id))
                    except ValueError:
                        continue
    
    if not all_unique_features:
        print("No features found to annotate. Exiting.")
        return
        
    print(f"Found {len(all_unique_features)} unique features to fetch explanations for.")

    explanation_cache = {}
    features_to_fetch = list(all_unique_features)
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        with tqdm(total=len(features_to_fetch), desc="Fetching explanations") as pbar:
            for key, explanation in executor.map(lambda f: fetch_explanation_worker(f[0], f[1]), features_to_fetch):
                explanation_cache[key] = explanation
                pbar.update(1)

    print("\nAll explanations fetched. Annotating results...")
    for example in tqdm(results, desc="Annotating examples"):
        for sol_type in ["positive_solution_details", "negative_solution_details"]:
            if sol_type in example and "active_features_by_layer" in example[sol_type]:
                for layer_str, features in example[sol_type]["active_features_by_layer"].items():
                    try:
                        layer_idx = int(layer_str)
                        for feature in features:
                            feature_id = feature.get("feature_id")
                            if feature_id is not None:
                                cache_key = (layer_idx, feature_id)
                                if cache_key in explanation_cache:
                                    feature["description"] = explanation_cache[cache_key]
                                else:
                                    feature["description"] = "Explanation not available (was not in initial scan)."
                    except ValueError:
                        continue

    print(f"\nAnnotation complete. Saving enriched data to '{args.output_file}'...")
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    
    print("\nProcess finished successfully.")
    print(f"You can now inspect '{args.output_file}' to see the feature descriptions.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Annotate SAE feature data with explanations from Neuronpedia.")
    parser.add_argument(
        "--input-file",
        type=str,
        default="piqa_sae_top100_all_layers.json",
        help="The JSON file containing the results from the feature extraction script."
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="piqa_results_with_explanations.json",
        help="The name of the new JSON file to save the annotated results to."
    )
    parser.add_argument(
        "--num-test-samples",
        type=int,
        default=None,
        help="Number of examples to process for a quick test. Processes all examples by default."
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=10,
        help="Number of parallel threads to use for fetching explanations."
    )
    args = parser.parse_args()
    main(args)
