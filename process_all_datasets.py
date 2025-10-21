import subprocess
import sys
import time
from datetime import datetime

DATASETS = [
    'gemma_boolq_answers.json',
    'gemma_winogrande_2000_each.json',
    'piqa_4000_balanced_train.json'
]

def run_dataset(dataset_file):
    print(f"\n{'='*80}")
    print(f"Starting processing for: {dataset_file}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}\n")
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            [sys.executable, 'dataset_processor.py', dataset_file],
            capture_output=True,
            text=True,
            check=True
        )
        
        print(result.stdout)
        
        if result.stderr:
            print("Warnings/Errors:")
            print(result.stderr)
        
        elapsed_time = time.time() - start_time
        print(f"\n✓ Successfully completed {dataset_file}")
        print(f"Time taken: {elapsed_time/60:.2f} minutes ({elapsed_time/3600:.2f} hours)")
        
        return True
        
    except subprocess.CalledProcessError as e:
        elapsed_time = time.time() - start_time
        print(f"\n✗ Failed to process {dataset_file}")
        print(f"Time taken: {elapsed_time/60:.2f} minutes")
        print(f"\nError output:")
        print(e.stdout)
        print(e.stderr)
        return False
    
    except FileNotFoundError:
        print(f"\n✗ Error: dataset_processor.py not found in current directory")
        print("Make sure dataset_processor.py is in the same folder as this script")
        return False

def main():
    print("="*80)
    print("DATASET PROCESSING")
    print("="*80)
    print(f"\nDatasets to process:")
    for i, dataset in enumerate(DATASETS, 1):
        print(f"  {i}. {dataset}")
    
    print(f"\nTotal datasets: {len(DATASETS)}")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nOutput files will be created:")
    for dataset in DATASETS:
        base = dataset.rsplit('.', 1)[0]
        print(f"  • {base}_hypotheses.txt")
        print(f"  • {base}_validation_results.json")
        print(f"  • {base}_test_results.json")
    
    print("\nPress Ctrl+C at any time to cancel.\n")
    
    print("Starting in 5 seconds...")
    for i in range(5, 0, -1):
        print(f"  {i}...")
        time.sleep(1)
    print()
    
    overall_start = time.time()
    results = {}
    
    for i, dataset in enumerate(DATASETS, 1):
        print(f"\n\n{'#'*80}")
        print(f"# DATASET {i} of {len(DATASETS)}")
        print(f"{'#'*80}")
        
        success = run_dataset(dataset)
        results[dataset] = success
        
        if i < len(DATASETS):
            print(f"\n⏸️  Waiting 10 seconds before next dataset...")
            time.sleep(10)
    
    overall_time = time.time() - overall_start
    
    print("\n\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    print(f"Total time: {overall_time/60:.2f} minutes ({overall_time/3600:.2f} hours)")
    print(f"Start: {datetime.fromtimestamp(overall_start).strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"End: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nResults:")
    
    successful = 0
    failed = 0
    
    for dataset, success in results.items():
        status = "✓ SUCCESS" if success else "✗ FAILED"
        print(f"  {status}: {dataset}")
        if success:
            successful += 1
        else:
            failed += 1
    
    print(f"\nSuccessful: {successful}/{len(DATASETS)}")
    print(f"Failed: {failed}/{len(DATASETS)}")
    
    if failed == 0:
        print("\nAll datasets processed successfully!")
        print("\nOutput files created:")
        for dataset in DATASETS:
            base = dataset.rsplit('.', 1)[0]
            print(f"\n  {dataset}:")
            print(f"    • {base}_hypotheses.txt")
            print(f"    • {base}_validation_results.json")
            print(f"    • {base}_test_results.json")
    else:
        print(f"\n⚠️  {failed} dataset(s) failed. Check the error messages above.")
    
    print("\n" + "="*80)
    print("Good morning! Your overnight run is complete. ☀️")
    print("="*80)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Process interrupted by user (Ctrl+C)")
        print("Partial results may be saved.")
        print("Exiting...")
        sys.exit(1)