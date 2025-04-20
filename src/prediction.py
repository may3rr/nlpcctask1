import json
import numpy as np
import argparse
import os
from tqdm import tqdm  # Optional: for progress bar

def predict_on_test(test_file_path, best_log_x, best_threshold, submission_file_path, den_epsilon=1e-6):
    """
    Loads test data, calculates Binoculars scores using provided parameters,
    generates a submission file in JSON format with predictions.

    Args:
        test_file_path (str): Path to the test_scores.json file.
        best_log_x (float): The best log_x value.
        best_threshold (float): The best classification threshold.
        submission_file_path (str): Path to save the submission JSON file.
    """
    print(f"Loading test data from {test_file_path}...")
    try:
        with open(test_file_path, 'r', encoding='utf-8') as f:
            test_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Test file not found at {test_file_path}")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {test_file_path}")
        return

    predictions = []

    # Process each sample in the test set
    print("Calculating scores and generating predictions...")
    for item in tqdm(test_data, desc="Predicting on test set"):
        performer_ppl = item['performer_perplexity']
        cross_ppl = item['cross_perplexity']
        sample_id = item.get('id', None)  # Use .get() to handle missing 'id' gracefully

        # Handle cases with invalid id
        if sample_id is None:
            print(f"Warning: Item is missing 'id'. Assigning a temporary id for prediction.")
            sample_id = "missing_id" # Or any appropriate placeholder

        denominator = best_log_x - cross_ppl

        # Classify based on score
        if denominator > den_epsilon:
            score = performer_ppl / denominator
            prediction = 1 if score < best_threshold else 0  # 1 for AI, 0 for Human
        else:
            print(f"Warning: Denominator close to zero for sample {sample_id}. Assigning a default prediction (0 = Human). Adjust den_epsilon if needed.")
            prediction = 0 # Default: classify as Human

        predictions.append({"id": sample_id, "prediction": prediction})

    # Write the predictions to a JSON file
    try:
        with open(submission_file_path, 'w', encoding='utf-8') as outfile:
            json.dump(predictions, outfile, indent=2, ensure_ascii=False)
        print(f"Predictions saved to {submission_file_path}")
    except IOError:
        print(f"Error: Could not write predictions to {submission_file_path}")


# Main execution block
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate submission JSON file for test data using hardcoded parameters.')
    parser.add_argument('--test_file', type=str, required=True, help='Path to the test_scores.json file.')
    parser.add_argument('--submission_file', type=str, default='submission.json', help='Path to save the submission JSON file.')

    args = parser.parse_args()

    # Hardcoded parameters from dev set optimization
    best_log_x = 7.4146
    best_threshold = 0.4118

    # Run prediction and create submission file
    predict_on_test(args.test_file, best_log_x, best_threshold, args.submission_file)
