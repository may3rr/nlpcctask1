import json
import numpy as np
from tqdm import tqdm
import argparse
import os
from sklearn.metrics import f1_score

# Helper function to calculate classification metrics
def calculate_metrics(scores, labels, threshold):
    """
    Calculates classification metrics (Accuracy, Precision, Recall, Macro F1)
    assuming scores < threshold predict positive (label 1, e.g., AI).
    """
    predictions = [1 if score < threshold else 0 for score in scores]

    # Assuming label 1 is positive class (AI), 0 is negative (Human)
    tp = sum(1 for pred, label in zip(predictions, labels) if pred == 1 and label == 1)
    fp = sum(1 for pred, label in zip(predictions, labels) if pred == 1 and label == 0)
    fn = sum(1 for pred, label in zip(predictions, labels) if pred == 0 and label == 1)
    tn = sum(1 for pred, label in zip(predictions, labels) if pred == 0 and label == 0)

    # Avoid division by zero for precision, recall
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    accuracy = (tp + tn) / len(labels) if len(labels) > 0 else 0

    # Calculate Macro F1 using sklearn
    macro_f1 = f1_score(labels, predictions, average='macro')

    return accuracy, precision, recall, macro_f1

def find_best_threshold(dev_file_path):
    """
    Finds the best threshold T for the original Binoculars formula
    B(s) = performer_perplexity / cross_perplexity
    using the dev set, optimizing for Macro F1 score.

    Args:
        dev_file_path (str): Path to the dev_scores.json file.

    Returns:
        dict: A dictionary containing 'best_threshold', 'best_macro_f1'.
              Returns None if unable to find threshold (e.g., no valid data).
    """
    print(f"Loading dev data from {dev_file_path}...")
    try:
        with open(dev_file_path, 'r', encoding='utf-8') as f:
            dev_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: File not found at {dev_file_path}")
        return None
    except json.JSONDecodeError:
         print(f"Error: Could not decode JSON from {dev_file_path}")
         return None


    binoculars_scores = []
    labels = []

    # Filter data to include only entries with necessary scores and label
    valid_data = [
        item for item in dev_data
        if 'performer_perplexity' in item and 'cross_perplexity' in item and 'label' in item
    ]

    if not valid_data:
        print("Error: No valid data points found in dev file with required keys ('performer_perplexity', 'cross_perplexity', 'label').")
        return None
        
    # Also check if labels are not all the same, otherwise F1 is ill-defined
    unique_labels = set([item['label'] for item in valid_data])
    if len(unique_labels) < 2:
         print("Error: All samples in dev file have the same label. Cannot calculate Macro F1 score.")
         return None

    for item in valid_data:
        performer_perplexity = item['performer_perplexity']
        cross_perplexity = item['cross_perplexity']
        label = item['label']

        # Calculate the binoculars score
        if cross_perplexity != 0:
            binoculars_score = performer_perplexity / cross_perplexity
            binoculars_scores.append(binoculars_score)
            labels.append(label)
        else:
            print(f"Warning: Skipping item due to cross_perplexity being zero.")
            continue


    if not binoculars_scores:
        print("Error: No valid binoculars scores could be computed.")
        return None

    print(f"Loaded {len(binoculars_scores)} valid data points.")

    # Find best threshold for the binoculars scores
    unique_scores = sorted(list(set(binoculars_scores)))
    best_macro_f1 = -1.0
    best_threshold = None

    # Check thresholds *between* unique scores or just the scores themselves?
    # Iterating through unique scores as thresholds is simpler and sufficient.
    for threshold in tqdm(unique_scores, desc="Searching for best threshold"):
        accuracy, precision, recall, macro_f1 = calculate_metrics(binoculars_scores, labels, threshold)
        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            best_threshold = threshold


    if best_threshold is None:
         print("Error: Could not find any valid threshold.")
         return None

    print("\n--- Best Parameters Found ---")
    print(f"Best Threshold T: {best_threshold:.4f}")
    print(f"Max Macro F1 on Dev Set: {best_macro_f1:.4f}")
    print("-----------------------------")

    return {
        'best_threshold': float(best_threshold),
        'best_macro_f1': float(best_macro_f1)
    }

# Main execution block
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Find optimal threshold for Binoculars formula using dev set, optimizing for Macro F1.')
    parser.add_argument('--dev_file', type=str, required=True, help='Path to the dev_scores.json file.')
    parser.add_argument('--output_params_file', type=str, default='best_binoculars_params.json', help='File to save the best parameters.')

    args = parser.parse_args()

    # Check if dev file exists
    if not os.path.exists(args.dev_file):
        print(f"Error: Dev file not found at {args.dev_file}")
        exit()

    best_params = find_best_threshold(args.dev_file)

    if best_params:
        try:
            with open(args.output_params_file, 'w', encoding='utf-8') as f:
                json.dump(best_params, f, indent=2)
            print(f"Best parameters saved to {args.output_params_file}")
        except IOError as e:
            print(f"Error saving parameters to file: {e}")
    else:
         print("Failed to find best parameters.")
