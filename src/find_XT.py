import json
import numpy as np
from tqdm import tqdm
import argparse
import os

# Helper function to calculate classification metrics
def calculate_metrics(scores, labels, threshold):
    """
    Calculates classification metrics (Accuracy, Precision, Recall, F1)
    assuming scores < threshold predict positive (label 1, e.g., AI).
    """
    predictions = [1 if score < threshold else 0 for score in scores]

    # Assuming label 1 is positive class (AI), 0 is negative (Human)
    tp = sum(1 for pred, label in zip(predictions, labels) if pred == 1 and label == 1)
    fp = sum(1 for pred, label in zip(predictions, labels) if pred == 1 and label == 0)
    fn = sum(1 for pred, label in zip(predictions, labels) if pred == 0 and label == 1)
    tn = sum(1 for pred, label in zip(predictions, labels) if pred == 0 and label == 0)

    # Avoid division by zero for precision, recall, f1
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (tp + tn) / len(labels) if len(labels) > 0 else 0

    return accuracy, precision, recall, f1

def find_best_x_and_t(dev_file_path, log_x_min=None, log_x_max=30.0, num_log_x_steps=200, den_epsilon=1e-6):
    """
    Finds the best log_x and threshold T for the formula
    B(s) = performer_perplexity / (log_x - cross_perplexity)
    using the dev set, optimizing for F1 score.

    Args:
        dev_file_path (str): Path to the dev_scores.json file.
        log_x_min (float, optional): Minimum value for log_x search. If None, determined from data.
        log_x_max (float, optional): Maximum value for log_x search. Defaults to 30.0.
        num_log_x_steps (int, optional): Number of steps in the log_x search range. Defaults to 200.
        den_epsilon (float, optional): Small value added to denominator check to avoid issues near zero.

    Returns:
        dict: A dictionary containing 'best_log_x', 'best_threshold', 'best_f1'.
              Returns None if unable to find parameters (e.g., no valid data).
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


    performer_perplexities = []
    cross_perplexities = []
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
         print("Error: All samples in dev file have the same label. Cannot calculate F1 score.")
         return None

    for item in valid_data:
        performer_perplexities.append(item['performer_perplexity'])
        cross_perplexities.append(item['cross_perplexity'])
        labels.append(item['label'])

    print(f"Loaded {len(valid_data)} valid data points.")

    # Determine log_x search range
    # Set lower bound slightly above max cross_perplexity to avoid common division issues
    max_cross_ppl = max(cross_perplexities)
    if log_x_min is None:
        log_x_min = max_cross_ppl + den_epsilon * 10 # Add a small margin

    if log_x_min >= log_x_max:
        print(f"Error: Calculated minimum log_x ({log_x_min:.4f}) is >= maximum log_x ({log_x_max:.4f}). Adjust search range.")
        return None

    print(f"Searching for best log_x in range [{log_x_min:.4f}, {log_x_max:.4f}] with {num_log_x_steps} steps.")

    best_f1 = -1.0
    best_log_x = None
    best_threshold = None
    best_scores = None # Store scores for the best log_x to find the threshold later

    log_x_candidates = np.linspace(log_x_min, log_x_max, num_log_x_steps)

    for current_log_x in tqdm(log_x_candidates, desc="Searching for best log_x"):
        current_scores = []
        current_labels = []

        # Calculate scores for the current log_x, only using samples where denominator is positive
        for pp, cp, lbl in zip(performer_perplexities, cross_perplexities, labels):
            denominator = current_log_x - cp
            if denominator > den_epsilon: # Check if denominator is positive and not too close to zero
                 score = pp / denominator
                 current_scores.append(score)
                 current_labels.append(lbl)
            # else: this sample is effectively ignored for this current_log_x,
            # which simplifies the metric calculation for this specific log_x trial.

        if not current_scores or len(set(current_labels)) < 2:
             # print(f"Warning: Not enough valid scores or labels for log_x={current_log_x:.4f}. Skipping.")
             continue # Skip this log_x if no valid scores could be computed or not enough label variety

        # Find best threshold for the current scores and labels subset
        unique_scores = sorted(list(set(current_scores)))
        temp_best_f1 = -1.0
        temp_best_threshold = None

        # Check thresholds *between* unique scores or just the scores themselves?
        # Iterating through unique scores as thresholds is simpler and sufficient.
        for threshold in unique_scores:
             _, _, _, f1 = calculate_metrics(current_scores, current_labels, threshold)
             if f1 > temp_best_f1:
                 temp_best_f1 = f1
                 temp_best_threshold = threshold

        # Update overall best if current log_x yields better F1
        if temp_best_f1 > best_f1:
            best_f1 = temp_best_f1
            best_log_x = current_log_x
            best_threshold = temp_best_threshold # This threshold corresponds to scores calculated with best_log_x
            best_scores = current_scores # Store the scores calculated with the best log_x to re-derive threshold accurately if needed

    if best_log_x is None:
         print("Error: Could not find any valid log_x value that resulted in computable scores for at least one sample.")
         return None

    # Re-calculate the best threshold precisely using the scores computed with the best log_x
    # (This step is technically optional if temp_best_threshold is stored correctly,
    # but it ensures the final threshold is found on the exact score distribution for the best log_x)
    if best_scores:
         unique_scores_best_x = sorted(list(set(best_scores)))
         final_best_f1 = -1.0
         final_best_threshold = None
         for threshold in unique_scores_best_x:
              _, _, _, f1 = calculate_metrics(best_scores, current_labels, threshold) # Use labels corresponding to valid samples
              if f1 > final_best_f1:
                  final_best_f1 = f1
                  final_best_threshold = threshold
         # Use the refined results
         best_f1 = final_best_f1
         best_threshold = final_best_threshold


    print("\n--- Best Parameters Found ---")
    print(f"Best log_x: {best_log_x:.4f}")
    # Convert log_x back to X if needed, X = exp(best_log_x)
    # print(f"Corresponding Best X: {np.exp(best_log_x):.4f}")
    print(f"Corresponding Best Threshold T: {best_threshold:.4f}")
    print(f"Max F1 on Dev Set: {best_f1:.4f}")
    print("-----------------------------")

    return {
        'best_log_x': float(best_log_x), # Ensure JSON serializable
        'best_threshold': float(best_threshold),
        'best_f1': float(best_f1)
    }

# Main execution block
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Find optimal log_x and threshold for Binoculars formula using dev set.')
    parser.add_argument('--dev_file', type=str, required=True, help='Path to the dev_scores.json file.')
    parser.add_argument('--log_x_min', type=float, default=None, help='Minimum value for log_x search (default: auto from data).')
    parser.add_argument('--log_x_max', type=float, default=30.0, help='Maximum value for log_x search.')
    parser.add_argument('--num_log_x_steps', type=int, default=200, help='Number of steps in log_x search.')
    parser.add_argument('--output_params_file', type=str, default='best_binoculars_params_optimized_x.json', help='File to save the best parameters.')

    args = parser.parse_args()

    # Check if dev file exists
    if not os.path.exists(args.dev_file):
        print(f"Error: Dev file not found at {args.dev_file}")
        exit()

    best_params = find_best_x_and_t(
        args.dev_file,
        log_x_min=args.log_x_min,
        log_x_max=args.log_x_max,
        num_log_x_steps=args.num_log_x_steps
    )

    if best_params:
        try:
            with open(args.output_params_file, 'w', encoding='utf-8') as f:
                json.dump(best_params, f, indent=2)
            print(f"Best parameters saved to {args.output_params_file}")
        except IOError as e:
            print(f"Error saving parameters to file: {e}")
    else:
         print("Failed to find best parameters.")

