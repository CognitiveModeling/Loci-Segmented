import re
import sys
from collections import defaultdict

def compute_means(log_file_path, prefix=""):
    # Initialize a dictionary to hold the sum and count of each metric (and its sub-values if applicable)
    metrics_data = defaultdict(lambda: defaultdict(lambda: [0, 0]))  # {metric: {sub_value_index: [sum, count]}}
    
    # Regular expression to match key-value pairs in the log
    key_value_regex = re.compile(r'([a-zA-Z0-9_-]+):\s*([\d.e+-]+[%]?(?:\|[\d.e+-]+[%]?)*)')
    
    # Read the log file
    with open(log_file_path, 'r') as f:
        lines = f.readlines()
    
    # Parse each line in the log file
    for line in lines:
        matches = key_value_regex.findall(line)
        for key, value_str in matches:
            # Remove "%" if present and split value_str by "|" if it contains multiple independent values
            values = list(map(lambda x: float(x.rstrip('%')), value_str.split("|")))
            for i, value in enumerate(values):
                metrics_data[key][i][0] += value  # Add to sum
                metrics_data[key][i][1] += 1  # Increment count
    
    # Compute the mean for each metric and its sub-values
    metrics = {key: {i: sum_value / count for i, (sum_value, count) in sub_values.items()} for key, sub_values in metrics_data.items()}

    print(f"{prefix} | {metrics['AMI'][0]:.3f} | {metrics['ARI'][0]:.3f} | {metrics['AMI'][1]:.3f} | {metrics['ARI'][1]:.3f} | {metrics['IoU'][1]/100:.3f} | {metrics['F1'][1]/100:.3f} | {metrics['OCA'][0]:.3f}")

if __name__ == "__main__":
    print("Validation set:\n")
    print("  Method     | AMI-A | ARI-A | AMI-O | ARI-O |  IoU  |  F1   |  OCA")
    compute_means("out/eval_review_test_random",       'Loci-s (rnd)')
    compute_means("out/eval_review_test_regularized",  'Loci-s (reg)')
    compute_means("out/eval_review_test_segmentation", 'Loci-s (seg)')

    print("\n\nGeneralization set:\n")
    print("  Method     | AMI-A | ARI-A | AMI-O | ARI-O |  IoU  |  F1   |  OCA")
    compute_means("out/eval_review_generalization_random",       'Loci-s (rnd)')
    compute_means("out/eval_review_generalization_regularized",  'Loci-s (reg)')
    compute_means("out/eval_review_generalization_segmentation", 'Loci-s (seg)')
