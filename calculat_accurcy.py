import json
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score

# Specify the paths to your JSON files
#  yolo_model_path = r"..\models\my_training\best.pt"
ground_truth_file = r'..\data\ground_truth_bags_auto.json'
predictions_file = r'..\data\predicted_bags.json'

def evaluate_model(gt_path, pred_path):
    """
    Loads ground truth and prediction data from JSON files and calculates
    key performance metrics for a binary classification task.
    """
    try:
        with open(gt_path, 'r') as f:
            y_true = json.load(f)

        with open(pred_path, 'r') as f:
            y_pred = json.load(f)
    except FileNotFoundError as e:
        print(f"Error: One of the files was not found. Please check the paths.")
        print(f"Details: {e}")
        return

    # Ensure both lists have the same length
    if len(y_true) != len(y_pred):
        print(f"Warning: Ground truth list (length {len(y_true)}) and predictions list (length {len(y_pred)}) have different sizes.")
        print("This can happen if the video processing was stopped prematurely.")
        # We will truncate the longer list to match the shorter one
        min_len = min(len(y_true), len(y_pred))
        y_true = y_true[:min_len]
        y_pred = y_pred[:min_len]
        
    if not y_true:
        print("Error: No data to evaluate. Both lists are empty after alignment.")
        return

    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    
    # Print the results
    print("-" * 50)
    print("Model Performance Metrics")
    print("-" * 50)
    print(f"Total samples evaluated: {len(y_true)}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print("-" * 50)
    print("Confusion Matrix:")
    print("           Predicted (No Bag)  Predicted (Bag)")
    print(f"True (No Bag)      {cm[0, 0]:<10}          {cm[0, 1]:<10}")
    print(f"True (Bag)         {cm[1, 0]:<10}          {cm[1, 1]:<10}")
    print("-" * 50)

# Run the evaluation
evaluate_model(ground_truth_file, predictions_file)