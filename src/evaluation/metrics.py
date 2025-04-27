import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def calculate_vqa_metrics(predictions, ground_truth):
    unique_answers = list(set(ground_truth + predictions))
    answer_to_id = {answer: i for i, answer in enumerate(unique_answers)}
    y_true = [answer_to_id[a] for a in ground_truth]
    y_pred = [answer_to_id[a] for a in predictions]
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')
    return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}
