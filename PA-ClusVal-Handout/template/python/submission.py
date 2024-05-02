# Submit this file to Gradescope
from typing import Dict, List, Tuple
# you may use other Python standard libraries, but not data
# science libraries, such as numpy, scikit-learn, etc.
import math

class Solution:

  def confusion_matrix(self, true_labels: List[int], pred_labels: List[int]) -> Dict[Tuple[int, int], int]:
    """Calculate the confusion matrix and return it as a sparse matrix in dictionary form.
    Args:
      true_labels: list of true labels
      pred_labels: list of predicted labels
    Returns:
      A dictionary of (true_label, pred_label): count
    """
    matrix = {}
    for t, p in zip(true_labels, pred_labels):
        matrix[(t, p)] = matrix.get((t, p), 0) + 1
    return matrix

  def jaccard(self, true_labels: List[int], pred_labels: List[int]) -> float:
    if not true_labels or not pred_labels or len(true_labels) != len(pred_labels):
        return 0.0

    # Initialize sets for true positives, false positives, and false negatives
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    # Compute the components of the Jaccard index
    for i in range(len(true_labels)):
        for j in range(i + 1, len(true_labels)):
            true_i_j = true_labels[i] == true_labels[j]
            pred_i_j = pred_labels[i] == pred_labels[j]
            if true_i_j and pred_i_j:
                true_positives += 1
            elif true_i_j and not pred_i_j:
                false_negatives += 1
            elif not true_i_j and pred_i_j:
                false_positives += 1

    # Calculate the Jaccard index
    jaccard_index = true_positives / (true_positives + false_positives + false_negatives) if true_positives + false_positives + false_negatives > 0 else 0

    return jaccard_index


  def nmi(self, true_labels: List[int], pred_labels: List[int]) -> float:
    """Calculate the normalized mutual information.
    Args:
      true_labels: list of true cluster labels
      pred_labels: list of predicted cluster labels
    Returns:
      The normalized mutual information. Do NOT round this value.
    """
    matrix = self.confusion_matrix(true_labels, pred_labels)
    total = len(true_labels)
    true_dist = {label: true_labels.count(label) for label in set(true_labels)}
    pred_dist = {label: pred_labels.count(label) for label in set(pred_labels)}
    MI = sum((count / total) * math.log((total * count) / (true_dist[t] * pred_dist[p]), 2)
            for (t, p), count in matrix.items() if t in true_dist and p in pred_dist)
    H_true = -sum((count / total) * math.log(count / total, 2) for count in true_dist.values())
    H_pred = -sum((count / total) * math.log(count / total, 2) for count in pred_dist.values())
    return MI / math.sqrt(H_true * H_pred) if H_true and H_pred else 0.0

