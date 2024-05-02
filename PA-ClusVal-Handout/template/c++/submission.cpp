// Submit this file to Gradescope
#include "submission.h"

using std::vector;
using std::unordered_map;
using std::string;

unordered_map<string, int> Solution::confusion_matrix(const vector<int> &true_labels, const vector<int> &predicted_labels) {
  // Calculate the confusion matrix and return it as a sparse matrix in dictionary format
  // Args:
  //   true_labels: a vector of true labels
  //   predicted_labels: a vector of predicted labels
  // Returns:
  //   a dictionary of "true_label-predicted_label": count. Use dash as the delimiter to connect true_label and predicted_label.

  // implement your code here
}

double Solution::jaccard(const vector<int> &true_labels, const vector<int> &predicted_labels) {
  // Calculate the Jaccard index
  // Args:
  //   true_labels: a vector of true labels
  //   predicted_labels: a vector of predicted labels
  // Returns:
  //   a float number of Jaccard index. Do NOT round this value.

  // implement your code here
}

double Solution::nmi(const vector<int> &true_labels, const vector<int> &predicted_labels) {
  // Calculate the normalized mutual information
  // Args:
  //   true_labels: a vector of true labels
  //   predicted_labels: a vector of predicted labels
  // Returns:
  //   a float number of normalized mutual information. Do NOT round this value.

  // implement your code here
}
