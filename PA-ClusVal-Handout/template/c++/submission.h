// This file will be available at autograder runtime
// Do NOT upload this file in your submission

#include <string>
#include <unordered_map>
#include <vector>

class Solution {
 public:
  std::unordered_map<std::string, int> confusion_matrix(const std::vector<int> &true_labels,
                                                        const std::vector<int> &predicted_labels);
  double jaccard(const std::vector<int> &true_labels, const std::vector<int> &predicted_labels);
  double nmi(const std::vector<int> &true_labels, const std::vector<int> &predicted_labels);
};
