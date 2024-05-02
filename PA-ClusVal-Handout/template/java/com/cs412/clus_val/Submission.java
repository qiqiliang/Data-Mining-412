// Submit this file to Gradescope
package com.cs412.clus_val; // do not change the package

// you may use other standard libraries
import java.util.Map;

public class Submission {
  public Map<String, Integer> getConfusionMatrix(
      int[] trueLabels, int[] predictedLabels) {
    // Calculate the confusion matrix and return it as a sparse matrix in dictoinary
    // form.
    // Args:
    // trueLabels: an array of true labels
    // predictedLabels: an array of predicted labels
    // Returns:
    // a dictionary of "i-j": count, where i and j are the true and predicted labels, use "-" as delimiter
  }

  public double getJaccard(int[] trueLabels, int[] predictedLabels) {
    // Calculate the Jaccard index.
    // Args:
    // trueLabels: an array of true labels
    // predictedLabels: an array of predicted labels
    // Returns:
    // the Jaccard index. Do NOT round this value.
  }

  public double getNMI(int[] trueLabels, int[] predictedLabels) {
    // Calculate the normalized mutual information.
    // Args:
    // trueLabels: an array of true labels
    // predictedLabels: an array of predicted labels
    // Returns:
    // the normalized mutual information. Do NOT round this value.
  }
}
