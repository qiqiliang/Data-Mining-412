from typing import List

class Node:
    """
    This class, Node, represents a single node in a decision tree. It is designed to store information about the tree
    structure and the specific split criteria at each node. It is important to note that this class should NOT be
    modified as it is part of the assignment and will be used by the autograder.

    The attributes of the Node class are:
    - split_dim: The dimension/feature along which the node splits the data (-1 by default, indicating uninitialized)
    - split_point: The value used for splitting the data at this node (-1 by default, indicating uninitialized)
    - label: The class label assigned to this node, which is the majority label of the data at this node. If there is a tie,
      the numerically smaller label is assigned (-1 by default, indicating uninitialized)
    - left: The left child node of this node (None by default). Either None or a Node object.
    - right: The right child node of this node (None by default) Either None or a Node object.
    """
    def __init__(self):
        self.split_dim = -1
        self.split_point = -1
        self.label = -1
        self.left = None
        self.right = None


class Solution:
    """
    This class uses the Node class to build a decision tree.
    """
    def __init__(self):
        self.root = None

    def fit(self, train_data: List[List[float]], train_label: List[int]) -> None:
        """
        Fit the decision tree model using the provided training data and labels.
        """
        self.root = Node()
        self.split_node(self.root, train_data, train_label, 0)

    def split_node(self, node, data, labels, depth):
        if depth == 2 or not data or len(set(labels)) == 1:
            node.label = max(set(labels), key=labels.count)
            return

        # Find the best split
        best_gain = -1
        best_split = None
        best_dim = None

        for dim in range(len(data[0])):
            values = [row[dim] for row in data]
            split_points = self.calculate_split_points(values)

            for split_point in split_points:
                gain = self.split_info(data, labels, dim, split_point)
                if gain > best_gain:
                    best_gain = gain
                    best_split = split_point
                    best_dim = dim

        if best_dim is None:  # No valid split was found
            node.label = max(set(labels), key=labels.count)
            return

        node.split_dim = best_dim
        node.split_point = best_split
        node.label = max(set(labels), key=labels.count)

        # Create child nodes
        left_data, left_labels, right_data, right_labels = self.partition(data, labels, best_dim, best_split)
        node.left = Node()
        node.right = Node()
        self.split_node(node.left, left_data, left_labels, depth + 1)
        self.split_node(node.right, right_data, right_labels, depth + 1)

    def split_info(self, data: List[List[float]], labels: List[int], split_dim: int, split_point: float) -> float:
        """
        Calculate the information gain from splitting the data at the specified dimension and split point.
        """
        left_labels = [labels[i] for i in range(len(data)) if data[i][split_dim] <= split_point]
        right_labels = [labels[i] for i in range(len(data)) if data[i][split_dim] > split_point]
        return self.information_gain(labels, left_labels, right_labels)

    def information_gain(self, total, left, right):
        """
        Helper method to calculate the information gain. Uses entropy to calculate information gain.
        """
        from math import log2

        def entropy(labels):
            if not labels:
                return 0
            probs = [labels.count(x) / len(labels) for x in set(labels)]
            return -sum(p * log2(p) for p in probs if p > 0)

        total_entropy = entropy(total)
        left_entropy = entropy(left)
        right_entropy = entropy(right)
        left_prob = len(left) / len(total)
        right_prob = len(right) / len(total)
        return total_entropy - (left_prob * left_entropy + right_prob * right_entropy)

    def partition(self, data, labels, dim, split_point):
        """
        Partition the data based on the split dimension and point.
        """
        left_data, right_data, left_labels, right_labels = [], [], [], []
        for i, row in enumerate(data):
            if row[dim] <= split_point:
                left_data.append(row)
                left_labels.append(labels[i])
            else:
                right_data.append(row)
                right_labels.append(labels[i])
        return left_data, left_labels, right_data, right_labels

    def classify(self, train_data: List[List[float]], train_label: List[int], test_data: List[List[float]]) -> List[int]:
        """
        Classify the test data using the decision tree model built from the provided training data and labels.
        """
        self.fit(train_data, train_label)
        return [self.predict(self.root, sample) for sample in test_data]

    def predict(self, node, sample):
        """
        Recursively predict the class of a given sample using the decision tree.
        """
        if node.left is None or node.right is None:
            return node.label
        elif sample[node.split_dim] <= node.split_point:
            return self.predict(node.left, sample)
        else:
            return self.predict(node.right, sample)
