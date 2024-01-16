"""
K Nearest Neighbours Model
"""
import numpy as np
from statistics import mode


class KNN(object):
    def __init__(self, num_class: int):
        self.num_class = num_class

    def train(self, x_train: np.ndarray, y_train: np.ndarray, k: int):
        """
        Train KNN Classifier

        KNN only need to remember training set during training

        Parameters:
            x_train: Training samples ; np.ndarray with shape (N, D)
            y_train: Training labels  ; snp.ndarray with shape (N,)
        """
        self._x_train = x_train
        self._y_train = y_train
        self.k = k

    def predict(self, x_test: np.ndarray, k: int = None, loop_count: int = 1) -> np.ndarray:
        """
        Use the contained training set to predict labels for test samples

        Parameters:
            x_test    : Test samples                                     ; np.ndarray with shape (N, D)
            k         : k to overwrite the one specificed during training; int
            loop_count: parameter to choose different knn implementation ; int

        Returns:
            predicted labels for the data in X_test; a 1-dimensional array of
                length N, where each element is an integer giving the predicted
                class.
        """
        # Fill this function in
        k_test = k if k is not None else self.k

        if loop_count == 1:
            distance = self.calc_dis_one_loop(x_test)
        elif loop_count == 2:
            distance = self.calc_dis_two_loop(x_test)

        # TODO: implement me
        result = np.empty(x_test.shape[0])
        for i, row in enumerate(distance):
            disToLabel = {}
            # Map each distance to its label
            for j in range(len(row)):
                disToLabel[row[j]] = self._y_train[j]
            # Get the smallest k distances
            smallest_k = np.partition(row, k_test)[:k_test]
            # Convert the distances to their labels
            labels = np.empty(len(smallest_k))
            for j in range(len(smallest_k)):
                labels[j] = disToLabel[smallest_k[j]]
            # Get the majority label
            majorityLabel = mode(labels)
            # Record the majority label
            result[i] = majorityLabel
        return result

    def calc_dis_one_loop(self, x_test: np.ndarray) -> np.ndarray:
        """
        Calculate distance between training samples and test samples

        This function could one for loop

        Parameters:
            x_test: Test samples; np.ndarray with shape (N, D)
        Returns:
            suppose self._x_train has n rows. the function returns
                a np.ndarray with shape (N, n). Each element (i,j)
                is the Euclidean distance between testIamge_i and
                trainImage_j.
        """

        # TODO: implement me
        distance = np.empty((x_test.shape[0], self._x_train.shape[0]))
        for i, testImage in enumerate(x_test):
            # Calculate all differences for each test image
            distance[i] = np.linalg.norm(testImage - self._x_train, axis=1)
        return distance

    def calc_dis_two_loop(self, x_test: np.ndarray) -> np.ndarray:
        """
        Calculate distance between training samples and test samples

        This function could contain two loop

        Parameters:
            x_test: Test samples; np.ndarray with shape (N, D)
        Returns:
            suppose self._x_train has n rows. the function returns
                a np.ndarray with shape (N, n). Each element (i,j)
                is the Euclidean distance between testIamge_i and
                trainImage_j.
        """
        # TODO: implement me
        distance = np.empty((x_test.shape[0], self._x_train.shape[0]))
        for i, testImage in enumerate(x_test):
            for j, trainImage in enumerate(self._x_train):
                # Calculate the Euclidean Distance between images
                distance[i, j] = np.linalg.norm(testImage - trainImage)
        return distance
