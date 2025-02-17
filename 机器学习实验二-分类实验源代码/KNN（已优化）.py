import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_csv('../KNN_11_11/playground-series-s4e2/train.csv')

X = df.drop(columns=['NObeyesdad', 'id'])  # 特征列
y = df['NObeyesdad']  # 标签列

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

X = pd.get_dummies(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 欧氏距离（Euclidean）和曼哈顿距离（Manhattan）
def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

def manhattan_distance(x1, x2):
    return np.sum(np.abs(x1 - x2))

class MyKNN:
    def __init__(self, k=5, metric='euclidean'):
        self.k = k
        self.metric = metric
        self.scaler = StandardScaler()

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

        self.X_train = self.scaler.fit_transform(X_train)

    def predict(self, X_test):
        X_test = self.scaler.transform(X_test)
        predictions = [self._predict(x) for x in X_test]
        return np.array(predictions)

    def _predict(self, x):
        # Compute distances between x and all points in the training set
        if self.metric == 'euclidean':
            distances = [euclidean_distance(x, x_train) for x_train in self.X_train]
        elif self.metric == 'manhattan':
            distances = [manhattan_distance(x, x_train) for x_train in self.X_train]
        else:
            raise ValueError("Unknown distance metric: choose 'euclidean' or 'manhattan'.")

        # k个最近邻的索引
        k_indices = np.argsort(distances)[:self.k]

        # 获取k个最近邻的标签
        k_nearest_labels = [self.y_train[i] for i in k_indices]

        # 加权投票
        k_nearest_distances = [distances[i] for i in k_indices]
        weights = 1 / (np.array(k_nearest_distances) + 1e-5)  # Add a small constant to avoid division by zero

        # 距离越近权重越大
        unique_labels = np.unique(k_nearest_labels)
        weighted_votes = {label: 0 for label in unique_labels}

        for label, weight in zip(k_nearest_labels, weights):
            weighted_votes[label] += weight

        return max(weighted_votes, key=weighted_votes.get)


knn_custom = MyKNN(k=5, metric='euclidean')
knn_custom.fit(X_train, y_train)
predictions = knn_custom.predict(X_test)

y_pred = knn_custom.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("Accuracy:", accuracy)
print("Classification Report:\n", report)
