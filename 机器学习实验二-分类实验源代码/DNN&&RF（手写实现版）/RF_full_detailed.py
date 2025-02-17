# 导入所需库
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import label_binarize

# 数据读取：加载训练和测试数据集
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# 合并数据集进行预处理，确保特征一致性
# 给训练集和测试集添加标识列，然后合并，方便统一的预处理
train['is_train'] = 1
test['is_train'] = 0
test['NObeyesdad'] = np.nan  # 给测试集添加目标变量列，以便与训练集一致
combined = pd.concat([train, test], ignore_index=True)

# 独热编码类别变量
# 将多类别的特征编码为独热编码，以便用于模型训练
combined = pd.get_dummies(combined, columns=['Gender', 'family_history_with_overweight', 'FAVC', 'CAEC', 'SMOKE', 'SCC', 'CALC', 'MTRANS'])

# 分离处理后的训练集和测试集
# 按标识列区分并分离原训练集和测试集
train = combined[combined['is_train'] == 1].drop(['is_train'], axis=1)
test = combined[combined['is_train'] == 0].drop(['is_train', 'NObeyesdad'], axis=1)

# 将目标变量转换为数值
# 使用LabelEncoder将类别型目标变量转换为数值编码
label_encoder = LabelEncoder()
train['NObeyesdad'] = label_encoder.fit_transform(train['NObeyesdad'])

# 分割特征和目标变量
X = train.drop(columns=['NObeyesdad', 'id'])  # 特征矩阵
y = train['NObeyesdad']  # 目标变量

# 标准化特征
# 使用StandardScaler标准化特征，以保证模型训练的稳定性
scaler = StandardScaler()
X = scaler.fit_transform(X)
X_test_final = scaler.transform(test.drop(columns=['id']))

# 划分训练集和最终评估测试集
# 从训练集中分出10%作为独立测试集，用于模型的最终评估
X_train_full, X_final_test, y_train_full, y_final_test = train_test_split(X, y, test_size=0.1, random_state=42)

# 手动实现的决策树分类器
class DecisionTreeClassifier:
    def __init__(self, max_depth=None, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None

    def fit(self, X, y):
        # 递归构建决策树
        self.tree = self._grow_tree(X, y, depth=0)

    def _grow_tree(self, X, y, depth):
        # 获取样本数量和特征数量
        num_samples, num_features = X.shape
        # 检查是否达到停止条件
        if depth >= self.max_depth or num_samples < self.min_samples_split or len(set(y)) == 1:
            return np.bincount(y).argmax()  # 返回叶节点的预测值

        # 找到最佳分裂点
        best_feat, best_thresh = self._best_split(X, y)
        if best_feat is None:
            return np.bincount(y).argmax()  # 若没有找到最佳分裂点，返回叶节点预测值

        # 递归构建左右子树
        left_idx = X[:, best_feat] <= best_thresh
        right_idx = X[:, best_feat] > best_thresh
        left = self._grow_tree(X[left_idx], y[left_idx], depth + 1)
        right = self._grow_tree(X[right_idx], y[right_idx], depth + 1)
        return {"feature": best_feat, "threshold": best_thresh, "left": left, "right": right}

    def _best_split(self, X, y):
        # 初始化最佳分裂点变量
        num_samples, num_features = X.shape
        if num_samples <= 1:
            return None, None

        best_feat, best_thresh = None, None
        best_gini = float('inf')

        # 遍历每个特征寻找最佳分裂点
        for feature in range(num_features):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                # 计算Gini不纯度
                left_idx = X[:, feature] <= threshold
                right_idx = X[:, feature] > threshold
                if len(left_idx) == 0 or len(right_idx) == 0:
                    continue

                gini = self._gini(y[left_idx], y[right_idx])
                if gini < best_gini:
                    best_gini = gini
                    best_feat, best_thresh = feature, threshold

        return best_feat, best_thresh

    def _gini(self, left_y, right_y):
        # 计算Gini不纯度
        def gini_impurity(y):
            m = len(y)
            return 1.0 - sum((np.sum(y == c) / m) ** 2 for c in np.unique(y))

        left_size, right_size = len(left_y), len(right_y)
        m = left_size + right_size
        return (left_size * gini_impurity(left_y) + right_size * gini_impurity(right_y)) / m

    def predict(self, X):
        # 对每个样本进行预测
        return np.array([self._predict(inputs, self.tree) for inputs in X])

    def _predict(self, inputs, tree):
        # 遍历决策树，获取叶节点的预测值
        if not isinstance(tree, dict):
            return tree
        if inputs[tree['feature']] <= tree['threshold']:
            return self._predict(inputs, tree['left'])
        else:
            return self._predict(inputs, tree['right'])

# 手动实现的随机森林分类器
class RandomForestClassifier:
    def __init__(self, n_estimators=10, max_depth=None, min_samples_split=2):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.trees = []

    def fit(self, X, y):
        # 构建多个决策树，组成随机森林
        self.trees = []
        for _ in range(self.n_estimators):
            tree = DecisionTreeClassifier(max_depth=self.max_depth, min_samples_split=self.min_samples_split)
            X_sample, y_sample = self._bootstrap_sample(X, y)  # 生成引导样本
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def _bootstrap_sample(self, X, y):
        # 生成引导样本
        n_samples = X.shape[0]
        indices = np.random.choice(n_samples, n_samples, replace=True)
        return X[indices], y[indices]

    def predict(self, X):
        # 获取所有决策树的预测结果，返回多数投票的结果
        tree_predictions = np.array([tree.predict(X) for tree in self.trees])
        return np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=tree_predictions)

# 定义模型参数
input_dim = X_train_full.shape[1]
n_estimators = 10
max_depth = 10
min_samples_split = 2

# K折交叉验证和模型训练
kf = KFold(n_splits=5, shuffle=True, random_state=42)
fold_accuracies = []

# 将 y 转换为 numpy 数组格式
y = y.to_numpy()

for fold, (train_index, val_index) in enumerate(kf.split(X)):
    print(f"\nFold {fold + 1}")
    X_train, X_val = X[train_index], X[val_index]
    y_train, y_val = y[train_index], y[val_index]

    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)
    accuracy = np.mean(y_pred == y_val)
    fold_accuracies.append(accuracy)
    print(f"Fold {fold + 1} Validation Accuracy: {accuracy}")

# 输出平均交叉验证准确率
print(f"\nAverage K-Fold Validation Accuracy: {np.mean(fold_accuracies)}")

# 使用独立测试集进行评估
y_test_pred = model.predict(X_final_test)
final_accuracy = np.mean(y_test_pred == y_final_test)
print(f"Final Test Accuracy on Unseen Data: {final_accuracy}")

# 混淆矩阵和分类报告
cm = confusion_matrix(y_final_test, y_test_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix on Final Test Set')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

print("Classification Report on Final Test Set:")
print(classification_report(y_final_test, y_test_pred))

# 将真实标签和预测标签转换为多类别的二进制格式
y_final_test_bin = label_binarize(y_final_test, classes=np.unique(y))
y_test_pred_bin = label_binarize(y_test_pred, classes=np.unique(y))
n_classes = y_final_test_bin.shape[1]  # 获取类别数量

# 初始化存储每个类别的FPR和TPR的字典
fpr = dict()
tpr = dict()
roc_auc = dict()

# 计算每个类别的ROC曲线和AUC
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_final_test_bin[:, i], y_test_pred_bin[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# 绘制多类别ROC曲线
plt.figure(figsize=(10, 8))
for i in range(n_classes):
    plt.plot(fpr[i], tpr[i], lw=2, label=f'ROC curve for class {i} (AUC = {roc_auc[i]:.2f})')

plt.plot([0, 1], [0, 1], linestyle='--', color='r', lw=2, label='Random Guess')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Multi-class ROC Curve on Final Test Set')
plt.legend(loc="lower right")
plt.show()

# 预测测试数据集
test_predictions = model.predict(X_test_final)
output = label_encoder.inverse_transform(test_predictions)  # 将预测结果转换回原始标签

# 创建提交文件
submission = pd.DataFrame({'id': test['id'], 'NObeyesdad': output})
submission.to_csv('submission.csv', index=False)
print(submission.head())
