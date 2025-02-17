# 导入所需库
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import label_binarize

# 数据读取
train = pd.read_csv("train.csv")  # 读取训练数据集
test = pd.read_csv("test.csv")    # 读取测试数据集

# 合并数据集进行预处理，确保特征一致性
train['is_train'] = 1             # 为训练集添加标识列
test['is_train'] = 0              # 为测试集添加标识列
test['NObeyesdad'] = np.nan       # 在测试集中添加目标变量列，值为NaN
combined = pd.concat([train, test], ignore_index=True)  # 合并训练集和测试集

# 独热编码类别变量
combined = pd.get_dummies(
    combined,
    columns=[
        'Gender', 'family_history_with_overweight', 'FAVC', 'CAEC', 'SMOKE',
        'SCC', 'CALC', 'MTRANS'
    ]
)  # 对分类特征进行独热编码，以便模型处理

# 分离处理后的训练集和测试集
train = combined[combined['is_train'] == 1].drop(['is_train'], axis=1)  # 提取训练集并删除标识列
test = combined[combined['is_train'] == 0].drop(['is_train', 'NObeyesdad'], axis=1)  # 提取测试集并删除标识列和目标变量列

# 将目标变量转换为数值
label_encoder = LabelEncoder()
train['NObeyesdad'] = label_encoder.fit_transform(train['NObeyesdad'])  # 对目标变量进行标签编码

# 分割特征和目标变量
X = train.drop(columns=['NObeyesdad', 'id'])  # 提取特征变量，删除目标变量和ID列
y = train['NObeyesdad']                      # 提取目标变量

# 标准化特征
scaler = StandardScaler()
X = scaler.fit_transform(X)                 # 对训练特征进行标准化
X_test_final = scaler.transform(test.drop(columns=['id']))  # 对测试特征进行标准化

# 划分训练集和最终评估测试集
X_train_full, X_final_test, y_train_full, y_final_test = train_test_split(
    X, y, test_size=0.1, random_state=42
)  # 将训练集划分为训练集和独立测试集，测试集占10%

# 定义DNN模型类（包括Adam优化器、Dropout和学习率调整）
class DNNModel:
    def __init__(
        self, input_dim, hidden_units, output_dim,
        learning_rate=0.0005, beta1=0.9, beta2=0.999, epsilon=1e-8
    ):
        self.layer_dims = [input_dim] + hidden_units + [output_dim]  # 定义每层的神经元数量
        self.learning_rate = learning_rate                           # 学习率
        self.beta1 = beta1                                           # Adam优化器的beta1参数
        self.beta2 = beta2                                           # Adam优化器的beta2参数
        self.epsilon = epsilon                                       # 防止除零的小数
        self.t = 0                                                   # 时间步，用于计算偏差校正
        self.parameters = self.initialize_parameters()               # 初始化网络参数
        self.m, self.v = self.initialize_adam()                      # 初始化Adam优化器的参数
        self.losses = []                                             # 存储损失值

    def initialize_parameters(self):
        np.random.seed(42)
        parameters = {}
        # 遍历每一层，初始化权重和偏置
        for l in range(1, len(self.layer_dims)):
            parameters[f"W{l}"] = np.random.randn(
                self.layer_dims[l - 1], self.layer_dims[l]
            ) * np.sqrt(2 / self.layer_dims[l - 1])  # He初始化
            parameters[f"b{l}"] = np.zeros((1, self.layer_dims[l]))   # 偏置初始化为0
        return parameters

    def initialize_adam(self):
        m = {}
        v = {}
        # 初始化一阶和二阶动量
        for l in range(1, len(self.layer_dims)):
            m[f"W{l}"], m[f"b{l}"] = (
                np.zeros_like(self.parameters[f"W{l}"]),
                np.zeros_like(self.parameters[f"b{l}"])
            )
            v[f"W{l}"], v[f"b{l}"] = (
                np.zeros_like(self.parameters[f"W{l}"]),
                np.zeros_like(self.parameters[f"b{l}"])
            )
        return m, v

    @staticmethod
    def relu(Z):
        return np.maximum(0, Z)  # ReLU激活函数

    @staticmethod
    def relu_derivative(Z):
        return Z > 0             # ReLU激活函数的导数

    @staticmethod
    def softmax(Z):
        exp_Z = np.exp(Z - np.max(Z, axis=1, keepdims=True))  # 防止指数爆炸
        return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)   # Softmax函数

    def forward_propagation(self, X, dropout_rate=0.1):
        cache = {}
        A = X
        num_layers = len(self.parameters) // 2  # 计算层数

        # 前向传播
        for l in range(1, num_layers):
            Z = np.dot(A, self.parameters[f"W{l}"]) + self.parameters[f"b{l}"]  # 线性部分
            A = self.relu(Z)                                                    # 非线性激活
            # Dropout正则化
            D = np.random.rand(*A.shape) < (1 - dropout_rate)  # 生成掩码矩阵
            A *= D                                             # 应用掩码
            A /= (1 - dropout_rate)                            # 缩放激活值
            cache[f"Z{l}"], cache[f"A{l}"], cache[f"D{l}"] = Z, A, D  # 存储缓存值

        # 输出层
        Z_final = np.dot(A, self.parameters[f"W{num_layers}"]) + self.parameters[f"b{num_layers}"]
        A_final = self.softmax(Z_final)  # 使用Softmax激活函数
        cache[f"Z{num_layers}"], cache[f"A{num_layers}"] = Z_final, A_final  # 存储输出层的缓存值

        return A_final, cache

    def compute_loss(self, Y, A_final):
        m = Y.shape[0]
        # 计算交叉熵损失
        log_probs = -np.log(A_final[range(m), Y] + 1e-9)  # 增加小数以防止log(0)
        loss = np.sum(log_probs) / m
        return loss

    def backward_propagation(self, X, Y, cache, dropout_rate=0.1):
        m = X.shape[0]
        gradients = {}
        num_layers = len(self.parameters) // 2

        # 初始化输出层的梯度
        dZ_final = cache[f"A{num_layers}"]
        dZ_final[range(m), Y] -= 1  # Softmax与交叉熵的导数结合

        # 反向传播
        for l in reversed(range(1, num_layers + 1)):
            # 计算梯度
            dW = (
                np.dot(cache[f"A{l - 1}"].T, dZ_final) / m
                if l > 1 else np.dot(X.T, dZ_final) / m
            )
            db = np.sum(dZ_final, axis=0, keepdims=True) / m
            gradients[f"dW{l}"], gradients[f"db{l}"] = dW, db

            if l > 1:
                # 计算前一层的梯度
                dA = np.dot(dZ_final, self.parameters[f"W{l}"].T)
                dA *= cache[f"D{l - 1}"]          # 应用Dropout掩码
                dA /= (1 - dropout_rate)          # 缩放梯度
                dZ_final = dA * self.relu_derivative(cache[f"Z{l - 1}"])  # 计算dZ

        return gradients

    def update_parameters(self, gradients):
        self.t += 1  # 时间步加1
        # 使用Adam优化器更新参数
        for l in range(1, len(self.layer_dims)):
            # 更新一阶和二阶动量
            self.m[f"W{l}"] = self.beta1 * self.m[f"W{l}"] + (1 - self.beta1) * gradients[f"dW{l}"]
            self.v[f"W{l}"] = self.beta2 * self.v[f"W{l}"] + (1 - self.beta2) * (gradients[f"dW{l}"] ** 2)
            self.m[f"b{l}"] = self.beta1 * self.m[f"b{l}"] + (1 - self.beta1) * gradients[f"db{l}"]
            self.v[f"b{l}"] = self.beta2 * self.v[f"b{l}"] + (1 - self.beta2) * (gradients[f"db{l}"] ** 2)

            # 计算偏差校正后的动量
            m_corrected_W = self.m[f"W{l}"] / (1 - self.beta1 ** self.t)
            v_corrected_W = self.v[f"W{l}"] / (1 - self.beta2 ** self.t)
            m_corrected_b = self.m[f"b{l}"] / (1 - self.beta1 ** self.t)
            v_corrected_b = self.v[f"b{l}"] / (1 - self.beta2 ** self.t)

            # 更新参数
            self.parameters[f"W{l}"] -= self.learning_rate * m_corrected_W / (
                np.sqrt(v_corrected_W) + self.epsilon
            )
            self.parameters[f"b{l}"] -= self.learning_rate * m_corrected_b / (
                np.sqrt(v_corrected_b) + self.epsilon
            )

    def train(self, X, y, num_epochs, batch_size, initial_lr=0.001, lr_decay=0.9):
        X = np.array(X)
        y = np.array(y)
        for epoch in range(num_epochs):
            self.learning_rate = initial_lr * (lr_decay ** epoch)  # 动态调整学习率
            permutation = np.random.permutation(len(X))            # 随机打乱数据
            X_shuffled, y_shuffled = X[permutation], y[permutation]
            for i in range(0, len(X), batch_size):
                # Mini-batch梯度下降
                X_batch, y_batch = X_shuffled[i:i + batch_size], y_shuffled[i:i + batch_size]
                A_final, cache = self.forward_propagation(X_batch)      # 前向传播
                loss = self.compute_loss(y_batch, A_final)              # 计算损失
                self.losses.append(loss)                                # 记录损失
                gradients = self.backward_propagation(X_batch, y_batch, cache)  # 反向传播
                self.update_parameters(gradients)                       # 更新参数
            if epoch % 10 == 0:
                print(f"Epoch {epoch}/{num_epochs}, Loss: {loss}")      # 每10个epoch打印一次损失

    def predict(self, X):
        A_final, _ = self.forward_propagation(X)  # 前向传播
        predictions = np.argmax(A_final, axis=1)  # 取最大概率对应的类别
        return predictions

# 定义模型参数
input_dim = X_train_full.shape[1]  # 输入层维度等于特征数量
hidden_units = [64, 128, 256, 512, 256, 128, 64, 32, 16]  # 隐藏层结构
output_dim = len(np.unique(y))     # 输出层维度等于类别数量
learning_rate = 0.0005
num_epochs = 100
batch_size = 32

# 将 y 转换为 numpy 数组格式
y = y.to_numpy()

# K折交叉验证和模型训练
kf = KFold(n_splits=5, shuffle=True, random_state=42)  # 定义5折交叉验证
fold_accuracies = []

for fold, (train_index, val_index) in enumerate(kf.split(X)):
    print(f"\nFold {fold + 1}")
    X_train, X_val = X[train_index], X[val_index]           # 获取训练集和验证集特征
    y_train, y_val = y[train_index], y[val_index]           # 获取训练集和验证集标签

    model = DNNModel(
        input_dim=input_dim, hidden_units=hidden_units,
        output_dim=output_dim, learning_rate=learning_rate
    )  # 初始化模型
    model.train(
        X_train, y_train, num_epochs=num_epochs,
        batch_size=batch_size
    )  # 训练模型

    y_pred = model.predict(X_val)                          # 在验证集上预测
    accuracy = np.mean(y_pred == y_val)                    # 计算准确率
    fold_accuracies.append(accuracy)
    print(f"Fold {fold + 1} Validation Accuracy: {accuracy}")

# 输出平均交叉验证准确率
print(f"\nAverage K-Fold Validation Accuracy: {np.mean(fold_accuracies)}")

# 绘制训练过程的损失曲线
plt.figure(figsize=(10, 6))
plt.plot(model.losses, label='Training Loss')             # 绘制损失曲线
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.legend()
plt.show()

# 使用独立测试集进行评估
y_test_pred = model.predict(X_final_test)                 # 在独立测试集上预测
final_accuracy = np.mean(y_test_pred == y_final_test)     # 计算最终测试集的准确率
print(f"Final Test Accuracy on Unseen Data: {final_accuracy}")

# 混淆矩阵和分类报告
cm = confusion_matrix(y_final_test, y_test_pred)          # 计算混淆矩阵
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')        # 绘制混淆矩阵
plt.title('Confusion Matrix on Final Test Set')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

print("Classification Report on Final Test Set:")
print(classification_report(y_final_test, y_test_pred))   # 打印分类报告

# ROC曲线
y_final_test_bin = label_binarize(y_final_test, classes=np.unique(y))    # 二值化真实标签
y_test_pred_bin = label_binarize(y_test_pred, classes=np.unique(y))      # 二值化预测标签
n_classes = y_final_test_bin.shape[1]                                    # 类别数量

fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    # 计算每个类别的ROC曲线和AUC值
    fpr[i], tpr[i], _ = roc_curve(y_final_test_bin[:, i], y_test_pred_bin[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

plt.figure(figsize=(10, 8))
for i in range(n_classes):
    # 绘制每个类别的ROC曲线
    plt.plot(fpr[i], tpr[i], lw=2, label=f'ROC curve for class {i} (AUC = {roc_auc[i]:.2f})')

plt.plot([0, 1], [0, 1], linestyle='--', color='r', lw=2, label='Random Guess')  # 绘制随机猜测的参考线
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Multi-class ROC Curve on Final Test Set')
plt.legend(loc="lower right")
plt.show()

# 预测测试数据集
test_predictions = model.predict(X_test_final)                      # 在实际测试集上进行预测
output = label_encoder.inverse_transform(test_predictions)          # 将预测结果转换回原始标签

# 创建提交文件
submission = pd.DataFrame({'id': test['id'], 'NObeyesdad': output})  # 创建提交文件
submission.to_csv('submission.csv', index=False)                     # 保存为CSV文件
print(submission.head())                                             # 打印前几行查看
