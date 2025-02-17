import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

# 1. 加载数据集
df = pd.read_csv('factor_scores.csv')  # 读取数据文件
X = df.iloc[:, :-1].values  # 排除"背景特征"列，仅选择前四列特征进行聚类

# 2. 手动实现K-Means聚类算法
def kmeans(X, k, max_iters=100, tol=1e-5):
    # 初始化质心（随机选取k个点）
    centroids = X[np.random.choice(X.shape[0], k, replace=False)]
    prev_centroids = centroids.copy()

    for i in range(max_iters):
        # 1) 计算每个点到每个质心的距离
        distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)

        # 2) 分配每个点到最近的质心
        labels = np.argmin(distances, axis=1)

        # 3) 更新质心
        new_centroids = []
        for j in range(k):
            # 找到当前簇的数据点
            points_in_cluster = X[labels == j]
            if len(points_in_cluster) > 0:
                # 如果簇不为空，计算均值
                new_centroids.append(points_in_cluster.mean(axis=0))
            else:
                # 如果簇为空，随机选择一个点作为新的质心
                new_centroids.append(X[np.random.choice(X.shape[0])])

        new_centroids = np.array(new_centroids)

        # 判断质心是否收敛
        centroid_shift = np.linalg.norm(new_centroids - prev_centroids)
        if centroid_shift < tol:
            break

        prev_centroids = new_centroids.copy()

    return labels, new_centroids, i + 1  # 返回新的质心和迭代次数


# 3. 遍历不同的k和max_iters，计算不同情况下的轮廓系数和迭代次数
best_silhouette_score = -1
best_k = 2
best_max_iters = 100
best_labels = None
best_centroids = None
best_iter = None

silhouette_scores = []
k_range = range(2, 11)  # 测试聚类数目从2到10
max_iters_range = range(10, 501, 10)  # 测试最大迭代次数从50到300，步长为50

for k in k_range:
    for max_iters in max_iters_range:
        labels, centroids, num_iters = kmeans(X, k, max_iters=max_iters)
        score = silhouette_score(X, labels)
        silhouette_scores.append(score)

        # 如果轮廓系数更好，更新最佳结果
        if score > best_silhouette_score:
            best_silhouette_score = score
            best_k = k
            best_max_iters = max_iters
            best_labels = labels
            best_centroids = centroids
            best_iter = num_iters

# 4. 输出最佳k和最佳迭代次数
print(f"Best k: {best_k}, Best max_iters: {best_max_iters}, Best silhouette score: {best_silhouette_score}")
print(f"Converged in {best_iter} iterations.")

# 5. 使用最佳k进行K-Means聚类
labels, centroids, _ = kmeans(X, best_k, max_iters=best_max_iters)

# 6. 使用PCA降维并可视化聚类结果
pca = PCA(n_components=2)
X_2D = pca.fit_transform(X)

# 可视化聚类结果
plt.figure(figsize=(8, 6))
scatter = plt.scatter(X_2D[:, 0], X_2D[:, 1], c=labels, cmap='viridis', marker='.')
# 高亮显示质心
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='x', s=200, label='Centroids')
plt.title(f'K-Means Clustering (k={best_k})')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.legend()
plt.colorbar(scatter, label='Cluster')
plt.show()

# 7. 绘制轮廓系数图，选择最佳聚类数目
plt.figure(figsize=(8, 6))
# 我们可以将x轴标签调整为每个k对应的最大轮廓系数，而不是每个k和max_iters的组合
mean_silhouette_scores = [np.max(silhouette_scores[i:i+len(max_iters_range)]) for i in range(0, len(silhouette_scores), len(max_iters_range))]

plt.plot(k_range, mean_silhouette_scores, marker='o', linestyle='-', color='b')
plt.title('Silhouette Scores for Different k')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Silhouette Score')
plt.xticks(k_range)
plt.grid(True)
plt.show()
