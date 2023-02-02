"""
    实现了主成分分析(PCA)降维的功能，使用PCA将原始光谱数据降维到指定维度。
    具体来说，它对输入的光谱数据进行预处理，然后使用PCA类将特征数量降为指定
    数量，并返回降维后的数据。
"""

from sklearn.decomposition import PCA

def Pca(X, nums=20):
    """
       :param X: raw spectrum data, shape (n_samples, n_features)
       :param nums: Number of principal components retained
       :return: X_reduction：Spectral data after dimensionality reduction
    """
    pca = PCA(n_components=nums)  # 保留的特征数码
    pca.fit(X)
    X_reduction = pca.transform(X)

    return X_reduction
