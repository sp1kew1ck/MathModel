import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
# 导入 PCA 模块
from sklearn.decomposition import PCA
# 导入 chi2 模块
from scipy.stats import chi2

def ewm(data: pd.DataFrame):
    """
    熵权法生成权重系数
    :param data: 正向化并标准化的矩阵，行为样本列为指标
    :return: 权重，列为指标，只有一行
    """
    cdata = data.copy().replace(0, 1e-8)
    s = data.sum(axis=0).replace(0, 1e-8)
    
    # 计算熵值
    E = np.nansum((-cdata*(np.log(cdata)-np.log(s)))/(s*np.log(cdata.shape[0])).replace(0, 1e-8), axis=0)
    # 计算权重
    E1 = 1-E
    return pd.Series(E1/E1.sum(), index=data.columns)

def critic(data: pd.DataFrame):
    """
    CRITIC 方法生成权重
    :param data: 正向化且标准化的二维矩阵，行为样本列为指标
    :return: 权重（pandas Series）
    """
    # 1. 计算每个指标的标准差
    std_devs = data.std()

    # 2. 计算相关系数矩阵
    corr_matrix = data.corr()

    # 3. 计算信息量
    # 计算每个指标与其他所有指标的相关性惩罚因子
    correlation_penalty = 1 - corr_matrix.abs().sum(axis=1)

    # 4. 计算最终信息量：标准差 * 相关性惩罚因子
    info = std_devs * correlation_penalty

    # 5. 计算权重：信息量标准化
    weights = info / info.sum()

    return weights

def kmo_test(data):
    X = np.corrcoef(data, rowvar=False)
    iX = np.linalg.inv(X)
    S2 = np.diag(np.diag(np.linalg.inv(iX)))
    AIS = np.dot(np.dot(S2, iX), S2)
    IS = X + AIS - 2 * S2
    Dai = np.diag(np.diag(np.sqrt(AIS)))
    IR = np.dot(np.dot(np.linalg.inv(Dai), IS), np.linalg.inv(Dai))
    AIR = np.dot(np.dot(np.linalg.inv(Dai), AIS), np.linalg.inv(Dai))
    a = np.sum((AIR - np.diag(np.diag(AIR))) ** 2)
    AA = np.sum(a)
    b = np.sum((X - np.eye(X.shape[0])) ** 2)
    BB = np.sum(b)
    kmo_statistic = BB / (AA + BB)
    return kmo_statistic

def bartlett_test(data):
    sample, variable = data.shape
    data_corr = np.corrcoef(data, rowvar=False)
    data_det = np.linalg.det(data_corr)
    chi2_statistic = -(sample - 1 - (2 * variable + 5) / 6) * np.log(data_det)
    df = variable * (variable - 1) // 2
    p_value = 1 - chi2.cdf(chi2_statistic, df)
    return chi2_statistic, df, p_value

def pca(data: pd.DataFrame):
    """_summary_
    PCA 降维，将大量的底层指标抽象出少量的高层指标以描述数据
    Args:
        data (pd.DataFrame): 标准化的的二维矩阵，行为样本列为指标

    Returns:
        _type_: 每个样本的评分
    """
    # KMO 测试
    kmo_statistic = kmo_test(data)
    print(f'KMO 测试统计量: {kmo_statistic}')
    print("0.90 - 1.00：非常适合进行因子分析\n0.80 - 0.89：适合进行因子分析\n0.70 - 0.79：有一定适合性，结果较好")
    # Bartlett 测试
    chi2_statistic, df, p_value = bartlett_test(data)
    print(f'Bartlett p-value: {p_value}')
    print("p 值：如果 p 值小于 0.05，拒绝零假设，表示数据变量之间有足够的相关性")
    
    if kmo_statistic < 0.7 and p_value > 0.05:
        print("数据不适合进行主成分分析")
        return data
    
    pca = PCA(n_components=0.85)
    pac_data = pca.fit_transform(data)
    # 返回 PCA 变换后的数据，PCA 的权重
    print(f'所保留的主成分个数: {pca.n_components_}')
    print(f'所保留的主成分各自的方差百分比（权重）:\n{pca.explained_variance_ratio_}')
    # 计算每个主成分中各个属性的权重
    print(f'每个主成分中各个属性的权重:\n{pca.components_.T}')
    return np.array(pac_data)@pca.explained_variance_ratio_
    
    

def topsis(data: pd.DataFrame, w: pd.DataFrame):
    """
    TOPSIS 评价多指标多样本，指标需正向化
    :param data: 正向化且标准化的行为样本列为指标的二维矩阵
    :param w: 指标的权重
    :return: 第 i 个样本的得分，排名第 i 的样本的索引
    """

    # 1. 乘以权重 (Weighted Normalization)
    weighted_data = data * w

    # 2. 计算理想解和负理想解 (Ideal and Negative-Ideal Solutions)
    ideal_best = weighted_data.max(axis=0)  # 理想解
    ideal_worst = weighted_data.min(axis=0)  # 负理想解

    # 3. 计算与理想解和负理想解的欧氏距离 (Euclidean Distance)
    dist_best = np.sqrt(((weighted_data - ideal_best) ** 2).sum(axis=1))
    dist_worst = np.sqrt(((weighted_data - ideal_worst) ** 2).sum(axis=1))

    # 4. 计算 TOPSIS 得分 (Score)
    scores = dist_worst / (dist_best + dist_worst)

    # 5. 返回排序后的得分
    return scores.sort_values(ascending=False)

if __name__ == "__main__":

    # 示例数据
    data = pd.DataFrame({
        '指标1': [0.2, 0.3, 0.4, 0.5],
        '指标2': [0.1, 0.4, 0.6, 0.9],
        '指标3': [0.8, 0.7, 0.9, 0.8]
    }, index=['样本1', '样本2', '样本3', '样本4'])

    print(data)
    
    # 标准化
    scaler = StandardScaler()
    data = pd.DataFrame(scaler.fit_transform(data), columns=data.columns, index=data.index)
    print(f'----------z-score 标准化后数据----------\n{data}')
    # data = scaler.inverse_transform(data)
    #data = pd.DataFrame(scaler.inverse_transform(data), columns=data.columns, index=data.index)
    #print(data)
    
    
    # 示例权重
    # weights = ewm(data)

    # 调用 topsis 方法
    print("-----------------PCA-----------------")
    scores = pca(data)
    print(f'PCA 得分:\n{scores}')
    print("-----------------TOPSIS-----------------")
    ewm_weights = ewm(data)
    print(f'熵权法权重:\n{ewm_weights}')
    scores = topsis(data, ewm_weights)
    print(f'熵权法+TOPSIS 得分:\n{scores}')
    critic_weights = critic(data)
    print(f'CRITIC 权重:\n{critic_weights}')
    scores = topsis(data, critic_weights)
    print(f'CRITIC+TOPSIS 得分:\n{scores}')
    

    

