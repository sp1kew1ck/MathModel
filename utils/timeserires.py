import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from visualize import plot_line, set_sci_style, plot_heatmap

# 引入 arima 模型和时间序列处理工具
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import arma_order_select_ic, adfuller, acf, pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# xgboost 模型
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.metrics import mean_squared_error
import xgboost as xgb

from mytimer import Timer

def auto_diff(data: pd.DataFrame, alpha: float = 0.05):
    """
    自动差分非平稳序列直至序列平稳，使用 adf 检验平稳性
    Args:
        data (pd.DataFrame): 原始时间序列
        alpha (float): ADF 检验的显著性水平，默认值为 0.05

    Returns:
        pd.DataFrame: 平稳时间序列
        int: 差分阶数
    """
    if data.empty:
        raise ValueError("输入的时间序列数据为空")
    
    d = int(0)
    diff_data = data.copy()  # 复制数据，避免原始数据被修改
    
    # 检查是否初始就是平稳的
    p_value = adfuller(diff_data)[1]
    if p_value <= alpha:
        return diff_data, d
    
    # 迭代差分，直到序列平稳
    while p_value > alpha:
        diff_data = diff_data.diff().iloc[1:]  # 进行差分并去除NaN
        d += 1
        p_value = adfuller(diff_data)[1]  # 重新计算ADF检验的p-value
    
    return diff_data, d

def white_noise_test(diff_data: pd.DataFrame, lags: int = 20, alpha: float = 0.05):
    """_summary_
    白噪声检验，若为白噪声则返回 True，使用 ljungbox 和 boxpierce 检验
    Args:
        diff_data (pd.DataFrame): _description_
        lags (int, optional): _description_. Defaults to 20.
        alpha (float, optional): _description_. Defaults to 0.05.

    Raises:
        ValueError: _description_

    Returns:
        _type_: _description_
    """
    if diff_data.empty:
        raise ValueError("输入的时间序列数据为空")
    ljungbox = acorr_ljungbox(diff_data, lags=20, return_df=True, boxpierce=True)
    print(ljungbox)
    # 寻找同时小于 alpha 的滞后期
    matching_lags = ljungbox[((ljungbox["lb_pvalue"] < alpha) & (ljungbox["bp_pvalue"] < alpha))].index
    if matching_lags.empty:
        return True
    return False

def plot_acf_pacf(diff_data: pd.DataFrame, lags: int = 20, alpha: float = 0.05, save_path: str = "img/acf_pacf"):
    set_sci_style()
    fig, axs = plt.subplots(2, 1, figsize=(12, 8))
    plot_acf(diff_data, lags=20, alpha=0.05, ax=axs[0])
    plot_pacf(diff_data, lags=20, alpha=0.05, ax=axs[1])
    plt.savefig(save_path+'.pdf', format='pdf', bbox_inches='tight', dpi=300)
    plt.show()
    
def auto_arima(time_series: pd.DataFrame):
    # 1. 平稳性检验
    diff_data, d = auto_diff(time_series)
    # 2. 白噪声检验
    if white_noise_test(diff_data):
        return None
    # 3. 绘制 ACF 和 PACF 图
    plot_acf_pacf(diff_data) 
    # 4. 参数估计
    result = arma_order_select_ic(diff_data, ic=['bic'], trend='c', max_ar=8, max_ma=8)
    p, q = result.bic_min_order
    print(f"p: {p}, d: {d}, q: {q}")
    result.bic.columns = ['MA' + str(col) for col in result.bic.columns]
    result.bic.index = ['AR' + str(ind) for ind in result.bic.index]
    plot_heatmap(result.bic, '', '', 'img/bic')
    model = ARIMA(time_series, order=(p,d,q)).fit()
    # 5. 模型检验: 残差诊断
    # 绘制 ACF 和 PACF 
    # 无显著自相关：如果 ACF 图中的条形图都位于置信区间范围内，并且没有显著的峰值
    # 说明残差序列没有自相关，模型拟合良好，残差可以认为是白噪声。
    plot_acf_pacf(model.resid, save_path="img/resid_acf_pacf")
    # 白噪声检验 ljungbox & boxpierce
    print(white_noise_test(model.resid))
    # 模型总结
    print(model.summary)
    # 历史数据折线图
    history = pd.DataFrame({"actual": time_series.iloc[:, 0], 'predicted':model.predict(dynamic=False)}, index=time_series.index)
    plot_line(history, 'Time', 'Value', 'img/arima_history')
    
    return model

def create_multistep_features(data: np.ndarray, window_size: int, forecast_horizon: int):
    """_summary_
    创建滑动窗口特征和多步预测标签
    Args:
        data (np.ndarray): _description_
        window_size (int): _description_
        forecast_horizon (int): _description_

    Returns:
        _type_: 特征, 标签
    """
    X = np.zeros((len(data) - window_size - forecast_horizon, window_size))
    y = np.zeros((len(data) - window_size - forecast_horizon, forecast_horizon))
    for i in range(len(data) - window_size - forecast_horizon):
        X[i, :] = data[i:i + window_size].reshape(window_size)  # 特征是过去 window_size 个数据点
        y[i, :] = data[i + window_size:i + window_size + forecast_horizon].reshape(forecast_horizon)  # 标签是未来 forecast_horizon 个数据点
    return X, y

def auto_xgboost(data: pd.DataFrame, window_size: int, forecast_horizon: int):
    # 1. 创建特征和标签
    X, y = create_multistep_features(np.array(data), window_size, forecast_horizon)
    # 2. 划分数据集为训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # 3. 将数据转为DMatrix格式，适用于xgboost
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    # 4. 调参
    print("-----------------GridSearch-----------------")
    # 参数字典
    params_grid = {
        'eta': [0.03, 0.06, 0.1, 0.3, 0.6],
        'max_depth': [6, 7, 8],
        'n_estimators': [100, 200, 300]
    }
    gsxgb = GridSearchCV(xgb.XGBRegressor(objective='reg:squarederror', n_jobs=-1, eval_metric='rmse'), # 模型
                        params_grid, # 待调参数（字典）
                        scoring="neg_mean_squared_error",
                        n_jobs=-1, # -1表示使用全部的cpu运算
                        cv=KFold(n_splits=5, shuffle=True, random_state=42))
    gsxgb_result = gsxgb.fit(X_train, y_train)
    ### summarize results
    means = gsxgb_result.cv_results_['mean_test_score']
    params =  gsxgb_result.cv_results_['params']
    for mean, param in zip(means,params):
        print("%f with: %r" % (mean,param))
    print("best:%f using %s" % (gsxgb_result.best_score_,gsxgb_result.best_params_))
    
    # 5. 训练模型
    params = {'objective': 'reg:squarederror', 'n_jobs':-1, 'eval_metric': 'rmse'}.update(gsxgb_result.best_params_)
    num_epoch = 100  # 迭代次数
    bst = xgb.train(params, dtrain, num_epoch)
    
    # 6. 测试集预测
    y_pred = bst.predict(dtest)

    # 7. 评估模型
    mse = mean_squared_error(y_test, y_pred)
    print(f'Mean Squared Error: {mse}')

    # 8. 绘制预测结果
    # 连接两个一维数组 X_test[0, :] 和 y_test[0, :]
    test_data = np.concatenate((X_test[0, :], y_test[0, :]))
    test_result = pd.DataFrame({"actual": test_data, "predicted": [np.nan] * len(X_test[0, :]) + list(y_pred[0, :])}, index=range(0, len(test_data)))
    plot_line(test_result, 'Time', 'Value', 'img/xgboost_result')
    return bst
   
if __name__ == '__main__':
    # 1. 观察时间序列数据
    x = np.linspace(-32*np.pi, 32*np.pi, 500)
    # 生成不平稳的时间序列：加入线性趋势和噪声
    trend = 0.05 * x  # 线性趋势项
    noise = np.random.normal(0, 0.5, len(x))  # 正态分布噪声
    y = np.sin(x) + trend + noise  # 生成不平稳的序列
    time_series = pd.DataFrame({"actual":y}, index=range(1, len(y)+1))
    # print(time_series)
    plot_line(time_series, "time", "value", "img/time_series")
    
    timer = Timer()
    # # 2. 平稳性检验
    # # 差分直至平稳
    # diff_data, d = auto_diff(time_series)
    # print(d)
    # plot_line(diff_data, "time", "value", "img/time_series")
    
    # # 3. 白噪声检验
    # print(white_noise_test(diff_data))
    
    # # 4. 绘制 ACF 和 PACF 图
    # plot_acf_pacf(diff_data)
    
    # 5. 模型识别
    # 6. 参数估计
    model = auto_arima(time_series)
    print(f'{timer.stop():.5f} sec')
    
    model = auto_xgboost(time_series, 50, 5)
    print(f'{timer.stop():.5f} sec')
    # 7. 模型检验
    # 8. 模型优化
    # 9. 模型预测