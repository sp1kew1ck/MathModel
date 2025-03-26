"""_summary_
用于数据预处理的工具函数
针对 pd.DataFrame 形式的数据
数据预处理过程：
1. 数据清洗(Data Cleaning)：缺失值处理、异常值处理。获得 *_cleaned 命名的数据集
2. 数据变换(Data Transformation)：数据规范化、数据离散化、数据归约、分类特征编码。获得 *_trans 命名的数据集
"""

import pandas as pd
import numpy as np
from scipy.interpolate import CubicSpline

"""
数据清洗：
1. 三次样条插值填充缺失值
-------------------------------------------------------------------------------
"""

def fillna_CubicSpline(data: pd.DataFrame) -> pd.DataFrame:
    """
    Process missing values in DataFrame using vectorized operations
    1e8 data need : 120 s
    Args:
        data: Input DataFrame
    Returns:
        DataFrame with processed missing values
    """
    # Create copy only once
    result = data.copy()
    
    # Get all numeric columns with NaN values in one operation
    numeric_cols = result.select_dtypes(include=[np.number]).columns
    null_masks = result[numeric_cols].isnull()
    cols_with_nan = result[numeric_cols].columns[null_masks.any()]
    
    # Process all columns with NaN values
    for col in cols_with_nan:
        series = result[col]
        null_mask = null_masks[col]
        
        valid_indices = np.arange(len(series))[~null_mask]
        valid_values = series[~null_mask].values
        
        if len(valid_values) > 3:
            # Vectorized cubic spline interpolation
            cs = CubicSpline(valid_indices, valid_values)
            null_indices = np.arange(len(series))[null_mask]
            result.loc[null_mask, col] = cs(null_indices)
        else:
            # Vectorized linear interpolation
            result[col] = series.interpolate(method='linear')
    
    return result

"""
数据变换：
1. 数据缩放 Minmax
2. 数据规范化 Z-score
3. 分类特征编码 独热编码
-------------------------------------------------------------------------------
"""

def minmax_scale(data: pd.DataFrame, cols) -> pd.DataFrame:
    """_summary_
    min-max缩放数据, 将数据缩放到[0, 1]范围内
    1e8 data need : 5 s
    Args:
        data (pd.DataFrame): _description_
        cols (_type_): _description_

    Returns:
        pd.DataFrame: _description_
    """
    min_vals = data[cols].min()
    max_vals = data[cols].max()
    scaled_data = (data[cols] - min_vals) / (max_vals - min_vals)
    return scaled_data

def zscore_normalize(data: pd.DataFrame, cols: list) -> pd.DataFrame:
    """Z-score标准化数据，将数据转换为均值为0，标准差为1的分布。
    1e8 data need : 5 s
    Args:
        data (pd.DataFrame): 输入数据框。
        cols (list): 需要规范化的列名列表。
    
    Returns:
        pd.DataFrame: 规范化后的数据框。
    """
    means = data[cols].mean()
    stds = data[cols].std()
    normalized_data = (data[cols] - means) / stds
    return normalized_data

def one_hot_encode(data: pd.DataFrame, cols: list) -> pd.DataFrame:
    """
    对指定列进行独热编码。
    1e8 need : 60 s
    Args:
        data (pd.DataFrame): 输入数据框。
        cols (list): 需要独热编码的列名列表。

    Returns:
        pd.DataFrame: 独热编码后的数据框。
    """
    encoded_data = pd.get_dummies(data, columns=cols, drop_first=False)
    return encoded_data