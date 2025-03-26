import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold

# 1. 数据加载与处理
# 这里假设已经加载了数据
# df = pd.read_csv('your_data.csv')
# X = df.drop(columns=['target'])  # 特征
# y = df['target']  # 目标变量

# 使用 sklearn 自带的iris数据集作为例子
from sklearn.datasets import load_iris
data = load_iris()
X = data.data
y = data.target

# 数据划分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 2. 数据预处理
# 标准化特征
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 3. 定义模型
rf_model = RandomForestClassifier(random_state=42)

# 4. 定义参数网格进行GridSearchCV调参
param_grid = {
    'n_estimators': [100, 200, 300],  # 树的数量
    'max_depth': [None, 10, 20, 30],  # 最大深度
    'min_samples_split': [2, 5, 10],  # 分裂一个节点所需的最小样本数
    'min_samples_leaf': [1, 2, 4],    # 叶子节点的最小样本数
    'max_features': ['auto', 'sqrt', 'log2'],  # 每棵树使用的最大特征数
    'bootstrap': [True, False]  # 是否使用自助法
}

# 5. 使用GridSearchCV进行超参数调优
grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, 
                           cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42), 
                           scoring='accuracy', n_jobs=-1, verbose=2)

# 在训练数据上进行搜索
grid_search.fit(X_train, y_train)

# 6. 输出最佳参数和最佳得分
print(f"Best parameters found: {grid_search.best_params_}")
print(f"Best cross-validation score: {grid_search.best_score_:.4f}")

# 7. 使用最佳模型进行预测
best_rf_model = grid_search.best_estimator_
y_pred = best_rf_model.predict(X_test)

# 8. 输出评估指标
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print(f"Accuracy on test set: {accuracy_score(y_test, y_pred):.4f}")
