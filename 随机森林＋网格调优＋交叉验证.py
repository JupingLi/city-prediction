import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# 解决画图中文显示问题
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 14

# 加载数据
data = pd.read_excel(r'C:\Users\86138\Desktop\mm.xlsx', engine='openpyxl')

# 将特征和目标变量分离
X = data.drop('target1', axis=1)
y = data['target1']

# 创建随机森林回归模型
rf_model = RandomForestRegressor()

# 定义参数空间
param_grid = {
    'n_estimators': [30, 50, 80],  # 减小树的数量
    'min_samples_split': [2, 3, 5],
    'min_samples_leaf': [1, 2, 3],
    'max_depth': [None, 5, 10],  # 减小max_depth的值
    'max_features': ['auto', 'sqrt']
}

# 创建GridSearchCV对象，加入交叉验证
grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error')

# 执行网格搜索和交叉验证
grid_search.fit(X, y)

# 获取最佳参数
best_params = grid_search.best_params_
print("最佳参数:", best_params)

# 使用最佳参数的模型进行训练
best_rf_model = grid_search.best_estimator_

# 交叉验证评估模型性能
cv_scores = cross_val_score(best_rf_model, X, y, cv=5)  # 使用5折交叉验证
print("交叉验证得分:", cv_scores)
print("平均交叉验证得分:", np.mean(cv_scores))

# 将数据分割成训练集和测试集，设置random_state
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 在测试集上进行预测
y_pred = best_rf_model.predict(X_test)

# 计算R方（R²）
r2 = r2_score(y_test, y_pred)

# 计算均方误差（MSE）
mse = mean_squared_error(y_test, y_pred)

# 计算平均绝对误差（MAE）
mae = mean_absolute_error(y_test, y_pred)

# 输出R方、MSE和MAE
print("测试集R方（R²）:", r2)
print("测试集均方误差（MSE）:", mse)
print("测试集平均绝对误差（MAE）:", mae)

# 假设已知未来一年的13个特征，存储在future_features变量中
future_features = pd.DataFrame({
    'feature1': [4.00],
    'feature2': [2023],
    'feature3': [149],
    'feature4': [100],
    'feature5': [73],
    'feature6': [15],
    'feature7': [32],
    'feature8': [144],
    'feature9': [34],
    'feature10': [95],
    'feature11': [33],
    'feature12': [73],
    'feature13': [0.45]
})

# 使用训练好的模型进行未来一年的目标变量预测
future_target = best_rf_model.predict(future_features)

# 打印未来一年的目标变量预测结果
print("未来一年的目标变量预测结果:", future_target)