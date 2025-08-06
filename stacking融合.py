import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.ensemble import StackingRegressor

# 解决画图中文显示问题
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 14

# 加载数据
data = pd.read_excel(r'D:\project\city prediction\mm.xlsx', engine='openpyxl')

# 将特征和目标变量分离
X = data.drop('常住人口（万人）', axis=1)
y = data['常住人口（万人）']

# 创建XGBoost回归模型
xgb_model = XGBRegressor()

# 定义XGBoost的参数空间
param_grid_xgb = {
    'n_estimators': [30, 50, 80],
    'learning_rate': [0.1, 0.01, 0.001],
    'max_depth': [3, 5, 7],
    'min_child_weight': [1, 3, 5],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.8, 0.9, 1.0],
}

# 创建GridSearchCV对象，加入交叉验证
grid_search_xgb = GridSearchCV(estimator=xgb_model, param_grid=param_grid_xgb, cv=5, scoring='neg_mean_squared_error')

# 执行网格搜索和交叉验证
grid_search_xgb.fit(X, y)

# 获取最佳参数
best_params_xgb = grid_search_xgb.best_params_
print("XGBoost最佳参数:", best_params_xgb)

# 使用最佳参数的XGBoost模型进行训练
best_xgb_model = grid_search_xgb.best_estimator_

# 创建随机森林回归模型
rf_model = RandomForestRegressor()

# 定义随机森林的参数空间
param_grid_rf = {
    'n_estimators': [30, 50, 80],
    'min_samples_split': [2, 3, 5],
    'min_samples_leaf': [1, 2, 3],
    'max_depth': [None, 5, 10],
    'max_features': ['auto', 'sqrt']
}

# 创建GridSearchCV对象，加入交叉验证
grid_search_rf = GridSearchCV(estimator=rf_model, param_grid=param_grid_rf, cv=5, scoring='neg_mean_squared_error')

# 执行网格搜索和交叉验证
grid_search_rf.fit(X, y)

# 获取最佳参数
best_params_rf = grid_search_rf.best_params_
print("随机森林最佳参数:", best_params_rf)

# 使用最佳参数的随机森林模型进行训练
best_rf_model = grid_search_rf.best_estimator_
# 创建模型融合(Stacking)
estimators = [
    ('xgb', best_xgb_model),
    ('rf', best_rf_model),
]
# 使用StackingRegressor进行模型融合
stacking_model = StackingRegressor(estimators=estimators, final_estimator=XGBRegressor())
# 交叉验证评估模型性能
cv_scores_stacking = cross_val_score(stacking_model, X, y, cv=5)
print("Stacking交叉验证得分:", cv_scores_stacking)
print("Stacking平均交叉验证得分:", np.mean(cv_scores_stacking))

# 将数据分割成训练集和测试集，设置random_state
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 在测试集上进行Stacking模型的预测
stacking_model.fit(X_train, y_train)
y_pred_stacking = stacking_model.predict(X_test)

# 计算R方（R²）
r2_stacking = r2_score(y_test, y_pred_stacking)

# 计算均方误差（MSE）
mse_stacking = mean_squared_error(y_test, y_pred_stacking)

# 计算平均绝对误差（MAE）
mae_stacking = mean_absolute_error(y_test, y_pred_stacking)

# 输出Stacking模型的性能指标
print("Stacking测试集R方（R²）:", r2_stacking)
print("Stacking测试集均方误差（MSE）:", mse_stacking)
print("Stacking测试集平均绝对误差（MAE）:", mae_stacking)

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

# 使用模型融合进行未来一年的目标变量预测
future_target_stacking = stacking_model.predict(future_features)

# 打印未来一年的目标变量预测结果
print("Stacking未来一年的目标变量预测结果:", future_target_stacking)