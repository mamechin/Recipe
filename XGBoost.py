import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
import requests
df_data = pd.read_csv("../input/secondrecipe/.csv")
# df_data
X=df_data.drop(labels = ['recipe','type'],axis=1).values
y = df_data['type'].values
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = .2, random_state=42)
from xgboost import XGBClassifier
# 建立XGBClassifier模型
xgboostModel = XGBClassifier(n_estimators=100,learning_rate=0.2)
# # 使用訓練資料訓練模型
xgboostModel.fit(X_train,y_train)
# # 使用訓練資料進行模型分類
predicted = xgboostModel.predict(X_train)
print('訓練集: ',xgboostModel.score(X_train,y_train))
print('測試集: ',xgboostModel.score(X_test,y_test))
from xgboost import plot_importance
from xgboost import plot_tree
plot_importance(xgboostModel)
print('特徵程度重要性: ',xgboostModel.feature_importances_)
df_train = pd.DataFrame(X_train)
df_train['type']=y_train
# 建立測試集的dataframe
df_test = pd.DataFrame(X_test)
df_test['type'] = y_test
df_train.columns = ['salt(g)','sugar(g)','chili(g)','vinegar(ml)']+list(df_train.columns[4:])
from sklearn.manifold import TSNE

tsneModel = TSNE(n_components=2, random_state=42,n_iter=1000)
train_reduced = tsneModel.fit_transform(X_train)
plt.figure(figsize=(8,6))
plt.scatter(train_reduced[:, 0], train_reduced[:, 1], c=y_train, alpha=0.5,
            cmap=plt.cm.get_cmap('nipy_spectral', 10))

plt.colorbar()
plt.show()