# 五、思考题与课后作业 —— 第 3 题 Python 实操
# Carseats 数据集：Sales ~ Price + Income + Advertising + ShelveLoc
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor
import pandas as pd

# 1. 加载 Carseats 数据集（ISLR）
carseats = sm.datasets.get_rdataset("Carseats", "ISLR").data
print(carseats.head())
print(carseats["ShelveLoc"].value_counts())

# 2. 建立多元线性回归模型（ShelveLoc 为定性特征，自动虚拟变量化）
model = smf.ols("Sales ~ Price + Income + Advertising + ShelveLoc", data=carseats).fit()
print(model.summary())

# 3. 各变量 VIF
X = model.model.exog
names = model.model.exog_names
vif = pd.DataFrame({
    "variable": names,
    "VIF": [variance_inflation_factor(X, i) for i in range(X.shape[1])],
})
print(vif)
