# 五、思考题与课后作业（第 3 题解答：Python 实操）

**题目：** 使用 Python 加载 Carseats 数据集，以销售额 Sales 为响应变量，选取 Price（价格）、Income（收入）、Advertising（广告）以及定性特征 ShelveLoc（货架位置），建立多元线性回归模型。

- 提取模型拟合报告，指出 ShelveLoc 的基准组是什么？
- 解读 ShelveLoc[Good] 系数的实际商业含义。
- 计算各变量的 VIF，评估是否存在多重共线性风险。

---

## 1. 代码实现

见同目录脚本 `03_python实操_carseats.py`，核心代码如下：

```python
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor
import pandas as pd

# 加载 Carseats 数据集（ISLR）
carseats = sm.datasets.get_rdataset("Carseats", "ISLR").data

# 建立多元线性回归（ShelveLoc 为定性变量，自动转为虚拟变量）
model = smf.ols("Sales ~ Price + Income + Advertising + ShelveLoc",
                data=carseats).fit()
print(model.summary())

# 计算各变量 VIF
X = model.model.exog
vif = pd.DataFrame({
    "variable": model.model.exog_names,
    "VIF": [variance_inflation_factor(X, i) for i in range(X.shape[1])],
})
print(vif)
```

数据基本情况：400 条样本，ShelveLoc 三水平分布为 Medium 219、Bad 96、Good 85。

## 2. 模型拟合报告

```
                            OLS Regression Results
==============================================================================
Dep. Variable:                  Sales   R-squared:                       0.627
Model:                            OLS   Adj. R-squared:                  0.622
Method:                 Least Squares   F-statistic:                     132.4
No. Observations:                 400   Prob (F-statistic):           4.89e-82
Df Residuals:                     394   Df Model:                          5
=======================================================================================
                          coef    std err          t      P>|t|      [0.025      0.975]
---------------------------------------------------------------------------------------
Intercept              10.3916      0.521     19.933      0.000       9.367      11.417
ShelveLoc[T.Good]       4.8359      0.260     18.624      0.000       4.325       5.346
ShelveLoc[T.Medium]     1.8959      0.213      8.894      0.000       1.477       2.315
Price                  -0.0570      0.004    -15.480      0.000      -0.064      -0.050
Income                  0.0137      0.003      4.399      0.000       0.008       0.020
Advertising             0.1056      0.013      8.044      0.000       0.080       0.131
=======================================================================================
```

模型整体显著（F = 132.4，p ≈ 4.9e-82），$R^2 = 0.627$，$R^2_{adj} = 0.622$，所有系数均在 1% 水平下显著。

### （1）ShelveLoc 的基准组

定性变量 ShelveLoc 有三个水平（Bad / Medium / Good），回归报告里只出现了 `ShelveLoc[T.Good]` 和 `ShelveLoc[T.Medium]` 两个虚拟变量，**没有出现 Bad**。

因此基准组（参照组）是 **ShelveLoc = Bad（货架位置差）**。Intercept（10.3916）就是当 Price、Income、Advertising 均为 0 且货架位置为 Bad 时的基准销售额。

### （2）ShelveLoc[Good] 系数的商业含义

`ShelveLoc[T.Good]` 系数为 **4.836**（p < 0.001）。

含义：**在价格、收入、广告投入完全相同的情况下**，把商品摆放在"好"的货架位置，相比摆放在"差"的货架位置（基准组），销售额平均多出约 **4.84 个单位**（即约 4836 件，因为 Sales 以千件为单位）。

商业解读：货架位置是极其重要的销售杠杆——好位置带来的销量提升（+4.84）甚至相当于广告预算增加约 46 个单位（4.836 / 0.1056 ≈ 45.8）的效果。同理，`ShelveLoc[T.Medium]` 系数 1.896 表示中等位置比差位置平均多卖约 1.90 个单位。对企业而言，争取优质货架位的投入回报可能远高于单纯砸广告。

### （3）VIF 多重共线性检验

| variable | VIF |
|---|---|
| Intercept | 1.000 |
| ShelveLoc[T.Good] | 1.497 |
| ShelveLoc[T.Medium] | 1.494 |
| Price | 1.008 |
| Income | 1.012 |
| Advertising | 1.009 |

**结论：** 除截距项外所有变量的 VIF 均约为 **1.0 ~ 1.5**，远小于常用警戒线（VIF > 5 或 10），说明各解释变量之间几乎线性无关，**不存在多重共线性风险**，回归系数估计稳定可靠。

## 3. 总结

- 基准组：**Bad（差的货架位置）**
- ShelveLoc[Good] = 4.836：其他条件不变，好货架位比差货架位平均多卖约 4.84 千件
- 全部 VIF < 1.5，无多重共线性问题
