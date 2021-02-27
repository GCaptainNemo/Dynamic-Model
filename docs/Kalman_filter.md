# Linear Dynamic System(Kalman filter)
## 一、介绍
[README.md](../README.md)中介绍了线性动态系统与其它两个模型之间的区别与联系，线性动态系统认为观测变量和状态变量服从高斯分布，
且变量之间满足如下线性关系(故又称为线性高斯模型):

Z<sub>t</sub> = AZ<sub>t-1</sub> + B + ε (1)

Z<sub>t</sub> = CX<sub>t</sub> + D + δ (2)

ε和δ代表噪声变量，且ε~ N(0, Q), δ~ N(0, R)。则Z<sub>t</sub> | Z<sub>t-1</sub>~N(AZ<sub>t-1</sub> + B, Q)，
X<sub>t</sub> | Z<sub>t</sub> ~ N(CZ<sub>t</sub> + D, R)，另外，初始状态Z<sub>1</sub>~N(μ<sub>1</sub>, Σ<sub>1</sub>)。

综上：线性动态系统的参数为***θ*** = (A, B, C, D, Q, R, μ<sub>1</sub>, Σ<sub>1</sub>)

## 二、解决的问题
### 2.1 Inference问题
[README.md]4.2节的Inference问题中，线性动态系统要解决的就是filtering问题，即求P(Z<sub>t</sub>|X<sub>1</sub>, X<sub>2</sub>, ... ,X<sub>t</sub>)。
由Bayes公式和动态模型的假设可得:

P(Z<sub>t</sub>|X<sub>1</sub>, X<sub>2</sub>, ... ,X<sub>t</sub>) ∝ P(Z<sub>t</sub>|X<sub>1</sub>, X<sub>2</sub>, ... ,X<sub>t-1</sub>)P(X<sub>t</sub>|Z<sub>t</sub>) (1)

其中求P(Z<sub>t</sub>|X<sub>1</sub>, X<sub>2</sub>, ... ,X<sub>t-1</sub>)是一个prediction问题，且满足:

P(Z<sub>t</sub>|X<sub>1</sub>, X<sub>2</sub>, ... ,X<sub>t-1</sub>) = ∫ P(Z<sub>t-1</sub>|X<sub>1</sub>, X<sub>2</sub>, ... ,X<sub>t-1</sub>)P(Z<sub>t</sub>|Z<sub>t-1</sub>)dZ<sub>t-1</sub> (2)

### 2.2 求解线性动态系统模型的步骤

根据2.1节的公式(1)和(2)，线性动态系统模型分为预测(prediction)和更新(update)两步:

step 1. 预测(prediction)

P(Z<sub>t</sub>|X<sub>1</sub>, X<sub>2</sub>, ... ,X<sub>t-1</sub>) = ∫ P(Z<sub>t-1</sub>|X<sub>1</sub>, X<sub>2</sub>, ... ,X<sub>t-1</sub>)P(Z<sub>t</sub>|Z<sub>t-1</sub>)dZ<sub>t-1</sub> 

代入高斯分布假设得：

N()

step 2. 更新(update)




