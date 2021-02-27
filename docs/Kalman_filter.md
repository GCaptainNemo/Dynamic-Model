# Linear Dynamic System(Kalman filter)
## 一、介绍
[README.md](../README.md)中介绍了线性动态系统与其它两个模型之间的区别与联系，线性动态系统认为观测变量和状态变量服从高斯分布，
且变量之间满足如下线性关系(故又称为线性高斯模型):

Z<sub>t</sub> = AZ<sub>t-1</sub> + B + ε (1)

Z<sub>t</sub> = CX<sub>t</sub> + D + δ (2)

ε和δ代表噪声变量，且ε~ N(0, Q), δ~ N(0, R)。则Z<sub>t</sub> | Z<sub>t-1</sub>~N(AZ<sub>t-1</sub> + B, Q)，
X<sub>t</sub> | Z<sub>t</sub> ~ N(CZ<sub>t</sub> + D, R)，另外，初始状态Z<sub>1</sub>~N(μ<sub>1</sub>, Σ<sub>1</sub>)。

综上：线性动态系统的参数为***θ*** = (A, B, C, D, Q, R, μ<sub>1</sub>, Σ<sub>1</sub>)

