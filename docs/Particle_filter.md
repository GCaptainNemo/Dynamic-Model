# Particle Filter
## 一、介绍
[README.md](../README.md)中介绍了Particle Filter与其它两个模型之间的区别与联系，Particle Filter认为观测变量和状态变量可以是任意分布，
噪声可以是任意分布，且变量之间的关系可以是非线性的:


Z<sub>t</sub> = g(Z<sub>t-1</sub> , u, ε) &nbsp;&nbsp;&nbsp;&nbsp;(1)

Z<sub>t</sub> = h(X<sub>t</sub>, u, δ)  &nbsp;&nbsp;&nbsp;&nbsp;(2)

通过与[Kalman filter](Kalman_filter.md)关系式比较可以发现，Particle Filter的适用范围要大得多，但Particle filter
没法像Kalman filter一样对参数更新得到漂亮的闭式解，只能通过MCMC采样的方式得到近似解。







## 二、解决的问题
### 2.1 Filtering问题
[README.md](../README.md)4.2节的Inference问题中，Particle filter要解决的就是Filtering问题，即求P(Z<sub>t</sub>|X<sub>1</sub>, X<sub>2</sub>, ... ,X<sub>t</sub>)。
由Bayes公式和动态模型的假设可得:

P(Z<sub>t</sub>|X<sub>1</sub>, X<sub>2</sub>, ... ,X<sub>t</sub>) ∝ P(Z<sub>t</sub>|X<sub>1</sub>, X<sub>2</sub>, ... ,X<sub>t-1</sub>)P(X<sub>t</sub>|Z<sub>t</sub>) &nbsp;&nbsp;&nbsp;&nbsp;(1)

其中求P(Z<sub>t</sub>|X<sub>1</sub>, X<sub>2</sub>, ... ,X<sub>t-1</sub>)是一个prediction问题，且满足:

P(Z<sub>t</sub>|X<sub>1</sub>, X<sub>2</sub>, ... ,X<sub>t-1</sub>) = ∫ P(Z<sub>t-1</sub>|X<sub>1</sub>, X<sub>2</sub>, ... ,X<sub>t-1</sub>)P(Z<sub>t</sub>|Z<sub>t-1</sub>)dZ<sub>t-1</sub> &nbsp;&nbsp;&nbsp;&nbsp;(2)

### 2.2 求解Filtering问题递推形式

根据2.1节的公式(1)和(2)，我们得到了一个求解Filtering问题的递推形式，分为预测(prediction)和更新(update)两步:

***step 1. 预测(prediction)***

P(Z<sub>t</sub>|X<sub>1</sub>, X<sub>2</sub>, ... ,X<sub>t-1</sub>) = ∫ P(Z<sub>t-1</sub>|X<sub>1</sub>, X<sub>2</sub>, ... ,X<sub>t-1</sub>)P(Z<sub>t</sub>|Z<sub>t-1</sub>)dZ<sub>t-1</sub> 


***step 2. 更新(update)***

P(Z<sub>t</sub>|X<sub>1</sub>, X<sub>2</sub>, ... ,X<sub>t</sub>) ∝ P(Z<sub>t</sub>|X<sub>1</sub>, X<sub>2</sub>, ... ,X<sub>t-1</sub>)P(X<sub>t</sub>|Z<sub>t</sub>) 

