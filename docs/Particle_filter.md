# Particle Filter
## 一、介绍
[README.md](../README.md)中介绍了Particle Filter与其它两个模型之间的区别与联系，Particle Filter认为观测变量和状态变量可以是任意分布，
噪声可以是任意分布，且变量之间的关系可以是非线性的:


Z<sub>t</sub> = g(Z<sub>t-1</sub> , u, ε) &nbsp;&nbsp;&nbsp;&nbsp;(1)

Z<sub>t</sub> = h(X<sub>t</sub>, u, δ)  &nbsp;&nbsp;&nbsp;&nbsp;(2)

通过与[Kalman filter](Kalman_filter.md)关系式比较可以发现，Particle Filter的适用范围要大得多，但Particle filter
没法像Kalman filter一样对状态后验概率分布更新得到闭式解，只能通过Monte Carlo采样的方式得到近似解。具体来说，就是用一组粒子(采样样本)和粒子的权重来逼近系统状态的后验概率分布，
即通过一个离散的分布列来逼近一个连续的分布函数。

## 二、采样算法
这里主要介绍一下Particle Filter用到的一些采样算法。采样本质上是要解决求积分的问题，而所有积分问题都可以写成某个概率分布下期望的形式，
所以也可以等价地说采样是求某个期望的方法。

### 2.1 Monte Carlo Sampling(MCS)

![Monte Carlo Sampling](../resources/particle_filter/Monte_Carlo_sampling.jpg)

### 2.2 Inportance Sampling(IS)

![Importance_sampling](../resources/particle_filter/Importance_Sampling.jpg)

其中每个采样样本f(x<sub>i</sub>)的权重P(x<sub>i</sub>)/P'(x<sub>i</sub>)，称为样本的重要性权重。对应的采样分布P'称为
提议分布(proposal distribution)。需要注意的是，对于一般的分布P我们很难直接从中进行采样，而Importance Sampling让我们可以
通过一个已知的分布P'进行采样。

### 2.3 Sequential Importance Sampling (SIS)
考虑一个高维空间，真实分布为p(X<sub>1:n</sub>)其中X<sub>1:n</sub>:=(X1， X2, ..., Xn)，提议分布为q(X<sub>1:n</sub>)，权重为w(X<sub>1:n</sub>)。
考虑一个特殊的情景，即我们需要从1维到n维依次采样，有没有办法用上次的权重进行新权重的计算(权重之间的递推关系式)？

![SIS](../resources/particle_filter/SIS.png)

得到如下关于权重的递推关系式，这样我们就可以通过上一维的权重直接计算当前维度的权重。

![SIS](../resources/particle_filter/SIS_2.png)

如果我们将1:n转换成1:t，赋予维度一个时序的意义，就可以用它来解决particle filter中的重要性权重计算更新问题。


### 2.4 Sampling Importance Resampling (SIR)
传统粒子滤波使用SIS会引起粒子退化(particle degeneracy)，即随着迭代的进行，一些粒子归一化权重会趋于1，而很多粒子权重则趋于0。
这样的分布列并不能很好地代表系统状态后验分布，因为如果某粒子的归一化权重很小，它就不太可能被抽到，不该存在有很多粒子的归一化权重趋于0的情况，
因此以这个分布列的期望作为Filtering的估计是不准确的。

重采样就是解决粒子退化的一种方法，具体来说就是以各个粒子的归一化权重作为分布列，从中采出N个样本，采出的所有样本权重均为1/N。这样相当于复制权重较大的粒子，
抛弃权重较小的粒子，用新得到的粒子来拟合分布，这样的采样技术称为重要性重采样(SIR)


## 二、解决的问题
### 2.1 Filtering问题
[README.md](../README.md)4.2节的Inference问题中，Particle filter要解决的就是Filtering问题，即求P(Z<sub>t</sub>|X<sub>1</sub>, X<sub>2</sub>, ... ,X<sub>t</sub>)。

### 2.2 求解Filtering问题递推形式

根据2.1节的公式(1)和(2)，我们得到了一个求解Filtering问题的递推形式，分为预测(prediction)和更新(update)两步:

***step 1. 预测(prediction)***

P(Z<sub>t</sub>|X<sub>1</sub>, X<sub>2</sub>, ... ,X<sub>t-1</sub>) = ∫ P(Z<sub>t-1</sub>|X<sub>1</sub>, X<sub>2</sub>, ... ,X<sub>t-1</sub>)P(Z<sub>t</sub>|Z<sub>t-1</sub>)dZ<sub>t-1</sub> 


***step 2. 更新(update)***

P(Z<sub>t</sub>|X<sub>1</sub>, X<sub>2</sub>, ... ,X<sub>t</sub>) ∝ P(Z<sub>t</sub>|X<sub>1</sub>, X<sub>2</sub>, ... ,X<sub>t-1</sub>)P(X<sub>t</sub>|Z<sub>t</sub>) 

