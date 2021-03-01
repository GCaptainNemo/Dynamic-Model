# Dynamic-model
## 一、介绍
机器学习领域有频率学派与贝叶斯学派两大学派，其中频率学派把模型参数当成固定未知的数，学习问题是一个对参数进行点估计的优化问题，由此发展出了统计机器学习这一分支。
而贝叶斯学派把模型参数也当成随机变量，非完全贝叶斯(non-full bayesian)对参数进行最大后验估计，本质上还是一类优化问题，因此也有的学者把非完全贝叶斯划入频率学派的范畴。
而完全贝叶斯(full-bayesian)则把学习问题看成一个积分问题，用MCMC等数值积分算法进行参数估计。

贝叶斯学派最大的成果就是概率图模型，包括贝叶斯网络(Bayesian network)和马尔科夫网络(Markov network)。而在概率图中加入时间序列，就是动态模型(Dynamic model)。
这里介绍三种动态模型，包括:

1. [HMM(Hidden Markov Model)](docs/HMM.md)
2. [Linear Dynamic system (Kalman filter)](docs/Kalman_filter.md)
3. [Particle filter](docs/Particle_filter.md)

这三种动态模型的概率图相同，如下图所示:

![PGM](resources/DS_PGM.jfif)

其中X<sub>i</sub>是观测变量(X<sub>i</sub>之间不是独立同分布的)，Z<sub>i</sub>称为系统状态变量，也是模型的隐变量，因此动态模型也被称为状态空间模型(State Space Model)。

## 二、比较
|            |       HMM|   Kalman Filter|      Particle Filter|
|------      | ---------|----------------|---------------|
|状态空间     |     离散  |      连续      | 连续|
|观测空间     |  连续或离散|      连续      |  连续|
|P(Z<sub>t</sub>&#124;Z<sub>t-1</sub>)   |  A(转移矩阵)|   N(AZ<sub>t-1</sub> + B, Q)    |   f(Z<sub>t-1</sub>) |
|P(X<sub>t</sub>&#124;Z<sub>t</sub>)   |  B(发射矩阵)|   N(CZ<sub>t-1</sub> , R)    |   g(Z<sub>t</sub>) |
|P(Z<sub>1</sub>)|                π    | N(μ1, ∑1) |  f(Z<sub>1</sub>)|
---
1. HMM模型的系统状态变量的取值是离散的，对于观测变量的取值离散或连续没有要求。
2. 线性动态系统模型(Kalman filter)的状态变量和观测变量的取值都是连续的，而且状态变量和观测变量都服从高斯分布，状态Z<sub>t</sub>、Z<sub>t-1</sub>之间
和状态与观测Z<sub>t</sub>、X<sub>t</sub>之间是一个线性关系(故又称为 Linear Gaussian Model)。
3. Particle filter模型的状态变量和观测变量的取值都是连续的，但是不服从高斯分布(non Gaussian)且非线性(non-linear)。

## 三、动态系统的假设
动态模型有两个基本的假设
### 3.1 同质(齐次)马尔可夫假设
在给定t时刻状态的情况下，t+1时刻的状态与t时刻之前的状态和观测无关。表示成概率的形式即:

P(Z<sub>t+1</sub>|Z<sub>t</sub>) = P(Z<sub>t+1</sub>|Z<sub>t</sub>, Z<sub>t-1</sub>, ..., Z<sub>1</sub>, X<sub>t</sub>, ..., X<sub>1</sub>)

### 3.1.2 观测独立性假设

|给定t时刻状态的情况下，t时刻的观测与t之前的状态和观测无关。
P(X<sub>t</sub>|Z<sub>t</sub>) = P(X<sub>t</sub>|Z<sub>t</sub>, Z<sub>t-1</sub>, ..., Z<sub>1</sub>, X<sub>t-1</sub>, ..., X<sub>1</sub>)

## 四、动态模型问题
### 4.1 Learning问题
Learning问题就是要估计出模型的参数。

λ<sub>MLE</sub> = argmax P(X|λ)

对于含有隐变量的极大似然估计，常用EM算法求解。

### 4.2 Inference问题
#### 4.2.1 分类
Inference问题本质上就是求关于隐变量的后验概率P(Z|X)，但由于动态系统的特殊性，Inference又可以细分成以下几类:

1. Decoding: 求P(Z<sub>1</sub>, Z<sub>2</sub>, ... ,Z<sub>t</sub>|X<sub>1</sub>, X<sub>2</sub>, ... ,X<sub>t</sub>)
2. Filtering: 求 P(Z<sub>t</sub>|X<sub>1</sub>, X<sub>2</sub>, ... ,X<sub>t</sub>) (online)
3. Smoothing: 求 P(Z<sub>t</sub>|X<sub>1</sub>, X<sub>2</sub>, ... ,X<sub>T</sub>) (offline)
4. Prediction: 求 P(Z<sub>t+1</sub>|X<sub>1</sub>, X<sub>2</sub>, ... ,X<sub>t</sub>) 或者 P(X<sub>t+1</sub>|X<sub>1</sub>, X<sub>2</sub>, ... ,X<sub>t</sub>) 

其中HMM关注Decoding问题，而Kalman filter和Particle filter更关注Filtering问题。

#### 4.2.2 求Inference问题的方法
根据后验概率是否能写成解析形式，可以把推断分成精确性推断和近似推断。近似推断又可以分为确定性近似推断(e.g., 变分推理)和随机近似推断(e.g., Monte Carlo采样)。

### 4.3 Evaluation问题
Evaluation问题即在给定参数**θ**的情况下求似然函数P(X|**θ**)。


## 五、参考资料
1. [https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python](https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python)

