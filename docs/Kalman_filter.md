# Linear Dynamic System(Kalman filter)
## 一、介绍
[README.md](../README.md)中介绍了线性动态系统与其它两个模型之间的区别与联系，线性动态系统认为观测变量和状态变量服从高斯分布，
且变量之间满足如下线性关系(故又称为线性高斯模型):

Z<sub>t</sub> = AZ<sub>t-1</sub> + B + ε &nbsp;&nbsp;&nbsp;&nbsp;(1)

Z<sub>t</sub> = CX<sub>t</sub> + δ  &nbsp;&nbsp;&nbsp;&nbsp;(2)

ε和δ代表噪声变量，且ε~ N(0, Q), δ~ N(0, R)。则Z<sub>t</sub> | Z<sub>t-1</sub>~N(AZ<sub>t-1</sub> + B, Q)，
X<sub>t</sub> | Z<sub>t</sub> ~ N(CZ<sub>t</sub>, R)，另外，初始状态Z<sub>1</sub>~N(μ<sub>1</sub>, Σ<sub>1</sub>)。

综上，线性动态系统的参数为***θ*** = (A, B, C, Q, R, μ<sub>1</sub>, Σ<sub>1</sub>)。**注意**公式2没有常数项，
因为可以给状态变量添加一个齐次坐标，这样做可以让推出的公式更简洁。

## 二、解决的问题
### 2.1 Inference问题
[README.md](../README.md)4.2节的Inference问题中，线性动态系统要解决的就是filtering问题，即求P(Z<sub>t</sub>|X<sub>1</sub>, X<sub>2</sub>, ... ,X<sub>t</sub>)。
由Bayes公式和动态模型的假设可得:

P(Z<sub>t</sub>|X<sub>1</sub>, X<sub>2</sub>, ... ,X<sub>t</sub>) ∝ P(Z<sub>t</sub>|X<sub>1</sub>, X<sub>2</sub>, ... ,X<sub>t-1</sub>)P(X<sub>t</sub>|Z<sub>t</sub>) &nbsp;&nbsp;&nbsp;&nbsp;(1)

其中求P(Z<sub>t</sub>|X<sub>1</sub>, X<sub>2</sub>, ... ,X<sub>t-1</sub>)是一个prediction问题，且满足:

P(Z<sub>t</sub>|X<sub>1</sub>, X<sub>2</sub>, ... ,X<sub>t-1</sub>) = ∫ P(Z<sub>t-1</sub>|X<sub>1</sub>, X<sub>2</sub>, ... ,X<sub>t-1</sub>)P(Z<sub>t</sub>|Z<sub>t-1</sub>)dZ<sub>t-1</sub> &nbsp;&nbsp;&nbsp;&nbsp;(2)

### 2.2 求解线性动态系统模型的步骤

根据2.1节的公式(1)和(2)，线性动态系统模型分为预测(prediction)和更新(update)两步:

***step 1. 预测(prediction)***

P(Z<sub>t</sub>|X<sub>1</sub>, X<sub>2</sub>, ... ,X<sub>t-1</sub>) = ∫ P(Z<sub>t-1</sub>|X<sub>1</sub>, X<sub>2</sub>, ... ,X<sub>t-1</sub>)P(Z<sub>t</sub>|Z<sub>t-1</sub>)dZ<sub>t-1</sub> 

代入高斯分布得：

N(Z<sub>t</sub> | μ<sub>t|t-1</sub>, Σ<sub>t|t-1</sub>) = ∫N(Z<sub>t-1</sub> | AZ<sub>t-1</sub> + B, Q) * 
N(Z<sub>t</sub> | μ<sub>t-1|t-1</sub>, Σ<sub>t-1|t-1</sub>)dZ<sub>t-1</sub> 


***step 2. 更新(update)***

P(Z<sub>t</sub>|X<sub>1</sub>, X<sub>2</sub>, ... ,X<sub>t</sub>) ∝ P(Z<sub>t</sub>|X<sub>1</sub>, X<sub>2</sub>, ... ,X<sub>t-1</sub>)P(X<sub>t</sub>|Z<sub>t</sub>) 

代入高斯分布得：

N(Z<sub>t</sub> | μ<sub>t|t</sub>, Σ<sub>t|t</sub>) ∝ N(Z<sub>t</sub> | μ<sub>t|t-1</sub>, Σ<sub>t|t-1</sub>) * 
N(X<sub>t</sub> | CZ<sub>t</sub>, R) 

### 2.3 线性动态系统的闭式解

由于多元高斯的联合、边缘、条件分布都是高斯分布，因此参数的更新有闭式解，对应如下：

***step 1. 预测(prediction)***

![prediction step](../resources/linear_dynamic_system/kalman_filter_prediction_step.jpg)

由如上关系可得，预测步参数更新公式为:

(1) μ<sub>t|t-1</sub> = A * μ<sub>t-1|t-1</sub> + B

(2) Σ<sub>t|t-1</sub> = A * Σ<sub>t-1|t-1</sub> * A<sup>T</sup> + Q

***step 2. 更新(prediction)***

对于相同量纲的两个高斯分布的乘积

N(x|μ*, Σ*) = N(x|μ1, Σ1) * N(x|μ2, Σ2)

参数关系如下:

μ* =  (Σ2<sup>-1</sup> + Σ1<sup>-1</sup>)<sup>-1</sup>
</sup>(Σ1<sup>-1</sup> * μ1 + Σ2<sup>-1</sup> * μ2) = μ1 + Σ1(Σ1 + Σ2)<sup>-1</sup>(μ2 - μ1)

Σ* =  (Σ2<sup>-1</sup> + Σ1<sup>-1</sup>)<sup>-1</sup> = Σ1 - Σ1(Σ1 + Σ2)<sup>-1</sup>Σ1

更新步即两个高斯分布相，但要注意的是需要统一量纲：

N(X<sub>t</sub> | C * μ<sub>t|t</sub>, C * Σ<sub>t|t</sub> * C<sup>T</sup>) ∝ N(X<sub>t</sub> | C * μ<sub>t|t-1</sub>,  C * Σ<sub>t|t-1</sub>* C<sup>T</sup>) 
N(X<sub>t</sub> | CZ<sub>t</sub>, R) 

综上，更新步的参数更新规则为：


(1) 计算Kalman增益
 
K<sub>t</sub> = Σ<sub>t|t-1</sub> * C<sup>T</sup> * (R + C * Σ<sub>t|t-1</sub> * C<sup>T</sup>)<sup>-1</sup>

(2) 更新方差矩阵

Σ<sub>t|t</sub> = (Σ<sub>t|t-1</sub><sup>-1</sup> + C<sup>T</sup>R<sup>-1</sup>C)<sup>-1</sup> = Σ<sub>t|t-1</sub> - K<sub>t</sub> * C * Σ<sub>t|t-1</sub>

(3) 更新均值向量

μ<sub>t|t</sub> = μ<sub>t|t-1</sub> + K<sub>t</sub>(X<sub>t</sub> - C*μ<sub>t|t-1</sub>)
 

## 三、效果
参考资料[1]中给的估计小车一维运动速度和时间的例子：

![kalman filter](../results/linear_dynamic_system/kalman_filter.png)

## 四、总结
Kalman filter的初始参数Q,R,Z0,SIGMA0都会对最终滤波后的效果产生影响，其中Q,R的影响对滤波效果的影响较大，往往需要调参。如果状态转移矩阵的准确度较高，则设置较小的Q矩阵；如果观测矩阵的准确度较高，则设置较小的R矩阵。

## 五、参考资料
[1] Faragher R . Understanding the Basis of the Kalman Filter Via a Simple and Intuitive Derivation [Lecture Notes][J]. IEEE Signal Processing Magazine, 2012, 29(5):128-132.

[2] Machine Learning: A Probabilistic Perspective (第四章)


