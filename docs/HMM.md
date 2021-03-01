# Hidden Markov Model(HMM)
## 一、介绍
[README.md](../README.md)中介绍了HMM与其它两个模型之间的区别与联系。HMM与其它两个模型最大的区别在于其状态空间是离散的，因此HMM的状态转移概率可以用
一个矩阵A表示，其中a<sub>ij</sub> = P(Zt=q<sub>j</sub> | Zt=q<sub>i</sub>)，该矩阵称为***状态转移矩阵***(Transition Matrix)，满足每一行的和为1。HMM对观测空间
的取值离散或连续没有规定，这里假设它也是离散的，可以用矩阵B表示观测概率，b<sub>ij</sub> = P(Xt=v<sub>j</sub>|Zt=q<sub>i</sub>)，称为***发射矩阵***(Emission Matrix)，
发射矩阵同样满足每行和为1。再加上Z<sub>1</sub>系统状态的初始分布π，称λ = (π, A, B)为HMM模型的参数。

## 二、HMM关注的问题

HMM关注三个问题：

##### 1. Evaluation 

给定模型参数λ的情况下，求似然函数P(X|λ)，相应的算法为前向算法(Forward Algorithm)和后向算法(Backward Algorithm)。

---
##### 2. Learning

λ<sub>MLE</sub> = argmax P(X|λ)

由于模型中含有隐变量，故用EM算法求解。

---
##### 3. Decoding

***<u>Z</u>*** = argmax P(Z<sub>1</sub>, Z<sub>2</sub>, ... ,Z<sub>t</sub>|X<sub>1</sub>, X<sub>2</sub>, ... ,X<sub>t</sub>)

---



