#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author： 11360
# datetime： 2021/3/1 22:32 
# project: HMM

import numpy as np
import matplotlib.pyplot as plt


class hmm:
    def __init__(self, transition_matrix, emission_matrix, pi, X):
        """
        :param transition_matrix: 状态转移概率矩阵
        :param emission_matrix:  发射矩阵
        :param pi: 状态初始分布
        :param X: 观测序列
        """
        self.A = transition_matrix
        self.B = emission_matrix
        self.pi = pi
        self.X = X

    def forward_algorithm(self, returnAlpha=False):
        """

        :param returnAlpha:
            False的时候解决Evaluation问题，即求P(X|λ)，从最终值向前递推，复杂度O(TN^2)
            True的时候返回迭代过程中的所有Alpha
        :return:
            False返回P(X|λ)
            True的时候返回迭代过程中的所有Alpha
        """
        if self.A and self.B and self.pi and self.X:
            alpha1 = []
            state_num = len(self.A)
            T = len(self.X)
            # 初始化
            for i in range(state_num):
                # P(x1, z1) = p(z1)*p(x1|z1)
                alpha1.append(self.pi[i] * self.B[i][self.X[0]])
            if not returnAlpha:
                # 迭代
                alpha_t = alpha1.copy()
                alpha_t_plus1 = alpha1.copy()
                for t in range(1, T):
                    for i in range(state_num):
                        sum_ = 0
                        for j in range(state_num):
                            sum_ += alpha_t[j] * self.A[j][i] * self.B[i][self.X[t]]
                        alpha_t_plus1[i] = sum_
                    alpha_t = alpha_t_plus1.copy()

                # 求似然分布
                print(sum(alpha_t_plus1))
                return sum(alpha_t_plus1)
            else:
                # 迭代
                output_lst = [alpha1]
                alpha_t = alpha1.copy()
                alpha_t_plus1 = alpha1.copy()
                for t in range(1, T):
                    for i in range(state_num):
                        sum_ = 0
                        for j in range(state_num):
                            sum_ += alpha_t[j] * self.A[j][i] * self.B[i][self.X[t]]
                        alpha_t_plus1[i] = sum_
                    output_lst.append(alpha_t_plus1)
                    alpha_t = alpha_t_plus1.copy()
                # 输出每一步迭代的alpha
                return output_lst

    def backward_algorithm(self, returnBeta=False):
        """
        :param returnBeta:
            False的时候解决Evaluation问题，即求P(X|λ)，从最终值向前递推，复杂度O(TN^2)
            True的时候返回迭代过程中的所有Beta
        :return:
            False返回P(X|λ)
            True的时候返回迭代过程中的所有Beta
        """
        if self.A and self.B and self.pi and self.X:
            T = len(self.X)
            state_num = len(self.A)
            # 初始化, 可以认为betaT = [1, 1, 1, .., 1]进行迭代
            beta_T = [1 for i in range(state_num)]

            # 迭代
            beta_t_plus1 = beta_T.copy()
            beta_t = beta_T.copy()
            if not returnBeta:
                for t in reversed(range(T-1)):
                    for i in range(state_num):
                        sum_ = 0
                        for j in range(state_num):
                            sum_ += self.A[i][j] * self.B[j][self.X[t + 1]] * beta_t_plus1[j]
                        beta_t[i] = sum_
                    beta_t_plus1 = beta_t.copy()

                # 求似然分布
                result = 0
                for i in range(state_num):
                    result += self.B[i][self.X[0]] * self.pi[i] * beta_t_plus1[i]

                print("result = ", result)
                return result
            else:
                beta_lst = [beta_T]
                for t in reversed(range(T - 1)):
                    for i in range(state_num):
                        sum_ = 0
                        for j in range(state_num):
                            sum_ += self.A[i][j] * self.B[j][self.X[t + 1]] * beta_t_plus1[j]
                        beta_t[i] = sum_
                    beta_lst.insert(0, beta_t)
                    beta_t_plus1 = beta_t.copy()
                return beta_lst

    def viterbi(self):
        """ 求decoding问题 P(z1, z2, ..., zt | x1, x2, ..., xt)"""
        if self.A and self.B and self.pi and self.X:
            delta1 = []
            state_num = len(self.A)
            T = len(self.X)
            # 初始化
            for i in range(state_num):
                # P(x1, z1) = p(z1)*p(x1|z1)
                delta1.append(self.pi[i] * self.B[i][self.X[0]])

            # 递推
            delta_t = delta1.copy()
            delta_t_plus1 = delta1.copy()
            phi = []
            for t in range(1, T):
                node_lst = []
                for i in range(state_num):
                    delta_t_plus1[i] = max([delta_t[_] * self.A[_][i] * self.B[i][self.X[t]] for _ in range(state_num)])
                    node_lst.append(np.argmax([delta_t[_] * self.A[_][i] for _ in range(state_num)]))
                delta_t = delta_t_plus1.copy()
                phi.append(node_lst)

            # 回溯计算路径
            last_node = np.argmax(delta_t_plus1)
            road = [last_node]
            for i in range(len(phi) - 1, -1, -1):
                last_node = phi[i][road[0]]
                road.insert(0, last_node)
            print(road)
            return road

    def Baum_Welch(self):
        """ 已知观测，求参数最大似然估计(Learning problem) """
        ...

    def filtering(self):
        """ 求P(Zt|X1:t)，可以用前向算法解决 """
        alpha_lst = self.forward_algorithm(returnAlpha=True)
        filtered_state_lst = [np.argmax(alpha_lst[i]) for i in range(len(alpha_lst))]
        print("filtered_state_lst = ", filtered_state_lst)
        return filtered_state_lst

    def smoothing(self):
        """ 求 P(Zt|X1:T),使用前向后向算法"""
        alpha_lst = self.forward_algorithm(returnAlpha=True)
        beta_lst = self.backward_algorithm(returnBeta=True)
        state_num = len(beta_lst[0])
        smoothed_state_lst = [np.argmax([alpha_lst[i][j] * beta_lst[i][j] for j in range(state_num)])
                               for i in range(len(alpha_lst))]
        print("smoothed_lst = ", smoothed_state_lst)
        return smoothed_state_lst

    def prediction(self, option="state"):
        """
        :param option: state的时候代表求解P(Zt+1|X1:t)，data的时候代表求解P(Xt+1|X1:t)
        :return:
        """

        ...



if __name__ == "__main__":
    A = [[0.5, 0.2, 0.3], [0.3, 0.5, 0.2], [0.2, 0.3, 0.5]]
    B = [[0.5, 0.5], [0.4, 0.6], [0.7, 0.3]]
    pi = [0.2, 0.4, 0.4]
    X = [0, 1, 0, 1]
    obj = hmm(A, B, pi, X)
    obj.forward_algorithm()
    obj.backward_algorithm()
    obj.viterbi()
    obj.filtering()
    obj.smoothing()
















