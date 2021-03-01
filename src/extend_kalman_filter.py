#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author： 11360
# datetime： 2021/3/1 11:20 

import numpy as np
import matplotlib.pyplot as plt


class ExtendKalmanfilter:
    def __init__(self, transition_operator, observe_operator, transis_gradient, observe_gradient, Q, R):
        self.transition_operator = transition_operator
        self.transition_noise_variance = Q
        self.observe_noise_variance = R
        self.observe_operator = observe_operator
        self.observe_g = observe_gradient
        self.transis_g = transis_gradient

    def make_data(self):
        """  """
        self.real_state = []
        z = np.random.normal(0, 1, 1)
        for i in range(100):
            self.real_state.append(z)
            z = self.transition_operator(z) + np.random.normal(0, self.transition_noise_variance)

        self.data = [self.observe_operator(self.real_state[i]) + np.random.normal(0, self.observe_noise_variance)
                             for i in range(len(self.real_state))]

    def filter(self, z0, sigma0):
        """ z0: 估计系统初始状态， sigma0：估计初始协方差矩阵"""
        # z_old = z0
        z_old = self.real_state[0]
        sigma_old = sigma0
        self.filtered_state_lst = []
        for i in range(len(self.data)):
            z_new, sigma_new = self.prediction_step(z_old, sigma_old)
            z_old, sigma_old = self.update_step(z_new, sigma_new, self.data[i])
            self.filtered_state_lst.append(z_old)

    def prediction_step(self, z_old, sigma_old):
        transis_matrix = self.transis_g(z_old)
        z_new = transis_matrix * z_old
        sigma_new = transis_matrix * sigma_old * transis_matrix + self.transition_noise_variance
        return z_new, sigma_new

    def update_step(self, z_new, sigma_new, observe_data):
        observe_matrix = self.observe_g(z_new)
        kalman_gain = sigma_new * observe_matrix / (observe_matrix * sigma_new * observe_matrix + self.observe_noise_variance)
        z_ = z_new + kalman_gain * (observe_data - observe_matrix * z_new)
        sigma_ = sigma_new - kalman_gain * observe_matrix * sigma_new
        return z_, sigma_

    def plot(self):
        plt.figure()
        num = len(self.data)
        x_ = [i for i in range(num)]
        plt.scatter(x_, self.real_state, c="r", s=5)
        plt.scatter(x_, self.filtered_state_lst, c="g", s=5)
        # plt.scatter(x_, self.filtered_state_lst, c="b", s=5)
        p1, = plt.plot(x_, self.real_state, c="r", )
        p2, = plt.plot(x_, self.filtered_state_lst, c="g")
        # p3, = plt.plot([i for i in range(num)], [self.filtered_state_lst[i][1, 0] for i in range(num)], c="b")
        plt.legend([p1, p2], ["Real state", "Estimated state"])
        plt.show()


if __name__ == "__main__":
    # Q_mat = np.mat([0.3]) # 状态转移噪声协方差矩阵，这里设置较小的值，因为觉得状态转移矩阵准确度高
    # R_mat = np.mat([1])  # 观测噪声协方差矩阵
    Q_mat = 0.3
    R_mat = 1
    dt = 0.1
    transis_operator = lambda x: x + x * np.cos(x) * dt
    observe_operator = lambda x: x ** 3 * dt
    transis_gradient = lambda x: 1 + (np.cos(x) - x * np.sin(x)) * dt
    observe_gradient = lambda x: 3 * x ** 2 * dt
    obj = ExtendKalmanfilter(transis_operator, observe_operator, transis_gradient,
                             observe_gradient, Q_mat, R_mat)
    obj.make_data()
    z0 = 0.1            # 系统初始估计状态
    sigma0 = 1  # 初始协方差矩阵
    obj.filter(z0, sigma0)
    obj.plot()
















