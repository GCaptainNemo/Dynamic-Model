#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author： 11360
# datetime： 2021/2/28 23:43
# project: Particle Filter

import numpy as np
import matplotlib.pyplot as plt

dt = 0.4
# 认为噪声仍服从高斯分布
variance_Q = 0.1 * dt  # 状态转移噪声协方差
variance_R = 1 * dt  # 测量协方差


class Particle_filter:
    def __init__(self, particle_num):
        self.particle_num = particle_num

    def estimate(self, particles, weights):
        """ 每一步对系统状态进行估计，求z关于后验概率的期望，对应即求Particles与权重的加权平均 """
        mean = np.average(particles, weights=weights)
        var = np.average((particles - mean) ** 2, weights=weights)
        return mean, var

    def simple_resample(self, particles, weights):
        """ 通过累计分布函数进行重采样 """
        N = len(particles)
        cumulative_sum = np.cumsum(weights)
        cumulative_sum[-1] = 1.  # 避免误差
        rn = np.random.rand(N)  # 0-1之间的均匀分布采样
        indexes = np.searchsorted(cumulative_sum, rn)   # 在数组A中插入数组B，返回索引
        # 根据索引采样
        particles[:] = particles[indexes]
        weights.fill(1.0 / N)
        return particles, weights

    def make_data(self):
        """  构造观测值 """
        z = np.random.normal(0, 1)  # 初始真实状态
        self.hidden_data = [z]
        self.observe_data = [dt * z ** 3 + np.random.normal(0, variance_R)]

        for i in range(99):
            # 100个数据点
            z = z + z * dt * np.cos(z) + np.random.normal(0, variance_Q)
            x = dt * z ** 3 + np.random.normal(0, variance_R)
            self.hidden_data.append(z)
            self.observe_data.append(x)

    def filter(self, resampling=True):
        z_estimate = 0  # 初始状态估计值
        self.z_estimate_lst = [z_estimate]
        V = 1    # 初始协方差
        z_particles = z_estimate + np.random.normal(0, V, self.particle_num)
        Particles_weights = np.array([1 / self.particle_num for _ in range(self.particle_num)])
        self.z_particles_lst = [z_particles]
        for i in range(1, len(self.observe_data)):
            # 从p(zt|zt-1)中采样
            z_particles_sampling = self.sampling(z_particles)
            x_particles_sampling = dt * z_particles_sampling ** 3
            # 计算权重
            Particles_weights = self.cal_weights(self.observe_data[i], x_particles_sampling, Particles_weights)

            # 估计
            z_est, z_var_ssd = self.estimate(z_particles_sampling, Particles_weights)
            self.z_estimate_lst.append(z_est)

            # 重采样
            if resampling:
                z_particles, Particles_weights = self.simple_resample(z_particles_sampling, Particles_weights)
                self.z_particles_lst.append(z_particles)
            else:
                z_particles = z_particles_sampling
                self.z_particles_lst.append(z_particles)

    def sampling(self, z_particles):
        """ 从p(zt|zt-1)中采样 """
        z_particles_sampling = z_particles + dt * z_particles * np.cos(z_particles) + \
                               np.random.normal(0, variance_Q, self.particle_num)
        return z_particles_sampling

    def cal_weights(self, observed_data, x_particles_sampling, old_par_weights):
        """ 计算p(xt|zt)w(t-1), 由于每次都进行重采样，w(t-1)是常数 """
        variance_R_guji = variance_R + 2
        Particles_weights = (1 / np.sqrt(2 * np.pi * variance_R_guji)) * np.exp(-(observed_data - x_particles_sampling) ** 2 /
                                                                           (2 * variance_R_guji))
        Particles_weights = Particles_weights * old_par_weights
        Particles_weights /= np.sum(Particles_weights)
        return Particles_weights

    def plot(self):
        plt.figure()
        num = len(self.observe_data)
        x = [i for i in range(num)]
        plt.scatter(x, self.observe_data, c="r", s=5)
        plt.scatter(x, self.z_estimate_lst, c="g", s=5)
        p1, = plt.plot(x, self.observe_data, c="r", )
        for i in range(self.particle_num):
            p3, = plt.plot(x, [self.z_particles_lst[j][i] for j in range(num)], color='gray')
        p2, = plt.plot(x, self.z_estimate_lst, c="g")

        plt.legend([p1, p2, p3], ["Observed data", "Estimated state", "Particle trajectory"])
        plt.show()


if __name__ == "__main__":
    obj = Particle_filter(100)
    obj.make_data()
    obj.filter(resampling=False)
    obj.plot()
