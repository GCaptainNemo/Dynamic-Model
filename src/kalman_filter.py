import numpy as np
import matplotlib.pyplot as plt

class Kalman_filter:
    def __init__(self, transition_matrix, input_vector, observe_matrix, Q, R):
        self.transition_matrix = transition_matrix
        self.input_vector = input_vector
        self.transition_noise_variance = Q
        self.observe_noise_variance = R
        self.observe_matirx = observe_matrix

    def make_data(self):
        """ 预测小车一维运动的位置和速度，状态变量有位置、速度；观测变量为位置 """
        x = np.mat([i for i in range(30)])
        noise = np.mat(np.round(np.random.normal(0, 1, 30), 2))
        self.data = x + noise   # 观测值

    def filter(self, z0, sigma0):
        """ z0: 系统初始状态， sigma0：初始协方差矩阵"""
        z_old = z0
        sigma_old = sigma0
        self.filtered_state_lst = []
        for i in range(self.data.shape[1]):
            z_new, sigma_new = self.prediction_step(z_old, sigma_old)
            z_old, sigma_old = self.update_step(z_new, sigma_new, self.data[0, i])
            self.filtered_state_lst.append(z_old)

    def prediction_step(self, z_old, sigma_old):
        z_new = self.transition_matrix * z_old + self.input_vector
        sigma_new = self.transition_matrix * sigma_old * self.transition_matrix.T + self.transition_noise_variance
        return z_new, sigma_new

    def update_step(self, z_new, sigma_new, observe_data):
        kalman_gain = sigma_new * self.observe_matirx.T / (self.observe_matirx * sigma_new * self.observe_matirx.T + self.observe_noise_variance)
        z_ = z_new + kalman_gain * (observe_data - self.observe_matirx * z_new)
        sigma_ = sigma_new - kalman_gain * self.observe_matirx * sigma_new
        return z_, sigma_

    def plot(self):
        plt.figure()
        num = self.data.shape[1]
        plt.scatter([i for i in range(num)], [self.data[0, i] for i in range(num)], c="r", s=5)
        plt.scatter([i for i in range(num)], [self.filtered_state_lst[i][0, 0] for i in range(num)], c="g", s=5)
        plt.scatter([i for i in range(num)], [self.filtered_state_lst[i][1, 0] for i in range(num)], c="b", s=5)
        p1, = plt.plot([i for i in range(num)], [self.data[0, i] for i in range(num)], c="r", )
        p2, = plt.plot([i for i in range(num)], [self.filtered_state_lst[i][0, 0] for i in range(num)], c="g")
        p3, = plt.plot([i for i in range(num)], [self.filtered_state_lst[i][1, 0] for i in range(num)], c="b")
        plt.legend([p1, p2, p3], ["Observed data(position)", "Estimated position", "Estimated velocity"])
        plt.show()


if __name__ == "__main__":
    transition_matrix = np.mat([[1, 1], [0, 1]])  # 设采样间隔为1s
    input_matrix = np.mat([[0], [0]])
    Q_mat = np.mat([[0.001, 0], [0, 0.001]]) # 状态转移噪声协方差矩阵，这里设置较小的值，因为觉得状态转移矩阵准确度高
    observe_matrix = np.mat([1, 0])
    R_mat = np.mat([1])  # 观测噪声协方差矩阵

    obj = Kalman_filter(transition_matrix, input_matrix, observe_matrix, Q_mat, R_mat)
    obj.make_data()
    z0 = np.mat([[5], [6]])  # 系统初始状态(position = 5, velocity = 6)
    sigma0 = np.mat([[1, 2], [3, 1]])  # 初始协方差矩阵
    obj.filter(z0, sigma0)
    obj.plot()











