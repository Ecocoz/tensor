import math, copy
from lab_utils_uni import plt_intuition, plt_stationary, plt_update_onclick, soup_bowl
import numpy as np
import matplotlib.pyplot as plt

# Load our data set
x_train = np.array([1.0, 2.0])  # features
y_train = np.array([300.0, 500.0])  # target value


# 计算成本函数
def compute_cost(x, y, w, b):
    m = x.shape[0]
    total_cost = 0
    cost_sum = 0
    for i in range(m):
        f_wb = w * x[i] - y[i]
        cost = (f_wb - y[i]) ** 2
        cost_sum += cost
    total_cost = 1 / (2 * m) * cost_sum
    return total_cost


# 计算梯度

def compute_gradient(x, y, w, b):
    m = x.shape[0]
    dj_dw = 0
    dj_db = 0

    for i in range(m):
        f_wb = w * x[i] + b
        dj_dw_i = (f_wb - y[i]) * x[i]
        dj_db_i = f_wb - y[i]
        dj_dw = dj_dw + dj_dw_i
        dj_db = dj_db + dj_db_i

        dj_dw = dj_dw / m
        dj_db = dj_db / m

    return dj_dw, dj_db


# 梯度下降
def gradient_descent(x, y, w_in, b_in, alpha, num_iters, cost_function, gradient_function):
    w = copy.deepcopy(w_in)

    j_history = []  # cost values 的记录
    p_history = []  # 参数 w,b 的记录
    b = b_in
    w = w_in

    for i in range(num_iters):
        dj_dw, dj_db = gradient_function(x, y, w, b)

        b = b - alpha * dj_db
        w = w - alpha * dj_dw

        if i < 10000:
            j_history.append(cost_function(x, y, w, b))
            p_history.append([w, b])

            # Print cost every at intervals 10 times or as many iterations if < 10
        if i % math.ceil(num_iters / 10) == 0:
            print(f"Iteration {i:4}: Cost {j_history[-1]:0.2e} ",
                  f"dj_dw: {dj_dw: 0.3e}, dj_db: {dj_db: 0.3e}  ",
                  f"w: {w: 0.3e}, b:{b: 0.5e}")

    return w, b, j_history, p_history


w_init = 0
b_init = 0
iterations = 10000
tmp_alpha = 1.0e-2
# 调用运行
w_final, b_final, J_history, P_history = gradient_descent(x_train, y_train, w_init, b_init, tmp_alpha, iterations,
                                                          compute_cost, compute_gradient)

print(f"(w,b) found by gradient descent: ({w_final:8.4f},{b_final:8.4f})")
