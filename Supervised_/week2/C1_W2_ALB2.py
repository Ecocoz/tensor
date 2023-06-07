import numpy as np
import copy, math
import matplotlib.pyplot as plt

# 准备数据
X_train = np.array([[2104, 5, 1, 45], [1416, 3, 2, 40], [852, 2, 1, 35]])
y_train = np.array([460, 232, 178])
print(X_train[1])
print(y_train[0])

# 数据存放在numpy数组中
print(f"X Shape: {X_train.shape}, X Type:{type(X_train)})")
print(X_train)
print(f"y Shape: {y_train.shape}, y Type:{type(y_train)})")
print(y_train)

# 为了演示，w和b将选取较优值
b_init = 785.1811367994083
w_init = np.array([0.39133535, 18.75376741, -53.36032453, -26.42131618])
print(f"w_init shape: {w_init.shape}, b_init type: {type(b_init)}")
print("********************")
print(w_init[0])


# 3.1逐个元素的单一预测

def predict_single_loop(x, w, b):
    price = 0
    for i in range(x.shape[0]):
        p_i = x[i] * w[i]
        price = price + p_i
    price += b

    return price


# 从训练数据中获取一行,并预测
x_vec = X_train[0]
print(predict_single_loop(x_vec, w_init, b_init))


# 3.2使用向量的方法进行预测
def predict(x, w, b):
    price = 0
    price = np.dot(x, w)
    price = price + b
    return price


# 同样做一次预测
x_vec = X_train[0]
print(predict(x_vec, w_init, b_init))

X_train = np.array([[2104, 5, 1, 45], [1416, 3, 2, 40], [852, 2, 1, 35]])
y_train = np.array([460, 232, 178])
b_init = 785.1811367994083
w_init = np.array([0.39133535, 18.75376741, -53.36032453, -26.42131618])


# 4多变量的计算成本
def compute_cost(X, y, w, b):
    j_wb = 0
    sum_j = 0
    m = X.shape[0]
    for i in range(X.shape[0]):
        f_wb_i = sum(X[i] * w) + b
        j_wb = j_wb + (f_wb_i - y[i]) ** 2
    j_wb = 1 / 2 * m * j_wb
    return j_wb


def compute_cost2(X, y, w, b):
    m = X.shape[0]
    cost2 = 0.0
    for i in range(m):
        f_wb_i = np.dot(X[i], w) + b  # (n,)(n,) = scalar (see np.dot)
        cost2 = cost2 + (f_wb_i - y[i]) ** 2  # scalar
    cost2 = cost2 / (2 * m)  # scalar
    return cost


# 使用我们预先选择的最优参数计算和显示成本。

cost = compute_cost(X_train, y_train, w_init, b_init)
cost2 = compute_cost2(X_train, y_train, w_init, b_init)
print(f'Cost at optimal w : {cost}')
print(f'Cost at optimal w2 : {cost2}')


# 计算梯度
def compute_gradient(X, y, w, b):
    m, n = X.shape
    err = 0.0
    dj_db = 0
    dj_dw = np.zeros((n,))
    for i in range(m):
        err = np.dot(X[i], w) + b - y[i]
        for j in range(n):
            dj_dw[j] = dj_dw[j] + err * X[i, j]
        dj_db = dj_db + err

    dj_dw = dj_dw / m
    dj_db = dj_db / m

    return dj_db, dj_dw


# Compute and display gradient
tmp_dj_db, tmp_dj_dw = compute_gradient(X_train, y_train, w_init, b_init)
print(f'dj_db at initial w,b: {tmp_dj_db}')
print(f'dj_dw at initial w,b: \n {tmp_dj_dw}')


def gradient_descent(X, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters):
    J_history = []
    w = copy.deepcopy(w_in)
    b = b_in

    for i in range(num_iters):
        dj_db, dj_dw = gradient_function(X, y, w, b)

        w = w - alpha * dj_dw
        b = b - alpha * dj_db

    if i < 200000:
        J_history.append(cost_function(X, y, w, b))
    if i % math.ceil(num_iters / 10) == 0:
        print(f"Iteration {i:4d}: Cost {J_history[-1]:8.2f}   ")

    return w, b, J_history


print("**************************************************")
initial_w = np.zeros_like(w_init)
initial_b = 0.0
iterations = 1000
alpha = 0.0000005

w_final, b_final, J_list = gradient_descent(X_train, y_train, initial_w, initial_b,
                                            compute_cost, compute_gradient,
                                            alpha, iterations)

print(f"b,w found by gradient descent: {b_final:0.2f},{w_final} ")

m, _ = X_train.shape
for i in range(m):
    print(f"prediction: {np.dot(X_train[i], w_final) + b_final:0.2f}, target value: {y_train[i]}")

b_init = 0.84
w_init = np.array([0.18, 16.3, -45.63, 0.77])
cost = compute_cost(X_train, y_train, w_init, b_init)
print(cost)
