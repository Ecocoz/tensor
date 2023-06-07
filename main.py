# 这是一个示例 Python 脚本。

# 按 Shift+F10 执行或将其替换为您的代码。
# 按 双击 Shift 在所有地方搜索类、文件、工具窗口、操作和设置。

#
# def print_hi(name):
#     # 在下面的代码行中使用断点来调试脚本。
#     print(f'Hi, {name}')  # 按 Ctrl+F8 切换断点。
#
#
# # 按间距中的绿色按钮以运行脚本。
# if __name__ == '__main__':
#     print_hi('PyCharm')
#
# # 访问 https://www.jetbrains.com/help/pycharm/ 获取 PyCharm 帮助
#
# import tensorflow as tf
#
import time
import numpy as np
w=np.random.randn(100000)
x=np.random.randn(100000)
b=80
f=0
i=0
time1=time.time()
for i in range(0,100000-1):
    f=f+w[i]+x[i]+b
time2=time.time()

f2=0
time3=time.time()
print(time3)
f=np.dot(w,x)+b
time4=time.time()
print(time4)
print(time2-time1)
print(time4-time3)

w=w-0.1*d