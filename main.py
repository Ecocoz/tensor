# # 这是一个示例 Python 脚本。
#
# # 按 Shift+F10 执行或将其替换为您的代码。
# # 按 双击 Shift 在所有地方搜索类、文件、工具窗口、操作和设置。
#
# #
# # def print_hi(name):
# #     # 在下面的代码行中使用断点来调试脚本。
# #     print(f'Hi, {name}')  # 按 Ctrl+F8 切换断点。
# #
# #
# # # 按间距中的绿色按钮以运行脚本。
# # if __name__ == '__main__':
# #     print_hi('PyCharm')
# #
# # # 访问 https://www.jetbrains.com/help/pycharm/ 获取 PyCharm 帮助
# #
# # import tensorflow as tf
# #
# import time
# import numpy as np
# w=np.random.randn(100000)
# x=np.random.randn(100000)
# b=80
# f=0
# i=0
# time1=time.time()
# for i in range(0,100000-1):
#     f=f+w[i]+x[i]+b
# time2=time.time()
#
# f2=0
# time3=time.time()
# print(time3)
# f=np.dot(w,x)+b
# time4=time.time()
# print(time4)
# print(time2-time1)
# print(time4-time3)
#
# w=w-0.1*d


# 导入pandas库和matplotlib库
# import pandas as pd
# import matplotlib.pyplot as plt
# import matplotlib.font_manager as font_manager
#
# import matplotlib.pyplot as plt
#
# plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
# plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
#
# # 读取表格数据
# knn_df = pd.read_csv('KNN.csv')
# dt_df = pd.read_csv('dt.csv')
#
# # 设置图表的大小和风格
# plt.figure(figsize=(10, 6))
# plt.style.use('ggplot')
#
# # 绘制KNN算法的折线图
# # plt.subplot(2, 1, 1)
# # plt.plot(knn_df['k值'], knn_df['准确率'], label='准确率')
# # plt.plot(knn_df['k值'], knn_df['精确率'], label='精确率')
# # plt.plot(knn_df['k值'], knn_df['召回率'], label='召回率')
# # plt.plot(knn_df['k值'], knn_df['F1值'], label='F1值')
# # plt.title('KNN算法在不同参数设置下的性能比较')
# # plt.xlabel('k值')
# # plt.ylabel('评估指标')
# # plt.legend()
#
# # # 绘制决策树算法的折线图
# plt.subplot(2, 1, 2)
# plt.plot(dt_df['最大深度'], dt_df['准确率'], label='准确率')
# plt.plot(dt_df['最大深度'], dt_df['精确率'], label='精确率')
# plt.plot(dt_df['最大深度'], dt_df['召回率'], label='召回率')
# plt.plot(dt_df['最大深度'], dt_df['F1值'], label='F1值')
# plt.title('决策树算法在不同参数设置下的性能比较')
# plt.xlabel('最大深度')
# plt.ylabel('评估指标')
# plt.legend()
#
# # 显示图表
# plt.show()


# import matplotlib.pyplot as plt
#
# plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
# plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
#
# data = {'TLS客户端指纹信息': {'客户端支持的加密套件数组': 0.87,
#                             '服务器端选择的加密套件': 0.85,
#                             '支持的扩展': 0.83,
#                             '客户端公钥长度': 0.81,
#                             'Client version': 0.79,
#                             '是否非CA自签名': 0.77},
#         '数据包元数据': {'数据包的大小': 0.32,
#                       '到达时间序列': 0.31,
#                       '字节分布': 0.29},
#         'HTTP头部信息': {'Content-Type': 0.28,
#                       'User-Agent': 0.27,
#                       'Accept-Language': 0.26,
#                       'Server': 0.25,
#                       'HTTP响应码': 0.24},
#         'DNS响应信息': {'域名的长度': 0.76,
#                       '数字字符及非字母数字字符的占比': 0.74,
#                       'DNS解析出的IP数量': 0.72,
#                       'TTL值': 0.71,
#                       '域名是否收录在Alexa网站中': 0.69}}
#
# fig, ax = plt.subplots(figsize=(8, 6))
# colors = plt.cm.Dark2(range(6))
#
# for i, (feature, scores) in enumerate(data.items()):
#     scores = sorted(scores.items(), key=lambda x: x[1])
#     features = [x[0] for x in scores]
#     scores = [x[1] for x in scores]
#     ax.barh(feature, scores, color=colors[i % len(colors)])
#     for j, score in enumerate(scores):
#         ax.text(score+0.01, j, f'{score:.2f}')
#     ax.set_xlim([0, 1])
#     ax.set_xlabel('Chi-Square Score')
#
# plt.tight_layout()
# plt.show()
# import matplotlib.pyplot as plt
#

#
# data = {'TLS客户端指纹信息': {'客户端支持的加密套件数组': 0.87,
#                             '服务器端选择的加密套件': 0.85,
#                             '支持的扩展': 0.83,
#                             '客户端公钥长度': 0.81,
#                             'Client version': 0.79,
#                             '是否非CA自签名': 0.77},
#         '数据包元数据': {'数据包的大小': 0.32,
#                       '到达时间序列': 0.31,
#                       '字节分布': 0.29},
#         'HTTP头部信息': {'Content-Type': 0.28,
#                       'User-Agent': 0.27,
#                       'Accept-Language': 0.26,
#                       'Server': 0.25,
#                       'HTTP响应码': 0.24},
#         'DNS响应信息': {'域名的长度': 0.76,
#                       '数字字符及非字母数字字符的占比': 0.74,
#                       'DNS解析出的IP数量': 0.72,
#                       'TTL值': 0.71,
#                       '域名是否收录在Alexa网站中': 0.69}}
#
# fig, ax = plt.subplots(figsize=(8, 6))
# colors = plt.cm.Dark2(range(6))
#
# for i, (feature, scores) in enumerate(data.items()):
#     scores = sorted(scores.items(), key=lambda x: x[1])
#     features = [x[0] for x in scores]
#     scores = [x[1] for x in scores]
#     ax.barh(feature, scores, color=colors[i % len(colors)], height=0.5)
#     for j, score in enumerate(scores):
#         ax.text(score+0.01, j-0.1, f'{score:.2f}', fontsize=8)
#     ax.set_xlim([0, 1])
#     ax.set_xlabel('Chi-Square Score')
#
# plt.tight_layout()
# plt.show()




# 导入pandas库和matplotlib库
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 读取数据
df = pd.read_csv('label.csv')

# 设置图表的大小和风格
plt.figure(figsize=(10, 6))
plt.style.use('ggplot')

# 绘制条形图
plt.barh(df['特征名称'], df['卡方检验得分'], color='orange')
plt.title('特征选择过程中的卡方检验得分')
plt.xlabel('卡方检验得分')
plt.ylabel('特征名称')

# 显示图表
plt.show()

