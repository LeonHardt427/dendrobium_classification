# -*- coding: utf-8 -*-
# @Time    : 2017/12/9 10:33
# @Author  : LeonHardt
# @File    : accuracy.py

import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':

    data = np.loadtxt('result_lda_knn_3_100')
    X = data[:, 0]
    y = data[:, 1]

    # label在图示(legend)中显示。若为数学公式，则最好在字符串前后添加"$"符号
    # color：b:blue、g:green、r:red、c:cyan、m:magenta、y:yellow、k:black、w:white、、、
    # 线型：-  --   -.  :    ,
    # marker：.  ,   o   v    <    *    +    1
    plt.figure(figsize=(10, 5))
    plt.grid(linestyle="--")  # 设置背景网格线为虚线
    ax = plt.gca()
    ax.spines['top'].set_visible(False)  # 去掉上边框
    ax.spines['right'].set_visible(False)  # 去掉右边框

    plt.plot(X, y, color="red", label="A algorithm", linewidth=1.5)
    # plt.plot(x, B, "k--", label="B algorithm", linewidth=1.5)
    # plt.plot(x, C, color="red", label="C algorithm", linewidth=1.5)
    # plt.plot(x, D, "r--", label="D algorithm", linewidth=1.5)

    # group_labels = ['dataset1', 'dataset2', 'dataset3', 'dataset4', 'dataset5', ' dataset6', 'dataset7', 'dataset8',
    #                 'dataset9', 'dataset10']  # x轴刻度的标识
    plt.xticks(range(0, 100, 5), fontsize=12, fontweight='bold')  # 默认字体大小为10
    plt.yticks(fontsize=12, fontweight='bold')
    plt.title('LDA-KNN', fontsize=12, fontweight='bold')  # 默认字体大小为12
    plt.xlabel("n_estimators", fontsize=13, fontweight='bold')
    plt.ylabel("Accuracy", fontsize=13, fontweight='bold')
    plt.xlim(0, 100)  # 设置x轴的范围
    # plt.ylim(0.5,1)

    # plt.legend()          #显示各曲线的图例
    plt.legend(loc=0, numpoints=1)
    leg = plt.gca().get_legend()
    ltext = leg.get_texts()
    plt.setp(ltext, fontsize=12, fontweight='bold')  # 设置图例字体的大小和粗细

    # plt.savefig('D:\\filename.svg', format='svg')  # 建议保存为svg格式，再用inkscape转为矢量图emf后插入word中
    plt.show()



