import pretreatment as pre
import matplotlib.pyplot as plt
import numpy as np
import argparse
def plt_plot(list):
    # 坐标设为排名和频率
    ranks = np.arange(1, len(list) + 1)
    frequencies = np.array([freq for _, freq in list])
    # 绘制双对数坐标系上的排名和频率
    plt.figure(figsize=(20, 10))
    plt.loglog(ranks, frequencies, marker='o', linestyle='-')

    plt.xlabel('rank')
    plt.ylabel('frequency')
    plt.title('rank-frequency')

    plt.grid(True, which="both", ls="--")
    plt.show()
if __name__ == '__main__':
    stopwords = pre.load_list('.\DLNLP2023-main\cn_stopwords.txt')
    #添加模式，1为按字检索，2为按词检索 这部分代码用来传参数用的
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-mode', type=int, default=2, help='using character(1) or words(2)')
    args = parser.parse_args()
    print(f'mode: {args.mode}')
    #预处理
    list =pre.pre('.\\cn_words',stopwords,max_length=args.mode)
    #画图
    plt_plot(list)