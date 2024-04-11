import pretreatment as pre
import matplotlib.pyplot as plt
import math
from collections import Counter
import argparse
def split_wordlist_to_seq(input_list, end_word='<end>'):
    # 初始化结果列表和临时子列表
    result = []
    seq_list = []

    for word in input_list:
        if word == end_word:
            # 遇到分割单词，将句子列表添加到结果列表中，并清空子列表
            if seq_list:
                result.append(seq_list)
            seq_list = []
        else:
            # 正常单词，将其添加到当前句子列表中
            seq_list.append(word)

    # 处理最后一个子列表（最后没标点情况）
    if seq_list:
        result.append(seq_list)
    return result
def get_entropy(seq_list):
    entropy = 0
    seq_list = split_wordlist_to_seq(seq_list)
    # 二元组和三元组
    bi_gram_counts = Counter()
    tri_gram_counts = Counter()
    for seq in seq_list:
        # 生成二元组和三元组
        bi_grams = [(seq[i], seq[i + 1]) for i in range(len(seq) - 1)]
        tri_grams = [(seq[i], seq[i + 1], seq[i + 2]) for i in range(len(seq) - 2)]
        bi_gram_counts.update(bi_grams)
        tri_gram_counts.update(tri_grams)
    total_bi_grams = sum(bi_gram_counts.values())
    total_tri_grams = sum(tri_gram_counts.values())
    # 计算信息熵
    for tri_gram in tri_gram_counts:
        # 计算tri-gram和对应bi-gram的概率
        tri_gram_prob = tri_gram_counts[tri_gram] / total_tri_grams
        bi_gram = tri_gram[:-1]
        bi_gram_prob = bi_gram_counts[bi_gram] / total_bi_grams

        # 计算条件概率
        cond_prob = tri_gram_prob / bi_gram_prob

        # 累加信息熵
        entropy -= tri_gram_prob * math.log2(cond_prob)
    return entropy

if __name__ == '__main__':
    stopwords = pre.load_list('.\DLNLP2023-main\cn_stopwords.txt')
    punctuation = pre.load_list('.\DLNLP2023-main\cn_punctuation.txt')
    #添加模式，1为按字检索，2为按词检索
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-mode', type=int, default=2, help='using character(1) or words(2)')
    args = parser.parse_args()
    print(f'mode: {args.mode}')

    
    list = pre.pre2('.\\cn_words', stopwords, punctuation, max_length=args.mode)
    # list = pre.pre2('.\\words', stopwords, punctuation, max_length=args.mode)
    entropy = get_entropy(list)
    print(f"基于三元组模型的中文平均信息熵: {entropy}")