import jieba
import os
import torch
import re
from collections import Counter
from sklearn.model_selection import train_test_split
import pandas as pd

#文本读取
def load_texts(directory):
    corpus = ""
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            with open(os.path.join(directory, filename), 'r', encoding='ANSI') as file:
                test= file.read()
                test=test.replace('\u3000','').replace('\n', '').replace(' ', '')
                corpus +=test
    return corpus


#分句
def split_sentences(text):
    sentences = re.split(r'([。！？])', text)
    sentences = [sentence + punct for sentence, punct in zip(sentences[0::2], sentences[1::2])]
    return sentences


#分词
def tokenize_sentences(sentences):
    tokenized_sentences = []
    for sentence in sentences:
        tokens = jieba.lcut(sentence)
        tokens = ['<sos>'] + tokens + ['<eos>']
        tokenized_sentences.append(tokens)
    return tokenized_sentences


#构建词汇表
def build_vocab(tokenized_sentences):
    counter = Counter()
    for sentence in tokenized_sentences:
        counter.update(sentence)
    vocab = {word: i for i, (word, _) in enumerate(counter.items(), start=1)}
    vocab['<unk>'] = 0
    return vocab

# 将句子转换为索引
def sentences_to_indices(tokenized_sentences, vocab):
    indexed_sentences = []
    for sentence in tokenized_sentences:
        indexed_sentence = [vocab.get(word, vocab['<unk>']) for word in sentence]
        indexed_sentences.append(indexed_sentence)
    return indexed_sentences

# 将索引转换回字符串形式的句子对


# 分割数据集
#train_sentences, valid_sentences = train_test_split(indexed_sentence_pairs, test_size=0.1, random_state=42)
#保存train_sentences


def pretrain():
    novel_dir = './cn_words/'
    punctuation_path = "./DLNLP2023-main/cn_punctuation.txt"
    stopwords_path = "./DLNLP2023-main/cn_stopwords.txt"
    corpus = load_texts(novel_dir)
    sentences = split_sentences(corpus)
    tokenized_sentences = tokenize_sentences(sentences)
    vocab = build_vocab(tokenized_sentences)
    indexed_sentences = sentences_to_indices(tokenized_sentences, vocab)
    #indexed_sentence_pairs = [(sentence, sentence) for sentence in indexed_sentences]
    indexed_sentence_pairs = [(indexed_sentences[i], indexed_sentences[i+1]) for i in range(len(indexed_sentences) - 1)]
    return vocab,indexed_sentence_pairs
#pretrain()
# 打印一些预处理后的结果
# print("Sample sentence:", sentences[0])
# print("Tokenized sentence:", tokenized_sentences[0])
# print("Indexed sentence:", indexed_sentences[0])

