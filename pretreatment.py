#预处理
import jieba
from collections import Counter
import os
#从文本中划分词
def find_words(text, max_length):
    words = []
    start_index = 0
    if (max_length > 1):
        words = list(jieba.cut(text))
    else:
        while start_index < len(text):
            # max_length =1说明按字划分
            words.append(text[start_index])
            start_index += 1

    return words
def load_list(path):
    list = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            list.append(line.strip())
    return list

def remove_stopwords(words, stopwords):
    return [word for word in words if word not in stopwords]

#句末添加分隔符
def add_end(words,punctuation):
    return ['<end>' if word in punctuation else word for word in words]
def pre(path,stopwords,max_length):
    total_counts = Counter()
    for filename in os.listdir(path):
        # 确保只处理.txt文件
        if filename.endswith('.txt'):
            # 构建完整的文件路径
            file_path = os.path.join(path, filename)
            print(file_path)
            with open(file_path, 'r', encoding='ANSI') as f:
                # 读取文本内容
                text = f.read()
                # 使用 replace() 方法去除所有空格
                text = text.replace('\u3000', '')
                text = text.replace('\n', '')
                text = text.replace(' ', '')
                # 按字划分
                words = find_words(text, max_length)
                # 移除停用词
                words = remove_stopwords(words, stopwords)
                # 统计词频
                total_counts.update(words)
    #汇总一下排名
    sort_words = sorted(total_counts.items(), key=lambda x: x[1], reverse=True)
    return sort_words

def pre2(path,stopwords,punctuation,max_length):
    total_text =[]
    for filename in os.listdir(path):
        # 确保只处理.txt文件
        if filename.endswith('.txt'):
            # 构建完整的文件路径
            file_path = os.path.join(path, filename)
            print(file_path)
            with open(file_path, 'r', encoding='ANSI') as f:
            #with open(file_path, 'r', encoding='utf8') as f:
                # 读取文本内容
                text = f.read()
                # 使用 replace() 方法去除所有空格和换行
                text = text.replace('\u3000', '')
                text = text.replace('\n', '')
                text = text.replace(' ', '')
                # 按字划分
                words = find_words(text, max_length)
                # 添加分隔符断句
                words = add_end(words, punctuation)
                # 移除停用词
                words = remove_stopwords(words, stopwords)
                # 添加到总语料列表中
                total_text.extend(words)

    return total_text

