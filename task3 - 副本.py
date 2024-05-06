import os
import jieba
import random
import numpy as np
import pandas as pd
import argparse
import pretreatment as pre
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics import precision_score, f1_score

# 假设小说数据集目录和文件列表
novel_dir = './cn_words'
novel_files = os.listdir(novel_dir)
#删除inf文件
novel_files = [file for file in novel_files if file != 'inf.txt']
#停用词
stopwords = pre.load_list('.\DLNLP2023-main\cn_stopwords.txt')

# 读取小说文本数据
def load_novels(novel_dir, novel_files):
    novels = {}
    for novel_file in novel_files:
        with open(os.path.join(novel_dir, novel_file), 'r', encoding='ANSI') as f:
            text= f.read()
            text = text.replace('\u3000', '')
            text = text.replace('\n', '')
            text = text.replace(' ', '')
            novels[novel_file] = text

    return novels
# 将小说分段
def extract_paragraphs(novels, paragraph_len,max_length=2):
    paragraphs = []
    labels = []
    for label, text in novels.items():
        words = []
        start_index = 0
        if (max_length > 1):
            words = list(jieba.cut(text))
        else:
            while start_index < len(text):
                # max_length =1说明按字划分
                words.append(text[start_index])
                start_index += 1
        words = pre.remove_stopwords(words, stopwords)
        paragraphs += [' '.join(words[i:i+paragraph_len]) for i in range(0, len(words), paragraph_len)]
        labels += [label] * ((len(words) // paragraph_len)+1)
    #完成
    print("finished")
    return paragraphs, labels
# 过滤停用词
def filter_stop_words(text):
    return ' '.join([word for word in text.split() if word not in stopwords])
# LDA主题建模
def get_lda_features(paragraphs, num_topics):
    vectorizer = CountVectorizer(analyzer='char')
    dt_matrix = vectorizer.fit_transform(paragraphs)
    lda = LatentDirichletAllocation(n_components=num_topics, random_state=114)
    topic_matrix = lda.fit_transform(dt_matrix)
    return topic_matrix, vectorizer


# 分类并评估结果

def classify_and_evaluate(X, y, classifier='lr'):
    kf = KFold(n_splits=10, shuffle=True, random_state=514)
    accuracies = []
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y_encoded[train_index], y_encoded[test_index]
        if classifier == 'lr':
            clf = LogisticRegression(random_state=514)
        elif classifier == 'nb':
            clf = MultinomialNB()
        elif classifier == 'svm':
            clf = SVC(random_state=514)
        elif classifier == 'rf':
            clf = RandomForestClassifier(n_estimators=100)
        else:
            raise ValueError("Unsupported classifier type")
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        accuracies.append(accuracy_score(y_test, y_pred))


    print(f"Classification accuracy: {np.mean(accuracies):.4f}")
    print(f"Classification report:")
    print(classification_report(y_encoded, clf.predict(X), target_names=label_encoder.classes_))


if __name__ == "__main__":
    novels = load_novels(novel_dir, novel_files)
    #第2个任务：以"词"和以"字"为基本单元下分类结果有什么差异？
    paragraph_lengths =1000
    num_topics = 20  # 20个主题
    print(f"Paragraph length: {paragraph_lengths}")
    paragraphs, labels = extract_paragraphs(novels, paragraph_lengths,max_length=1)
    topic_matrix, vectorizer = get_lda_features(paragraphs, num_topics)
    classify_and_evaluate(topic_matrix, labels, classifier='rf')
    print("finish 2")
