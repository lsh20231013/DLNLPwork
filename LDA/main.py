import os
import random
import jieba
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# 设置参数
FOLDER_PATH = "/home/omnisky/lsh/DLNLP/NLPwork/LDA/train"  # 金庸小说全集存放的文件夹
TOKEN_PER_PARAGRAPH = 100  # 选择段落长度 K（20, 100, 500, 1000, 3000）
NUM_TOPICS = 10  # 主题数 T
USE_WORDS = False  # True: 以 "词" 为单位；False: 以 "字" 为单位
NUM_EXPERIMENTS = 10  # 10 折交叉验证


# 1. 加载小说文件，并分词
def load_novels_from_folder(folder_path, token_per_paragraph, use_words=True):
    file_list = os.listdir(folder_path)
    paragraphs = []
    labels = []

    print(f"总共加载 {len(file_list)} 个文件")

    for file_name in file_list:
        file_path = os.path.join(folder_path, file_name)
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

        print(f"\n加载文件: {file_name}, 文件长度: {len(text)} 字符")

        # **修正：根据 USE_WORDS 选择分词方式**
        tokens = list(text) if not use_words else list(jieba.cut(text))
        print(f"分词完成，共分词：{len(tokens)}")

        # 按固定 token 数拆分段落
        num_paragraphs = len(tokens) // token_per_paragraph
        for i in range(num_paragraphs):
            paragraphs.append(" ".join(tokens[i * token_per_paragraph: (i + 1) * token_per_paragraph]))
            labels.append(file_name)  # 每个段落的标签为小说文件名

        print(f"划分文件 {file_name} 为 {num_paragraphs} 个段落")

    print(f"\n共划分了 {len(paragraphs)} 个段落")

    return paragraphs, labels


# 2. LDA 主题建模 + 分类
def run_experiment(paragraphs, labels, num_topics, num_experiments):
    print(f"\n开始执行实验：主题数={num_topics}, 段落长度={TOKEN_PER_PARAGRAPH}, 基本单元={'词' if USE_WORDS else '字'}")

    # **修正：适应按“字”拆分**
    vectorizer = CountVectorizer(token_pattern=r"(?u).", max_features=50000) if not USE_WORDS else CountVectorizer(
        max_features=50000)

    X = vectorizer.fit_transform(paragraphs)
    y = np.array(labels)

    # 10 折交叉验证
    skf = StratifiedKFold(n_splits=num_experiments, shuffle=True, random_state=42)
    accuracies = []

    with tqdm(total=num_experiments, desc="交叉验证进度") as pbar:
        for i, (train_idx, test_idx) in enumerate(skf.split(X, y)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # **修正：转换为密集矩阵**
            X_train = X_train.toarray()
            X_test = X_test.toarray()

            # 训练 LDA
            lda = LatentDirichletAllocation(n_components=num_topics, random_state=42, learning_method="batch")
            X_train_topics = lda.fit_transform(X_train)
            X_test_topics = lda.transform(X_test)

            # 训练 Naive Bayes 分类器
            classifier = MultinomialNB()
            classifier.fit(X_train_topics, y_train)

            # 预测
            y_pred = classifier.predict(X_test_topics)
            acc = accuracy_score(y_test, y_pred)
            accuracies.append(acc)

            # 更新进度条
            pbar.update(1)
            tqdm.write(f"完成第 {i + 1} 次交叉验证，当前准确率: {acc:.4f}")

    # 计算最终准确率
    final_acc = np.mean(accuracies)
    print(f"\n主题数 {num_topics}，最终分类准确率: {final_acc:.4f}")
    return final_acc


# 3. 运行多个实验，观察不同参数影响
def main():
    paragraphs, labels = load_novels_from_folder(FOLDER_PATH, TOKEN_PER_PARAGRAPH, USE_WORDS)

    topic_counts = [5, 10, 20, 50]
    results = []

    for num_topics in topic_counts:
        acc = run_experiment(paragraphs, labels, num_topics, NUM_EXPERIMENTS)
        results.append((num_topics, acc))

    df_results = pd.DataFrame(results, columns=["主题数", "分类准确率"])
    print("\n实验结果汇总：")
    print(df_results)


# 运行主程序
main()