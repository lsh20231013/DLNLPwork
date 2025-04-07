import os
import jieba
import numpy as np
from gensim.models import Word2Vec
from sklearn.manifold import TSNE
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams['font.family'] = 'SimHei'
matplotlib.rcParams['axes.unicode_minus'] = False  # 正确显示负号

# === 1. 加载金庸小说语料 ===
def load_corpus_from_folder(folder_path):
    all_sentences = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):
            filepath = os.path.join(folder_path, filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                text = f.read()
                sentences = text.split('。')
                tokenized = [list(jieba.cut(sent.strip())) for sent in sentences if sent.strip()]
                all_sentences.extend(tokenized)
    return all_sentences

# === 2. 训练 Word2Vec 模型 ===
def train_word2vec(sentences):
    model = Word2Vec(sentences, vector_size=100, window=5, min_count=3, sg=1)
    return model

# === 3. 显示与指定词最相似的词 ===
def show_similar_words(model, keyword, topn=10):
    print(f"\n与「{keyword}」最相似的词：")
    try:
        similar = model.wv.most_similar(keyword, topn=topn)
        for word, score in similar:
            print(f"{word}: {score:.4f}")
    except KeyError:
        print(f"词语「{keyword}」不在词表中")

# === 4. 类比推理：A 之于 B，如同 C 之于 ? ===
def analogy_test(model, word_a, word_b, word_c):
    print(f"\n「{word_a}」之于「{word_b}」，如同「{word_c}」之于：")
    try:
        result = model.wv.most_similar(positive=[word_c, word_b], negative=[word_a])
        for word, score in result[:5]:
            print(f"{word}: {score:.4f}")
    except KeyError as e:
        print(f"词语错误：{e}")

# === 5. 词向量聚类并可视化 ===
def plot_words(model, words):
    vectors = []
    valid_words = []
    for word in words:
        if word in model.wv:
            vectors.append(model.wv[word])
            valid_words.append(word)

    if len(vectors) < 2:
        print("可用的词向量太少，无法可视化。")
        return

    vectors = np.array(vectors)

    tsne = TSNE(n_components=2, perplexity=min(30, len(vectors) - 1), random_state=42)
    reduced = tsne.fit_transform(vectors)

    plt.figure(figsize=(10, 8))
    for i, word in enumerate(valid_words):
        plt.scatter(reduced[i, 0], reduced[i, 1])
        plt.annotate(word, (reduced[i, 0], reduced[i, 1]))
    plt.title("金庸人物词向量聚类")
    plt.show()

# === 主程序 ===
if __name__ == "__main__":
    folder_path = '/home/omnisky/lsh/DLNLP/NLPwork/word_embeddings/train'  # 小说文件夹路径

    print("加载金庸小说语料中...")
    corpus = load_corpus_from_folder(folder_path)

    print(f"共加载 {len(corpus)} 个句子，开始训练 Word2Vec...")
    model = train_word2vec(corpus)
    print("训练完成！")

    # === 相似词演示 ===
    show_similar_words(model, '郭靖')
    show_similar_words(model, '令狐冲')
    show_similar_words(model, '乔峰')

    # === 类比推理演示 ===
    analogy_test(model, '郭靖', '黄蓉', '杨过')  # 郭靖:黄蓉 = 杨过:?

    # === 聚类可视化 ===
    characters = ['郭靖', '黄蓉', '杨过', '小龙女', '令狐冲', '任盈盈',
                  '张无忌', '赵敏', '周芷若', '乔峰', '段誉', '阿朱']
    plot_words(model, characters)