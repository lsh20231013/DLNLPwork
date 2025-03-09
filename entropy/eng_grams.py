import nltk
import numpy
import math
import time
import matplotlib.pyplot as plt
from collections import Counter
from nltk.corpus import gutenberg, stopwords
from nltk.tokenize import word_tokenize
from nltk.util import ngrams

# 计算N-gram信息熵
def calculate_ngram_entropy(text, n):
    start_time = time.time()
    stop_words = set(stopwords.words('english'))
    word_counts = Counter()
    total_ngrams = 0
    # 处理文本
    words = word_tokenize(text.lower())  # 分词
    words = [word for word in words if word.isalpha() and word not in stop_words]  # 过滤非字母单词和停用词
    # 生成N-grams
    n_grams = ngrams(words, n)
    word_counts.update(n_grams)
    total_ngrams = sum(word_counts.values())
    # 计算信息熵
    entropy = -sum((count / total_ngrams) * math.log2(count / total_ngrams) for count in word_counts.values())
    elapsed_time = time.time() - start_time
    return entropy, total_ngrams, elapsed_time, word_counts

# **加载语料库**
corpus_texts = [gutenberg.raw(file_id) for file_id in gutenberg.fileids()]
english_text = " ".join(corpus_texts)  # 合并所有文本

# **计算1-gram信息熵**
unigram_entropy, total_unigrams, unigram_time, unigram_counts = calculate_ngram_entropy(english_text, 1)
print(f"1-gram 信息熵: {unigram_entropy:.4f}")
print(f"语料库总1-gram数: {total_unigrams}")
print(f"1-gram熵计算时间: {unigram_time:.4f} 秒")

# **计算2-gram信息熵**
bigram_entropy, total_bigrams, bigram_time, bigram_counts = calculate_ngram_entropy(english_text, 2)
print(f"\n2-gram 信息熵: {bigram_entropy:.4f}")
print(f"语料库总2-gram数: {total_bigrams}")
print(f"2-gram熵计算时间: {bigram_time:.4f} 秒")

# **计算3-gram信息熵**
trigram_entropy, total_trigrams, trigram_time, trigram_counts = calculate_ngram_entropy(english_text, 3)
print(f"\n3-gram 信息熵: {trigram_entropy:.4f}")
print(f"语料库总3-gram数: {total_trigrams}")
print(f"3-gram熵计算时间: {trigram_time:.4f} 秒")

# **绘制1-gram频率直方图（前50个高频单词）**
plt.figure(figsize=(12, 6))
unigram_freq_sorted = unigram_counts.most_common(50)
x_unigrams = [' '.join(gram[0]) for gram in unigram_freq_sorted]  # 提取n-gram并将其连接为字符串
y_unigrams = [count for _, count in unigram_freq_sorted]
plt.bar(x_unigrams, y_unigrams, color='orange')
plt.title("1-gram Frequency Distribution (Top 50)")
plt.xlabel("1-gram")
plt.ylabel("Occurrences")
plt.xticks(rotation=45)
plt.show()

# **绘制2-gram频率直方图（前50个高频2-grams）**
plt.figure(figsize=(12, 6))
bigram_freq_sorted = bigram_counts.most_common(50)
x_bigrams = [' '.join(gram[0]) for gram in bigram_freq_sorted]  # 提取n-gram并将其连接为字符串
y_bigrams = [count for _, count in bigram_freq_sorted]
plt.bar(x_bigrams, y_bigrams, color='pink')
plt.title("2-gram Frequency Distribution (Top 50)")
plt.xlabel("2-gram")
plt.ylabel("Occurrences")
plt.xticks(rotation=45)
plt.show()

# **绘制3-gram频率直方图（前50个高频3-grams）**
plt.figure(figsize=(12, 6))
trigram_freq_sorted = trigram_counts.most_common(50)
x_trigrams = [' '.join(gram[0]) for gram in trigram_freq_sorted]  # 提取n-gram并将其连接为字符串
y_trigrams = [count for _, count in trigram_freq_sorted]
plt.bar(x_trigrams, y_trigrams, color='lightblue')
plt.title("3-gram Frequency Distribution (Top 50)")
plt.xlabel("3-gram")
plt.ylabel("Occurrences")
plt.xticks(rotation=45)
plt.show()







