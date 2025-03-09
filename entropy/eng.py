import nltk
import numpy
import math
import time
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import subprocess
from collections import Counter
from nltk.corpus import gutenberg, stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag

# 词性转换函数，将 nltk 的 POS 标签转换为 wordnet 词性
def get_wordnet_pos(tag):
    if tag.startswith('J'):  # 形容词
        return wordnet.ADJ
    elif tag.startswith('V'):  # 动词
        return wordnet.VERB
    elif tag.startswith('N'):  # 名词
        return wordnet.NOUN
    elif tag.startswith('R'):  # 副词
        return wordnet.ADV
    else:
        return wordnet.NOUN  # 默认为名词

# 计算字母级信息熵
def calculate_letter_entropy(text):
    start_time = time.time()
    # 统计字母频率
    letters = [char.lower() for char in text if char.isalpha()]
    total_letters = len(letters)
    letter_counts = Counter(letters)
    # 计算信息熵
    entropy = -sum((count / total_letters) * math.log2(count / total_letters) for count in letter_counts.values())
    elapsed_time = time.time() - start_time
    return entropy, total_letters, elapsed_time, letter_counts

# 计算单词级信息熵
def calculate_word_entropy(text):
    start_time = time.time()
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    word_counts = Counter()
    total_words = 0
    # 处理文本
    words = word_tokenize(text.lower())  # 分词
    words = [word for word in words if word.isalpha() and word not in stop_words]  # 过滤非字母单词和停用词
    total_words = len(words)
    # 词形归一化
    word_pos_tags = pos_tag(words)
    lemmatized_words = [lemmatizer.lemmatize(word, get_wordnet_pos(tag)) for word, tag in word_pos_tags]
    # 统计词频
    word_counts.update(lemmatized_words)
    # 计算信息熵
    entropy = -sum((count / total_words) * math.log2(count / total_words) for count in word_counts.values())
    elapsed_time = time.time() - start_time
    return entropy, total_words, elapsed_time, word_counts


corpus_texts = [gutenberg.raw(file_id) for file_id in gutenberg.fileids()]
english_text = " ".join(corpus_texts)  # 合并所有文本

# **计算字母级信息熵**
letter_entropy, total_letters, letter_time, letter_counts = calculate_letter_entropy(english_text)
print(f"英文字母级信息熵: {letter_entropy:.4f}")
print(f"语料库总字母数: {total_letters}")
print(f"字母熵计算时间: {letter_time:.4f} 秒")

# **计算单词级信息熵**
word_entropy, total_words, word_time, word_counts = calculate_word_entropy(english_text)
print(f"\n英文单词级信息熵: {word_entropy:.4f}")
print(f"语料库总单词数（去停用词和标点）: {total_words}")
print(f"单词熵计算时间: {word_time:.4f} 秒")

# **绘制字母频率直方图**
plt.figure(figsize=(12, 6))
letter_freq_sorted = sorted(letter_counts.items(), key=lambda x: x[1], reverse=True)
x_letters = [letter[0].upper() for letter in letter_freq_sorted]
y_letters = [letter[1] for letter in letter_freq_sorted]
plt.bar(x_letters, y_letters, color='orange')
plt.title("English Letter Frequency Distribution")
plt.xlabel("Letter")
plt.ylabel("Occurrences")
plt.show()

# **绘制单词频率直方图(前50个高频单词)**
plt.figure(figsize=(12, 6))
word_freq_sorted = word_counts.most_common(50)
x_words = [word[0] for word in word_freq_sorted]
y_words = [word[1] for word in word_freq_sorted]
plt.bar(x_words, y_words, color='pink')
plt.title("English Word Frequency Distribution (Top 50)")
plt.xlabel("Word")
plt.ylabel("Occurrences")
plt.xticks(rotation=45)
plt.show()



