import os
import jieba
import math
import time
import opencc
import re
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import subprocess
from collections import Counter
from tqdm import tqdm

# 获取系统默认的中文字体
def get_chinese_font():
    try:
        font_path = subprocess.getoutput("fc-list :lang=zh | head -n 1 | cut -d':' -f1")  # 取第一行字体路径
        return fm.FontProperties(fname=font_path)
    except Exception as e:
        print("未找到中文字体，请检查系统字体安装", e)
        return None
# 设置字体
font_prop = get_chinese_font()
if font_prop:
    plt.rcParams["font.family"] = font_prop.get_name()
    plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

# **创建 OpenCC 繁体转简体转换器**
converter = opencc.OpenCC('t2s')


# **加载停用词**
def load_stopwords(stopwords_path):
    stopwords = set()
    if os.path.exists(stopwords_path):
        with open(stopwords_path, "r", encoding="utf-8") as f:
            stopwords = set(f.read().splitlines())
    return stopwords


# **繁体转简体**
def convert_to_simplified(text):
    print("正在进行繁体转简体...")
    converted_text = []
    chunk_size = max(1, len(text) // 100)  # 100 个批次处理，防止文本过长卡死
    for i in tqdm(range(0, len(text), chunk_size), desc="转换进度"):
        converted_text.append(converter.convert(text[i:i + chunk_size]))
    return "".join(converted_text)

# 过滤非中文字符（去掉英文、数字、特殊字符）
def filter_chinese(text):
    return "".join(re.findall(r'[\u4e00-\u9fff]+', text))

# **计算 N-Gram 信息熵**
def calculate_ngram_entropy(text, stopwords, n=1):
    print(f"\n计算 {n}-gram 信息熵...")
    start_time = time.time()
    words = jieba.lcut(text)

    # 过滤停用词
    words = [word for word in tqdm(words, desc="分词进度") if word.strip() and word not in stopwords]

    total_ngrams = len(words) - (n - 1)  # 计算总的 N-Grams 数量
    if total_ngrams <= 0:
        print(f"⚠️  {n}-gram 词数过少，无法计算信息熵！")
        return 0, 0, 0, Counter()

    # 生成 N-Grams
    ngram_counts = Counter()
    for i in tqdm(range(total_ngrams), desc=f"统计 {n}-gram 频率"):
        ngram = tuple(words[i:i + n])  # 生成 N-Gram
        ngram_counts[ngram] += 1

    # 计算信息熵
    entropy = -sum((count / total_ngrams) * math.log2(count / total_ngrams) for count in ngram_counts.values())
    elapsed_time = time.time() - start_time
    return entropy, total_ngrams, elapsed_time, ngram_counts


# **读取文件夹 a 内所有无后缀的文本文件**
def load_corpus_from_folder(folder_path):
    corpus_texts = []
    file_list = []

    # 遍历文件夹及子目录
    for root, _, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            if os.path.isfile(file_path):  # 确保是文件，而不是目录
                file_list.append(file_path)

    if not file_list:
        print("⚠️ 没有找到任何文本文件，请检查路径！")
        return ""

    print(f"📂 找到 {len(file_list)} 个文本文件，正在读取...")

    for file_path in tqdm(file_list, desc="读取文件进度"):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                corpus_texts.append(f.read())
        except Exception as e:
            print(f"❌ 无法读取文件 {file_path}: {e}")

    return "\n".join(corpus_texts)


# **设置路径**
folder_path = "/home/omnisky/lsh/DLNLP/NLPwork/wiki_zh_2019/wiki_zh"  # 文件夹路径
stopwords_path = "/home/omnisky/lsh/DLNLP/NLPwork/entropy/cn_stopwords.txt"  # 停用词文件路径

# **加载停用词**
stopwords = load_stopwords(stopwords_path)
print(f"✅ 加载 {len(stopwords)} 个停用词")

# **加载无后缀文件**
chinese_text = load_corpus_from_folder(folder_path)

if chinese_text:
    # **繁体转简体**
    simplified_text = convert_to_simplified(chinese_text)

    # 仅保留中文字符
    cleaned_text = filter_chinese(simplified_text)

    # **计算 1-gram 信息熵**
    unigram_entropy, total_unigrams, unigram_time, unigram_counts = calculate_ngram_entropy(cleaned_text, stopwords,
                                                                                            n=1)
    print(f"\n🔤 1-gram 信息熵: {unigram_entropy:.4f}")
    print(f"📖 1-gram 总数: {total_unigrams}")
    print(f"⏳ 计算时间: {unigram_time:.4f} 秒")

    # **计算 2-gram 信息熵**
    bigram_entropy, total_bigrams, bigram_time, bigram_counts = calculate_ngram_entropy(cleaned_text, stopwords, n=2)
    print(f"\n🔠 2-gram 信息熵: {bigram_entropy:.4f}")
    print(f"📖 2-gram 总数: {total_bigrams}")
    print(f"⏳ 计算时间: {bigram_time:.4f} 秒")

    # **计算 3-gram 信息熵**
    trigram_entropy, total_trigrams, trigram_time, trigram_counts = calculate_ngram_entropy(cleaned_text, stopwords,
                                                                                            n=3)
    print(f"\n🔢 3-gram 信息熵: {trigram_entropy:.4f}")
    print(f"📖 3-gram 总数: {total_trigrams}")
    print(f"⏳ 计算时间: {trigram_time:.4f} 秒")

    # 修正 1-gram 直方图
    plt.figure(figsize=(12, 6))
    unigram_freq_sorted = unigram_counts.most_common(50)
    x_unigrams = [''.join(word[0]) for word in unigram_freq_sorted]  # 确保 x 轴是字符串
    y_unigrams = [word[1] for word in unigram_freq_sorted]
    plt.bar(x_unigrams, y_unigrams, color='blue')
    plt.title("Chinese 1-Gram Frequency Distribution (Top 50)", fontproperties=font_prop)
    plt.xlabel("Character")
    plt.ylabel("Occurrences")
    plt.show()

    # 修正 2-gram 直方图
    plt.figure(figsize=(12, 6))
    bigram_freq_sorted = bigram_counts.most_common(50)
    x_bigrams = [''.join(word[0]) for word in bigram_freq_sorted]  # 确保 x 轴是字符串
    y_bigrams = [word[1] for word in bigram_freq_sorted]
    plt.bar(x_bigrams, y_bigrams, color='red')
    plt.title("Chinese 2-Gram Frequency Distribution (Top 50)", fontproperties=font_prop)
    plt.xlabel("Bigram")
    plt.ylabel("Occurrences")
    plt.xticks(rotation=45)
    plt.show()

    # 修正 3-gram 直方图
    plt.figure(figsize=(12, 6))
    trigram_freq_sorted = trigram_counts.most_common(50)
    x_trigrams = [''.join(word[0]) for word in trigram_freq_sorted]  # 确保 x 轴是字符串
    y_trigrams = [word[1] for word in trigram_freq_sorted]
    plt.bar(x_trigrams, y_trigrams, color='green')
    plt.title("Chinese 3-Gram Frequency Distribution (Top 50)", fontproperties=font_prop)
    plt.xlabel("Trigram")
    plt.ylabel("Occurrences")
    plt.xticks(rotation=45)
    plt.show()


