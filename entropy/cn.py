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

# 创建 OpenCC 转换器（繁体 -> 简体）
converter = opencc.OpenCC('t2s')


# 加载停用词
def load_stopwords(stopwords_path):
    if os.path.exists(stopwords_path):
        with open(stopwords_path, "r", encoding="utf-8") as f:
            return set(f.read().splitlines())
    return set()


# 繁体转简体（带进度条）
def convert_to_simplified(text):
    print("正在进行繁体转简体...")
    chunk_size = max(1, len(text) // 100)
    return "".join(
        converter.convert(text[i:i + chunk_size]) for i in tqdm(range(0, len(text), chunk_size), desc="转换进度"))


# 过滤非中文字符（去掉英文、数字、特殊字符）
def filter_chinese(text):
    return "".join(re.findall(r'[\u4e00-\u9fff]+', text))


# 计算字符级信息熵
def calculate_character_entropy(text):
    print("\n计算字符级信息熵...")
    start_time = time.time()
    char_counts = Counter(text)
    total_chars = sum(char_counts.values())

    entropy = -sum((count / total_chars) * math.log2(count / total_chars) for count in char_counts.values())
    return entropy, total_chars, time.time() - start_time, char_counts


# 计算词级信息熵
def calculate_word_entropy(text, stopwords):
    print("\n计算词级信息熵...")
    start_time = time.time()
    words = [word for word in jieba.lcut(text) if word.strip() and word not in stopwords]
    word_counts = Counter(words)
    total_words = sum(word_counts.values())

    entropy = -sum((count / total_words) * math.log2(count / total_words) for count in word_counts.values())
    return entropy, total_words, time.time() - start_time, word_counts


# 读取文件夹 a 内所有无后缀的文本文件
def load_corpus_from_folder(folder_path):
    corpus_texts = []
    file_list = [os.path.join(root, file) for root, _, files in os.walk(folder_path) for file in files]

    if not file_list:
        print("⚠️ 没有找到任何文本文件，请检查路径和目录结构！")
        return ""

    print(f"📂 找到 {len(file_list)} 个文本文件，正在读取...")

    for file_path in tqdm(file_list, desc="读取文件进度"):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                corpus_texts.append(f.read())
        except Exception as e:
            print(f"❌ 无法读取文件 {file_path}: {e}")

    return "\n".join(corpus_texts)


# 设置语料库文件夹路径
folder_path = "/home/omnisky/lsh/DLNLP/NLPwork/wiki_zh_2019/wiki_zh"  # 文件夹路径
stopwords_path = "/home/omnisky/lsh/DLNLP/NLPwork/entropy/cn_stopwords.txt"  # 停用词文件路径

# 加载停用词
stopwords = load_stopwords(stopwords_path)
print(f"✅ 加载 {len(stopwords)} 个停用词")

# 加载所有无后缀文件
chinese_text = load_corpus_from_folder(folder_path)

if chinese_text:
    # 繁体转简体
    simplified_text = convert_to_simplified(chinese_text)

    # 仅保留中文字符
    cleaned_text = filter_chinese(simplified_text)

    # 计算字符级信息熵
    char_entropy, total_chars, char_time, char_counts = calculate_character_entropy(cleaned_text)
    print(f"\n🔤 中文字符级信息熵: {char_entropy:.4f}")
    print(f"📖 语料库总字符数: {total_chars}")
    print(f"⏳ 字符熵计算时间: {char_time:.4f} 秒")

    # 计算词级信息熵
    word_entropy, total_words, word_time, word_counts = calculate_word_entropy(cleaned_text, stopwords)
    print(f"\n📝 中文词级信息熵: {word_entropy:.4f}")
    print(f"📖 语料库总单词数（去停用词）: {total_words}")
    print(f"⏳ 词熵计算时间: {word_time:.4f} 秒")

    # 绘制字符频率直方图
    plt.figure(figsize=(12, 6))
    char_freq_sorted = char_counts.most_common(50)
    plt.bar([char[0] for char in char_freq_sorted], [char[1] for char in char_freq_sorted], color='blue')
    plt.title("Chinese Character Frequency Distribution (Top 50)", fontproperties=font_prop)
    plt.xlabel("Character")
    plt.ylabel("Occurrences")
    plt.show()

    # 绘制词频直方图
    plt.figure(figsize=(12, 6))
    word_freq_sorted = word_counts.most_common(50)
    plt.bar([word[0] for word in word_freq_sorted], [word[1] for word in word_freq_sorted], color='green')
    plt.title("Chinese Word Frequency Distribution (Top 50)", fontproperties=font_prop)
    plt.xlabel("Word")
    plt.ylabel("Occurrences")
    plt.xticks(rotation=45)
    plt.show()