import os
import re
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Embedding, LSTM, Dense, Input, MultiHeadAttention, GlobalAveragePooling1D
from tensorflow.keras.optimizers import Adam
from tqdm.auto import tqdm
import matplotlib
import matplotlib.pyplot as plt
# ✅ 设置 matplotlib 使用中文字体（Noto Sans CJK SC）
matplotlib.rcParams['font.family'] = 'SimHei'
matplotlib.rcParams['axes.unicode_minus'] = False  # 正确显示负号
# 自定义训练进度条
class TqdmProgressCallback(tf.keras.callbacks.Callback):
    def __init__(self, total_epochs):
        super().__init__()
        self.epoch_pbar = None
        self.total_epochs = total_epochs

    def on_train_begin(self, logs=None):
        self.epoch_pbar = tqdm(total=self.total_epochs, desc='Epochs', position=0)

    def on_epoch_end(self, epoch, logs=None):
        self.epoch_pbar.update(1)
        logs = logs or {}
        self.epoch_pbar.set_postfix({
            'loss': f"{logs.get('loss', '?') :.4f}",
            'val_loss': f"{logs.get('val_loss', '?') :.4f}"
        })

    def on_train_end(self, logs=None):
        self.epoch_pbar.close()

# 数据加载和预处理
def load_and_preprocess_data(folder_path, seq_length=100):
    print("\n[1/4] 正在加载和预处理数据...")
    corpus = ""
    files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]
    print(f"发现 {len(files)} 个文本文件")

    with tqdm(files, desc='读取文件', unit='file') as file_pbar:
        for filename in file_pbar:
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
                text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9]', '', text)  # 去除标点
                corpus += text + '\n'
            file_pbar.set_postfix({'当前文件': filename[:10] + '...'})

    print("\n正在分词...")
    tokenizer = Tokenizer(char_level=True)
    tokenizer.fit_on_texts([corpus])
    total_chars = len(tokenizer.word_index) + 1

    sequences = []
    print("\n正在生成训练序列...")
    for i in range(0, len(corpus) - seq_length, 1):
        sequences.append(corpus[i:i + seq_length])

    subset_size = 1000000  # 限制训练规模
    if len(sequences) > subset_size:
        indices = np.random.choice(len(sequences), subset_size, replace=False)
        sequences = [sequences[i] for i in indices]

    X, y = [], []
    print("\n正在准备训练数据...")
    for seq in tqdm(sequences, desc='序列编码'):
        encoded = tokenizer.texts_to_sequences([seq])[0]
        X.append(encoded[:-1])
        y.append(encoded[-1])

    X = np.array(pad_sequences(X, maxlen=seq_length - 1, padding='pre'))
    y = np.array(y)

    print(f"\n数据预处理完成！总字符数: {total_chars}，训练样本数: {len(X)}")
    return X, y, tokenizer, total_chars, seq_length

# LSTM模型
def build_lstm_model(total_chars, seq_length):
    model = Sequential([
        Embedding(total_chars, 128, input_length=seq_length - 1),
        LSTM(128),
        Dense(total_chars, activation='softmax')
    ])
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# Transformer模型
def build_transformer_model(total_chars, seq_length):
    inputs = Input(shape=(seq_length - 1,))
    x = Embedding(total_chars, 128)(inputs)
    x = MultiHeadAttention(key_dim=32, num_heads=2)(x, x)
    x = GlobalAveragePooling1D()(x)
    outputs = Dense(total_chars, activation="softmax")(x)
    model = Model(inputs, outputs)
    model.compile(optimizer=Adam(1e-3), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model

# 模型训练
def train_model(model, X, y, name, epochs, batch_size):
    print(f"\n[3/4] 正在训练 {name} 模型...")
    callbacks = [
        TqdmProgressCallback(epochs),
        tf.keras.callbacks.TensorBoard(log_dir=f'logs/{name}')
    ]
    history = model.fit(
        X, y,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=0.2,
        callbacks=callbacks,
        verbose=0
    )
    print(f"{name} 训练完成！")
    return history

# 文本生成并保存
def generate_and_save_text(seeds, models, tokenizer, seq_length, num_gen_chars=50, output_file="generated_results.txt"):
    with open(output_file, 'w', encoding='utf-8') as f_out:
        for seed in seeds:
            f_out.write(f"\n种子文本: '{seed}'\n")
            print(f"\n种子文本: '{seed}'")
            for name, model in models.items():
                generated = seed
                for _ in tqdm(range(num_gen_chars), desc=f'{name}生成', unit='char'):
                    token_list = tokenizer.texts_to_sequences([generated])[0][-(seq_length - 1):]
                    token_list = pad_sequences([token_list], maxlen=seq_length - 1, padding='pre')
                    predicted = model.predict(token_list, verbose=0)[0]
                    predicted_id = np.argmax(predicted)
                    output_word = tokenizer.index_word.get(predicted_id, '')
                    generated += output_word
                f_out.write(f"{name} 生成结果:\n{generated}\n\n")
                print(f"{name} 生成结果:\n{generated}\n")
    print(f"生成文本已保存至：{output_file}")

# 绘制训练损失曲线
def plot_training_history(histories):
    plt.figure(figsize=(12, 5))
    for name, history in histories.items():
        plt.plot(history.history['loss'], label=f'{name} Train Loss')
        plt.plot(history.history['val_loss'], label=f'{name} Val Loss')
    plt.title('模型训练损失')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

# 主流程
def main():
    folder_path = '/home/omnisky/lsh/DLNLP/NLPwork/transformer/train'  # 请替换为你的文本文件夹
    seq_length = 500
    epochs = 75
    batch_size = 1024

    X, y, tokenizer, total_chars, seq_length = load_and_preprocess_data(folder_path, seq_length)

    print("\n[2/4] 正在构建模型...")
    models = {
        'LSTM': build_lstm_model(total_chars, seq_length),
        'Transformer': build_transformer_model(total_chars, seq_length)
    }

    histories = {}
    for name, model in models.items():
        histories[name] = train_model(model, X, y, name, epochs, batch_size)

    print("\n[4/4] 正在生成示例文本并写入文件...")
    seeds = ["杨过", "郭靖", "张无忌", "乔峰"]
    generate_and_save_text(seeds, models, tokenizer, seq_length, num_gen_chars=50)

    plot_training_history(histories)

if __name__ == "__main__":
    main()