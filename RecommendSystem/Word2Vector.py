import collections
import math
import random
import zipfile
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


def read_data(filename):
    with zipfile.ZipFile(filename) as f:
        data = tf.compat.as_str(f.read(f.namelist()[0])).split()
    return data


def build_dataset(words, vocabulary_size):
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(vocabulary_size - 1))
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    data = list()
    unk_count = 0
    for word in words:
        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reverse_dictionary


def generate_batch(batch_size, num_skips, skip_window):
    global data_index
    # 利用断言来判断参数是否满足条件
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    span = 2 * skip_window + 1
    buffer = collections.deque(maxlen=span)

    # 初始填充buffer数据
    for _ in range(span):
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    # 对每一个目标单词生成数据
    for i in range(batch_size // num_skips):
        target_to_avoid = [skip_window]
        # 只是为了给target初值
        target = skip_window
        # 对确定的目标单词buffer[skip_window]生成样本数据
        for j in range(num_skips):
            while target in target_to_avoid:
                target = random.randint(0, span - 1)
            batch[i * num_skips + j] = buffer[target]
            labels[i * num_skips + j] = buffer[skip_window]
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    return batch, labels


filename = "E:\\CCIR\\text8.zip"
vocabulary_size = 50000

batch_size = 128
embedding_size = 128
skip_window = 1
num_skips = 2

valid_size = 16
valid_window = 100
valid_examples = np.random.choice(valid_window, valid_size, replace=False)
num_sampled = 64

graph = tf.Graph()
with graph.as_default():
    train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
    train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
    valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

    with tf.device('/cpu:0'):
        embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
        embed = tf.nn.embedding_lookup(embeddings, train_inputs)

        nce_weights = tf.Variable(tf.truncated_normal([vocabulary_size, embedding_size],
                                                      stddev=1.0 / math.sqrt(embedding_size)))
        nce_biases = tf.Variable(tf.zeros([vocabulary_size]))
        loss = tf.reduce_mean(tf.nn.nce_loss(weights=nce_weights,
                                             biases=nce_biases,
                                             labels=train_labels,
                                             inputs=embed,
                                             num_sampled=num_sampled,
                                             num_classes=vocabulary_size))
        optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)

        norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
        normalized_embeddings = embeddings / norm
        valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
        similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True)

        init = tf.global_variables_initializer()


def plot_with_labels(low_dim_embs, labels, filename='tsne.png'):
    assert low_dim_embs.shape[0] >= len(labels), "More labels than enbeddings"
    plt.figure(figsize=(18, 18))
    for i, label in enumerate(labels):
        x, y = low_dim_embs[i, :]
        plt.scatter(x, y)
        # (x,y)点注释，以描点为参考，向左偏移5，向上偏移2，
        plt.annotate(label, xy=(x, y), xytext=(5, 2), textcoords='offset points', ha='right', va='bottom')
        plt.savefig(filename)


if __name__ == '__main__':
    words = read_data("E:\\CCIR\\text8.zip")
    # print('Data size:', len(words))
    # 保留top50000的单词
    data, count, index_dict, reverse_index_dict = build_dataset(words, 50000)
    # 删除单词列表，节省空间
    del words

    data_index = 0

    with tf.Session(graph=graph) as sess:
        sess.run(init)

        average_loss = 0.0
        for step in range(100001):
            batch_inputs, batch_labels = generate_batch(batch_size, skip_window, num_skips)
            _, loss_val = sess.run([optimizer, loss], feed_dict={train_inputs: batch_inputs,
                                                                 train_labels: batch_labels})
            average_loss += loss_val
            if step % 2000 == 0:
                print("Average loss at step:", step, ":", average_loss / 2000)
                average_loss = 0.0

            if step % 10000 == 0:
                sim = similarity.eval()
                for i in range(valid_size):
                    valid_word = reverse_index_dict[valid_examples[i]]
                    # 展示最近的8个单词，标号为0的为单词本身
                    nearest = (-sim[i, :]).argsort()[1: 9]
                    print(sim[i, :])
                    log_str = "Nearest to " + valid_word + " :"
                    for k in range(8):
                        close_word = reverse_index_dict[nearest[k]]
                        log_str = log_str + close_word + ','
                    print(log_str)

        final_embeddings = normalized_embeddings.eval()
        tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
        plot_only = 100
        low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only, :])
        labels = [reverse_index_dict[i] for i in range(plot_only)]
        plot_with_labels(low_dim_embs, labels)
