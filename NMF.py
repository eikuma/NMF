import numpy as np
import matplotlib.pyplot as plt
import collections


with open('data2020.txt', encoding='utf-8') as f:  # data2020.txtから取り出す
    articles = [i.strip() for i in f.readlines()]  # 改行を無くす

words_list = [[0]]*len(articles)  # 記事ごとに1単語ずつ格納する2次元リスト
all_words_list = []  # すべての記事の単語を1つずつ格納するリスト
words = []  # 最終的な単語のリスト
words_dict = {}  # 単語と出現数の辞書
changed_articles = [0]*len(articles)  # 単語の処理をした後の記事を格納するリスト


for i in range(0, len(articles)):  # 単語の処理
    changed_articles[i] = articles[i].replace(',', '').replace(
        '.', '').replace('?', '').lower()
    words_list[i] = changed_articles[i].split(' ')
    all_words_list += words_list[i]

words_dict = collections.Counter(all_words_list)  # 出現頻度

words = [i for i, j in words_dict.items() if 11 > j > 3]  # 選ばれた単語


matrix = np.zeros(len(articles)*len(words)
                  ).reshape(len(articles), len(words))  # 行列を0で初期化

for i in range(0, len(articles)):  # 記事ごとの単語数を数え，行列を完成させる
    for j in range(0, len(words_list[i])):
        if words_list[i][j] in words:
            matrix[i][words.index(words_list[i][j])] += 1

k = 10  # 特徴数
n = 100  # 繰り返し回数


def update_wh(V, W, H):  # 更新の関数
    H = H*(np.dot(W.T, V)/np.dot(np.dot(W.T, W), H))
    W = W*(np.dot(V, H.T)/np.dot(np.dot(W, H), H.T))
    return H, W


cost = [0]*n  # cost初期化

# WとHの初期化
W = abs(np.random.uniform(low=0, high=1, size=(len(articles), k)))  # 重みの行列
H = abs(np.random.uniform(low=0, high=1, size=(k, len(words))))  # 特徴の行列


for i in range(0, n):  # NMF
    H, W = update_wh(matrix, W, H)
    X = np.dot(W, H)
    for j in range(len(articles)):
        for l in range(len(words)):
            cost[i] += (matrix[j][l]-X[j][l])**2


W_sort_col_index = np.argsort(W, axis=0)[::-1]  # Wの列の降順インデックス
W_sort_row_index = np.argsort(W)[:, ::-1]  # Wの行の降順インデックス
H_sort_row_index = np.argsort(H)[:, ::-1]  # Hの行の降順インデックス


for i in range(k):
    for j in range(6):  # 特徴の行列の上位6単語を表示
        print("'{}'".format(words[H_sort_row_index[i][j]]), end=" ")
    print('\n')
    for l in range(3):  # 重みの行列の上位3記事を表示
        print('{}, {}'.format(W[W_sort_col_index[l][i]][i],
                              ' '.join(articles[W_sort_col_index[l][i]].split(' ')[:15])))
    print('\n\n')

for i in range(len(articles)):
    print('\n\n')
    print(' '.join(articles[i].split(' ')[:15]))
    for j in range(3):  # 重みの行列の上位3位の値を表示
        print('\n{} '.format(W[i][W_sort_row_index[i][j]]), end=" ")
        for l in range(6):  # 特徴の行列の上位6単語を表示
            print("'{}'".format(
                words[H_sort_row_index[W_sort_row_index[i][j]][l]]), end=" ")


# 折れ線グラフを表示
plt.plot(range(n), cost)
plt.xlabel('繰り返し数', fontname="MS Gothic")
plt.ylabel('コスト関数の値', fontname="MS Gothic")
plt.show()
