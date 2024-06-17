import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sentence_transformers import SentenceTransformer

# 事前学習済み日本語BERTモデルの読み込み
# https://huggingface.co/sonoisa/sentence-bert-base-ja-mean-tokens-v2
model = SentenceTransformer('sonoisa/sentence-bert-base-ja-mean-tokens-v2')

# CSVファイルを読み込む
df = pd.read_csv('data/categorize_input.csv')

# キーワードをベクトル化
keyword_vectors = model.encode(df['target'])

# データ数に対して最低データ数で割った数のクラスターを最大クラスター数として設定.
data_count = df.shape[0]
min_data = 5
_max_cluster = data_count // min_data + 1
print(f"_max_cluster[{_max_cluster}] = data_count[{data_count}] // min_data[{min_data}] + 1")

max_data = 30
_mincluster = data_count // max_data
print(f"_mincluster[{_mincluster}] = data_count[{data_count}] // max_data[{max_data}]")


# 最適なクラスタ数を見つけるためにシルエットスコアを用いる
def find_optimal_clusters(data, max_k=_max_cluster, min_k=_mincluster):
    iters = range(min_k, max_k + 1, 1)
    optimal_kmeans = None
    optimal_cluster = 0
    max_score = -1
    for k in iters:
        kmeans = KMeans(n_clusters=k, random_state=0)
        kmeans.fit(data)
        s_score = silhouette_score(data, kmeans.labels_)
        print(f"Fit {k} clusters：silhouette score = [{s_score}]")

        if max_score < s_score:
            max_score = s_score
            optimal_kmeans = kmeans
            optimal_cluster = k
            print(f"Update optimal. cluster=[{optimal_cluster}]")

    print(f"Optimal number of clusters: {optimal_cluster} with silhouette score: {max_score}")
    return optimal_kmeans


# 最適クラスタ数を求める
_optimal_kmeans = find_optimal_clusters(keyword_vectors)

# キーワードとクラスタをデータフレームにまとめる
df['category'] = _optimal_kmeans.labels_

# 結果をCSVファイルに保存
df.to_csv('data/categorize_output.csv')
