import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

# CSVファイルを読み込む
df = pd.read_csv('categorize_input.csv')

# TfidfVectorizerを初期化します
vectorizer = TfidfVectorizer()

# 検索ワードをfit_transformします
X = vectorizer.fit_transform(df['target'])

_min_data = 5
_max_data = 50


# クラスタ数を調整する関数を定義します
def adjust_clusters(data, min_data=_min_data, max_data=_max_data):
    n_samples = data.shape[0]
    tmp_clusters = None
    tmp_labels = None
    tmp_min = 0
    max_data_count = n_samples

    min_cluster = n_samples // max_data
    max_cluster = n_samples // min_data + 1

    print(f"クラスター数[{min_cluster}]～[{max_cluster}]でK-meansを試行します。")

    for n_clusters in range(min_cluster, max_cluster):
        kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42)
        kmeans.fit(data)
        cluster_counts = pd.Series(kmeans.labels_).value_counts()

        # Pattern1. 各クラスターの数が指定のデータ数の範囲だった場合.
        if cluster_counts.between(min_data, max_data).all():
            print(f"[{n_clusters}]：Pattern1")
            return n_clusters, kmeans.labels_, min_data

        # Pattern3. 最大データ数がなるべく少なくなっているもので分類.
        if cluster_counts.max() < max_data_count:
            tmp_clusters = n_clusters
            tmp_labels = kmeans.labels_
            max_data_count = cluster_counts.max()
            print(f"[{n_clusters}]：Pattern3 max_data_count=[{max_data_count}]")
        else:
            print(f"[{n_clusters}]：Pattern4 max_data_count=[{cluster_counts.max()}]")

    return tmp_clusters, tmp_labels, tmp_min, max_data_count


# クラスタ数を調整します
optimal_clusters, labels, min_count, max_count = adjust_clusters(X, _min_data, _max_data)

# クラスタ数が見つかった場合、結果をDataFrameに追加します
if optimal_clusters:
    df['category'] = labels
    # 結果のクラスタを表示します
    print(df)
    # クラスタをCSVファイルに保存します
    df.to_csv('categorize_output.csv')
    # 成功メッセージを表示します
    message = f"検索ワードは{optimal_clusters}個のクラスタに正常に分類され、"
    message += f"各クラスタは{min_count}～{max_count}個のデータセットを含みます。"
    message += "結果は'search_keyword_clusters.csv'に保存されました。"
    print(message)
else:
    print("クラスタ分割のループがまわっていない想定外のエラーです。")
