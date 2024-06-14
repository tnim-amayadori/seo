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
    for n_clusters in range(n_samples // max_data, n_samples // min_data + 1):
        kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42)
        kmeans.fit(data)
        cluster_counts = pd.Series(kmeans.labels_).value_counts()
        if cluster_counts.between(min_data, max_data).all():
            return n_clusters, kmeans.labels_, min_data

        if cluster_counts.between(0, max_data).all():
            tmp_clusters = n_clusters
            tmp_labels = kmeans.labels_

    return tmp_clusters, tmp_labels, tmp_min


# クラスタ数を調整します
optimal_clusters, labels, min_count = adjust_clusters(X, _min_data, _max_data)

# クラスタ数が見つかった場合、結果をDataFrameに追加します
if optimal_clusters:
    df['category'] = labels
    # 結果のクラスタを表示します
    print(df)
    # クラスタをCSVファイルに保存します
    df.to_csv('categorize_output.csv')
    # 成功メッセージを表示します
    message = f"検索ワードは{optimal_clusters}個のクラスタに正常に分類され、"
    message += f"各クラスタは{min_count}～{_max_data}個のデータセットを含みます。"
    message += "結果は'search_keyword_clusters.csv'に保存されました。"
    print(message)
else:
    print(
        "適切なクラスタ数が見つかりませんでした。データセットの数を調整するか、クラスタ当たりのデータの数の範囲を変更してください。")
