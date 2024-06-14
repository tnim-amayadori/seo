import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

# CSVファイルを読み込む
df = pd.read_csv('categorize_input.csv')

# TF-IDFを用いて特徴抽出
vectorizer = TfidfVectorizer(max_features=1000)
X = vectorizer.fit_transform(df['target'])

# K-Meansクラスタリングを適用
kmeans = KMeans(n_clusters=40, random_state=42)
kmeans.fit(X)

# クラスタリング結果をDataFrameに追加
df['category'] = kmeans.labels_

# 結果をCSVファイルに保存
df.to_csv('categorize_output.csv')
