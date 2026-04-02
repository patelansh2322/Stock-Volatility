import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

df = pd.read_csv('OHLCV/unclean_OHLCV.csv', usecols=lambda col: not col.startswith('Unnamed:'))

return_pivot = df.pivot(index='date', columns='ticker', values='ret')

corr = return_pivot.corr()

pca = PCA(n_components=4)
embeddings = pca.fit_transform(corr)

stock2vec = pd.DataFrame(embeddings, index=corr.index, columns=["stock2vec_1", "stock2vec_2", "stock2vec_3", "stock2vec_4"]).reset_index().rename(columns={"index": "ticker"})

df = df.merge(stock2vec, on="ticker", how="left")
df['date'] = pd.to_datetime(df['date'])
df['date_ordinal'] = df['date'].map(pd.Timestamp.toordinal)

df = df.drop(columns=['date', 'ticker'])

df.to_csv('OHLCV/clean_OHLCV.csv', index=False)
