
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

class TextEmbedder:
    def __init__(self, max_features=5000):
        self.vectorizer = TfidfVectorizer(max_features=max_features)

    def fit_transform(self, texts):
        return self.vectorizer.fit_transform(texts)

    def transform(self, texts):
        return self.vectorizer.transform(texts)

def build_text_embeddings(df, columns=['Subject', 'Body']):
    df.fillna('', inplace=True)
    df['combined_text'] = df[columns].apply(lambda row: ' '.join(row.values.astype(str)), axis=1)
    
    embedder = TextEmbedder()
    X_vectorized = embedder.fit_transform(df['combined_text'])

    return X_vectorized, embedder
