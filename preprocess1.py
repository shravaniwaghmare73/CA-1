import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from Config import config

def load_and_prepare_data():
    # Load the CSV and drop completely empty columns
    df = pd.read_csv(config['data_path'])
    df = df.dropna(axis=1, how='all')


    df['combined_text'] = df['Ticket Summary'].astype(str) + ' ' + df['Interaction content'].astype(str)


    df = df.dropna(subset=config['label_columns'])

    label_encoders = {}
    for col in config['label_columns']:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

    # Extract labels
    labels = df[config['label_columns']]


    vectorizer = TfidfVectorizer(max_features=5000)
    features = vectorizer.fit_transform(df['combined_text'])

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.25, random_state=42)

    return X_train, X_test, y_train, y_test