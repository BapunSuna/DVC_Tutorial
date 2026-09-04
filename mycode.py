import os
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

def create_dataframe():
    data = {
        "id":[1,2,3,4,5,6,7,8,9,10],
        "review":[
            "This product is great!v2",
            "I love this item.",
            "Not worth the price.",
            "Excellent quality and fast shipping.",
            "Terrible experience, will not buy again.",
            "Highly recommend this to everyone.",
            "The product broke after one use.",
            "Fantastic! Exceeded my expectations.",
            "Mediocre at best, not impressed.",
            "Five stars for sure!"
        ]
    }
    df = pd.DataFrame(data)
    return df

def save_dataframe(df):
    if not os.path.exists('data'):
        os.makedirs('data')
        df.to_csv('data/data.csv', index=False)
        print("DataFrame saved to 'data/data.csv'.")

def process_data(k):
    df = pd.read_csv('data/data.csv')
    vectorizer = CountVectorizer(max_features=k)
    vectorized_data = vectorizer.fit_transform(df['review'])
    feature_names = vectorizer.get_feature_names_out()

    vectorized_df = pd.DataFrame(vectorized_data.toarray(), columns=feature_names)
    processed_df = pd.concat([df, vectorized_df], axis=1)

    processed_df.to_csv('data/processed_data.csv', index=False)
    print(f"Processed DataFrame saved to 'data/processed_data.csv' with top {k} features.")
    return processed_df

if __name__ == "__main__":
    df = create_dataframe()
    save_dataframe(df)
    k = 5  # Number of top features to extract
    processed_df = process_data(k)
    print(f"data shape: {df.shape}")
    print(f"Processed data shape: {processed_df.shape}")