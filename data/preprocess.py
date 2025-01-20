import argparse

import data
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def transform_singleton_users(df):
    # Replace users and books with only 1 rating to a default user and book
    user_counts = df['User_id'].value_counts()
    book_counts = df['Id'].value_counts()
    singleton_users = user_counts[user_counts == 1].index
    singleton_books = book_counts[book_counts == 1].index

    default_user_id = df['User_id'].max() + 1
    default_book_id = df['Id'].max() + 1

    df.loc[df['User_id'].isin(singleton_users), 'User_id'] = default_user_id
    df.loc[df['Id'].isin(singleton_books), 'Id'] = default_book_id
    return df


def preprocess(input, output):
    user_id_encoder = LabelEncoder()
    book_id_encoder = LabelEncoder()
    df = data.df(input)
    df['User_id'] = user_id_encoder.fit_transform(df['User_id'])
    df['Id'] = book_id_encoder.fit_transform(df['Id'])

    df.to_csv(output, index=False)

    train_df, test_df, val_df = data.split_pd(df)
    train_df = transform_singleton_users(train_df)
    test_df = transform_singleton_users(test_df)
    val_df = transform_singleton_users(val_df)

    train_df.to_csv('./data/train_encoded.csv', index=False)
    test_df.to_csv('./data/test_encoded.csv', index=False)
    val_df.to_csv('./data/val_encoded.csv', index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("output")
    args = parser.parse_args()

    preprocess(args.input, args.output)
