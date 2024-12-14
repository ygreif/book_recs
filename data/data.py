import numpy as np
import pandas as pd
import torch

from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset
from torch.utils.data import random_split

def df(file):
    df = pd.read_csv(file, usecols=['Id', 'User_id', 'review/score', 'review/time'])
    # only keep most recent review by Id and User_id
    df = df.sort_values('review/time', ascending=False).drop_duplicates(['Id', 'User_id'])
    return df

class BookDataset(Dataset):
    def __init__(self, file, transform=None, target_transform=None):
        self._data = df(file)
        user_id_encoder = LabelEncoder()
        book_id_encoder = LabelEncoder()
        self._data['User_id'] = user_id_encoder.fit_transform(self._data['User_id'])
        self._data['Id'] = book_id_encoder.fit_transform(self._data['Id'])

        self.num_users = len(user_id_encoder.classes_)
        self.num_books = len(book_id_encoder.classes_)

        self._ids = torch.tensor(self._data['Id'].values, dtype=torch.int64)
        self._user_ids = torch.tensor(self._data['User_id'].values, dtype=torch.int64)
        self._times = torch.tensor(self._data['review/time'].values, dtype=torch.int64)
        self._scores = torch.tensor(self._data['review/score'].values, dtype=torch.int8)


        self._transform = transform
        self._target_transform = target_transform

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        features = (self._ids[idx], self._user_ids[idx], self._times[idx])
        if self._transform:
            features = self._transform(features)
        target = self._scores[idx]
        if self._target_transform:
            target = self._target_transform(target)
        return features, target

def split_pd(df):
    size = len(df)
    train_size = int(0.8 * size)
    test_size = int(.1 * size)
    val_size = size - train_size - test_size

    indices = np.random.permutation(size)
    train_indices = indices[:train_size]
    test_indices = indices[train_size:train_size+test_size]
    val_indices = indices[train_size+test_size:]

    return df.iloc[train_indices], df.iloc[test_indices], df.iloc[val_indices]

def split_data(data):
    size = len(data)
    train_size = int(0.8 * size)
    test_size = int(.1 * size)
    val_size = size - train_size - test_size
    return random_split(data, [train_size, test_size, val_size])

if __name__ == '__main__':
    data = BookDataset('./data/Books_rating.csv')
    import pdb;pdb.set_trace()
