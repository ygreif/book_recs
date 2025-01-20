from itertools import chain

import torch
import torch.nn as nn

class ConcatModel(nn.Module):
    def __init__(self, num_users, num_books, embedding_dim, hidden_sizes=[128], dropout_prob=0, sparse=False):
        super(ConcatModel, self).__init__()
        print("Setting up embeddings", sparse)
        self.user_embeddings = nn.Embedding(num_users, embedding_dim, sparse=sparse)
        self.book_embeddings = nn.Embedding(num_books, embedding_dim, sparse=sparse)
        print("Done setting up embeddings")

        prev_size = embedding_dim * 2
        layers = []
        for hidden_dim in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_dim))
            layers.append(nn.LeakyReLU())
            if dropout_prob > 0:
                layers.append(nn.Dropout(dropout_prob))
            prev_size = hidden_dim
        layers.append(nn.Linear(prev_size, 1))
        self.mlp = nn.Sequential(*layers)

        nn.init.xavier_uniform_(self.user_embeddings.weight)
        nn.init.xavier_uniform_(self.book_embeddings.weight)

    def sparse_l2_penalty(self, user_id, book_id, weight_decay):
        user_embeds = self.user_embeddings(user_id)
        book_embeds = self.book_embeddings(book_id)
        return weight_decay * (user_embeds.pow(2).sum() + book_embeds.pow(2).sum())

    def forward(self, user_id, book_id):
        inp = torch.cat((self.user_embeddings(user_id), self.book_embeddings(book_id)), dim=1)
        raw_output = self.mlp(inp)
        return 1 + 4 * torch.sigmoid(raw_output)

    def sparse_parameters(self):
        return chain(self.user_embeddings.parameters(), self.book_embeddings.parameters())

    def dense_parameters(self):
        return list(self.mlp.parameters())
