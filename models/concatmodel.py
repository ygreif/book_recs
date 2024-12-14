import torch
import torch.nn as nn

class ConcatModel(nn.Module):
    def __init__(self, num_users, num_books, embedding_dim, hidden_sizes=[128]):
        super(ConcatModel, self).__init__()
        self.user_embeddings = nn.Embedding(num_users, embedding_dim)
        self.book_embeddings = nn.Embedding(num_books, embedding_dim)

        prev_size = embedding_dim * 2
        layers = []
        for hidden_dim in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_dim))
 #           layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.LeakyReLU())
            prev_size = hidden_dim
        layers.append(nn.Linear(prev_size, 1))
        self.mlp = nn.Sequential(*layers)

        nn.init.xavier_uniform_(self.user_embeddings.weight)
        nn.init.xavier_uniform_(self.book_embeddings.weight)

    def forward(self, user_id, book_id):
        inp = torch.cat((self.user_embeddings(user_id), self.book_embeddings(book_id)), dim=1)
        raw_output = self.mlp(inp)
        return 1 + 4 * torch.sigmoid(raw_output)
