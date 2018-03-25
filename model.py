import torch
import torch.nn as nn
import numpy as np
import os
from random import shuffle
from torch.utils.data import DataLoader


def args_packer():
    args = {}
    args['batch_size'] = 80  # paper
    args['rnn_dropout'] = 0.3  # paper
    args['epochs'] = 15
    args['embedding_dim'] = 300  # this and below need to be equal to tie weights
    args['hidden_dim'] = 300  # this and above need to be equal to tie weights
    args['num_layers'] = 3
    args['vocabulary_size'] = 33278  # calculated from vocab
    args['sequence_length'] = 50
    return args


def dataset_path(name):
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'dataset', name)


def to_variable(tensor):
    if torch.cuda.is_available():
        tensor = tensor.cuda()
    return torch.autograd.Variable(tensor)


def shuffle_and_concatenate(data):
    shuffle(data)
    flat_list = [item for sublist in data for item in sublist]
    list_array = np.asarray(flat_list)
    return list_array


class MyLSTMModel(nn.Module):

    def __init__(self, args):
        super(MyLSTMModel, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=args['vocabulary_size'], embedding_dim=args['embedding_dim'])
        self.rnn = nn.LSTM(input_size=args['embedding_dim'], hidden_size=args['hidden_dim'],
                           num_layers=args['num_layers'], dropout=args['rnn_dropout'])
        self.projection = nn.Linear(in_features=args['hidden_dim'], out_features=args['vocabulary_size'])
        self.projection.weight = self.embedding.weight  # tie up weight

    def forward(self, input):
        h = input
        h = self.embedding(h)
        h, state = self.rnn(h)
        h = self.projection(h)
        return h


class MyLoader(DataLoader):

    def __init__(self, data, batch_size, sequence_length):
        # we don't need the super constructor
        self.data = data[: int(data.shape[0] / batch_size) * batch_size]
        self.data = np.reshape(self.data, (batch_size, -1))
        self.data = self.data.T
        self.target = data[1: (int(data.shape[0] / batch_size) * batch_size) + 1]
        self.target = np.reshape(self.target, (batch_size, -1))
        self.target = self.target.T
        self.sequence_length = sequence_length

    def __iter__(self):
        pos = 0
        while self.data.shape[0] - pos > self.sequence_length:
            yield [torch.from_numpy(self.data[pos:pos + self.sequence_length, :]).long(),
                   torch.from_numpy(self.target[pos:pos + self.sequence_length, :]).long()]
            pos += self.sequence_length


def trainer():
    vocab = np.load(dataset_path('vocab.npy'))
    train = np.load(dataset_path('wiki.train.npy'))
    valid = np.load(dataset_path('wiki.valid.npy'))
    args = args_packer()
    model = MyLSTMModel(args)
    optim = torch.optim.Adam(model.parameters())
    loss_fn = torch.nn.CrossEntropyLoss()

    if torch.cuda.is_available():
        model = model.cuda()
        loss_fn = loss_fn.cuda()

    for epoch in range(args['epochs']):
        losses = []
        list_array = shuffle_and_concatenate(train)
        data_loader = MyLoader(list_array, args['batch_size'], args['sequence_length'])
        for (input, label) in data_loader:
            optim.zero_grad()
            prediction = model(to_variable(input))  # forward pass
            loss = loss_fn(prediction.view(-1, prediction.size(2)),
                           to_variable(label).view(-1))  # Compute losses
            loss.backward()  # Backpropagate the gradients
            losses.append(loss.data.cpu().numpy())
            optim.step()
        torch.save(model.state_dict(), 'model.pt')
        print("Epoch {} Loss: {:.4f}".format(epoch, np.asscalar(np.mean(losses))))

if __name__ == "__main__":
    trainer()


# TODO: Dropout in embedding layer, final output, word-vector
# TODO: change seq_length acc to paper


# For prediction return the scores for just the next word. Shape batch size, vocab.

# zeroing out hidden state every time?
