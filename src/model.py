# import
import torch
import torch.nn as nn

# def


def calculate_dim(Lin, kernel_size, stride=1, padding=0, dilation=1):
    Lin = ((Lin+2*padding-dilation*(kernel_size-1)-1)/stride)+1
    return int(Lin)


# class
class DEEPCNN(nn.Module):
    def __init__(self, in_channels, kernel_size, in_dim, out_dim, hidden_dim, n_hidden, dropout=0.5):
        super(DEEPCNN, self).__init__()
        self.conv = nn.Conv1d(in_channels=in_channels,
                              out_channels=1, kernel_size=kernel_size)
        self.pool = nn.MaxPool1d(kernel_size=kernel_size, stride=1)
        in_dim = calculate_dim(calculate_dim(in_dim, 3), 3)
        in_sizes = [in_dim]+[hidden_dim]*(n_hidden-1)
        out_sizes = [hidden_dim]*n_hidden
        self.layers = nn.ModuleList([nn.Linear(in_size, out_size) for (
            in_size, out_size) in zip(in_sizes, out_sizes)])
        self.last_layer = nn.Linear(hidden_dim, out_dim)
        self.relu = nn.ReLU(inplace=True)
        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.pool(self.conv(x))
        x = x.squeeze(1)
        for layer in self.layers:
            x = self.relu(self.dropout(layer(x)))
        x = self.softmax(self.last_layer(x))
        return x


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim, n_hidden, dropout=0.5):
        super(MLP, self).__init__()
        in_sizes = [in_dim]+[hidden_dim]*(n_hidden-1)
        out_sizes = [hidden_dim]*n_hidden
        self.layers = nn.ModuleList([nn.Linear(in_size, out_size) for (
            in_size, out_size) in zip(in_sizes, out_sizes)])
        self.last_layer = nn.Linear(hidden_dim, out_dim)
        self.leakyrelu = nn.LeakyReLU(inplace=True)
        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        for layer in self.layers:
            x = self.leakyrelu(self.dropout(layer(x)))
        x = self.softmax(self.last_layer(x))
        return x
