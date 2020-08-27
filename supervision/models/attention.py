import numpy as np
import  torch
from torch import nn
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data.dataset import Dataset,
from torch.utils.data import DataLoader


class ISSIDataset(Dataset):
    def __init__(self, patterns, snipets, labels):

        self.patterns = patterns
        self.snipets = snipets
        self.labels = labels

        assert len(self.patterns) == len(self.snipets)
        self.data_len = len(self.patterns)

    def __getitem__(self, index):

        pattern = torch.FloatTensor(self.patterns[index])
        snipet = torch.FloatTensor(self.snipets[index])
        label = self.labels[index]
        return (pattern, snipet, label)

    def __len__(self):
        return self.data_len


class ISSINet(nn.Module):

    def __init__(self, n_filters,
                       time_dim,
                       embedding_dim,
                       attention='dot'):
        super(ISSINet, self).__init__()

        self.n_filters = n_filters
        self.time_dim = time_dim
        self.embedding_dim = embedding_dim

        self.attention = attention
        self.conv_1 = nn.Conv1d(1, self.n_filters, kernel_size=(self.time_dim, self.embedding_dim), stride=1)
        self.linear_1 = nn.Linear(in_features=2* self.n_filters, out_features=1)

    def convolve(self, input):
        return self.conv_1(input)

    def dot_product_attention(self, query, values):
        query_shape = query.shape
        scores = torch.matmul(query.view(-1, query_shape[2], query_shape[1]), values)
        return scores

    def classifier(self, query, context):
        concat = torch.cat((query, context), dim=-1)
        out = self.linear_1(concat)
        return out

    def multiplicative_attention(self):
        scores = None

    def additive_attention(self):
        scores = None

    def forward(self, pattern, snipet):

        convolved_pattern = torch.squeeze(self.conv_1(pattern), dim=-1)
        convolved_snipet = torch.squeeze(self.conv_1(snipet), dim=-1)
        if self.attention == 'dot':
            scores = self.dot_product_attention(convolved_pattern, convolved_snipet)
        else:
            print("Not allowed attention type")

        scores = torch.softmax(scores, dim=-1)
        context = torch.mul(convolved_snipet, scores)
        context = torch.sum(context, dim=-1)

        convolved_pattern = torch.squeeze(convolved_pattern)
        out = self.classifier(convolved_pattern, context)
        return out


def train(model, device, criterion, train_loader, optimizer, epoch, log_interval):

    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def test(model, device, criterion, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def get_data_loaders(**kwargs):

    patterns =  np.load('tracks/snipets/patterns.npy', allow_pickle=True)
    snipets =  np.load('tracks/snipets/snipets.npy', allow_pickle=True)
    y =  np.load('tracks/snipets/labels.npy', allow_pickle=True)

    indecies = np.arange(len(y))
    np.random.shuffle(indecies)

    train_size = 0.8
    train_indecies = indecies[:int(len(y)*train_size)]
    test_indencies = indecies[int(len(y)*train_size):]

    train_dataset = ISSIDataset(patterns=patterns[train_indecies], snipets=snipets[train_indecies], labels=y[train_indecies])
    test_dataset = ISSIDataset(patterns=patterns[test_indencies], snipets=snipets[test_indencies], labels=y[test_indencies])
    train_loader = DataLoader(train_dataset, **kwargs)
    test_loader = DataLoader(test_dataset, **kwargs)

    return train_loader, test_loader


def main():

    seed = 2020
    save_model = True
    log_interval = 10
    epochs = 200
    lr = 1e-3
    batch_size = 128


    use_cuda = True and torch.cuda.is_available()
    torch.manual_seed(seed)
    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'batch_size': batch_size}
    if use_cuda:
        kwargs.update({'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True},
                     )

    train_loader, test_loader = get_data_loaders(**kwargs)

    model = ISSINet(8, 2, 128)
    model = model.to(device)

    optimizer = AdamW(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(1, epochs + 1):
        train(model, device, criterion, train_loader, optimizer, epoch, log_interval)
        test(model, device, criterion, test_loader)

    if save_model==True:
        torch.save(model.state_dict(), "issi_net.pt")


if __name__ == '__main__':
    main()