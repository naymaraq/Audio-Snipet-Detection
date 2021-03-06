import numpy as np
import  torch
from torch import nn
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader


class ISSIDataset(Dataset):
    def __init__(self, patterns, snipets, labels):

        self.patterns = patterns.reshape(-1, 1, patterns.shape[1], patterns.shape[2])
        self.snipets = snipets.reshape(-1, 1, snipets.shape[1], snipets.shape[2])
        self.labels = labels
        
        print(self.snipets.shape)
        print(self.labels.shape)
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
                       attention='dot'):
        super(ISSINet, self).__init__()

        self.n_filters = n_filters
        self.attention = attention
        self.conv = nn.Conv1d(1, self.n_filters, kernel_size=(2,2), stride=1)
        self.max_pool = nn.MaxPool2d(kernel_size=(2,2), stride=(1,2))

        embedding_dim = 441

        if attention == 'multiplicative':
            temp_tensor = torch.rand(embedding_dim, embedding_dim)
            nn.init.xavier_uniform_(temp_tensor, gain=1.0)
            self.attention_matrix = nn.Parameter(temp_tensor, requires_grad = True)
        elif attention == 'additive':
            pass

        self.linear_1 = nn.Linear(in_features=embedding_dim*2, out_features=64)
        self.linear_2 = nn.Linear(in_features=64, out_features=1)

        self.dropout = nn.Dropout(0.4)

    def convolve(self, input):
        return self.conv_1(input)

    def classifier(self, query, context):
        concat = torch.cat((query, context), dim=-1)
        concat = self.dropout(concat)
        out = torch.tanh(self.linear_1(concat))
        out = self.linear_2(out)
        return out

    def dot_product_attention(self, query, values):
        scores = torch.matmul(values, query.transpose(2,1))
        return scores

    def multiplicative_attention(self, query, values):
        w_s = torch.matmul(values, self.attention_matrix)
        scores = torch.matmul(w_s, query.transpose(2, 1))
        return scores

    def additive_attention(self):
        scores = None

    def forward(self, pattern, snipet):

        convolved_pattern = torch.relu(self.conv(pattern))
        convolved_pattern = convolved_pattern.transpose(1,2)
        convolved_snipet = torch.relu(self.conv(snipet))
        convolved_snipet = convolved_snipet.transpose(1,2)


        q = self.max_pool(convolved_pattern)
        vs = self.max_pool(convolved_snipet)

        q = torch.flatten(q, start_dim=2)
        vs = torch.flatten(vs, start_dim=2)

        if self.attention == 'dot':
            scores = self.dot_product_attention(q, vs)
        elif self.attention == "multiplicative":
            scores = self.multiplicative_attention(q, vs)
        else:
            print("Not allowed attention type")


        scores = torch.softmax(scores, dim=1)
        context = torch.mul(vs, scores)
        context = torch.sum(context, dim=1)
        q = torch.squeeze(q, dim=1)
        out = self.classifier(q, context)
        return torch.squeeze(out)


def train(model, device, criterion, train_loader, optimizer, epoch, log_interval):

    model.train()
    correct = 0
    for batch_idx, (pat, snip, target) in enumerate(train_loader):
        pat, snip, target = pat.to(device), snip.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(pat, snip)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(pat), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def test(model, device, criterion, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for pat, snip, target in test_loader:
            pat, snip, target = pat.to(device), snip.to(device), target.to(device)
            output = model(pat, snip)
            test_loss += criterion(output, target).item()  # sum up batch loss
            y_pred = [int(i>=0) for i in output.cpu().numpy()]
            y_true = [i for i in target.cpu().numpy()]
            correct += sum([i==j for i,j in zip(y_pred, y_true)])

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    return 100*correct / len(test_loader.dataset)

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

    print("Define parameters")
    seed = 2020
    save_model = True
    log_interval = 10
    epochs = 1000
    lr = 1e-3
    batch_size = 256
    use_cuda = True and torch.cuda.is_available()
    print("Use_cuda: ", use_cuda)

    torch.manual_seed(seed)
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'batch_size': batch_size}
    if use_cuda:
        kwargs.update({'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True},
                     )
    print("Construct train/test data loaders")
    train_loader, test_loader = get_data_loaders(**kwargs)

    print("Construct model")
    model = ISSINet(8, attention='multiplicative')
    model = model.to(device)

    optimizer = AdamW(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()


    best_model, best_acc = None, 0
    for epoch in range(1, epochs + 1):
        train(model, device, criterion, train_loader, optimizer, epoch, log_interval)
        acc = test(model, device, criterion, test_loader)
        if save_model == True:
            if acc > best_acc:
                best_acc = acc
                print("new best :" , best_acc)
                torch.save(model.state_dict(), "issi_net.pt")


if __name__ == '__main__':
    main()



