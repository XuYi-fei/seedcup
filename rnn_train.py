# -*- coding: utf-8 -*-#

# -------------------------------------------------------------------------------
# Name:         rnn_train
# Description:
# Author:       梁超
# Date:         2021/11/1
# -------------------------------------------------------------------------------
# -*- coding: utf-8 -*-#

# -------------------------------------------------------------------------------
# Name:         conv_train
# Description:
# Author:       梁超
# Date:         2021/10/23
# -------------------------------------------------------------------------------
import torch
import torch.nn as nn
from rnn_model import RnnNet
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from conv_model import *
from colorama import Fore
from metric import *
import pandas as pd


class SeedDataset(Dataset):

    def __init__(self, annotations_file):
        super().__init__()
        self.data: pd.DataFrame = pd.read_csv(annotations_file)
        self.data: pd.DataFrame = self.data[self.data['label'].notna()]
        self.Y = self.data['label']
        self.X = self.data.drop(columns=['id', 'label']).fillna(value=-1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.as_tensor(self.X.iloc[idx].values).type(torch.LongTensor), torch.as_tensor(self.Y.iloc[idx]).type(
            torch.LongTensor)


def train(dataloader, model, loss_fn, optimizer, device, positive_weight):
    model.train()

    Y = []
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        X = torch.clip(X, 0, 10000)
        logit = model(X)
        positive_index = y == 1

        loss = (positive_weight * loss_fn(logit[positive_index], y[positive_index]) + loss_fn(logit[~positive_index], y[
            ~positive_index])) / len(X)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def valid(dataloader, model, loss_fn, device):
    model.eval()
    model = model.to(device)
    num_dataset = len(dataloader.dataset)
    loss = 0

    with torch.no_grad():
        pred, Y = [], []
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)
            X = torch.clip(X, 0, 10000)

            logit = model(X)
            loss += loss_fn(logit, y).item()

            pred.append(logit.argmax(1))
            Y.append(y)

        loss /= num_dataset

        pred = torch.cat(pred)
        Y = torch.cat(Y)
        print(f"{Fore.CYAN + '[valid]===>'} " \
              f"loss: {loss}  acc: {100 * Accuracy(pred, Y)}%  precision: {Precision(pred, Y)}  recall: {Recall(pred, Y)}   fscore: {Fscore(pred, Y)}" \
              f"{'' + Fore.RESET}")


if __name__ == '__main__':
    torch.manual_seed(777)
    device = torch.device('cpu')

    batch_size, in_features, out_features = 30, 33, 2
    # 原数据：1e-3 2.33
    lr, positive_weight = 1e-3, 2.33
    epochs = 150

    model = RnnNet(embedding_dim=256, num_embeddings=100000, output_size=2, hidden_size=16,
                   layer_num=1, bidirectional=True, device=device)
    model = model.to(device)
    loss_fn = nn.CrossEntropyLoss()
    loss_fn = loss_fn.to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_dataset = SeedDataset("./data/33_dimension/train.csv")
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_dataset = SeedDataset("./data/33_dimension/valid_banlanced.csv")
    valid_dataloader = DataLoader(valid_dataset, batch_size=1, shuffle=False)

    print("Start")
    for t in range(epochs):
        print(f"{Fore.GREEN + '===>'} Epoch {t + 1} {'' + Fore.RESET}\n"
              "---------------------------------------")
        train(train_dataloader, model, loss_fn, optimizer, device, positive_weight)
        valid(valid_dataloader, model, loss_fn, device)
        torch.save(model.state_dict(), f"./test_history/{t}_epoc.pt")

