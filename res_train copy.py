import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from res_model import *
from colorama import Fore
from metric import *
import pandas as pd

import os
import argparse
from hyp_evol import *


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
        return torch.as_tensor(self.X.iloc[idx].values).type(torch.FloatTensor), torch.as_tensor(self.Y.iloc[idx]).type(
            torch.LongTensor)


def train(dataloader, model, loss_fn, optimizer, device, positive_weight):
    model.train()

    Y = []
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        logit = model(X)
        positive_index = y == 1

        loss = loss_fn(logit, y)
        loss = (positive_weight * loss_fn(logit[positive_index], y[positive_index]) + loss_fn(logit[~positive_index], y[
            ~positive_index])) / len(X)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # if batch % 100 == 0:
        #     loss = loss.item()
        #     print(f"{Fore.GREEN + '[train]===>'} loss: {loss} {'' + Fore.RESET}")


def valid(dataloader, model, loss_fn, device):
    model.eval()
    model = model.to(device)
    num_dataset = len(dataloader.dataset)
    loss = 0

    with torch.no_grad():
        pred, Y = [], []
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)

            logit = model(X)
            loss += loss_fn(logit, y).item()

            pred.append(logit.argmax(1))
            Y.append(y)

        loss /= num_dataset

        pred = torch.cat(pred)
        Y = torch.cat(Y)
        print(f"{Fore.CYAN + '[valid]===>'} "
              f"loss: {loss}  acc: {100 * Accuracy(pred, Y)}%  precision: {Precision(pred, Y)}  recall: {Recall(pred, Y)}   fscore: {Fscore(pred, Y)}"
              f"{'' + Fore.RESET}")


# For updating learning rate
def update_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--evol', action='store_true',
                        help="hyperparameters auto evolve")
    # parser.add_argument('--model', help="train with last model",
    #                     type=str, default="./checkpoints/unevol/24_epoc.pt")

    return parser.parse_args()


if __name__ == '__main__':

    args = parse_args()

    torch.manual_seed(777)
    device = torch.device('cuda')

    batch_size, in_features, out_features = 30, 28, 2
    lr, positive_weight = 1e-3, 2.33
    epochs = 300

    model = ResNet(ResidualBlock, [2, 2, 2])
    model = model.to(device)
    loss_fn = nn.CrossEntropyLoss()
    loss_fn = loss_fn.to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_dataset = SeedDataset("./data/v1/train.csv")
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)

    valid_dataset = SeedDataset("./data/v1/valid.csv")
    valid_dataloader = DataLoader(valid_dataset, batch_size=1, shuffle=False)

    for t in range(epochs):
        print(f"{Fore.GREEN + '===>'} Epoch {t + 1} {'' + Fore.RESET}\n"
              "---------------------------------------")
        train(train_dataloader, model, loss_fn,
              optimizer, device, positive_weight)
        valid(valid_dataloader, model, loss_fn, device)
        torch.save(model.state_dict(), f"./checkpoints/{t}_epoc.pt")
