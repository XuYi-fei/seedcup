# -*- coding: utf-8 -*-#

# -------------------------------------------------------------------------------
# Name:         conv_train
# Description:
# Author:       梁超
# Date:         2021/10/23
# -------------------------------------------------------------------------------
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from XXNet_model import *
from colorama import Fore
from metric import *
import pandas as pd
import os
import argparse


rate = "0.5"  # 默认为6：4的正负样本比例，若要改为1：1则取rate=“0.5”


class SeedDataset(Dataset):

    def __init__(self, annotations_file):
        super().__init__()
        self.data: pd.DataFrame = pd.read_csv(annotations_file)
        self.data: pd.DataFrame = self.data[self.data['label'].notna()]
        self.Y = self.data['label']
        self.X = self.data.drop(columns=['id', 'label']).fillna(value=-1)
        self.data = torch.ones(len(self.X), 3, 33, 33)
        # self.transform = transforms.Resize([224, 224])
        for i in range(len(self.X)):
            x = torch.as_tensor(self.X.iloc[i].values).type(torch.FloatTensor).unsqueeze(0)
            y = x.permute(1, 0)
            self.data[i, :, ] = torch.mm(x, y)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], torch.as_tensor(self.Y.iloc[idx]).type(
            torch.LongTensor)


def train(dataloader, model, loss_fn, optimizer, device, positive_weight):
    model.train()

    Y = []
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        logit = model(X)
        positive_index = y == 1

        loss = loss_fn(logit, y)
        # loss = (positive_weight * loss_fn(logit[positive_index], y[positive_index]) + loss_fn(logit[~positive_index], y[
        #     ~positive_index])) / len(X)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # if batch % 100 == 0:
        loss = loss.item()
        print(f"{Fore.GREEN + '[train]===>'} loss: {loss} {'' + Fore.RESET}")


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
        return loss, Fscore(pred, Y)

# For updating learning rate
def update_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=str,
                        default="../../data/33_normalized/train.csv")
    parser.add_argument('--valid', type=str,
                        default=f"../../data/33_normalized/valid.csv")
    parser.add_argument('--in_feature', type=int,
                        default=33)
    parser.add_argument('--device', type=str,
                        default='cpu')

    return parser.parse_args()


if __name__ == '__main__':

    args = parse_args()

    torch.manual_seed(777)
    device = torch.device(args.device)

    batch_size, in_features, out_features = 8, args.in_feature, 2
    # 原数据：1e-3 2.33
    lr, positive_weight = 1e-3, 1.33
    epochs = 150

    model = FastCTNetPlus()
    model = model.to(device)
    loss_fn = nn.CrossEntropyLoss()
    loss_fn = loss_fn.to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_dataset = SeedDataset(args.train)
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)

    valid_dataset = SeedDataset(args.valid)
    valid_dataloader = DataLoader(valid_dataset, batch_size=1, shuffle=False)

    if(os.path.isdir("../../checkpoints") == 0):
        os.mkdir("../../checkpoints")
    if(os.path.isdir("../../checkpoints/XXNet") == 0):
        os.mkdir("../../checkpoints/XXNet")

    loss = 10000
    fscore = 0
    for t in range(epochs):
        print(f"{Fore.GREEN + '===>'} Epoch {t + 1} {'' + Fore.RESET}\n"
              "---------------------------------------")
        if t == epochs // 5:
            optimizer = optim.Adam(model.parameters(), lr=lr/5)
        if t == epochs // 4:
            optimizer = optim.Adam(model.parameters(), lr=lr/10)
        if t == epochs // 2:
            optimizer = optim.Adam(model.parameters(), lr=lr/20)
        train(train_dataloader, model, loss_fn,
              optimizer, device, positive_weight)
        new_loss, new_fscore = valid(valid_dataloader, model, loss_fn, device)
        # if new_loss < loss or new_fscore > fscore:
        loss = new_loss
        fscore = new_fscore
        torch.save(model.state_dict(), f"../../checkpoints/XXNet/{t}_epoc_loss_{loss}_fscore_{fscore}.pt")
