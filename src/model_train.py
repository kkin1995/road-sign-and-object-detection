from dotenv import load_dotenv
import os
from datetime import datetime
from torch.utils.data import DataLoader
from classification_datasets import TrafficSignDataset
from torch.optim import Adam
import torch
import torch.nn as nn
from model import TrafficSignNet

EPOCHS = 2

load_dotenv()
data_dir = os.environ.get("DATA_DIR")
train_set = TrafficSignDataset("train_dataset.csv", data_dir)
val_set = TrafficSignDataset("val_dataset.csv", data_dir)
train_dataloader = DataLoader(train_set, 16)
print(f"Number of Training Batches: {len(train_dataloader)}")
val_dataloader = DataLoader(val_set, 16)
print(f"Number of Validation Batches: {len(val_dataloader)}")
clf = TrafficSignNet().to("cpu")
opt = Adam(clf.parameters(), lr=0.01)
loss_function = nn.CrossEntropyLoss()


def train_one_epoch():
    running_loss = 0.0
    for idx, batch in enumerate(train_dataloader):
        X, y = batch
        X, y = X.to("cpu"), y.to("cpu")
        yhat = clf(X)
        loss = loss_function(yhat, y)
        running_loss += loss

        opt.zero_grad()
        loss.backward()
        opt.step()

        running_loss += loss.item()
        last_loss = running_loss / len(train_dataloader)
        print(f"BATCH: {idx + 1} LOSS: {last_loss}")
        running_loss = 0.0
    return last_loss


timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
best_vloss = 1e10

for epoch in range(EPOCHS):
    print(f"EPOCH: {epoch + 1}")

    clf.train(True)
    avg_loss = train_one_epoch()

    running_vloss = 0.0

    clf.eval()
    with torch.no_grad():
        for idx, vdata in enumerate(val_dataloader):
            vinput, vlabel = vdata
            voutput = clf(vinput)
            vloss = loss_function(voutput, vlabel)
            running_vloss += vloss

    avg_vloss = running_vloss / (idx + 1)
    print(f"LOSS train: {avg_loss} valid: {avg_vloss}")

    if avg_vloss < best_vloss:
        best_vloss = avg_vloss
        model_path = f"model_{timestamp}_{epoch}"
        torch.save(clf.state_dict(), model_path)
