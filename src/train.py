import numpy as np
from model import BERTClassifier
from custom_data import CustomDataset
from pathlib import Path
import torch
import logging
import pandas as pd
import argparse
from transformers import BertTokenizer
from torch.utils.data import DataLoader
import shutil
import json
import time
import wandb

wandb.init(project='BERT-ML-classifcation', entity='yatika-paliwal-06')

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# device = torch.device("mps")
device = torch.device("cpu")
model = BERTClassifier()
model.to(device)

log = logging.getLogger(__name__)


def main(cfg):
    train_loader, validation_loader = get_data(data_path=cfg.data_path, max_len=cfg.max_len,
                                               train_batch_size=cfg.train_batch_size,
                                               valid_batch_size=cfg.valid_batch_size)
    train(cfg.n_epoch, train_loader, validation_loader, cfg.learning_rate,
          cfg.checkpoint_path, cfg.best_model_path)


def train(n_epoch: int, training_loader, validation_loader, learning_rate: float,
          checkpoint_path: Path, best_model_path: Path):
    import torch
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        x = torch.ones(1, device=device)
        print(x)
    else:
        print("MPS device not found.")
    device = torch.device("cpu")

    training_start_time = time.time()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)
    min_val_loss = np.Inf
    val_targets = []
    val_outputs = []

    for epoch in range(1, n_epoch + 1):
        running_loss = 0
        train_loss = 0
        val_loss = 0
        model.train()
        print("**********Epoch {}: Model Training starts***********".format(epoch))
        for batch_idx, data in enumerate(training_loader):
            ids = data['ids'].to(device, dtype=torch.long)
            mask = data['mask'].to(device, dtype=torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
            targets = data['targets'].to(device, dtype=torch.float)

            outputs = model(ids, mask, token_type_ids)
            optimizer.zero_grad()  # Making gradients to zero
            loss = loss_fn(outputs, targets)
            running_loss += loss.item()
            if batch_idx % 2000 == 1999:
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, batch_idx + 1, running_loss / 2000))
                wandb.log({'epoch': epoch + 1, 'loss': running_loss / 2000})
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()  # performs the single optimization step
            train_loss = train_loss + ((1 / (batch_idx + 1)) * (loss.item() - train_loss))

        print('*******Epoch {}: Training ends**********'.format(epoch))
        print('******Epoch {}: Model Evaluation Starts******'.format(epoch))

        model.eval()

        with torch.no_grad():
            for batch_idx, data in enumerate(validation_loader, 0):
                ids = data['ids'].to(device, dtype=torch.long)
                mask = data['mask'].to(device, dtype=torch.long)
                token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
                targets = data['targets'].to(device, dtype=torch.float)

                outputs = model(ids, mask, token_type_ids)
                loss = loss_fn(outputs, targets)
                val_loss = val_loss + ((1 / (batch_idx + 1)) * (loss.item() - val_loss))
                val_targets.extend(targets.cpu().detach().numpy().tolist())
                val_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())

        print('******Epoch {}: Validation Ends*******'.format(epoch))
        train_loss = train_loss / len(training_loader)  # calculate average losses
        val_loss = val_loss / len(validation_loader)
        print('*******Average Training and Validation loss for Epoch ', epoch, ' ', train_loss, ' ', val_loss)

        # create checkpoint for saving important data
        checkpoint = {
            'epoch': epoch + 1,
            'valid_loss_min': val_loss,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }
        save_ckp(checkpoint, False, checkpoint_path, best_model_path)
        if min_val_loss >= val_loss:
            log.info('Validation loss decreased from {:.6f}->{:.6f}. Saving model....'.format(min_val_loss, val_loss))
            save_ckp(checkpoint, True, checkpoint_path, best_model_path)
            min_val_loss = val_loss
        print('Everything done for {:.6f}'.format(epoch))
        print('Finished training in {:.2f}s'.format(time.time() - training_start_time))

    return model


def loss_fn(outputs, targets):
    return torch.nn.BCEWithLogitsLoss()(outputs, targets)


def save_ckp(state, is_best, checkpoint_path, best_model_path):
    torch.save(state, checkpoint_path)
    if is_best:
        shutil.copy(checkpoint_path, best_model_path)


def get_data(data_path: Path, max_len, train_batch_size, valid_batch_size):
    train_size = 0.8
    data = pd.read_csv(data_path)
    train_set = data.sample(frac=train_size, random_state=200).reset_index(drop=True)
    valid_set = data.drop(train_set.index).reset_index(drop=True)

    training_set = CustomDataset(train_set, tokenizer, max_len)
    validation_set = CustomDataset(valid_set, tokenizer, max_len)

    train_loader = DataLoader(training_set, batch_size=train_batch_size, shuffle=True, num_workers=0)
    validation_loader = DataLoader(validation_set, batch_size=valid_batch_size, shuffle=False, num_workers=0)
    return train_loader, validation_loader


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--filepath", required=True, type=Path)
    args = parser.parse_args()

    with open(args.filepath) as param_file:
        t_args = argparse.Namespace()
        t_args.__dict__.update(json.load(param_file))
        args = parser.parse_args(namespace=t_args)
        print(args)
    main(args)
