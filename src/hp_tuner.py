import wandb
import pandas as pd
from custom_data import CustomDataset
from transformers import BertTokenizer
from torch.utils.data import DataLoader
from model import BERTClassifier
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
import torch
import time, datetime
import random
import numpy as np
import argparse

wandb.login()


class HyperparameterTuner:
    def __init__(self, data_path):
        self.sweep_config = {
            'method': 'random',
            'metric': {
                'name': 'loss',
                'goal': 'minimize'
            },
            'parameters': {
                'optimizer': {
                    'values': ['adam', 'sgd']
                },
                'learning_rate': {
                    'values': [5e-5, 3e-5, 2e-5]
                },
                'drop_out': {
                    'values': [0.3, 0.4, 0.5]
                },
                'epochs': {
                    'values': [2, 3, 4]
                },
                'batch_size': {
                    'values': [16, 32]
                }
            }
        }
        self.data_path = data_path
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.max_len = 250

    def get_data(self, data_path):
        batch_size = wandb.config.batch_size
        data = pd.read_csv(data_path)
        data_set = data.sample(frac=0.4, random_state=200).reset_index(drop=True)
        train_set = data_set.sample(frac=0.9, random_state=200).reset_index(drop=True)
        valid_set = data_set.drop(train_set.index).reset_index(drop=True)

        training_set = CustomDataset(train_set, self.tokenizer, self.max_len)
        validation_set = CustomDataset(valid_set, self.tokenizer, self.max_len)

        train_loader = DataLoader(training_set, batch_size=batch_size, shuffle=True, num_workers=0)
        validation_loader = DataLoader(validation_set, batch_size=batch_size, shuffle=False, num_workers=0)
        return train_loader, validation_loader

    def ret_optim(self,model):
        print('Learning_rate = ', wandb.config.learning_rate)
        optimizer = AdamW(model.parameters(),
                          lr=wandb.config.learning_rate,
                          eps=1e-8
                          )
        return optimizer

    def ret_scheduler(self, train_dataloader, optimizer):
        epochs = wandb.config.epochs
        print('epochs =>', epochs)
        # Total number of training steps is [number of batches] x [number of epochs].
        # (Note that this is not the same as the number of training samples).
        total_steps = len(train_dataloader) * epochs

        # Create the learning rate scheduler.
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=0,  # Default value in run_glue.py
                                                    num_training_steps=total_steps)
        return scheduler

    def format_time(self, elapsed):
        elapsed_rounded = int(round((elapsed)))
        return str(datetime.timedelta(seconds=elapsed_rounded))

    def loss_fn(self, outputs, targets):
        return torch.nn.BCEWithLogitsLoss()(outputs, targets)

    def flat_accuracy(self, preds, labels):
        # pred_flat = np.argmax(preds, axis=1).flatten()
        # labels_flat = labels.flatten()
        return np.sum(preds == labels) / len(labels)

    def train(self):
        user = "yatika-paliwal-06"
        project = "BERT-ML-classifcation"
        display_name = "experiment-2023-04-12"

        wandb.init(entity=user, project=project, name=display_name)
        val_targets = []
        val_outputs = []
        training_stats = []
        seed_val = 42
        torch.manual_seed(seed_val)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = BERTClassifier(wandb.config.drop_out)
        model.to(device)
        train_dataloader, validation_dataloader = self.get_data(self.data_path)
        optimizer = self.ret_optim(model)  # optimizer with weight decay and learning rate
        scheduler = self.ret_scheduler(train_dataloader, optimizer)

        total_t0 = time.time()
        epochs = wandb.config.epochs
        for epoch_i in range(0, epochs):
            print("")
            print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
            print('Training...')

            t0 = time.time()
            total_train_loss = 0
            total_eval_loss = 0
            model.train()
            for step, batch in enumerate(train_dataloader):
                if step % 40 == 0 and not step == 0:
                    elapsed = self.format_time(time.time() - t0)
                    print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

                    ids = batch['ids'].to(device, dtype=torch.long)
                    mask = batch['mask'].to(device, dtype=torch.long)
                    token_type_ids = batch['token_type_ids'].to(device, dtype=torch.long)
                    targets = batch['targets'].to(device, dtype=torch.float)
                    model.zero_grad()
                    outputs = model(ids, mask, token_type_ids)
                    loss = self.loss_fn(outputs, targets)
                    wandb.log({'train_batch_loss': loss.item()})
                    total_train_loss += loss.item()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()
            avg_train_loss = total_train_loss / len(train_dataloader)
            training_time = self.format_time(time.time() - t0)  # Measure how long this epoch took.
            wandb.log({'avg_train_loss': avg_train_loss})
            print("")
            print("  Average training loss: {0:.2f}".format(avg_train_loss))
            print("  Training epcoh took: {:}".format(training_time))

            print("")
            print("Running Validation...")
            t0 = time.time()
            model.eval()
            for batch in validation_dataloader:
                ids = batch['ids'].to(device, dtype=torch.long)
                mask = batch['mask'].to(device, dtype=torch.long)
                token_type_ids = batch['token_type_ids'].to(device, dtype=torch.long)
                targets = batch['targets'].to(device, dtype=torch.float)

                with torch.no_grad():
                    outputs = model(ids, mask, token_type_ids)
                loss = self.loss_fn(outputs, targets)
                total_eval_loss += loss.item()
                val_targets.extend(targets.cpu().detach().numpy().tolist())
                val_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())
            validation_time = self.format_time(time.time() - t0)
            avg_val_loss = total_eval_loss / len(validation_dataloader)
            wandb.log({'avg_val_loss': avg_val_loss})
            print("  Validation Loss: {0:.2f}".format(avg_val_loss))
            print("  Validation took: {:}".format(validation_time))

            training_stats.append(
                {
                    'epoch': epoch_i + 1,
                    'Training Loss': avg_train_loss,
                    'Valid. Loss': avg_val_loss,
                    'Training Time': training_time,
                    'Validation Time': validation_time
                }
            )

        print("")
        print("Training complete!")

        print("Total training took {:} (h:mm:ss)".format(self.format_time(time.time() - total_t0)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Hyperparameter tuning utility",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data-path', type=str, help='Path to data file')

    config = parser.parse_args()
    tuner = HyperparameterTuner(config.data_path)
    sweep_id = wandb.sweep(tuner.sweep_config)
    wandb.agent(sweep_id, function=tuner.train)
