import os

os.system('pip install -r /workdir/requirements.txt')

import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, models
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F
from pytorch_metric_learning import losses
import wandb

os.mkdir('/workdir/model_dir')

batch_size = int(os.environ.get('batch_size'))
epochs = int(os.environ.get('epochs'))
seed = int(os.environ.get('seed'))
learning_rate = float(os.environ.get('learning_rate'))
model_name = os.environ.get('model_name')
distil_model_name = os.environ.get('distil_model_name')
dataset_name = os.environ.get('dataset_name')
temperature = float(os.environ.get('temperature'))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = SentenceTransformer(f'/workdir/models/{distil_model_name}', device=str(device))
model.train()
tokenizer = model.tokenizer


class NLIDataset(torch.utils.data.Dataset):
    def __init__(self, positive_samples, negative_samples, batch_size=16):
        self.positive_samples = positive_samples
        self.negative_samples = negative_samples
        self.negatives_count = batch_size - 1

    def __len__(self):
        return len(self.positive_samples)

    def __getitem__(self, idx):
        premise = f"query: {self.positive_samples['premise'].iloc[idx]}"
        positive = f"passage: {self.positive_samples['hypothesis'].iloc[idx]}"

        negatives = list(map(lambda x: f"passage: {x}",
                             self.negative_samples['hypothesis'].sample(n=self.negatives_count).values.tolist()))

        texts = [premise, positive] + negatives
        labels = [0, 0] + list(range(1, len(negatives) + 1))
        return labels, texts


train_data = pd.read_csv(f'/workdir/data/{dataset_name}.csv')
positive_samples = train_data[train_data['label'] == 0]
negative_samples = train_data[train_data['label'] == 1]

train_dataset = NLIDataset(positive_samples, negative_samples, batch_size)

train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=1, collate_fn=lambda x: x[0])

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
lr_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer)
# loss_func = InfoNCELoss(temperature=temperature).to(device)
loss_func = losses.NTXentLoss(temperature=temperature).to(device)
# loss_func = torch.nn.MSELoss().to(device)

config = {
    'lr': learning_rate,
    'epochs': epochs,
    'model name': model_name,
    'distil model name': distil_model_name,
    'seed': seed,
    'batch_size': batch_size,
    'loss_function': str(loss_func),
    'temperature': temperature,
    'optimizer': str(optimizer),
    'lr_scheduler': str(lr_scheduler),
    'dataset_name': dataset_name
}

print(config)

wandb.login()
run = wandb.init(
    project='embedders-distillation',
    name=f'{model_name}',
    config=config,
    save_code=True
)


def tokenize_inputs(texts_batch):
    student_inputs = tokenizer(texts_batch, padding=True, truncation=True, max_length=512,
                               return_tensors='pt')
    return student_inputs


best_avg_batch_loss = float('inf')
for epoch in range(epochs):
    loss_value = 0
    for labels_batch, texts_batch in tqdm(train_dataloader):
        optimizer.zero_grad()

        tokenized_inputs = tokenize_inputs(texts_batch).to(device)
        embeddings = model(tokenized_inputs)['sentence_embedding']
        labels = torch.tensor(labels_batch, dtype=torch.int16).to(device)
        # embeddings = torch.vstack((student_embeddings, t_embeddings)).to(device)

        # labels = torch.tensor(list(map(int, train_ids_batch)), dtype=torch.int32).repeat(2).to(device)
        loss = loss_func(embeddings, labels)
        loss_value += loss.item()
        loss.backward()

        optimizer.step()

        # del tokenized_inputs, embeddings, labels
        # torch.cuda.empty_cache()

    avg_batch_loss = loss_value / len(train_dataset)
    if avg_batch_loss < best_avg_batch_loss:
        best_avg_batch_loss = avg_batch_loss
        model.save(f'/workdir/model_dir/{model_name}')
    run.log({'avg train loss': avg_batch_loss,
             },
            step=epoch)
    lr_scheduler.step()

run.finish()
