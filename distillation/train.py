import os

os.system('pip install -r /workdir/requirements.txt')

import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, models
from modules import CKDDataset
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import torch.nn.functional as F
from pytorch_metric_learning import losses
from transformers import AutoConfig
from tqdm import tqdm
import wandb

os.mkdir('/workdir/model_dir')

teacher_emb_dim = int(os.environ.get('teacher_emb_dim'))
batch_size = int(os.environ.get('batch_size'))
epochs = int(os.environ.get('epochs'))
seed = int(os.environ.get('seed'))
learning_rate = float(os.environ.get('learning_rate'))
model_name = os.environ.get('model_name')
pooling_mode = os.environ.get('pooling_mode')
dataset_name = os.environ.get('dataset_name')
temperature = float(os.environ.get('temperature'))
blocks_count = int(os.environ.get('blocks_count'))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

config = AutoConfig.from_pretrained("d0rj/e5-small-en-ru")
config.num_hidden_layers = blocks_count

transformer = models.Transformer("d0rj/e5-small-en-ru", max_seq_length=512)
transformer.config = config
transformer._load_model("d0rj/e5-small-en-ru", config=config, cache_dir=None)
pooling = models.Pooling(transformer.get_word_embedding_dimension(), pooling_mode=pooling_mode)
norm = models.Normalize()
student_model = SentenceTransformer(modules=[transformer, pooling, norm], device=str(device))
student_tokenizer = student_model.tokenizer

print(student_model)

data = pd.read_csv(f'/workdir/data/{dataset_name}.csv')[['id', 'text']]

train_texts, test_texts = train_test_split(data, train_size=0.8, random_state=seed)

train_dataset = CKDDataset(train_texts)
test_dataset = CKDDataset(test_texts)

print(train_dataset[0])
print(train_dataset[1])

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

teacher_embeddings = torch.load('/workdir/data/teacher_embeddings_small.pt')

optimizer = torch.optim.Adam(student_model.parameters(), lr=learning_rate)
lr_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer)
loss_func = losses.NTXentLoss(temperature=temperature).to(device) # InfoNCE
# loss_func = torch.nn.MSELoss().to(device)

config = {
    'lr': learning_rate,
    'epochs': epochs,
    'model': model_name,
    'seed': seed,
    'batch_size': batch_size,
    'loss_function': str(loss_func),
    'temperature': temperature,
    'optimizer': str(optimizer),
    'lr_scheduler': str(lr_scheduler),
    'pooling_mode': pooling_mode,
    'dataset_name': dataset_name,
    'blocks_count': blocks_count
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
    student_inputs = student_tokenizer(texts_batch, padding=True, truncation=True, max_length=512,
                                       return_tensors='pt')
    return student_inputs


best_avg_cosine_sim = 0

for epoch in range(epochs):
    train_loss = 0
    student_model.train()
    for train_ids_batch, train_texts_batch in tqdm(train_dataloader):
        optimizer.zero_grad()

        tokenized_inputs = tokenize_inputs(train_texts_batch).to(device)
        t_embeddings = torch.cat([teacher_embeddings[idx] for idx in train_ids_batch]).detach().to(device)
        student_embeddings = student_model(tokenized_inputs)['sentence_embedding']
        embeddings = torch.cat((student_embeddings, t_embeddings)).to(device)

        labels = torch.tensor(list(map(int, train_ids_batch)), dtype=torch.int32)
        labels = torch.cat((labels, labels)).detach().to(device)
        loss = loss_func(embeddings, labels)
        # loss = loss_func(student_embeddings, t_embeddings)
        train_loss += loss.item()
        loss.backward()

        optimizer.step()

        del tokenized_inputs, t_embeddings, student_embeddings, embeddings, labels
        torch.cuda.empty_cache()

    avg_train_batch_loss = train_loss / len(train_dataloader)

    student_model.eval()

    test_loss = 0
    test_cosine_sum = 0
    for test_ids_batch, test_texts_batch in tqdm(test_dataloader):
        with torch.no_grad():
            tokenized_inputs = tokenize_inputs(test_texts_batch).to(device)
            t_embeddings = torch.cat([teacher_embeddings[idx] for idx in test_ids_batch]).detach().to(device)
            student_embeddings = student_model(tokenized_inputs)['sentence_embedding']
            embeddings = torch.cat((student_embeddings, t_embeddings)).to(device)

            labels = torch.tensor(list(map(int, test_ids_batch)), dtype=torch.int32)
            labels = torch.cat((labels, labels)).detach().to(device)
            loss = loss_func(embeddings, labels)
            # loss = loss_func(student_embeddings, t_embeddings)
            test_loss += loss.item()

            batch_cosine_sim = F.cosine_similarity(student_embeddings, t_embeddings, dim=1)
            test_cosine_sum += torch.sum(batch_cosine_sim).item()

            del tokenized_inputs, t_embeddings, student_embeddings, batch_cosine_sim, embeddings, labels
            torch.cuda.empty_cache()

    avg_test_batch_loss = test_loss / len(test_dataloader)
    avg_cosine_sim = test_cosine_sum / len(test_dataset)
    if avg_cosine_sim > best_avg_cosine_sim:
        best_avg_cosine_sim = avg_cosine_sim
        student_model.save(f'/workdir/model_dir/{model_name}')

    log_dict = {'avg train loss': avg_train_batch_loss,
                'avg test loss': avg_test_batch_loss,
                'avg test cosine similarity': avg_cosine_sim,
                'best avg test cosine similarity': best_avg_cosine_sim
                }
    run.log(log_dict, step=epoch)
    print(f'epoch: {epoch + 1}')
    for k, v in log_dict.items():
        print('\t' + f'{k}: {v}')
    lr_scheduler.step()

run.finish()
