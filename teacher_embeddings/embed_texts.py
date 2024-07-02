import os

os.system('pip install -r /workdir/requirements.txt')

import torch
from sentence_transformers import SentenceTransformer
import pandas as pd
from tensordict import TensorDict

dataset_name = os.environ.get('dataset_name')
batch_size = int(os.environ.get('batch_size'))
add_prefix = bool(os.environ.get('add_prefix'))
seed = int(os.environ.get('seed'))

print(f'dataset name: {dataset_name}')
print(f'batch size: {batch_size}')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
teacher_model = SentenceTransformer("d0rj/e5-small-en-ru", device=device)
tokenizer = teacher_model.tokenizer

print(f'device: {device}')

data = pd.read_csv(f'/workdir/data/{dataset_name}.csv').sample(frac=1, random_state=seed)[['id', 'text']].set_index(
    'id').to_dict(
    orient='index')

for k in data:
    data[k] = data[k]['text']

indices = []
texts = []

for k, v in data.items():
    indices.append(k)
    texts.append(v)

if add_prefix:
    for i, pair in enumerate(zip(indices, texts)):
        idx = pair[0]
        text = pair[1]
        if idx % 2 == 0:
            texts[i] = f'query: {text}'
        else:
            texts[i] = f'passage: {text}'

print(texts[:2])

print(f'indices length: {len(indices)}')
print(f'texts length: {len(texts)}')

embeddings_dict = {}


def tokenize(texts):
    return tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors='pt')


total_batches = len(texts) // batch_size + int(len(texts) % batch_size > 0)
print('start embedding process')
for idx in range(0, len(texts), batch_size):
    print(f'current batch: {idx // batch_size + 1} out of {total_batches}')
    tokenized_batch = tokenize(texts[idx:idx + batch_size]).to(device)
    with torch.no_grad():
        batch_embeddings = teacher_model(tokenized_batch)['sentence_embedding'].cpu()

    for tensor_idx, doc_idx in enumerate(indices[idx:idx + batch_size]):
        embeddings_dict[str(doc_idx)] = batch_embeddings[tensor_idx].view(1, -1)

tensor_dict = TensorDict(
    embeddings_dict,
    batch_size=[1]
)

torch.save(tensor_dict, '/workdir/teacher_embeddings_small.pt')
