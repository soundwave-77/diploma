from torch.utils.data import Dataset

class CKDDataset(Dataset):
    def __init__(self, data):
        super().__init__()
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ids = self.data.iloc[idx]['id']
        text = self.data.iloc[idx]['text']
        if ids % 2 == 0:
            text = f'query: {text}'
        else:
            text = f'passage: {text}'
        return str(ids), text
