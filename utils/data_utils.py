import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader

from .preprocessing_utils import text_cleansing, normalize_slang, slang_dict

class DocumentSentimentDataset(Dataset):

    LABEL2INDEX = {'negative': 0, 'neutral': 1, 'positive': 2}
    INDEX2LABEL = {0: 'negative', 1: 'neutral', 2: 'positive'}
    NUM_LABELS = 3

    def load_dataset(self, path):
        df = pd.read_csv(path, sep=',')
        df.columns = ['text', 'sentiment']
        df['sentiment'] = df['sentiment'].map(self.LABEL2INDEX)
        return df

    def __init__(self, dataset_path, tokenizer, no_special_token=False, *args, **kwargs):
        self.data = self.load_dataset(dataset_path)
        self.tokenizer = tokenizer
        self.no_special_token = no_special_token

    def __getitem__(self, index):
        data = self.data.iloc[index]
        text, sentiment = data['text'], data['sentiment']

        # Preprocessing
        text = normalize_slang(text, slang_dict)
        text = text_cleansing(text)

        subword = self.tokenizer.encode(text, add_special_tokens=not self.no_special_token)
        return np.array(subword), np.array(sentiment), text

    def __len__(self):
        return len(self.data)


class DocumentSentimentDataLoader(DataLoader):
    def __init__(self, max_seq_len=512, *args, **kwargs):
        super(DocumentSentimentDataLoader, self).__init__(*args, **kwargs)
        self.collate_fn = self._collate_fn
        self.max_seq_len = max_seq_len

    def _collate_fn(self, batch):
        batch_size = len(batch)
        max_seq_len = min(self.max_seq_len, max(len(data[0]) for data in batch))

        subword_batch = np.zeros((batch_size, max_seq_len), dtype=np.int64)
        mask_batch = np.zeros((batch_size, max_seq_len), dtype=np.int64)
        sentiment_batch = np.zeros((batch_size), dtype=np.int64)

        seq_lists = []
        for i, data in enumerate(batch):
            subword, sentiment, text = data
            subword_batch[i][:len(subword)] = subword[:max_seq_len]
            mask_batch[i][:len(subword)] = 1
            sentiment_batch[i] = sentiment
            seq_lists.append(text)

        return subword_batch, mask_batch, sentiment_batch, seq_lists
