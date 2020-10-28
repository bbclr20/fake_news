from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from fake_news_dataset import FakeNewsDataset
from transformers import BertTokenizer
import torch


def create_mini_batch(samples):
    tokens_tensors = [s[0] for s in samples]
    segments_tensors = [s[1] for s in samples]
    
    # 測試集有 labels
    if samples[0][2] is not None:
        label_ids = torch.stack([s[2] for s in samples])
    else:
        label_ids = None
    
    # zero pad 到同一序列長度
    tokens_tensors = pad_sequence(tokens_tensors, 
                                  batch_first=True)
    segments_tensors = pad_sequence(segments_tensors, 
                                    batch_first=True)
    
    # attention masks，將 tokens_tensors 裡頭不為 zero padding
    # 的位置設為 1 讓 BERT 只關注這些位置的 tokens
    masks_tensors = torch.zeros(tokens_tensors.shape, 
                                dtype=torch.long)
    masks_tensors = masks_tensors.masked_fill(
        tokens_tensors != 0, 1)
    
    return tokens_tensors, segments_tensors, masks_tensors, label_ids

if __name__ == "__main__":
    PRETRAINED_MODEL_NAME = "bert-base-chinese"
    tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)
    trainset = FakeNewsDataset("train", tokenizer)
    BATCH_SIZE = 2
    trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, 
                             collate_fn=create_mini_batch)
    for tokens_tensors, segments_tensors, masks_tensors, label_ids in trainloader:
        print(tokens_tensors)
        print(segments_tensors)
        break