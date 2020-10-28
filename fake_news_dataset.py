from torch.utils.data import Dataset
import torch
from transformers import BertTokenizer
import pandas as pd
import os

    
class FakeNewsDataset(Dataset):
    
    def __init__(self, mode, tokenizer, data_dir="dataset/FakeNews"):
        assert mode in ["train", "test"]
        self.mode = mode
        filename = os.path.join(data_dir, f"{mode}.tsv")
        self.df = pd.read_csv(filename, sep="\t").fillna("")
        self.len = len(self.df)
        self.label_map = {'agreed': 0, 'disagreed': 1, 'unrelated': 2}
        self.tokenizer = tokenizer

    def __getitem__(self, idx):
        if self.mode == "test":
            text_a, text_b = self.df.iloc[idx, :2].values
            label_tensor = None
        else:
            text_a, text_b, label = self.df.iloc[idx, :].values
            label_id = self.label_map[label]
            label_tensor = torch.tensor(label_id)
            
        # 建立第一個句子的 BERT tokens 並加入分隔符號 [SEP]
        word_pieces = ["[CLS]"]
        tokens_a = self.tokenizer.tokenize(text_a)
        word_pieces += tokens_a + ["[SEP]"]
        len_a = len(word_pieces)
        
        # 第二個句子的 BERT tokens
        tokens_b = self.tokenizer.tokenize(text_b)
        word_pieces += tokens_b + ["[SEP]"]
        len_b = len(word_pieces) - len_a
        
        # 將整個 token 序列轉換成索引序列
        ids = self.tokenizer.convert_tokens_to_ids(word_pieces)
        tokens_tensor = torch.tensor(ids)
        
        # 將第一句包含 [SEP] 的 token 位置設為 0，其他為 1 表示第二句
        segments_tensor = torch.tensor([0] * len_a + [1] * len_b, 
                                        dtype=torch.long)
        
        return (tokens_tensor, segments_tensor, label_tensor)
    
    def __len__(self):
        return self.len

if __name__ == "__main__":
    PRETRAINED_MODEL_NAME = "bert-base-chinese"
    tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)

    trainset = FakeNewsDataset("train", tokenizer=tokenizer)
    print(f"tokens_tensor: {trainset[0][0]}")
    print(f"segments_tensor: {trainset[0][1]}")
    print(f"label_tensor: {trainset[0][2]}")
    