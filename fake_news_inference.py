import torch
from torch.utils.data import DataLoader
from fake_news_dataset import FakeNewsDataset
from fake_news_dataloader import create_mini_batch
from transformers import BertTokenizer, BertForSequenceClassification
from fake_news_train import get_predictions
import pandas as pd


PRETRAINED_MODEL_NAME = "bert-base-chinese"
NUM_LABELS = 3
MODEL_PATH = "fake_news.pth"

tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)
    
testset = FakeNewsDataset("test", tokenizer=tokenizer)
testloader = DataLoader(testset, batch_size=256, collate_fn=create_mini_batch)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = BertForSequenceClassification.from_pretrained(PRETRAINED_MODEL_NAME, num_labels=NUM_LABELS)
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()
model = model.to(device)
predictions = get_predictions(model, testloader)

index_map = {v: k for k, v in testset.label_map.items()}

# 生成 Kaggle 繳交檔案
df = pd.DataFrame({"Category": predictions.tolist()})
df['Category'] = df.Category.apply(lambda x: index_map[x])
df_pred = pd.concat([testset.df.loc[:, ["Id"]], 
                          df.loc[:, 'Category']], axis=1)
df_pred.to_csv('prec_training_samples.csv', index=False)
df_pred.head()
