import torch
from torch.utils.data import DataLoader
from fake_news_dataset import FakeNewsDataset
from fake_news_dataloader import create_mini_batch
from transformers import BertTokenizer
from transformers import BertForSequenceClassification


def get_predictions(model, dataloader, compute_acc=False):
    predictions = None
    correct = 0
    total = 0
      
    with torch.no_grad():
        for data in dataloader:
            if next(model.parameters()).is_cuda:
                data = [t.to("cuda:0") for t in data if t is not None]
            
            # 別忘記前 3 個 tensors 分別為 tokens, segments 以及 masks
            # 且強烈建議在將這些 tensors 丟入 `model` 時指定對應的參數名稱
            tokens_tensors, segments_tensors, masks_tensors = data[:3]
            outputs = model(input_ids=tokens_tensors, 
                            token_type_ids=segments_tensors, 
                            attention_mask=masks_tensors)
            
            logits = outputs[0]
            _, pred = torch.max(logits.data, 1)
            
            if compute_acc:
                labels = data[3]
                total += labels.size(0)
                correct += (pred == labels).sum().item()
                
            if predictions is None:
                predictions = pred
            else:
                predictions = torch.cat((predictions, pred))
    
    if compute_acc:
        acc = correct / total
        return predictions, acc
    return predictions

if __name__ == "__main__":
    PRETRAINED_MODEL_NAME = "bert-base-chinese"
    NUM_LABELS = 3
    BATCH_SIZE = 64
    EPOCHS = 6  # 幸運數字
    
    #
    # load model and data
    #
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = BertForSequenceClassification.from_pretrained(PRETRAINED_MODEL_NAME, num_labels=NUM_LABELS)
    model = model.to(device)
    tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)
    trainset = FakeNewsDataset("train", tokenizer)
    trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, 
                             collate_fn=create_mini_batch)

    # # 讓模型跑在 GPU 上並取得訓練集的分類準確率
    # _, acc = get_predictions(model, trainloader, compute_acc=True)
    # print("classification acc:", acc)

    #
    # train model
    #
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    
    for epoch in range(EPOCHS):
        running_loss = 0.0
        for data in trainloader:
            tokens_tensors, segments_tensors, masks_tensors, labels = [t.to(device) for t in data]
            optimizer.zero_grad() 
            outputs = model(input_ids=tokens_tensors, 
                            token_type_ids=segments_tensors, 
                            attention_mask=masks_tensors, 
                            labels=labels)

            loss = outputs[0]
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
        _, acc = get_predictions(model, trainloader, compute_acc=True)
        print(f'[epoch {epoch+1}] loss: {running_loss:.3f}, acc: {acc:.3f}')

    torch.save(model.state_dict(), "fake_news.pth")
