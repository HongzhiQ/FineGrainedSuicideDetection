import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertForSequenceClassification
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score

class SuicideRiskDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.len = len(dataframe)
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __getitem__(self, index):
        comment = str(self.data.comment[index])
        comment = " ".join(comment.split())
        inputs = self.tokenizer.encode_plus(
            comment,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'targets': torch.tensor(self.data.myLabel[index], dtype=torch.long)
        }
    def __len__(self):
        return self.len
def prepare_data(train_path, tokenizer, max_len=150, batch_size=16):
    df = pd.read_csv(train_path, sep='\t')
    dataset = SuicideRiskDataset(df, tokenizer, max_len)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader

def compute_metrics(model, data_loader, device):
    model.eval()
    total_loss = 0
    total_samples = 0
    all_targets, all_predictions = [], []

    with torch.no_grad():
        for _, data in enumerate(data_loader, 0):
            ids = data['ids'].to(device, dtype=torch.long)
            mask = data['mask'].to(device, dtype=torch.long)
            targets = data['targets'].to(device, dtype=torch.long)

            outputs = model(ids, mask, labels=targets)
            loss = outputs[0]
            total_loss += loss.item() * ids.size(0)
            total_samples += ids.size(0)

            _, predicted = torch.max(outputs.logits, 1)
            all_targets.extend(targets.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    average_loss = total_loss / total_samples
    accuracy = accuracy_score(all_targets, all_predictions)
    report = classification_report(all_targets, all_predictions, output_dict=True, zero_division=1)
    return average_loss, accuracy, report


# train model
def train(model, train_loader, val_loader, optimizer, device, epochs=300,patience=15):
    # best_accuracy = 0.0
    # best_loss = float('inf')
    best_val_loss = float('inf')
    no_improvement = 0

    for epoch in range(epochs):
        model.train()
        total_loss, total_correct, total_samples = 0, 0, 0

        for _, data in enumerate(train_loader, 0):
            ids = data['ids'].to(device, dtype = torch.long)
            mask = data['mask'].to(device, dtype = torch.long)
            targets = data['targets'].to(device, dtype = torch.long)

            outputs = model(ids, mask, labels=targets)
            loss = outputs[0]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            _, predicted = torch.max(outputs.logits, 1)
            total_correct += (predicted == targets).sum().item()
            total_samples += targets.size(0)

        train_loss = total_loss / len(train_loader)
        train_accuracy = total_correct / total_samples

        val_loss, val_accuracy, val_report = compute_metrics(model, val_loader, device)
        print(
            f'Epoch: {epoch}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, '
            f'Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}'
            f'Val Precision: {val_report["weighted avg"]["precision"]:.4f}, Val Recall: {val_report["weighted avg"]["recall"]:.4f}, Val F1-Score: {val_report["weighted avg"]["f1-score"]:.4f}'
        )

        # Early Stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improvement = 0

            torch.save(model.state_dict(), 'The path to your training model.')
            print('Saved Best Model')
        else:
            no_improvement += 1
            if no_improvement == patience:
                print("Early stopping triggered")
                break


if __name__ == "__main__":

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    tokenizer = BertTokenizer.from_pretrained('hfl/chinese-bert-wwm-ext')

    model = BertForSequenceClassification.from_pretrained('hfl/chinese-bert-wwm-ext', num_labels=11)
    model.to(device)

    # prepare data
    train_loader = prepare_data('Your training set path', tokenizer)
    val_loader = prepare_data('your test set path', tokenizer)

    # optimizer
    #optimizer = torch.optim.AdamW(model.parameters(), lr=1e-6,weight_decay=1e-5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-6)
    # train model
    train(model, train_loader, val_loader, optimizer, device)
