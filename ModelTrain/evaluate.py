import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertForSequenceClassification
import pandas as pd
from sklearn.metrics import classification_report
import numpy as np
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
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            # pad_to_max_length=True,
            padding='max_length',
            return_token_type_ids=True
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


# prepare data
def prepare_data(train_path, tokenizer, max_len=150, batch_size=8):
    df = pd.read_csv(train_path, sep='\t')
    dataset = SuicideRiskDataset(df, tokenizer, max_len)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader

# Evaluate the model
def evaluate(model, data_loader, device):
    model.eval()
    predictions, true_labels = [], []

    with torch.no_grad():
        for _, data in enumerate(data_loader, 0):
            ids = data['ids'].to(device, dtype=torch.long)
            mask = data['mask'].to(device, dtype=torch.long)
            targets = data['targets'].to(device, dtype=torch.long)

            outputs = model(ids, mask)
            logits = outputs.logits
            logits = logits.detach().cpu().numpy()
            label_ids = targets.to('cpu').numpy()

            predictions.append(logits)
            true_labels.append(label_ids)

    # Calculate performance metrics
    predictions = [item for sublist in predictions for item in sublist]
    true_labels = [item for sublist in true_labels for item in sublist]
    predictions = np.argmax(predictions, axis=1)

    return classification_report(true_labels, predictions, digits=4)


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load tokenizer and model
    tokenizer = BertTokenizer.from_pretrained('hfl/chinese-bert-wwm-ext')

    model = BertForSequenceClassification.from_pretrained('hfl/chinese-bert-wwm-ext', num_labels=11)
    #The path to your trained model
    model.load_state_dict(torch.load('The path to your trained model'))
    model.to(device)

    test_loader = prepare_data('data/test_data.tsv', tokenizer)

    # evaluate the model
    report = evaluate(model, test_loader, device)
    print(report)
