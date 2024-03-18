import torch
from transformers import BertTokenizer, BertForSequenceClassification
import pandas as pd
import numpy as np

def load_model(model_path, tokenizer_path):
    tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
    model = BertForSequenceClassification.from_pretrained(tokenizer_path, num_labels=11)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model, tokenizer

def predict(model, tokenizer, comment, max_len=150):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    # Encode text
    inputs = tokenizer.encode_plus(
        comment,
        None,
        add_special_tokens=True,
        max_length=max_len,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )
    ids = inputs['input_ids'].to(device, dtype=torch.long)
    mask = inputs['attention_mask'].to(device, dtype=torch.long)

    # predict
    with torch.no_grad():
        outputs = model(ids, mask)
        preds = torch.softmax(outputs.logits, dim=1)
        preds = preds.detach().cpu().numpy()

    return preds

if __name__ == "__main__":
    model_path = 'your model path'
    tokenizer_path = 'hfl/chinese-macbert-base'


    model, tokenizer = load_model(model_path, tokenizer_path)

    # example_comment
    inputComment = input("请输入预测文本————————————————\n")

    # Predict
    preds = predict(model, tokenizer, inputComment)
    preds = np.array(preds)

    # Find the index of the class with the highest probability
    max_prob_index = np.argmax(preds, axis=1)

    print("预测结果:", preds)

    print(f"预测级别为第 {max_prob_index[0] + 1} 级别")

