import re
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

data = pd.read_csv("Rawdata/RawData-twoClass.tsv", sep='\t')

# 定义文本预处理函数
def preprocess_text(text):
    # text = re.sub(r'[^\w\s]', '', text)
    # text = re.sub(r'\d+', '', text)
    # text = text.replace(" ", "")
    text = text.replace("nbsp", "")
    text = text.replace("&;", "")
    return text

# 对数据中的 'text' 列应用预处理函数
data['comment'] = data['comment'].apply(preprocess_text)

# 分层抽样，划分数据集
train_df, test_df = train_test_split(data, test_size=0.2, stratify=data['twoClass'], random_state=888, shuffle=True)  # 80% 训练集，20% 测试集
train_df, val_df = train_test_split(train_df, test_size=0.25, stratify=train_df['twoClass'], random_state=888, shuffle=True)  # 将训练集再分为训练集和验证集

# 保存划分后的数据集
train_df.to_csv("twoClassData\\train_data_twoClass.tsv", sep='\t',index=False)
val_df.to_csv("twoClassData\\test_data_twoClass.tsv",  sep='\t',index=False)
test_df.to_csv("twoClassData\\val_data_twoClass.tsv",  sep='\t',index=False)
# 输出每个数据集中各类别的数量
train_counts = train_df['twoClass'].value_counts()
val_counts = val_df['twoClass'].value_counts()
test_counts = test_df['twoClass'].value_counts()

print("Training set class distribution:\n", train_counts)
print("\nValidation set class distribution:\n", val_counts)
print("\nTest set class distribution:\n", test_counts)
