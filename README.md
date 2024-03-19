# Social media suicide fine-grained classification

This is the data and code for the paper: Social media suicide fine-grained classification

* Paper link: https://arxiv.org/abs/2309.03564

## Introduction

We introduced two annotated datasets from Chinese social media: a fine-grained suicide risk classification dataset based on the urgency of suicide risk and a binary dataset distinguishing between high and low suicide risk. Based on these datasets, we evaluated the performance of pre-trained models and, considering class imbalance, employed three data augmentation methods—synonym replacement, round-trip translation, and large model-generated data—to achieve class balance. The models were then re-trained on the balanced data.

The specific code directory structure is as follows:

- `suicideDataProcessing/`:This folder contains two suicide risk classification datasets, the code for auxiliary annotation using GPT-4, and the data and code for class supplementation using data augmentation methods.
  - `data/`:Inside this folder, the "binary" folder contains the binary classification data for high and low suicide risk, and "fine-grained" stores the fine-grained suicide risk classification data.
  - `LLMs auxiliary annotation/`: This folder contains code for auxiliary data annotation using GPT-4, corresponding to the two prompt strategies mentioned in the paper: Scene-definition Prompting and Scene-definition Prompting+Description of Levels.
  - `data augmentation/`: This folder contains traditional data augmentation methods (synonym replacement and back translation) and methods for data augmentation using large model-generated data. Included are the augmented data generated for this project, which are attached here for everyone's use.
- `ModelTrain/`: Code for model training. Using the BERT model as an example, this folder stores the code for training BERT from this paper, along with the corresponding baseline data and data after applying augmentation strategies.
  - `train.py`: After preparing the data, use this code for training the model.
  - `evaluate.py`: After the model has been trained, replace your model path and then proceed with the model evaluation.

All the pre-trained models mentioned in this paper and their corresponding source code are as follows:

• ERNIE 3.0: https://huggingface.co/nghuyong/ernie-3.0-base-zh 

• NeZha: https://huggingface.co/sijunhe/nezha-cn-base 

• ELECTRA: https://huggingface.co/hfl/chinese-electra-base-discriminator 

• RoBERTa: https://huggingface.co/hfl/chinese-roberta-wwm-ext 

• BERT: https://huggingface.co/hfl/chinese-bert-wwm-ext 

• Chinese MentalBERT: https://github.com/zwzzzQAQ/Chinese-MentalBERT 

• MacBERT: https://github.com/ymcui/MacBERT

To quickly swap among the pre-trained models listed above using Hugging Face's Transformers library, you can easily load and use any model by specifying its name. The library provides a unified API across a wide range of model architectures, allowing for seamless model interchange. Here's a basic example:
```python
from transformers import BertTokenizer, BertForSequenceClassification

# Specify the model you want to use
model_name = 'zwzzz/Chinese-MentalBERT' # Change this to the model you want

# Load the model and tokenizer
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels = 11)

# Now you can use the model and tokenizer for your task


```
## Citation

If this repository helps you, please cite this paper:

```bibtex
@misc{qi2024social,
  title={Social media suicide fine-grained classification},
  author={Qi, Hongzhi and Liu, Hanfei and Zhao, Qing and Li, Jianqiang and Zhai, Wei and Yang, Bing Xiang and Fu, Guanghui},
  year={2024},
  eprint={arXivID},
  archivePrefix={arXiv},
  primaryClass={cs.CL}
}

```
