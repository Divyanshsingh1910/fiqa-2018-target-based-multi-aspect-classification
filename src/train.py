"""
Title: Barclays IB Analytics Case Study Submission
---------------------------------------------------

> Divyansh
> IIT Kanpur
> mail: divyansh21@iitk.ac.in
> roll: 210355


# Contents
    1. Problem Statement
    2. Solution Approach
    3. Running Instructions
    4. Evaluaton Results
    5. Code
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
train = pd.read_csv("train.csv")




plt.figure(figsize=(12, 6))
sns.histplot(data=train, x='sentiment_score', bins=30, kde=True)
plt.axvline(x=train['sentiment_score'].mean(), color='red', linestyle='--', label=f'Mean: {train["sentiment_score"].mean():.3f}')
plt.title('Distribution of Sentiment Scores', fontsize=14, pad=15)
plt.xlabel('Sentiment Score', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)

print("\nSentiment Score Summary Statistics:")
print(train['sentiment_score'].describe())

plt.tight_layout()
plt.show()





fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
format_counts = train['format'].value_counts()
format_counts.plot(kind='bar', ax=ax1)
ax1.set_title('Number of Articles by Format', fontsize=12, pad=15)
ax1.set_xlabel('Format')
ax1.set_ylabel('Count')

sns.boxplot(data=train, x='format', y='sentiment_score', ax=ax2)
ax2.set_title('Sentiment Score Distribution by Format', fontsize=12, pad=15)
ax2.set_xlabel('Format')
ax2.set_ylabel('Sentiment Score')

print("\nFormat-wise Sentiment Analysis:")
format_stats = train.groupby('format')['sentiment_score'].agg([
    'count', 'mean', 'std', 'min', 'max'
]).round(3)
print(format_stats)

plt.tight_layout()
plt.show()






plt.figure(figsize=(8, 6))
sns.boxplot(data=train, x='label', y='sentiment_score')
plt.title('Sentiment Score Distribution by Label', fontsize=12, pad=15)
plt.xlabel('Label')
plt.ylabel('Sentiment Score')

print("\nLabel-wise Sentiment Analysis:")
label_stats = train.groupby('label')['sentiment_score'].agg([
    'count', 'mean', 'std'
]).round(3)
print(label_stats)

plt.tight_layout()
plt.show()


aspect_list = []
aspect_sentiments = {}

for i in range(len(train)):
    aspects = train.iloc[i]['aspects'][2:-2].split('/')
    aspects = np.unique(aspects).tolist()
    sentiment = train.iloc[i]['sentiment_score']

    for aspect in aspects:
        aspect_list.append(aspect)
        if aspect not in aspect_sentiments:
            aspect_sentiments[aspect] = []
        aspect_sentiments[aspect].append(sentiment)

aspect_counts = pd.Series(aspect_list).value_counts()
aspect_mean_sentiment = {aspect: np.mean(sentiments)
                        for aspect, sentiments in aspect_sentiments.items()}

top_20_aspects = aspect_counts.head(25)
# top_20_aspects = aspect_counts.head(20)
top_20_sentiments = {aspect: aspect_mean_sentiment[aspect]
                    for aspect in top_20_aspects.index}
# top_20_sentiments = {aspect: aspect_mean_sentiment[aspect]
#                     for aspect in top_25_aspects.index}

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 15))
fig.suptitle('Aspect Analysis', fontsize=16, y=0.95)

bars = ax1.bar(range(len(top_20_aspects)), top_20_aspects.values)
ax1.set_xticks(range(len(top_20_aspects)))
# bars = ax1.bar(range(len(aspects)), aspects.values)
# ax1.set_xticks(range(len(aspects)))

ax1.set_xticklabels(top_20_aspects.index, rotation=45, ha='right')
ax1.set_title('Top 25 Most Frequent Aspects')
ax1.set_ylabel('Frequency')

for bar in bars:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
             f'{int(height)}',
             ha='center', va='bottom')

sentiment_values = list(top_20_sentiments.values())
bars = ax2.bar(range(len(top_20_sentiments)), sentiment_values)
ax2.set_xticks(range(len(top_20_sentiments)))
ax2.set_xticklabels(top_20_sentiments.keys(), rotation=45, ha='right')
# ax2.set_xticklabels(top_20_sentiments.keys(), rotation=45, ha='right')
ax2.set_title('Average Sentiment Score by Aspect')
ax2.set_ylabel('Average Sentiment Score')
ax2.axhline(y=0, color='r', linestyle='--', alpha=0.3)

for i, bar in enumerate(bars):
    if sentiment_values[i] < 0:
        bar.set_color('#ff9999')
    else:
        bar.set_color('#99ff99')

    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.2f}',
             ha='center', va='bottom' if height > 0 else 'top')

plt.tight_layout()
plt.show()






import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModel
import ast
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import precision_recall_fscore_support, mean_squared_error
import re 





# clean sentences for clean tokenization and less noise
def clean(text):
    text = str(text)
    text = re.sub(r'http\S+|www\S+', '', text)
    text = text.replace('$', '')
    text = text.replace('#', '')
    text = re.sub(r'\s+', ' ', text)

    text = text.strip()
    return text 





# Preprocess data to extract features and also prepare the Dataset loader
class FinancialAspectDataset(Dataset):
    def __init__(self, csv_file, tokenizer_name='ProsusAI/finbert', max_length=128):
        self.df = pd.read_csv(csv_file)

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length

        with open("unique_aspects", 'r') as file:
            aspects_list = [line.strip() for line in file]

        self.aspects = aspects_list #unique aspects

        self.mlb = MultiLabelBinarizer()
        self.aspect_labels = self.mlb.fit_transform(self.aspects)
        self.aspect_classes = self.mlb.classes_
        self.num_aspects = len(self.aspect_classes)

    def __len__(self):
        return len(self.df)

    def process_aspects(self, aspects):
        aspect_names = np.unique(aspects[2:-2].split('/')).tolist()
        true_mask = [self.aspects.index(a) for a in aspect_names if a != '']
        aspect_ids = [0]*len(self.aspects)
        for m in true_mask:
            aspect_ids[m] = 1
        return aspect_ids

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        sentence = str(row['sentence'])
        snippet = ast.literal_eval(row['snippets'])[0]
        # print("snippet: ", snippet)
        target = str(row['target'])
        # print("target: ", target)

        sentence_encoding = self.tokenizer(
            clean(sentence),
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        snippet_encoding = self.tokenizer(
            clean(snippet),
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        target_encoding = self.tokenizer(
            clean(target),
            padding='max_length',
            truncation=True,
            max_length=32,  # intentionally kam rakha hai
            return_tensors='pt'
        )

        aspect_label_ids = self.process_aspects(row['aspects'])
        return {
            'sentence_ids': sentence_encoding['input_ids'].squeeze(0),
            'sentence_mask': sentence_encoding['attention_mask'].squeeze(0),
            'snippet_ids': snippet_encoding['input_ids'].squeeze(0),
            'snippet_mask': snippet_encoding['attention_mask'].squeeze(0),
            'target_ids': target_encoding['input_ids'].squeeze(0),
            'target_mask': target_encoding['attention_mask'].squeeze(0),
            'aspect_label_ids': (torch.tensor(aspect_label_ids, dtype=float)).squeeze(0),
            'sentiment_score': torch.FloatTensor([float(row['sentiment_score'])])
        }



# Create data loaders
seed = 42
csv_file = "train.csv"
batch_size = 8
torch.manual_seed(seed)
dataset = FinancialAspectDataset(csv_file)
train_loader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True
)

for batch in train_loader:
    print(batch.keys())
    break






# Modelling target based attention
class TargetAttention(torch.nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        # 8 -> 2 or 4
        self.attention = torch.nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=2,
            dropout=0.1
        )

        self.hidden_dim = 768
        # self.dim_reducer = torch.nn.Linear(-1, self.hidden_dim)

    def forward(self, sentence_encoding, snippet_encoding, target_encoding):
        # sentence_reduced = self.dim_reducer(sentence_encoding)
        # snippet_reduced = self.dim_reducer(snippet_encoding)
        # target_reduced = self.dim_reducer(target_encoding)

        attn_output, attn_weights = self.attention(
            query   =   target_encoding,
            key     =   sentence_encoding,
            value   =   snippet_encoding
        )

        return attn_output, attn_weights





# Modelling the final model
# have both sentiment score and aspect classification

class AspectDetectionModel(torch.nn.Module):
    def __init__(self, num_aspects):
        super().__init__()
        self.finbert = AutoModel.from_pretrained('ProsusAI/finbert')

        for param in self.finbert.parameters():
            param.requires_grad = False

        for param in self.finbert.encoder.layer[-2:].parameters():
            param.requires_grad = True

        self.target_attention = TargetAttention(hidden_dim=768)

        self.aspect_classifier = torch.nn.Sequential(
            torch.nn.Linear(768, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(128, num_aspects),
        )
        self.sentiment_regressor = torch.nn.Sequential(
            torch.nn.Linear(768, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(128, 1),
            torch.nn.Tanh()
        )

    def forward(self, sentence_ids, sentence_mask,
                              snippet_ids, snippet_mask,
                              target_ids, target_mask):
        sentence_encoding = self.finbert(sentence_ids, sentence_mask)[0]  # [batch_size, seq_len, 768]
        sentence_encoding = sentence_encoding[:, 0, :] # B, H
        # print("sent_encoding: ",sentence_encoding.shape)
        snippet_encoding = self.finbert(snippet_ids, snippet_mask)[0]
        snippet_encoding = snippet_encoding[:, 0, :]
        # print("snippet_encoding: ",snippet_encoding.shape)
        target_encoding = self.finbert(target_ids, target_mask)[0]
        target_encoding = target_encoding[:,0,:]
        # print("target_encoding: ",target_encoding.shape)

        target_aware_output, _ = self.target_attention(
            sentence_encoding=sentence_encoding,
            snippet_encoding=snippet_encoding,
            target_encoding=target_encoding
        )

        # print("attn_output: ",target_aware_output.shape)

        aspect_logits = self.aspect_classifier(target_aware_output)
        sentiment_score = self.sentiment_regressor(target_aware_output)
        # print("output_logits: ",aspect_logits.shape)
        # print(aspect_logits[0])
        return aspect_logits, sentiment_score






device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

ASPECT_LIST = 119
num_epochs = 3



model = AspectDetectionModel(num_aspects=ASPECT_LIST).to(device)
criterion = torch.nn.BCEWithLogitsLoss()
sentiment_criterion = torch.nn.MSELoss()

optimizer = AdamW([
    {'params': model.finbert.parameters(), 'lr': 2e-5},
    {'params': model.target_attention.parameters(), 'lr': 1e-4},
    {'params': model.aspect_classifier.parameters(), 'lr': 1e-3},
    {'params': model.sentiment_regressor.parameters(), 'lr': 1e-3}
])

print(model)





# Training loop
aspect_loss_hist = []
sentiment_loss_hist = []
tot_loss_hist = []
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    cnt = 0
    for batch in train_loader:
        cnt += 1
        sentence_ids = batch['sentence_ids'].to(device)
        sentence_mask = batch['sentence_mask'].to(device)
        snippet_ids = batch['snippet_ids'].to(device)
        snippet_mask = batch['snippet_mask'].to(device)
        target_ids = batch['target_ids'].to(device)
        target_mask = batch['target_mask'].to(device)

        aspect_labels = batch['aspect_label_ids'].to(device)  # Binary matrix [batch_size, num_aspects]
        true_sentiment_scores = batch['sentiment_score'].to(device)
        # print("true: ", aspect_labels.shape)

        # print("forward pass: ")
        aspect_logits, pred_sentiment_score = model(sentence_ids, sentence_mask,
                              snippet_ids, snippet_mask,
                              target_ids, target_mask)

        # print("pred: ", aspect_logits.shape)

        aspect_loss = criterion(aspect_logits, aspect_labels)
        sentiment_loss = sentiment_criterion(pred_sentiment_score.squeeze(), true_sentiment_scores.squeeze())

        loss = aspect_loss + sentiment_loss

        aspect_loss_hist.append(aspect_loss.cpu())
        sentiment_loss_hist.append(sentiment_loss.cpu())
        tot_loss_hist.append(loss.cpu())

        if cnt%20==0:
            print(f"{int(cnt)}/{(epoch+1)}:")
            print("\t loss:           ", loss.item())
            print("\t aspect_loss:    ", aspect_loss.item())
            print("\t sentiment_loss: ", sentiment_loss.item())
        epoch_loss += loss

        optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

    print("\n----------------------------")
    print(f"epoch_loss @{epoch+1}: ", (epoch_loss/cnt).item())
    print("----------------------------\n")







def flat(arr):
    a = []
    for i in range(len(arr)):
        # print(i)
        if arr[i].shape == torch.Size([]):
            a.append(arr[i].item())
            break
        for j in range(len(arr[i])):
            a.append(arr[i][j].item())

    return a


def metric_eval(data_loader):
    model.eval()
    total_aspect_loss = 0
    total_sentiment_loss = 0
    val_steps = 0

    all_aspect_predictions = []
    all_true_aspects = []
    all_sentiment_predictions = []
    all_true_sentiments = []

    with torch.no_grad():
        for batch in data_loader:
            sentence_ids = batch['sentence_ids'].to(device)
            sentence_mask = batch['sentence_mask'].to(device)
            snippet_ids = batch['snippet_ids'].to(device)
            snippet_mask = batch['snippet_mask'].to(device)
            target_ids = batch['target_ids'].to(device)
            target_mask = batch['target_mask'].to(device)

            aspect_labels = batch['aspect_label_ids'].to(device)
            sentiment_scores = batch['sentiment_score'].to(device)

            aspect_logits, pred_sentiment = model(sentence_ids, sentence_mask,
                                                snippet_ids, snippet_mask,
                                                target_ids, target_mask)

            aspect_loss = criterion(aspect_logits, aspect_labels)
            sentiment_loss = sentiment_criterion(pred_sentiment.squeeze(), sentiment_scores.squeeze())

            total_aspect_loss += aspect_loss.item()
            total_sentiment_loss += sentiment_loss.item()
            val_steps += 1

            aspect_predictions = torch.sigmoid(aspect_logits) > 0.2


            aspect_predictions = aspect_predictions.cpu().numpy()
            true_aspects = aspect_labels.cpu().numpy()
            sentiment_predictions = pred_sentiment.squeeze().cpu()
            true_sentiments = sentiment_scores.squeeze().cpu()

            all_aspect_predictions.append(aspect_predictions)
            all_true_aspects.append(true_aspects)
            all_sentiment_predictions.append(sentiment_predictions)
            all_true_sentiments.append(true_sentiments)

    all_aspect_predictions = np.vstack(all_aspect_predictions)
    all_true_aspects = np.vstack(all_true_aspects)
    # all_sentiment_predictions = np.vstack(all_sentiment_predictions)
    # all_true_sentiments = np.vstack(all_true_sentiments)

    all_sentiment_predictions = flat(all_sentiment_predictions)
    all_true_sentiments = flat(all_true_sentiments)
    # Calculate metrics
    precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(
        all_true_aspects, all_aspect_predictions, average='micro'
    )

    sentiment_mse = mean_squared_error(all_true_sentiments, all_sentiment_predictions)
    avg_aspect_loss = total_aspect_loss / val_steps
    avg_sentiment_loss = total_sentiment_loss / val_steps

    print(f"Aspect Loss: {avg_aspect_loss:.4f}")
    print(f"Sentiment MSE Loss: {avg_sentiment_loss:.4f}")

    print("\nAspect Classification Metrics (Micro-averaged):")
    print(f"Precision: {precision_micro:.4f}")
    print(f"Recall: {recall_micro:.4f}")
    print(f"F1-score: {f1_micro:.4f}")

    print("\nSentiment Regression Metrics:")
    print(f"MSE: {sentiment_mse:.4f}")
    print('-'*60)



print("Eval Metric on Training Data")
print('-'*60)
metric_eval(train_loader)


val_data = FinancialAspectDataset("validation.csv")
val_loader = DataLoader(
    val_data,
    batch_size=batch_size,
    shuffle=True
)
print("Eval Metric on Validation Data")
print('-'*60)

model.eval()
total_aspect_loss = 0
total_sentiment_loss = 0
val_steps = 0

all_aspect_predictions = []
all_true_aspects = []
all_sentiment_predictions = []
all_true_sentiments = []

with torch.no_grad():
    for batch in val_loader:
        sentence_ids = batch['sentence_ids'].to(device)
        sentence_mask = batch['sentence_mask'].to(device)
        snippet_ids = batch['snippet_ids'].to(device)
        snippet_mask = batch['snippet_mask'].to(device)
        target_ids = batch['target_ids'].to(device)
        target_mask = batch['target_mask'].to(device)

        aspect_labels = batch['aspect_label_ids'].to(device)
        sentiment_scores = batch['sentiment_score'].to(device)

        aspect_logits, pred_sentiment = model(sentence_ids, sentence_mask,
                                            snippet_ids, snippet_mask,
                                            target_ids, target_mask)

        aspect_loss = criterion(aspect_logits, aspect_labels)
        sentiment_loss = sentiment_criterion(pred_sentiment.squeeze(), sentiment_scores.squeeze())

        total_aspect_loss += aspect_loss.item()
        total_sentiment_loss += sentiment_loss.item()
        val_steps += 1

        aspect_predictions = torch.sigmoid(aspect_logits) > 0.2


        aspect_predictions = aspect_predictions.cpu().numpy()
        true_aspects = aspect_labels.cpu().numpy()
        sentiment_predictions = pred_sentiment.squeeze().cpu()
        true_sentiments = sentiment_scores.squeeze().cpu()

        all_aspect_predictions.append(aspect_predictions)
        all_true_aspects.append(true_aspects)
        all_sentiment_predictions.append(sentiment_predictions)
        all_true_sentiments.append(true_sentiments)

all_aspect_predictions = np.vstack(all_aspect_predictions)
all_true_aspects = np.vstack(all_true_aspects)
# all_sentiment_predictions = np.vstack(all_sentiment_predictions)
# all_true_sentiments = np.vstack(all_true_sentiments)

all_sentiment_predictions = flat(all_sentiment_predictions)
all_true_sentiments = flat(all_true_sentiments)
# Calculate metrics
precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(
    all_true_aspects, all_aspect_predictions, average='micro'
)

sentiment_mse = mean_squared_error(all_true_sentiments, all_sentiment_predictions)
avg_aspect_loss = total_aspect_loss / val_steps
avg_sentiment_loss = total_sentiment_loss / val_steps

print(f"Aspect Loss: {avg_aspect_loss:.4f}")
print(f"Sentiment MSE Loss: {avg_sentiment_loss:.4f}")

print("\nAspect Classification Metrics (Micro-averaged):")
print(f"Precision: {precision_micro:.4f}")
print(f"Recall: {recall_micro:.4f}")
print(f"F1-score: {f1_micro:.4f}")

print("\nSentiment Regression Metrics:")
print(f"MSE: {sentiment_mse:.4f}")
