import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertTokenizer, BertForSequenceClassification, AdamW

# Load your data
df = pd.read_csv('path_to_your_file.csv')  # replace with your actual file path

# Prepare the data
df['Type'] = df['Type'].map({'SR': 0, 'Incident': 1})  # map labels to integers

# Split the data
X_train, X_test, y_train, y_test = train_test_split(df['description'], df['Type'], test_size=0.2, random_state=42)

# Load pre-trained BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Tokenize data
def tokenize_data(sentences, labels, max_length=128):
    inputs = tokenizer(
        sentences.tolist(),
        max_length=max_length,
        padding=True,
        truncation=True,
        return_tensors='pt'
    )
    inputs['labels'] = torch.tensor(labels.tolist())
    return inputs

train_inputs = tokenize_data(X_train, y_train)
test_inputs = tokenize_data(X_test, y_test)

# Create DataLoader
train_data = TensorDataset(train_inputs['input_ids'], train_inputs['attention_mask'], train_inputs['labels'])
test_data = TensorDataset(test_inputs['input_ids'], test_inputs['attention_mask'], test_inputs['labels'])

train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
test_loader = DataLoader(test_data, batch_size=16)

# Load pre-trained BERT model for classification
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# Use GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Optimizer
optimizer = AdamW(model.parameters(), lr=2e-5)

# Training loop
epochs = 3

for epoch in range(epochs):
    model.train()
    total_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()

        input_ids, attention_mask, labels = [x.to(device) for x in batch]

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()

        loss.backward()
        optimizer.step()

    avg_loss = total_loss / len(train_loader)
    print(f'Epoch {epoch+1}, Loss: {avg_loss}')

# Evaluation
model.eval()
predictions, true_labels = [], []

with torch.no_grad():
    for batch in test_loader:
        input_ids, attention_mask, labels = [x.to(device) for x in batch]
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        logits = outputs.logits
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        predictions.extend(preds)
        true_labels.extend(labels.cpu().numpy())

# Performance report
accuracy = accuracy_score(true_labels, predictions)
print(f'Accuracy: {accuracy:.4f}')
print(classification_report(true_labels, predictions, target_names=['SR', 'Incident']))
