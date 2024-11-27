import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from torch.utils.data import DataLoader
from transformers import AdamW
import pandas as pd

from sklearn.metrics import confusion_matrix
import seaborn as sns
    
# Creating PyTorch datasets and dataloaders
class TextDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        item = self.data[idx]
        return torch.tensor(item['input_ids']), torch.tensor(item['isToxic'])
    
# Padding function
def pad_sequence(seq, max_len=20):
    return seq + [0] * (max_len - len(seq)) if len(seq) < max_len else seq[:max_len]

# Basic Tokenizer
def simple_tokenizer(text):
    return text.split()

def text_to_indices(text):
    return [word_to_index.get(word, 0) for word in simple_tokenizer(text)]

# Custom Transformer model class
class TransformerTextClassifier(nn.Module):
    def __init__(self, vocab_size, embed_size, num_heads, num_layers, hidden_dim, num_classes):
        super(TransformerTextClassifier, self).__init__()
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        
        # Transformer Encoder layers
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=embed_size, nhead=num_heads, dim_feedforward=hidden_dim)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        
        # Fully connected layer for classification
        self.fc = nn.Linear(embed_size, num_classes)
    
    def forward(self, x):
        # Get embeddings for input
        x = self.embedding(x)
        
        # Apply transformer encoder
        x = self.transformer_encoder(x)
        
        # Use the mean of all token embeddings as the sentence representation
        x = x.mean(dim=1)
        
        # Pass through the fully connected layer for classification
        x = self.fc(x)
        x = torch.sigmoid(x)
        return x

## Main code
# Get data from our csv
csv1 = pd.read_csv('GamingToxicPhrases.csv')
csv1 = csv1.replace('Toxic', True)
csv1 = csv1.replace('Not Toxic', False)
csv1.rename(columns={'text': 'sentence', 'is_toxic': 'isToxic'}, inplace=True)

# Get data from the other csv 
csv2 = pd.read_csv('toxicity_en.csv')
csv2 = csv2.replace('Toxic', True)
csv2 = csv2.replace('Not Toxic', False)
csv2.rename(columns={'text': 'sentence', 'is_toxic': 'isToxic'}, inplace=True)

# Combine the 2 dataframes
dataset = pd.concat([csv1, csv2])
dataset['isToxic'] = dataset['isToxic'].astype(int)

# Split dataset into train and test sets
train_df, valid_df = train_test_split(dataset, test_size=0.2, random_state=42)

# Vocabulary setup: build a simple vocabulary based on the dataset
all_text = dataset['sentence'].tolist()
vocab = set(word for sentence in all_text for word in simple_tokenizer(sentence))

# Map the vocabulary to indices
word_to_index = {word: idx+1 for idx, word in enumerate(vocab)}  # idx=0 reserved for padding
index_to_word = {idx+1: word for idx, word in enumerate(vocab)}

# Tokenize the dataframes
train_data = train_df.apply(lambda x: {'input_ids': text_to_indices(x['sentence']), 'isToxic': x['isToxic']}, axis=1).tolist()
test_data = valid_df.apply(lambda x: {'input_ids': text_to_indices(x['sentence']), 'isToxic': x['isToxic']}, axis=1).tolist()

# Pad sequences to the same length
train_df['input_ids'] = train_df['sentence'].apply(lambda x: pad_sequence(text_to_indices(x)))
valid_df['input_ids'] = valid_df['sentence'].apply(lambda x: pad_sequence(text_to_indices(x)))

# Convert the dataframes to lists of dictionaries
train_data = train_df[['input_ids', 'isToxic']].to_dict('records')
test_data = valid_df[['input_ids', 'isToxic']].to_dict('records')

train_dataset = TextDataset(train_data)
test_dataset = TextDataset(test_data)

train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=2)

# Model parameters
vocab_size = len(vocab) + 1  # +1 for padding token
embed_size = 128  # Embedding size
num_heads = 4  # Number of attention heads.
num_layers = 2  # Number of transformer layers
hidden_dim = 256  # Hidden dimension of feed-forward layers
num_classes = 2  # Toxic or Not Toxic (binary classification)
learning_rate = 1e-4
num_epochs = 25

# Model, loss function, and optimizer
model = TransformerTextClassifier(vocab_size, embed_size, num_heads, num_layers, hidden_dim, num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = AdamW(model.parameters(), lr=learning_rate)

# Training loop
# Arrays for plotting
train_accuracies = []
val_accuracies = []
train_losses = []
val_losses = []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for input_ids, labels in train_loader:
        optimizer.zero_grad()

        # Forward pass
        outputs = model(input_ids)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    # Calculate training accuracy
    model.eval()
    train_predictions, train_labels = [], []
    with torch.no_grad():
        for input_ids, label in train_loader:
            outputs = model(input_ids)
            preds = torch.argmax(outputs, dim=1)
            train_predictions.extend(preds.cpu().numpy())
            train_labels.extend(label.cpu().numpy())
    train_accuracy = accuracy_score(train_labels, train_predictions)
    train_accuracies.append(train_accuracy)
    train_losses.append(running_loss / len(train_loader))
    

    # Calculate validation loss and accuracy
    model.eval()
    val_loss = 0.0
    val_predictions, val_labels = [], []
    with torch.no_grad():
        for input_ids, label in test_loader:
            outputs = model(input_ids)
            loss = criterion(outputs, label)
            val_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            val_predictions.extend(preds.cpu().numpy())
            val_labels.extend(label.cpu().numpy())
    val_accuracy = accuracy_score(val_labels, val_predictions)
    val_accuracies.append(val_accuracy)
    val_losses.append(val_loss / len(test_loader))
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}, Train Accuracy: {train_accuracy}, Validation Loss: {val_loss/len(test_loader)}, Validation Accuracy: {val_accuracy}")

# Calculate accuracy and print classification report
accuracy = accuracy_score(val_labels, val_predictions)
print(f"Accuracy: {accuracy}")
print(classification_report(val_labels, val_predictions, target_names=["Not Toxic", "Toxic"]))
# Generate confusion matrix
conf_matrix = confusion_matrix(val_labels, val_predictions)

import matplotlib.pyplot as plt
# Plotting the accuracies
plt.figure(figsize=(10, 5))
plt.plot(range(1, num_epochs + 1), train_accuracies, label='Training Accuracy')
plt.plot(range(1, num_epochs + 1), val_accuracies, label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy over Epochs')
plt.legend()
plt.show()

# Plotting the losses
plt.figure(figsize=(10, 5))
plt.plot(range(1, num_epochs + 1), train_losses, label='Training Loss')
plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss over Epochs')
plt.legend()
plt.show()

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=["Not Toxic", "Toxic"], yticklabels=["Not Toxic", "Toxic"])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()