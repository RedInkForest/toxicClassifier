import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from torch.utils.data import DataLoader
from transformers import AdamW
import psycopg2
import pandas as pd


# Adding in data from DB
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")

def fetch_data_from_db():
    try:
        # Connect to the database
        connection = psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD
        )
        print("Connected to the database successfully!")

        query = 'SELECT sentence, "isToxic" FROM vectorize.sentences;'
        df = pd.read_sql_query(query, connection)

        # Close the connection
        connection.close()
        return df
    except Exception as e:
        print(f"Error connecting to the database: {e}")
        return None
    
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
        
        return x
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
        
        return x

## Main code
# Get data from the database
df = fetch_data_from_db()

# Get data from the csv 
csv = pd.read_csv('Ctoxicity_en.csv')
csv = csv.replace('Toxic', True)
csv = csv.replace('Not Toxic', False)
csv.rename(columns={'text': 'sentence', 'is_toxic': 'isToxic'}, inplace=True)

# Combine the 2 dataframes
dataset = pd.concat([df, csv])
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
num_heads = 4  # Number of attention heads
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
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}, Train Accuracy: {train_accuracy}, Validation Loss: {val_loss/len(test_loader)}, Validation Accuracy: {val_accuracy}")


# Evaluation
model.eval()
predictions, labels = [], []
with torch.no_grad():
    for input_ids, label in test_loader:
        outputs = model(input_ids)
        preds = torch.argmax(outputs, dim=1)
        predictions.extend(preds.cpu().numpy())
        labels.extend(label.cpu().numpy())

# Calculate accuracy and print classification report
accuracy = accuracy_score(labels, predictions)
print(f"Accuracy: {accuracy}")
print(classification_report(labels, predictions, target_names=["Not Toxic", "Toxic"]))


