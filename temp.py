import psycopg2
import pandas as pd
from transformers import BertTokenizer, BertModel
import torch
import torch.nn as nn
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Get database connection details from environment variables
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")

# Define dataset
class BERTDataset:  # using BertTokenizer here
    def __init__(self, texts, labels, max_len=128):
        self.texts = texts
        self.labels = labels
        self.max_len = max_len
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.num_examples = len(self.texts)
    
    def __len__(self):
        return self.num_examples

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        tokenized_text = self.tokenizer(
            text, 
            add_special_tokens=True, 
            padding='max_length', 
            max_length=self.max_len,
            truncation=True,
        )
        ids = tokenized_text['input_ids']
        mask = tokenized_text['attention_mask']
        token_type_ids = tokenized_text['token_type_ids']

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'target': torch.tensor(label, dtype=torch.long)
        }

class ToxicModel(nn.Module):
    def __init__(self):
        super(ToxicModel, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased", return_dict=False)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(768, 2)

    def forward(self, input_ids, token_type_ids, attention_mask, labels=None):
        _, pooled_output = self.bert(input_ids, token_type_ids, attention_mask)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits

    def train_model(self, train_dataset, valid_dataset, epochs=1):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)

        param_optimizer = list(self.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_paramaters = [
            {
                'params' : [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                'weight_decay': 0.01,
            },
            {
                'params' : [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0,
            },
        ]

        optimizer = torch.optim.Adam(optimizer_grouped_paramaters, lr=5e-5)
        criterion = nn.CrossEntropyLoss()

        for i in range(epochs):
            self.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            for batch in train_dataset:
                optimizer.zero_grad()
                ids = batch['ids'].to(device)
                mask = batch['mask'].to(device)
                token_type_ids = batch['token_type_ids'].to(device)
                target = batch['target'].to(device)
                
                logits = self(ids, token_type_ids, mask)
                loss = criterion(logits, target)
                loss.backward()
                optimizer.step()
                
                # Calculate accuracy for training
                _, predicted = torch.max(logits, dim=1)
                train_correct += (predicted == target).sum().item()
                train_total += target.size(0)
                
                train_loss += loss.item() * batch['ids'].size(0)
            
            train_loss = train_loss / len(train_dataset)
            train_accuracy = 100 * train_correct / train_total  # Calculate training accuracy
            
            self.eval()
            valid_loss = 0
            valid_correct = 0
            valid_total = 0
            for batch in valid_dataset:
                ids = batch['ids'].to(device)
                mask = batch['mask'].to(device)
                token_type_ids = batch['token_type_ids'].to(device)
                target = batch['target'].to(device)
                
                logits = self(ids, token_type_ids, mask)
                loss = criterion(logits, target)
                
                # Calculate accuracy for validation
                _, predicted = torch.max(logits, dim=1)
                valid_correct += (predicted == target).sum().item()
                valid_total += target.size(0)
                
                valid_loss += loss.item() * batch['ids'].size(0)
            
            valid_loss = valid_loss / len(valid_dataset)
            valid_accuracy = 100 * valid_correct / valid_total  # Calculate validation accuracy
            
            print(f"Epoch {i+1}/{epochs}.. Train loss: {train_loss:.3f}.. Train accuracy: {train_accuracy:.2f}%.. Validation loss: {valid_loss:.3f}.. Validation accuracy: {valid_accuracy:.2f}%")


# Function to fetch data from the database
def fetch_data_from_db():
    try:
        # Connect to the database
        connection = psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            sslmode='require'
        )
        print("Connected to the database successfully!")
        
        # Query data
        query = 'SELECT sentence, "isToxic" FROM vectorize.sentences;'
        df = pd.read_sql_query(query, connection)
        
        # Close the connection
        connection.close()
        return df

    except Exception as e:
        print(f"Error connecting to the database: {e}")
        return None

# Fetch data from the database
df_train = fetch_data_from_db()

if df_train is not None:
    # Prepare train and validation datasets
    train_dataset = BERTDataset(
        df_train.sentence.values,  # Input texts
        df_train.isToxic.values    # Labels
    )

    valid_dataset = BERTDataset(
        df_train.sentence.values,  # Input texts
        df_train.isToxic.values    # Labels
    )

    train_data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
    )
    valid_data_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=32,
        shuffle=True,
    )

    # Create the model and start training
    model = ToxicModel()
    model.train_model(train_data_loader, valid_data_loader, epochs=10)
