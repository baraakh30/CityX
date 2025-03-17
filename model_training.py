import os
import joblib
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
from sklearn.model_selection import train_test_split

print("Loading dataset...")
df = pd.read_csv('Competition_Dataset.csv')

class CrimeDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

try:
        # Prepare data for transformer model
        print("Preparing data for transformer model...")
        
        # Convert text to list
        texts = df['Descript'].tolist()
        
        # Encode labels
        label_encoder = LabelEncoder()
        encoded_labels = label_encoder.fit_transform(df['Category'])
        
        # Save label encoder
        joblib.dump(label_encoder, 'app/static/transformer_label_encoder.pkl')
        
        # Split data
        train_texts, test_texts, train_labels, test_labels = train_test_split(
            texts, encoded_labels, test_size=0.2, random_state=42
        )
        

        
        # Load pre-trained tokenizer and model
        model_name = "distilbert-base-uncased"  # Smaller and faster than BERT
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, 
            num_labels=len(label_encoder.classes_)
        )
        model.to(device)
        
        # Create datasets
        train_dataset = CrimeDataset(train_texts, train_labels, tokenizer)
        test_dataset = CrimeDataset(test_texts, test_labels, tokenizer)
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=128,  # Adjust based on your GPU memory
            shuffle=True
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=128
        )
        
        # Training parameters
        optimizer = AdamW(model.parameters(), lr=5e-5)
        num_epochs = 3  # Start with fewer epochs to save time
        
        # Training loop
        print(f"Training transformer model for {num_epochs} epochs...")
        for epoch in range(num_epochs):
            model.train()
            total_loss = 0
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
            
            for batch in progress_bar:
                # Move batch to device
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                # Zero gradients
                optimizer.zero_grad()
                
                # Forward pass
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                total_loss += loss.item()
                
                # Backward pass and optimize
                loss.backward()
                optimizer.step()
                
                # Update progress bar
                progress_bar.set_postfix({'loss': total_loss / (progress_bar.n + 1)})
            
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(train_loader):.4f}")
        
        # Save the model and tokenizer
        print("Saving transformer model...")
        os.makedirs('app/static/transformer_model', exist_ok=True)
        model.save_pretrained('app/static/transformer_model')
        tokenizer.save_pretrained('app/static/transformer_tokenizer')
        
        # Evaluation
        print("Evaluating transformer model...")
        model.eval()
        predictions = []
        actual_labels = []
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Evaluating"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                _, preds = torch.max(outputs.logits, dim=1)
                
                predictions.extend(preds.cpu().tolist())
                actual_labels.extend(labels.cpu().tolist())
        
        # Convert back to original labels
        predictions = label_encoder.inverse_transform(predictions)
        actual_labels = label_encoder.inverse_transform(actual_labels)
        
        # Print classification report
        print("Transformer Classification Report:")
        transformer_report = classification_report(actual_labels, predictions)
        print(transformer_report)
        
        # Save report to file
        with open('app/static/transformer_classification_report.txt', 'w') as f:
            f.write(transformer_report)
            
        # Calculate and print accuracy
        accuracy = accuracy_score(actual_labels, predictions)
        print(f"Transformer model accuracy: {accuracy:.4f}")
     
        print("Transformer model training and evaluation completed!")
        
except Exception as e:
        print(f"Error in transformer model training: {str(e)}")
        import traceback
        traceback.print_exc()
        print("Transformer model training failed!")