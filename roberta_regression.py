import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaForSequenceClassification, RobertaTokenizer, RobertaConfig, RobertaModel
from transformers import AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import torch.nn as nn

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# Set device for computation - Change to use GPU 1
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define constants
OUTPUT_DIR = "friedman2_output"
MODEL_DIR = os.path.join(OUTPUT_DIR, "roberta_model")
os.makedirs(MODEL_DIR, exist_ok=True)

# Updated parameters
BATCH_SIZE = 16
EPOCHS = 20  # Increased from 5 to 20
LEARNING_RATE = 5e-5  # Increased from 2e-5 to 5e-5
MAX_LEN = 128
MODEL_NAME = "roberta-base"
GRADIENT_CLIP = 1.0

# Custom RoBERTa model with regression head
class RobertaRegression(nn.Module):
    def __init__(self, model_name, dropout_rate=0.3):
        super(RobertaRegression, self).__init__()
        self.roberta = RobertaModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout_rate)
        self.regressor = nn.Sequential(
            nn.Linear(self.roberta.config.hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 1)
        )
        
    def forward(self, input_ids, attention_mask, token_type_ids=None):
        outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        # Use the [CLS] token representation
        pooled_output = outputs.last_hidden_state[:, 0, :]
        pooled_output = self.dropout(pooled_output)
        return self.regressor(pooled_output)

class FriedmanTextDataset(Dataset):
    """Custom dataset for text-based Friedman #2 data"""
    
    def __init__(self, data_file, tokenizer, max_len=128):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.texts = []
        self.targets = []
        
        # Load data from file
        with open(data_file, 'r') as f:
            for line in f:
                text, target = line.strip().split('\t')
                # Enhanced text representation with more explicit feature relationships
                text = self.enhance_text_representation(text)
                self.texts.append(text)
                self.targets.append(float(target))
    
    def enhance_text_representation(self, text):
        """Enhance the text representation to better capture numerical relationships"""
        # Extract features from text format
        parts = text.split(', ')
        features = {}
        for part in parts:
            key, value = part.split(': ')
            features[key] = float(value)
        
        # Create enhanced text with more explicit relationships
        enhanced_text = (
            f"The value of {features['x1']:.2f} squared plus "
            f"({features['x2']:.2f} multiplied by {features['x3']:.4f} minus "
            f"1 divided by ({features['x2']:.2f} multiplied by {features['x4']:.2f})) squared. "
            f"Feature x1={features['x1']:.2f}, feature x2={features['x2']:.2f}, "
            f"feature x3={features['x3']:.4f}, feature x4={features['x4']:.2f}."
        )
        return enhanced_text
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        target = self.targets[idx]
        
        # Tokenize the text
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=True,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'token_type_ids': encoding['token_type_ids'].flatten(),
            'target': torch.tensor(target, dtype=torch.float)
        }

def train_epoch(model, data_loader, optimizer, scheduler, device):
    """Train the model for one epoch"""
    model.train()
    total_loss = 0
    
    for batch in data_loader:
        optimizer.zero_grad()
        
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        token_type_ids = batch['token_type_ids'].to(device)
        targets = batch['target'].to(device).unsqueeze(1)  # Shape: [batch_size, 1]
        
        # Use custom model
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        # MSE loss
        loss_fn = nn.MSELoss()
        loss = loss_fn(outputs, targets)
        total_loss += loss.item()
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRADIENT_CLIP)
        optimizer.step()
        scheduler.step()
    
    return total_loss / len(data_loader)

def evaluate(model, data_loader, device):
    """Evaluate the model on the given data loader"""
    model.eval()
    predictions = []
    actual_values = []
    
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            targets = batch['target'].to(device)
            
            # Use custom model
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            )
            
            # Get predictions from the model
            preds = outputs.squeeze(-1)  # Shape: [batch_size]
            
            predictions.extend(preds.cpu().numpy())
            actual_values.extend(targets.cpu().numpy())
    
    predictions = np.array(predictions)
    actual_values = np.array(actual_values)
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(actual_values, predictions))
    mae = mean_absolute_error(actual_values, predictions)
    r2 = r2_score(actual_values, predictions)
    
    return {
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'predictions': predictions,
        'actual_values': actual_values
    }

def plot_learning_curve(train_losses, val_metrics, output_path):
    """Plot learning curves showing training loss and validation metrics"""
    epochs = range(1, len(train_losses) + 1)
    
    plt.figure(figsize=(12, 8))
    
    # Plot training loss
    plt.subplot(2, 2, 1)
    plt.plot(epochs, train_losses, 'b-', label='Training Loss')
    plt.title('Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot validation RMSE
    plt.subplot(2, 2, 2)
    plt.plot(epochs, [m['rmse'] for m in val_metrics], 'r-', label='Validation RMSE')
    plt.title('Validation RMSE')
    plt.xlabel('Epochs')
    plt.ylabel('RMSE')
    plt.legend()
    
    # Plot validation MAE
    plt.subplot(2, 2, 3)
    plt.plot(epochs, [m['mae'] for m in val_metrics], 'g-', label='Validation MAE')
    plt.title('Validation MAE')
    plt.xlabel('Epochs')
    plt.ylabel('MAE')
    plt.legend()
    
    # Plot validation R²
    plt.subplot(2, 2, 4)
    plt.plot(epochs, [m['r2'] for m in val_metrics], 'y-', label='Validation R²')
    plt.title('Validation R²')
    plt.xlabel('Epochs')
    plt.ylabel('R²')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_predictions(predictions, actual_values, output_path):
    """Plot predicted vs actual values and residuals"""
    residuals = actual_values - predictions
    
    plt.figure(figsize=(12, 10))
    
    # Predicted vs Actual
    plt.subplot(2, 1, 1)
    plt.scatter(actual_values, predictions, alpha=0.5)
    plt.plot([min(actual_values), max(actual_values)], 
             [min(actual_values), max(actual_values)], 'r--')
    plt.title('Predicted vs Actual Values')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    
    # Residuals
    plt.subplot(2, 1, 2)
    plt.scatter(predictions, residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.title('Residuals vs Predicted Values')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    
    # Residual distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(residuals, kde=True)
    plt.title('Residual Distribution')
    plt.xlabel('Residual Value')
    plt.ylabel('Frequency')
    plt.savefig(output_path.replace('.png', '_distribution.png'))
    plt.close()

def train_random_forest_baseline(X_train, y_train, X_test, y_test):
    """Train a Random Forest model as a baseline for comparison"""
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Make predictions
    predictions = model.predict(X_test)
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    
    return {
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'predictions': predictions
    }

def main():
    print("Preparing improved RoBERTa regression model for Friedman #2 dataset...")
    
    # Load tokenizer
    tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME)
    
    # Load datasets
    print("Loading datasets with enhanced text representation...")
    train_dataset = FriedmanTextDataset(
        os.path.join(OUTPUT_DIR, "train_roberta.txt"),
        tokenizer,
        max_len=MAX_LEN
    )
    
    val_dataset = FriedmanTextDataset(
        os.path.join(OUTPUT_DIR, "validation_roberta.txt"),
        tokenizer,
        max_len=MAX_LEN
    )
    
    test_dataset = FriedmanTextDataset(
        os.path.join(OUTPUT_DIR, "test_roberta.txt"),
        tokenizer,
        max_len=MAX_LEN
    )
    
    print(f"Train size: {len(train_dataset)}")
    print(f"Validation size: {len(val_dataset)}")
    print(f"Test size: {len(test_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )
    
    # Initialize custom model
    print("Initializing custom RoBERTa model for regression...")
    model = RobertaRegression(MODEL_NAME)
    model.to(device)
    
    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),  # 10% warmup
        num_training_steps=total_steps
    )
    
    # Training loop
    print("Starting training...")
    best_val_rmse = float('inf')
    train_losses = []
    val_metrics = []
    
    for epoch in range(EPOCHS):
        print(f"Epoch {epoch + 1}/{EPOCHS}")
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device)
        train_losses.append(train_loss)
        
        # Validate
        val_results = evaluate(model, val_loader, device)
        val_metrics.append(val_results)
        
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Validation RMSE: {val_results['rmse']:.4f}")
        print(f"Validation MAE: {val_results['mae']:.4f}")
        print(f"Validation R²: {val_results['r2']:.4f}")
        
        # Save best model
        if val_results['rmse'] < best_val_rmse:
            best_val_rmse = val_results['rmse']
            print(f"Saving best model with RMSE: {best_val_rmse:.4f}")
            # Save custom model
            torch.save(model.state_dict(), os.path.join(MODEL_DIR, "pytorch_model.bin"))
            tokenizer.save_pretrained(MODEL_DIR)
    
    # Plot learning curves
    plot_learning_curve(
        train_losses,
        val_metrics,
        os.path.join(OUTPUT_DIR, "learning_curves.png")
    )
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_results = evaluate(model, test_loader, device)
    
    print(f"Test RMSE: {test_results['rmse']:.4f}")
    print(f"Test MAE: {test_results['mae']:.4f}")
    print(f"Test R²: {test_results['r2']:.4f}")
    
    # Plot predictions vs actual values
    plot_predictions(
        test_results['predictions'],
        test_results['actual_values'],
        os.path.join(OUTPUT_DIR, "predictions_vs_actual.png")
    )
    
    # Train and evaluate Random Forest baseline
    print("\nTraining Random Forest baseline...")
    
    # Load original numerical data for Random Forest
    train_df = pd.read_csv(os.path.join(OUTPUT_DIR, "train_data.csv"))
    test_df = pd.read_csv(os.path.join(OUTPUT_DIR, "test_data.csv"))
    
    feature_names = ['x1', 'x2', 'x3', 'x4']
    X_train = train_df[feature_names].values
    y_train = train_df['target'].values
    X_test = test_df[feature_names].values
    y_test = test_df['target'].values
    
    rf_results = train_random_forest_baseline(X_train, y_train, X_test, y_test)
    
    print(f"Random Forest Test RMSE: {rf_results['rmse']:.4f}")
    print(f"Random Forest Test MAE: {rf_results['mae']:.4f}")
    print(f"Random Forest Test R²: {rf_results['r2']:.4f}")
    
    # Save predictions and errors for later use in conformal prediction
    predictions_df = pd.DataFrame({
        'actual': test_results['actual_values'],
        'predicted': test_results['predictions'],
        'residual': test_results['actual_values'] - test_results['predictions'],
        'abs_error': np.abs(test_results['actual_values'] - test_results['predictions'])
    })
    
    predictions_df.to_csv(os.path.join(OUTPUT_DIR, "roberta_predictions.csv"), index=False)
    
    print("\nImproved RoBERTa regression model training and evaluation complete!")
    print(f"Model saved to {MODEL_DIR}")
    print(f"Predictions and errors saved to {os.path.join(OUTPUT_DIR, 'roberta_predictions.csv')}")

if __name__ == "__main__":
    main() 