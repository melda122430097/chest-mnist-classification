# train.py

import torch
import torch.nn as nn
import torch.optim as optim
from datareader import get_data_loaders, NEW_CLASS_NAMES
from model import DashNet
import matplotlib.pyplot as plt
from utils import plot_training_history, visualize_random_val_predictions

# --- Hyperparameter ---
EPOCHS = 16
BATCH_SIZE = 16
LEARNING_RATE = 0.001

def train():
    # 1. Memuat Data
    train_loader, val_loader, num_classes, in_channels = get_data_loaders(BATCH_SIZE)
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 2. Inisialisasi Model
    model = DashNet(in_channels=in_channels, num_classes=num_classes).to(device)
    print(model)
    
    # 3. Mendefinisikan Loss Function dan Optimizer
    if num_classes == 2:
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Inisialisasi list untuk menyimpan history
    train_losses_history = []
    val_losses_history = []
    train_accs_history = []
    val_accs_history = []
    
    print("\n--- Memulai Training ---")
    
    # 4. Training Loop
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for images, labels in train_loader:
            images = images.to(device)
            
            if num_classes == 2:
                # pastikan labels shape (N,1) dan float untuk BCEWithLogitsLoss
                labels = labels.view(-1, 1).float().to(device)
            else:
                labels = labels.long().to(device)
            
            outputs = model(images)
            
            # Loss
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            # Hitung training accuracy
            if num_classes == 2:
                probs = torch.sigmoid(outputs)
                predicted = (probs > 0.5).float()
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
            else:
                _, predicted = outputs.max(1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
        
        avg_train_loss = running_loss / max(1, len(train_loader))
        train_accuracy = 100 * train_correct / max(1, train_total)
        
        # --- Fase Validasi ---
        model.eval()
        val_correct = 0
        val_total = 0
        val_running_loss = 0.0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                
                if num_classes == 2:
                    labels = labels.view(-1, 1).float().to(device)
                else:
                    labels = labels.long().to(device)
                
                outputs = model(images)
                val_loss = criterion(outputs, labels)
                val_running_loss += val_loss.item()
                
                if num_classes == 2:
                    probs = torch.sigmoid(outputs)
                    predicted = (probs > 0.5).float()
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
                else:
                    _, predicted = outputs.max(1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
        
        avg_val_loss = val_running_loss / max(1, len(val_loader))
        val_accuracy = 100 * val_correct / max(1, val_total)
        
        # Simpan history
        train_losses_history.append(avg_train_loss)
        val_losses_history.append(avg_val_loss)
        train_accs_history.append(train_accuracy)
        val_accs_history.append(val_accuracy)
        
        print(f"Epoch [{epoch+1}/{EPOCHS}] | "
              f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_accuracy:.2f}% | "
              f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_accuracy:.2f}%")

    print("--- Training Selesai ---")
    
    # Tampilkan plot
    plot_training_history(train_losses_history, val_losses_history, 
                         train_accs_history, val_accs_history)

    # Visualisasi prediksi pada 10 gambar random dari validation set
    visualize_random_val_predictions(model, val_loader, num_classes, count=10)

if __name__ == '__main__':
    train()
