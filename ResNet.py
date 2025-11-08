import torch 
import torch.nn as nn
import torch.optim as optim
from datareader import get_data_loaders, NEW_CLASS_NAMES
from torchvision import models  # Menggunakan model ResNet dari torchvision
import matplotlib.pyplot as plt
from utils import plot_training_history, visualize_random_val_predictions

# --- Hyperparameter ---
EPOCHS = 20
BATCH_SIZE = 16
LEARNING_RATE = 0.0003

# Menampilkan plot riwayat training dan validasi setelah training selesai.

def train():
    # 1. Memuat Data
    train_loader, val_loader, num_classes, in_channels = get_data_loaders(BATCH_SIZE)
    
    # 2. Inisialisasi Model
    model = models.resnet18(pretrained=True)  # Menggunakan ResNet-18 dari torchvision
    
    # Modifikasi input channel agar sesuai dengan gambar grayscale (1 channel)
    if in_channels != 3:
        model.conv1 = nn.Conv2d(in_channels, model.conv1.out_channels, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    
    # Ganti classifier untuk sesuai dengan jumlah kelas
    in_features = model.fc.in_features
    if num_classes == 2:
        model.fc = nn.Linear(in_features, 1)  # Binary classification (BCEWithLogitsLoss)
    else:
        model.fc = nn.Linear(in_features, num_classes)  # Multi-class classification
    
    print(model)
    
    # 3. Mendefinisikan Loss Function dan Optimizer
    if num_classes == 2:
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss()  # Untuk multi-class classification
    
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
            images = images
            # Ubah tipe data label menjadi float untuk BCEWithLogitsLoss
            labels = labels.float() if num_classes == 2 else labels.long()
            
            outputs = model(images)
            loss = criterion(outputs, labels)  # Loss dihitung antara output tunggal dan label
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            # Hitung training accuracy
            predicted = (outputs > 0).float() if num_classes == 2 else outputs.max(1)[1]
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        avg_train_loss = running_loss / len(train_loader)
        train_accuracy = 100 * train_correct / train_total
        
        # --- Fase Validasi ---
        model.eval()
        val_correct = 0
        val_total = 0
        val_running_loss = 0.0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images = images
                labels = labels.float() if num_classes == 2 else labels.long()
                
                outputs = model(images)
                val_loss = criterion(outputs, labels)
                val_running_loss += val_loss.item()
                
                predicted = (outputs > 0).float() if num_classes == 2 else outputs.max(1)[1]
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        avg_val_loss = val_running_loss / len(val_loader)
        val_accuracy = 100 * val_correct / val_total
        
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
