import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from google.colab import files
import warnings
from tqdm import tqdm
warnings.filterwarnings('ignore')

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Model Classes
class PatchDescriptor(nn.Module):
    """Patch Descriptor untuk feature extraction"""
    def __init__(self, input_dim=3, hidden_dim=256, output_dim=384):
        super().__init__()
        self.conv1 = nn.Conv2d(input_dim, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv4 = nn.Conv2d(256, output_dim, 3, padding=1)

        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.relu(self.conv3(x))
        x = self.pool(x)
        x = self.relu(self.conv4(x))
        return x

class StudentTeacherNet(nn.Module):
    """Student-Teacher network untuk EfficientAD"""
    def __init__(self, input_dim=3, feature_dim=384):
        super().__init__()
        self.teacher = PatchDescriptor(input_dim, output_dim=feature_dim)
        self.student = PatchDescriptor(input_dim, output_dim=feature_dim)

        # Initialize teacher with student weights
        self.teacher.load_state_dict(self.student.state_dict())
        
        # Freeze teacher network
        for param in self.teacher.parameters():
            param.requires_grad = False

    def forward(self, x):
        teacher_features = self.teacher(x)
        student_features = self.student(x)
        return teacher_features, student_features

class EfficientAD_M(nn.Module):
    """EfficientAD-M model"""
    def __init__(self, input_size=256, feature_dim=384):
        super().__init__()
        self.input_size = input_size
        self.feature_dim = feature_dim

        # Student-Teacher networks
        self.st_net = StudentTeacherNet(input_dim=3, feature_dim=feature_dim)

        # Autoencoder untuk reconstruction
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.LeakyReLU(0.2),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # Student-Teacher features
        teacher_feat, student_feat = self.st_net(x)

        # Autoencoder reconstruction
        encoded = self.encoder(x)
        reconstructed = self.decoder(encoded)

        return {
            'teacher_features': teacher_feat,
            'student_features': student_feat,
            'reconstructed': reconstructed,
            'encoded': encoded
        }

    def compute_anomaly_map(self, x):
        """Compute anomaly heatmap"""
        with torch.no_grad():
            outputs = self.forward(x)

            # Student-Teacher anomaly score
            teacher_feat = outputs['teacher_features']
            student_feat = outputs['student_features']
            st_diff = torch.mean((teacher_feat - student_feat) ** 2, dim=1, keepdim=True)

            # Reconstruction anomaly score
            recon_diff = torch.mean((x - outputs['reconstructed']) ** 2, dim=1, keepdim=True)

            # Combine anomaly scores
            st_map = F.interpolate(st_diff, size=(self.input_size, self.input_size), mode='bilinear', align_corners=False)
            recon_map = F.interpolate(recon_diff, size=(self.input_size, self.input_size), mode='bilinear', align_corners=False)

            # Weighted combination
            anomaly_map = 0.7 * st_map + 0.3 * recon_map

            return anomaly_map.squeeze().cpu().numpy()

# Dataset Classes
class CustomTrainDataset(Dataset):
    """Dataset untuk training menggunakan data normal"""
    def __init__(self, root_dir='/content/data-normal', transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = []

        # Mengumpulkan semua gambar dari direktori
        if os.path.exists(self.root_dir):
            for img_name in os.listdir(self.root_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                    self.images.append(os.path.join(self.root_dir, img_name))
        
        if len(self.images) == 0:
            raise ValueError(f"No images found in {self.root_dir}. Please upload training images.")
        
        print(f"Found {len(self.images)} training images in {self.root_dir}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Load image
        image_path = self.images[idx]
        image = cv2.imread(image_path)
        
        if image is None:
            # Fallback jika gambar tidak bisa dibaca
            image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (256, 256))

        # Apply transforms
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed['image']

        return image

class CustomTestDataset(Dataset):
    def __init__(self, root_dir='/content/test_data', transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = []

        # Mengumpulkan semua gambar dari direktori /content/test_data
        if os.path.exists(self.root_dir):
            for img_path in os.listdir(self.root_dir):
                if img_path.endswith(('.png', '.jpg', '.jpeg')):
                    self.images.append(os.path.join(self.root_dir, img_path))

        if len(self.images) == 0:
            raise ValueError("No images found in /content/test_data. Please upload test images.")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Load image
        image_path = self.images[idx]
        image = cv2.imread(image_path)
        if image is None:
            image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (256, 256))

        # Simpan gambar asli untuk visualisasi
        original_image = image.copy()

        # Apply transforms
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed['image']

        return {
            'image': image,
            'original_image': original_image,
            'path': image_path
        }

# Transformasi data
train_transform = A.Compose([
    A.Resize(256, 256),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

test_transform = A.Compose([
    A.Resize(256, 256),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

# Training Functions
def train_model(model, train_loader, num_epochs=5, lr=0.001):
    """Training function untuk EfficientAD-M"""
    print(f"Starting training for {num_epochs} epochs...")
    
    # Optimizers - hanya untuk student network dan autoencoder
    student_params = list(model.st_net.student.parameters())
    autoencoder_params = list(model.encoder.parameters()) + list(model.decoder.parameters())
    all_params = student_params + autoencoder_params
    
    optimizer = optim.Adam(all_params, lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)
    
    # Loss functions
    mse_loss = nn.MSELoss()
    
    train_losses = []
    
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        
        for batch_idx, images in enumerate(progress_bar):
            images = images.to(device)
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(images)
            
            # Student-Teacher loss
            teacher_feat = outputs['teacher_features']
            student_feat = outputs['student_features']
            st_loss = mse_loss(student_feat, teacher_feat.detach())
            
            # Reconstruction loss
            recon_loss = mse_loss(outputs['reconstructed'], images)
            
            # Total loss (weighted combination)
            total_loss = 0.7 * st_loss + 0.3 * recon_loss
            
            # Backward pass
            total_loss.backward()
            optimizer.step()
            
            epoch_loss += total_loss.item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f'{total_loss.item():.4f}',
                'ST_Loss': f'{st_loss.item():.4f}',
                'Recon_Loss': f'{recon_loss.item():.4f}'
            })
        
        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)
        scheduler.step()
        
        print(f'Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}')
        
        # Save model setiap 10 epoch
        if (epoch + 1) % 10 == 0:
            os.makedirs('results', exist_ok=True)
            torch.save(model.state_dict(), f'results/efficientad_model_epoch_{epoch+1}.pth')
    
    # Save final model
    os.makedirs('results', exist_ok=True)
    torch.save(model.state_dict(), 'results/efficientad_model.pth')
    
    # Plot training loss
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.show()
    
    print("Training completed!")
    return train_losses

def upload_train_data():
    """Upload data training ke /content/data-normal"""
    print("Upload gambar training (data normal)...")
    !mkdir -p /content/data-normal
    uploaded = files.upload()
    for filename in uploaded.keys():
        !mv {filename} /content/data-normal/
    print(f"{len(uploaded)} gambar training diupload ke /content/data-normal.")

def upload_test_data():
    """Upload gambar test ke /content/test_data"""
    print("Upload gambar test (ponsel dengan kemungkinan defect)...")
    !mkdir -p /content/test_data
    uploaded = files.upload()
    for filename in uploaded.keys():
        !mv {filename} /content/test_data/
    print(f"{len(uploaded)} gambar test diupload ke /content/test_data.")

def load_model(model_path='results/efficientad_model.pth'):
    """Load trained EfficientAD-M model"""
    model = EfficientAD_M().to(device)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        model.eval()
        print(f"Model loaded from {model_path}")
    else:
        print(f"Model file {model_path} not found. Please train the model first.")
        return None
    return model

def visualize_heatmap_and_detect(model, image_tensor, original_image, threshold=0.5):
    """Visualisasi heatmap dan deteksi anomali dengan threshold rendah"""
    anomaly_map = model.compute_anomaly_map(image_tensor)

    # Hitung skor anomali maksimum
    max_score = np.max(anomaly_map)
    mean_score = np.mean(anomaly_map)

    # Indikator anomali (threshold rendah agar mudah mendeteksi anomali kecil)
    is_anomaly = max_score > threshold
    anomaly_status = "Anomali Terdeteksi (Defect Kemungkinan Ada)" if is_anomaly else "Tidak Ada Anomali (Normal)"

    # Visualisasi
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Gambar asli
    axes[0].imshow(original_image)
    axes[0].set_title('Gambar Asli')
    axes[0].axis('off')

    # Heatmap anomali
    im = axes[1].imshow(anomaly_map, cmap='hot', interpolation='bilinear')
    axes[1].set_title('Heatmap Anomali')
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    # Overlay heatmap pada gambar asli
    axes[2].imshow(original_image)
    axes[2].imshow(anomaly_map, cmap='hot', alpha=0.6, interpolation='bilinear')
    axes[2].set_title(f'Overlay Heatmap\nStatus: {anomaly_status}\nMax: {max_score:.4f}, Mean: {mean_score:.4f}')
    axes[2].axis('off')

    plt.tight_layout()
    plt.show()

    print(f"Status: {anomaly_status} (Threshold: {threshold})")
    print(f"Max Anomaly Score: {max_score:.4f}, Mean Anomaly Score: {mean_score:.4f}")

# Main Functions
def training_main():
    """Main function untuk training"""
    print("=== Training EfficientAD-M ===")
    
    # Check if training data exists, if not upload
    if not os.path.exists('/content/data-normal') or len(os.listdir('/content/data-normal')) == 0:
        print("Data training tidak ditemukan. Silakan upload data training terlebih dahulu.")
        upload_train_data()
    
    # Create training dataset dan loader
    train_dataset = CustomTrainDataset(root_dir='/content/data-normal', transform=train_transform)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2)
    
    print(f"Training samples: {len(train_dataset)}")
    
    # Initialize model
    model = EfficientAD_M().to(device)
    
    # Start training
    train_losses = train_model(model, train_loader, num_epochs=5, lr=0.001)
    
    print("=== Training Completed ===")
    return model, train_losses

def inference_main():
    """Main function untuk inferensi"""
    print("=== Inferensi EfficientAD-M pada Data Test ===")

    # Upload data test jika belum ada
    if not os.path.exists('/content/test_data') or len(os.listdir('/content/test_data')) == 0:
        upload_test_data()

    # Create test dataset dan loader
    test_dataset = CustomTestDataset(root_dir='/content/test_data', transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

    print(f"Test samples: {len(test_dataset)}")

    # Load model
    model = load_model()
    if model is None:
        print("Model tidak ditemukan. Silakan lakukan training terlebih dahulu.")
        return

    # Proses setiap gambar test
    for batch in test_loader:
        image_tensor = batch['image'].to(device)
        original_image = batch['original_image'][0]
        image_path = batch['path'][0]

        print(f"\nMemproses: {os.path.basename(image_path)}")
        visualize_heatmap_and_detect(model, image_tensor, original_image)

    print("\n=== Inferensi Selesai ===")

def main():
    """Main function dengan pilihan mode"""
    print("=== EfficientAD-M untuk Visual Anomaly Detection ===")
    print("Pilih mode:")
    print("1. Training (menggunakan data normal dari /content/data-normal)")
    print("2. Inference (menggunakan data test dari /content/test_data)")
    print("3. Training + Inference")
    
    mode = input("Masukkan pilihan (1/2/3): ").strip()
    
    if mode == "1":
        training_main()
    elif mode == "2":
        inference_main()
    elif mode == "3":
        print("Menjalankan training terlebih dahulu...")
        training_main()
        print("\nMelanjutkan ke inference...")
        inference_main()
    else:
        print("Pilihan tidak valid. Menjalankan mode training + inference...")
        training_main()
        inference_main()

# Auto-run functions (uncomment yang dibutuhkan)
if __name__ == "__main__":
    # Uncomment salah satu sesuai kebutuhan:
    
    main()  # Interactive mode dengan pilihan
    
    # training_main()  # Langsung training
    
    # inference_main()  # Langsung inference (pastikan model sudah ada)
