# Visual Anomaly Detection dengan Autoencoder
# Struktur folder: data/train/normal/, data/test/defect/, data/test/normal/

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from PIL import Image
import zipfile
import requests
from io import BytesIO

# ======================= SETUP DATA =======================

def create_sample_dataset():
    """Membuat dataset sample untuk testing"""
    print("Membuat struktur folder...")
    
    # Buat struktur folder
    os.makedirs('data/train/normal', exist_ok=True)
    os.makedirs('data/test/normal', exist_ok=True)
    os.makedirs('data/test/defect', exist_ok=True)
    
    # Generate gambar sample normal (kotak dengan variasi kecil)
    def create_normal_image(idx):
        img = np.ones((64, 64, 3)) * 0.8  # Background abu-abu
        # Tambah kotak di tengah
        img[20:44, 20:44] = [0.2, 0.3, 0.9]  # Kotak biru
        # Tambah noise kecil
        noise = np.random.normal(0, 0.05, img.shape)
        img = np.clip(img + noise, 0, 1)
        return (img * 255).astype(np.uint8)
    
    # Generate gambar sample defect (dengan anomali)
    def create_defect_image(idx):
        img = np.ones((64, 64, 3)) * 0.8  # Background abu-abu
        img[20:44, 20:44] = [0.2, 0.3, 0.9]  # Kotak biru
        
        # Tambah defect (spot merah, garis, dll)
        if idx % 3 == 0:
            # Red spot
            img[15:25, 15:25] = [0.9, 0.1, 0.1]
        elif idx % 3 == 1:
            # White line
            img[30, :] = [1.0, 1.0, 1.0]
        else:
            # Dark patch
            img[40:55, 10:25] = [0.1, 0.1, 0.1]
            
        # Tambah noise
        noise = np.random.normal(0, 0.05, img.shape)
        img = np.clip(img + noise, 0, 1)
        return (img * 255).astype(np.uint8)
    
    # Generate training normal images
    print("Generating training normal images...")
    for i in range(100):
        img = create_normal_image(i)
        Image.fromarray(img).save(f'data/train/normal/normal_{i:03d}.jpg')
    
    # Generate test normal images
    print("Generating test normal images...")
    for i in range(30):
        img = create_normal_image(i + 100)
        Image.fromarray(img).save(f'data/test/normal/normal_{i:03d}.jpg')
    
    # Generate test defect images
    print("Generating test defect images...")
    for i in range(30):
        img = create_defect_image(i)
        Image.fromarray(img).save(f'data/test/defect/defect_{i:03d}.jpg')
    
    print("Dataset sample berhasil dibuat!")

# ======================= DATA LOADING =======================

def load_images_from_folder(folder_path, target_size=(64, 64)):
    """Load gambar dari folder"""
    images = []
    filenames = []
    
    if not os.path.exists(folder_path):
        print(f"Folder {folder_path} tidak ditemukan!")
        return np.array([]), []
    
    for filename in sorted(os.listdir(folder_path)):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(folder_path, filename)
            try:
                img = Image.open(img_path).convert('RGB')
                img = img.resize(target_size)
                img_array = np.array(img) / 255.0  # Normalize ke [0,1]
                images.append(img_array)
                filenames.append(filename)
            except Exception as e:
                print(f"Error loading {filename}: {e}")
    
    return np.array(images), filenames

def load_data():
    """Load semua data"""
    print("Loading training data...")
    train_normal, train_filenames = load_images_from_folder('data/train/normal')
    
    print("Loading test data...")
    test_normal, test_normal_filenames = load_images_from_folder('data/test/normal')
    test_defect, test_defect_filenames = load_images_from_folder('data/test/defect')
    
    print(f"Training normal: {len(train_normal)} images")
    print(f"Test normal: {len(test_normal)} images")
    print(f"Test defect: {len(test_defect)} images")
    
    return train_normal, test_normal, test_defect, test_normal_filenames, test_defect_filenames

# ======================= MODEL AUTOENCODER =======================

def build_autoencoder(input_shape=(64, 64, 3)):
    """Membangun model autoencoder"""
    
    # Encoder
    input_img = keras.Input(shape=input_shape)
    
    # Encoding layers
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    encoded = layers.MaxPooling2D((2, 2), padding='same')(x)
    
    # Decoder
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(encoded)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x)
    decoded = layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)
    
    autoencoder = keras.Model(input_img, decoded)
    autoencoder.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
    return autoencoder

# ======================= TRAINING =======================

def train_autoencoder(model, train_data, epochs=50, batch_size=16):
    """Training autoencoder"""
    print("Training autoencoder...")
    
    # Callback untuk early stopping
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='loss', patience=10, restore_best_weights=True
    )
    
    # Training
    history = model.fit(
        train_data, train_data,
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,
        validation_split=0.1,
        callbacks=[early_stopping],
        verbose=1
    )
    
    return history

# ======================= ANOMALY DETECTION =======================

def calculate_reconstruction_error(model, images):
    """Menghitung reconstruction error"""
    reconstructed = model.predict(images)
    mse = np.mean(np.square(images - reconstructed), axis=(1, 2, 3))
    return mse, reconstructed

def find_threshold(model, normal_data, percentile=95):
    """Menentukan threshold untuk anomaly detection"""
    errors, _ = calculate_reconstruction_error(model, normal_data)
    threshold = np.percentile(errors, percentile)
    return threshold

# ======================= VISUALISASI =======================

def plot_training_history(history):
    """Plot training history"""
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='Training MAE')
    plt.plot(history.history['val_mae'], label='Validation MAE')
    plt.title('Model MAE')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

def visualize_reconstructions(model, images, titles, n_samples=5):
    """Visualisasi original vs reconstructed images"""
    if len(images) == 0:
        print("No images to visualize!")
        return
        
    reconstructed, _ = calculate_reconstruction_error(model, images)
    reconstructed_imgs = model.predict(images)
    
    n_samples = min(n_samples, len(images))
    
    plt.figure(figsize=(15, 6))
    for i in range(n_samples):
        # Original
        plt.subplot(3, n_samples, i + 1)
        plt.imshow(images[i])
        plt.title(f'Original\n{titles[i]}')
        plt.axis('off')
        
        # Reconstructed
        plt.subplot(3, n_samples, i + 1 + n_samples)
        plt.imshow(reconstructed_imgs[i])
        plt.title(f'Reconstructed')
        plt.axis('off')
        
        # Difference
        plt.subplot(3, n_samples, i + 1 + 2*n_samples)
        diff = np.abs(images[i] - reconstructed_imgs[i])
        plt.imshow(diff, cmap='hot')
        plt.title(f'Difference\nError: {reconstructed[i]:.4f}')
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def plot_error_distribution(normal_errors, defect_errors, threshold):
    """Plot distribusi error"""
    plt.figure(figsize=(10, 6))
    
    plt.hist(normal_errors, bins=30, alpha=0.7, label='Normal', color='blue')
    plt.hist(defect_errors, bins=30, alpha=0.7, label='Defect', color='red')
    plt.axvline(threshold, color='green', linestyle='--', linewidth=2, 
                label=f'Threshold: {threshold:.4f}')
    
    plt.xlabel('Reconstruction Error')
    plt.ylabel('Frequency')
    plt.title('Distribution of Reconstruction Errors')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

def plot_confusion_matrix(y_true, y_pred, labels=['Normal', 'Defect']):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

def evaluate_model(model, test_normal, test_defect, threshold):
    """Evaluasi model"""
    # Hitung errors
    normal_errors, _ = calculate_reconstruction_error(model, test_normal)
    defect_errors, _ = calculate_reconstruction_error(model, test_defect)
    
    # Prediksi
    normal_pred = (normal_errors > threshold).astype(int)  # 0 = normal, 1 = anomaly
    defect_pred = (defect_errors > threshold).astype(int)
    
    # Ground truth
    y_true = np.concatenate([np.zeros(len(test_normal)), np.ones(len(test_defect))])
    y_pred = np.concatenate([normal_pred, defect_pred])
    
    # Print classification report
    print("\n=== EVALUATION RESULTS ===")
    print(f"Threshold: {threshold:.4f}")
    print(f"Normal images classified as anomaly: {np.sum(normal_pred)}/{len(normal_pred)}")
    print(f"Defect images classified as anomaly: {np.sum(defect_pred)}/{len(defect_pred)}")
    
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=['Normal', 'Defect']))
    
    # Plot confusion matrix
    plot_confusion_matrix(y_true, y_pred)
    
    # Plot error distribution
    plot_error_distribution(normal_errors, defect_errors, threshold)
    
    return normal_errors, defect_errors

# ======================= MAIN EXECUTION =======================

def main():
    """Fungsi utama"""
    print("=== VISUAL ANOMALY DETECTION ===\n")
    
    # 1. Create sample dataset (jika belum ada)
    if not os.path.exists('data'):
        create_sample_dataset()
    
    # 2. Load data
    train_normal, test_normal, test_defect, test_normal_names, test_defect_names = load_data()
    
    if len(train_normal) == 0:
        print("Tidak ada data training! Pastikan struktur folder benar.")
        return
    
    # 3. Build dan train model
    print("\n=== BUILDING MODEL ===")
    model = build_autoencoder()
    model.summary()
    
    print("\n=== TRAINING ===")
    history = train_autoencoder(model, train_normal, epochs=30)
    plot_training_history(history)
    
    # 4. Tentukan threshold
    print("\n=== DETERMINING THRESHOLD ===")
    threshold = find_threshold(model, train_normal, percentile=95)
    print(f"Threshold determined: {threshold:.4f}")
    
    # 5. Visualisasi reconstruction
    print("\n=== VISUALIZING RECONSTRUCTIONS ===")
    if len(test_normal) > 0:
        print("Normal test images:")
        visualize_reconstructions(model, test_normal, test_normal_names)
    
    if len(test_defect) > 0:
        print("Defect test images:")
        visualize_reconstructions(model, test_defect, test_defect_names)
    
    # 6. Evaluasi model
    print("\n=== EVALUATION ===")
    if len(test_normal) > 0 and len(test_defect) > 0:
        normal_errors, defect_errors = evaluate_model(model, test_normal, test_defect, threshold)
    else:
        print("Tidak cukup data test untuk evaluasi!")
    
    # 7. Save model
    print("\n=== SAVING MODEL ===")
    model.save('anomaly_detection_model.h5')
    print("Model saved as 'anomaly_detection_model.h5'")
    
    print("\n=== SELESAI ===")
    print("Anomaly detection berhasil dijalankan!")

# ======================= RUN =======================

if __name__ == "__main__":
    main()
