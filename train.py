import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from models.gaze_cnn import GazeCNN
from utils.dataset import GazeDataset
from torchvision import transforms
from tqdm import tqdm

# --- 引数のパース ---
parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs')
args = parser.parse_args()

# --- ハイパーパラメータ ---
BATCH_SIZE = 32
EPOCHS = args.epochs
LEARNING_RATE = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_SAVE_DIR = "models/results"
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
print(f"Using device: {DEVICE}")

# --- データセットの準備 ---
train_dataset = GazeDataset(
    image_dir="data/frames",
    csv_path="data/gaze_labels.csv",
    transform=transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# --- モデル準備 ---
model = GazeCNN().to(DEVICE)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# --- 学習ループ ---
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {avg_loss:.4f}")

    save_path = os.path.join(MODEL_SAVE_DIR, f"gaze_model_epoch{epoch+1}.pth")
    torch.save(model.state_dict(), save_path)
