import torch
import cv2
import numpy as np
from torchvision import transforms
from models.gaze_cnn import GazeCNN

# --- 設定 ---
VIDEO_PATH = "data/raw_videos/Ahmad_American.avi"
MODEL_PATH = "models/results/gaze_model_epoch20.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = (224, 224)

# --- モデルの読み込み ---
model = GazeCNN()
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model = model.to(DEVICE)
model.eval()

# --- 変換処理 ---
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor()
])

# --- 動画読み込み ---
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print(f"Error: {VIDEO_PATH} を開けませんでした")
    exit()

# --- FPS 調整（60FPS再生） ---
TARGET_FPS = 60
frame_interval_ms = int(1000 / TARGET_FPS)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    orig_h, orig_w, _ = frame.shape

    # 前処理
    img = transform(frame).unsqueeze(0).to(DEVICE)

    # 推論
    with torch.no_grad():
        output = model(img)
        x_norm, y_norm = output[0].cpu().numpy()

    # 元サイズに変換
    x = int(x_norm * orig_w)
    y = int(y_norm * orig_h)

    # 描画
    cv2.circle(frame, (x, y), 8, (0, 0, 255), -1)
    cv2.imshow("Gaze Prediction", frame)

    if cv2.waitKey(frame_interval_ms) & 0xFF == 27:  # Escキーで終了
        break

cap.release()
cv2.destroyAllWindows()
