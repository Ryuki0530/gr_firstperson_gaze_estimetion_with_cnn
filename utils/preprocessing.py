import cv2
import os

PRINT_LOG = True

def extract_frames_from_video(video_path, output_dir, frame_interval=1):
    """
    動画ファイルからフレームを抽出して保存

    Parameters:
        video_path (str): 動画ファイルへのパス
        output_dir (str): フレーム画像の保存先ディレクトリ
        frame_interval (int): 何フレームごとに保存するか（デフォルト=1で全保存）
    """
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"動画ファイルを開けません: {video_path}")

    frame_count = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            filename = f"frame_{saved_count:05d}.jpg"
            path = os.path.join(output_dir, filename)
            cv2.imwrite(path, frame)
            if PRINT_LOG:
                print("frameNumber:"+str(saved_count)+" saved")
            saved_count += 1

        frame_count += 1

    cap.release()
    print(f"保存完了: {saved_count} 枚のフレームを {output_dir} に保存しました。")

if __name__ == "__main__":
    # 使用例
    video_file = "data/raw_videos/Ahmad_American.avi"
    output_folder = "data/frames"
    extract_frames_from_video(video_file, output_folder, frame_interval=1)
