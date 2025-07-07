import cv2
import os
import pandas as pd
import sys

PRINT_LOG = True

def extract_frames_from_video(video_path, output_dir, frame_interval=1):
    """
    動画ファイルからフレームを抽出して保存

    Parameters:
        video_path (str): 動画ファイルへのパス
        output_dir (str): フレーム画像の保存先ディレクトリ
        frame_interval (int): 何フレームごとに保存するか（デフォルト=1で全保存）
    
    Returns:
        int: 保存されたフレーム数
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
                print(f"frameNumber: {saved_count} saved")
            saved_count += 1

        frame_count += 1

    cap.release()
    print(f"保存完了: {saved_count} 枚のフレームを {output_dir} に保存しました。")
    return saved_count

def process_gaze_data(txt_path, output_csv="gaze_labels.csv", frame_width=1280, frame_height=960, max_frame=None):
    """
    視線データを処理してCSVファイルに出力

    Parameters:
        txt_path (str): 視線データファイル（TSV）へのパス
        output_csv (str): 出力CSVファイル名
        frame_width (int): フレーム幅
        frame_height (int): フレーム高さ
        max_frame (int): 最大フレーム数（Noneの場合は自動検出）
    
    Returns:
        pd.DataFrame: 処理済みの視線データ
    """
    # データを読み込む（タブ区切り）
    df = pd.read_csv(txt_path, sep='\t', header=None, 
                     names=['Time', 'Type', 'Trial', 'L POR X [px]', 'L POR Y [px]', 
                           'Frame', 'Aux1', 'L Event Info'])

    # Fixation のみ抽出
    df = df[df['L Event Info'] == 'Fixation']

    # 平均 gaze をフレーム単位で計算
    df_grouped = df.groupby('Frame')[['L POR X [px]', 'L POR Y [px]']].mean().reset_index()

    # 正規化
    df_grouped['gaze_x'] = df_grouped['L POR X [px]'] / frame_width
    df_grouped['gaze_y'] = df_grouped['L POR Y [px]'] / frame_height
    df_grouped['image_name'] = df_grouped['Frame'].apply(lambda x: f"frame_{int(x):05d}.jpg")

    # 最大フレーム数を決定
    if max_frame is None:
        max_frame = int(df_grouped['Frame'].max())
    
    # 全フレーム範囲のデータフレームを作成
    all_frames = pd.DataFrame({
        'Frame': range(0, max_frame + 1),
        'image_name': [f"frame_{i:05d}.jpg" for i in range(0, max_frame + 1)]
    })
    
    # 視線データとマージ
    merged_df = all_frames.merge(df_grouped[['Frame', 'gaze_x', 'gaze_y']], on='Frame', how='left')
    
    # 欠損値を直前の値で埋める（forward fill）
    merged_df['gaze_x'] = merged_df['gaze_x'].fillna(method='ffill')
    merged_df['gaze_y'] = merged_df['gaze_y'].fillna(method='ffill')
    
    # 最初のフレームで視線データがない場合は0.5（画面中央）で初期化
    merged_df['gaze_x'] = merged_df['gaze_x'].fillna(0.5)
    merged_df['gaze_y'] = merged_df['gaze_y'].fillna(0.5)

    # 最終出力
    result_df = merged_df[['image_name', 'gaze_x', 'gaze_y']]
    result_df.to_csv(output_csv, index=False, float_format='%.5f')
    
    missing_count = df_grouped.shape[0]
    total_count = result_df.shape[0]
    filled_count = total_count - missing_count
    
    print(f"視線データ処理完了: {total_count} 件のデータを {output_csv} に保存しました。")
    print(f"  - 元データ: {missing_count} 件")
    print(f"  - 補完データ: {filled_count} 件")
    return result_df

def process_video_and_gaze(video_path, gaze_txt_path, output_frames_dir, output_csv="gaze_labels.csv", 
                          frame_interval=1, frame_width=1280, frame_height=960):
    """
    動画とアイトラッキングデータを統合処理

    Parameters:
        video_path (str): 動画ファイルへのパス
        gaze_txt_path (str): 視線データファイル（TSV）へのパス
        output_frames_dir (str): フレーム画像の保存先ディレクトリ
        output_csv (str): 出力CSVファイル名
        frame_interval (int): フレーム抽出間隔
        frame_width (int): フレーム幅
        frame_height (int): フレーム高さ
    """
    print("=== 動画フレーム抽出開始 ===")
    saved_frames = extract_frames_from_video(video_path, output_frames_dir, frame_interval)
    
    print("\n=== 視線データ処理開始 ===")
    gaze_df = process_gaze_data(gaze_txt_path, output_csv, frame_width, frame_height, saved_frames)
    
    print(f"\n=== 処理完了 ===")
    print(f"抽出フレーム数: {saved_frames}")
    print(f"視線データ数: {len(gaze_df)}")
    
    return saved_frames, gaze_df

if __name__ == "__main__":
    if len(sys.argv) >= 3:
        # コマンドライン引数での実行
        if len(sys.argv) < 3:
            print("Usage: python data_processor.py <video_path> <gaze_txt_path> [output_frames_dir] [output_csv]")
            sys.exit(1)
        
        video_file = sys.argv[1]
        gaze_file = sys.argv[2]
        frames_dir = sys.argv[3] if len(sys.argv) > 3 else "data/frames"
        csv_file = sys.argv[4] if len(sys.argv) > 4 else "gaze_labels.csv"
        
        process_video_and_gaze(video_file, gaze_file, frames_dir, csv_file)
    else:
  
        video_file = "data/raw_videos/Ahmad_American.avi"
        gaze_file = "data/raw_gaze_labels/Ahmad_American.csv"
        output_folder = "data/frames"
        csv_output = "gaze_labels.csv"
        
        process_video_and_gaze(video_file, gaze_file, output_folder, csv_output)
