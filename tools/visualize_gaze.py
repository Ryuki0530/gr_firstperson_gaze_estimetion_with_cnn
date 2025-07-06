import cv2
import pandas as pd
import numpy as np
import os
import sys
import argparse
from pathlib import Path

def draw_gaze_point(image, gaze_x, gaze_y, radius=10, color=(0, 255, 0), thickness=3):
    """
    画像上に視線点を描画
    
    Parameters:
        image: OpenCV画像
        gaze_x: 正規化された視線X座標 (0-1)
        gaze_y: 正規化された視線Y座標 (0-1)
        radius: 円の半径
        color: 色 (B, G, R)
        thickness: 線の太さ
    """
    height, width = image.shape[:2]
    
    # 正規化座標を画像座標に変換
    pixel_x = int(gaze_x * width)
    pixel_y = int(gaze_y * height)
    
    # 視線点を円で描画
    cv2.circle(image, (pixel_x, pixel_y), radius, color, thickness)
    
    # 十字線を描画（より見やすくするため）
    cross_size = radius + 5
    cv2.line(image, (pixel_x - cross_size, pixel_y), (pixel_x + cross_size, pixel_y), color, thickness//2)
    cv2.line(image, (pixel_x, pixel_y - cross_size), (pixel_x, pixel_y + cross_size), color, thickness//2)
    
    return image

def add_info_text(image, frame_name, gaze_x, gaze_y, frame_idx, total_frames):
    """
    画像に情報テキストを追加
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    color = (255, 255, 255)
    thickness = 2
    
    # 背景用の黒い矩形を描画
    text_bg_color = (0, 0, 0)
    cv2.rectangle(image, (10, 10), (400, 120), text_bg_color, -1)
    
    # テキスト情報を描画
    texts = [
        f"Frame: {frame_name} ({frame_idx+1}/{total_frames})",
        f"Gaze: ({gaze_x:.3f}, {gaze_y:.3f})",
        f"Pixel: ({int(gaze_x * image.shape[1])}, {int(gaze_y * image.shape[0])})",
        "Press 'q' to quit, 'n' for next, 'p' for previous"
    ]
    
    for i, text in enumerate(texts):
        y_pos = 30 + i * 25
        cv2.putText(image, text, (15, y_pos), font, font_scale, color, thickness)
    
    return image

def visualize_gaze_data(frames_dir, csv_path, output_dir=None, auto_play=False, delay=1000):
    """
    視線データと画像を重ねて表示
    
    Parameters:
        frames_dir: フレーム画像のディレクトリ
        csv_path: 視線データCSVファイル
        output_dir: 出力ディレクトリ（Noneの場合は表示のみ）
        auto_play: 自動再生モード
        delay: 自動再生時の遅延時間（ミリ秒）
    """
    # CSVデータを読み込み
    try:
        df = pd.read_csv(csv_path)
        print(f"視線データを読み込みました: {len(df)} 件")
    except Exception as e:
        print(f"CSVファイル読み込みエラー: {e}")
        return
    
    # 出力ディレクトリの作成
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        print(f"可視化画像を {output_dir} に保存します")
    
    frame_idx = 0
    total_frames = len(df)
    
    print("操作方法:")
    print("  'n' または 'd': 次のフレーム")
    print("  'p' または 'a': 前のフレーム") 
    print("  'q' または ESC: 終了")
    print("  's': 現在のフレームを保存")
    print("  SPACE: 自動再生の開始/停止")
    
    auto_mode = auto_play
    
    while True:
        if frame_idx < 0:
            frame_idx = 0
        elif frame_idx >= total_frames:
            frame_idx = total_frames - 1
        
        # 現在のフレームデータを取得
        row = df.iloc[frame_idx]
        image_name = row['image_name']
        gaze_x = row['gaze_x']
        gaze_y = row['gaze_y']
        
        # 画像ファイルパス
        image_path = os.path.join(frames_dir, image_name)
        
        if not os.path.exists(image_path):
            print(f"警告: 画像ファイルが見つかりません: {image_path}")
            frame_idx += 1
            continue
        
        # 画像を読み込み
        image = cv2.imread(image_path)
        if image is None:
            print(f"警告: 画像を読み込めません: {image_path}")
            frame_idx += 1
            continue
        
        # 視線点を描画
        vis_image = image.copy()
        vis_image = draw_gaze_point(vis_image, gaze_x, gaze_y)
        vis_image = add_info_text(vis_image, image_name, gaze_x, gaze_y, frame_idx, total_frames)
        
        # 画像を表示
        cv2.imshow('Gaze Visualization', vis_image)
        
        # 自動再生モードでない場合のキー入力待機
        if auto_mode:
            key = cv2.waitKey(delay) & 0xFF
        else:
            key = cv2.waitKey(0) & 0xFF
        
        # キー操作の処理
        if key == ord('q') or key == 27:  # 'q' または ESC
            break
        elif key == ord('n') or key == ord('d'):  # 次のフレーム
            frame_idx += 1
            auto_mode = False
        elif key == ord('p') or key == ord('a'):  # 前のフレーム
            frame_idx -= 1
            auto_mode = False
        elif key == ord('s'):  # 保存
            if output_dir:
                output_path = os.path.join(output_dir, f"gaze_vis_{image_name}")
                cv2.imwrite(output_path, vis_image)
                print(f"保存しました: {output_path}")
            else:
                print("出力ディレクトリが指定されていません")
        elif key == ord(' '):  # スペースキー: 自動再生切り替え
            auto_mode = not auto_mode
            print(f"自動再生モード: {'ON' if auto_mode else 'OFF'}")
        
        # 自動再生モードで次のフレームへ
        if auto_mode:
            frame_idx += 1
            if frame_idx >= total_frames:
                print("最後のフレームに到達しました。自動再生を停止します。")
                auto_mode = False
                frame_idx = total_frames - 1
    
    cv2.destroyAllWindows()
    print("可視化ツールを終了します")

def batch_save_visualizations(frames_dir, csv_path, output_dir, start_frame=0, end_frame=None, step=1):
    """
    指定範囲のフレームの可視化画像を一括保存
    
    Parameters:
        frames_dir: フレーム画像のディレクトリ
        csv_path: 視線データCSVファイル
        output_dir: 出力ディレクトリ
        start_frame: 開始フレーム
        end_frame: 終了フレーム（Noneの場合は最後まで）
        step: ステップ数
    """
    # CSVデータを読み込み
    df = pd.read_csv(csv_path)
    
    if end_frame is None:
        end_frame = len(df)
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"一括保存開始: フレーム {start_frame} から {end_frame-1} まで（ステップ: {step}）")
    
    for frame_idx in range(start_frame, min(end_frame, len(df)), step):
        row = df.iloc[frame_idx]
        image_name = row['image_name']
        gaze_x = row['gaze_x']
        gaze_y = row['gaze_y']
        
        image_path = os.path.join(frames_dir, image_name)
        
        if not os.path.exists(image_path):
            print(f"スキップ: {image_path}")
            continue
        
        image = cv2.imread(image_path)
        if image is None:
            print(f"スキップ: {image_path}")
            continue
        
        # 視線点を描画
        vis_image = image.copy()
        vis_image = draw_gaze_point(vis_image, gaze_x, gaze_y)
        vis_image = add_info_text(vis_image, image_name, gaze_x, gaze_y, frame_idx, len(df))
        
        # 保存
        output_path = os.path.join(output_dir, f"gaze_vis_{image_name}")
        cv2.imwrite(output_path, vis_image)
        
        if frame_idx % 10 == 0:
            print(f"処理中: {frame_idx}/{end_frame}")
    
    print(f"一括保存完了: {output_dir}")

def main():
    parser = argparse.ArgumentParser(description='視線データと画像の可視化ツール')
    parser.add_argument('frames_dir', help='フレーム画像のディレクトリ')
    parser.add_argument('csv_path', help='視線データCSVファイル')
    parser.add_argument('--output', '-o', help='出力ディレクトリ（指定すると保存可能）')
    parser.add_argument('--auto', '-a', action='store_true', help='自動再生モード')
    parser.add_argument('--delay', '-d', type=int, default=1000, help='自動再生時の遅延時間（ミリ秒）')
    parser.add_argument('--batch', '-b', action='store_true', help='一括保存モード')
    parser.add_argument('--start', type=int, default=0, help='開始フレーム（一括保存時）')
    parser.add_argument('--end', type=int, help='終了フレーム（一括保存時）')
    parser.add_argument('--step', type=int, default=1, help='ステップ数（一括保存時）')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.frames_dir):
        print(f"エラー: フレームディレクトリが見つかりません: {args.frames_dir}")
        return
    
    if not os.path.exists(args.csv_path):
        print(f"エラー: CSVファイルが見つかりません: {args.csv_path}")
        return
    
    if args.batch:
        if not args.output:
            print("エラー: 一括保存モードでは出力ディレクトリの指定が必要です")
            return
        batch_save_visualizations(args.frames_dir, args.csv_path, args.output, 
                                args.start, args.end, args.step)
    else:
        visualize_gaze_data(args.frames_dir, args.csv_path, args.output, args.auto, args.delay)

if __name__ == "__main__":
    if len(sys.argv) == 1:
        # デフォルト実行（引数なしの場合）
        frames_dir = "data/frames"
        csv_path = "gaze_labels.csv"
        output_dir = "visualization_output"
        
        print("デフォルト設定で実行します:")
        print(f"  フレームディレクトリ: {frames_dir}")
        print(f"  CSVファイル: {csv_path}")
        print(f"  出力ディレクトリ: {output_dir}")
        
        if os.path.exists(frames_dir) and os.path.exists(csv_path):
            visualize_gaze_data(frames_dir, csv_path, output_dir)
        else:
            print("デフォルトファイルが見つかりません。引数を指定してください。")
            print("使用法: python visualize_gaze.py <frames_dir> <csv_path> [--output output_dir]")
    else:
        main()




# # USEGE
# # インタラクティブ表示
# python tools/visualize_gaze.py data/frames gaze_labels.csv
# # 出力ディレクトリを指定して保存可能にする
# python tools/visualize_gaze.py data/frames gaze_labels.csv --output visualization_output
# # 自動再生モード（1秒間隔）
# python tools/visualize_gaze.py data/frames gaze_labels.csv --auto --delay 1000
# # 全フレームを一括保存
# python tools/visualize_gaze.py data/frames gaze_labels.csv --batch --output vis_output
# # 特定範囲を10フレームごとに保存
# python tools/visualize_gaze.py data/frames gaze_labels.csv --batch --output vis_output --start 0 --end 100 --step 10    
# # 全フレームを一括保存
# python tools/visualize_gaze.py data/frames gaze_labels.csv --batch --output vis_output
# # 特定範囲を10フレームごとに保存
# python tools/visualize_gaze.py data/frames gaze_labels.csv --batch --output vis_output --start 0 --end 100 --step 10    

