import pandas as pd
import sys

# コマンドライン引数でファイルパスを取得
if len(sys.argv) < 2:
    print("Usage: python dataSetChange.py <txt_file_path>")
    sys.exit(1)

txt_path = sys.argv[1]

# 画像サイズ設定
FRAME_WIDTH = 1920
FRAME_HEIGHT = 1080

# データを読み込む（タブ区切り）
df = pd.read_csv(txt_path, sep='\t', header=None, names=['Time', 'Type', 'Trial', 'L POR X [px]', 'L POR Y [px]', 'Frame', 'Aux1', 'L Event Info'])

# Fixation のみ抽出
df = df[df['L Event Info'] == 'Fixation']

# 平均 gaze をフレーム単位で計算
df_grouped = df.groupby('Frame')[['L POR X [px]', 'L POR Y [px]']].mean().reset_index()

# 正規化
df_grouped['gaze_x'] = df_grouped['L POR X [px]'] / FRAME_WIDTH
df_grouped['gaze_y'] = df_grouped['L POR Y [px]'] / FRAME_HEIGHT
df_grouped['image_name'] = df_grouped['Frame'].apply(lambda x: f"frame_{int(x):05d}.jpg")

# 最終出力
df_grouped[['image_name', 'gaze_x', 'gaze_y']].to_csv('gaze_labels.csv', index=False, float_format='%.5f')