# プロジェクト概要

このプロジェクトは、畳み込みニューラルネットワーク (CNN) を使用して、ファーストパーソン視点の視線推定を行うためのものです。

## 特徴

- ファーストパーソン視点の画像データを使用
- CNNを活用したリアルタイム前提の視線推定

## 必要条件

以下のソフトウェアとライブラリが必要です：

- Python 3.8以上
- TensorFlow 2.x
- OpenCV
- NumPy
- その他の依存関係は`requirements.txt`をご確認ください。

## インストール

1. リポジトリをクローンします：
    ```bash
    git clone https://github.com/Ryuki053/gr_firstperson_gaze_estimation_with_cnn.git
    cd gr_firstperson_gaze_estimation_with_cnn
    ```

2. 必要なライブラリをインストールします：
    ```bash
    pip install -r requirements.txt
    ```

## 使用方法

1. データセットを準備し、`data/`ディレクトリに配置します。
データセットの形式は、`data/readme.txt`を参照してください。

2. 以下のコマンドでモデルをトレーニングします：
    ```bash
    python train.py
    ```
3. 推論を実行するには、以下を使用します：
    ```bash
    python realtime_infer.py
    ```

## データセット

このプロジェクトでは、独自のデータセットまたは公開されている視線推定用データセットを使用できます。

## 貢献

バグ報告や機能提案は歓迎します。プルリクエストを送る前に、必ずIssueを作成してください。

## ライセンス
このプロジェクトはMITライセンスの下で提供されています。詳細は`LICENSE`ファイルをご確認ください。

## 著者

- 名前: Ryuki Fujita
- GitHub: [Ryuki053](https://github.com/Ryuki053)