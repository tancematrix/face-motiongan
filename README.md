大体の部分は以下のリポジトリからの借り物です。こちらのREADMEにも目を通していただくとありがたいです。

https://github.com/B-Step62/pytorch-motiongan-open

## Requirement
- Python3 (>=3.6)
- Pytorch (>=0.4.1)
- cv2
- scipy
- tensorboardX
- ImageMagick
- Matplotlib
- transforms3d
- Easydict
- Pillow


***
# データセット
RAVDESSデータセットに特殊化しています。

(T, 68, 3): 時系列、3次元、68特徴点　です。


# 各スクリプトの説明
## データセットの準備
PCAで圧縮したデータセットを作るためには以下の二つのスクリプトを走らせる必要があります。

- openface2data.py
openfaceのデータ（csv）からmotion部分のデータを取り出すスクリプト。`--use2d`フラグを付けると、二次元特徴点を出力する。そうでなければ三次元。以下では三次元を前提としています。

```
python openface2data.py --source OPENFACEの結果のcsvファイルが存在するディレクトリ --target 出力ディレクトリ
```

- make_pca_dataset.py
PCAによって次元圧縮を行う。このスクリプトはもともと書き捨てで、ちゃんと整理できていないです...すみません。
PATHなどハードコードしています。大した処理はしていないのでコードを見てみてください、すみません。

## 学習
- train.py, train_pca.py

学習を実行する。train_pca.pyは入力データがPCAによる変換データである場合（デフォルト）に使う。

何が違うかというと、train_pca.pyの方はPCAの逆変換によって3次元データに復元してロスをとったりしている。

学習の設定は`configs?MotionGAN/*.py`に記述する。train_pca.pyを使う場合、pca_rootという項目を設定しないといけない。pca_rootにはPCAの結果の主成分ベクトルと平均ベクトルを置く。

実行例
```
python train_pca.py config/MotionGAN/FaseAnalysis.py
```

## 生成
- transfer.py, transfer_pca.py
学習結果の重みを使って感情変換を行う。`--source_npy`で指定したファイル（任意の感情）を、学習時に存在した8感情それぞれに変換した結果がresult_dirに保存される。

```
python transfer_pca.py config/MotionGAN/FaseAnalysis.py --source_npy 変換元のnpyファイル --result_dir 結果格納ディレクトリ
```

`--source ディレクトリ名` でディレクトリ以下のnpyファイルをまとめて変換。（`--source_npy`とどちらかだけをつかう）

- sample_generation.py, sample_generation_pca.py
transfer*.pyの原型です。多分使う必要はないです。

## 評価
参考までに。多分使わないと思います。

- test.py
テスト。竹内は使っていない（上記MotionGANを引き継いでおいてあるだけ）

- classify.py
ディスクリミネーターの評価スクリプト

- hosvd.py
Facial Expression Decomposition(https://ieeexplore.ieee.org/document/1238452)
の再実装。比較のために用いた。

- eval_transfer.py
変換の結果の評価スクリプト。変換前後の距離の比較など。必要があったら竹内に聞いてください。結局ボツにした評価結果で、使わなくていいと思います。


# 動画生成
ここまでで扱ってきたデータは点群だった。点群で表されるmotionデータを元に、動画の感情変換をするスクリプトは`script/`以下にまとまっている。

warp_face.py, animate_face_from_csv.pyが本体。

もともとの動画の各フレームについて、そのフレームの特徴点座標（二次元）->変換後のmotionデータの特徴点座標（二次元）というワーピングを施す。

warp_face.pyがワーピングのライブラリ的な位置づけで、実行するのはanimate_face_from_csv.py

```
python animate_face_from_csv.py -s SOURCE_CSV -t TARGET_NPY -m SOURCE_MP4 --title TITLE --out OUT_FILENAME
```

SOURCE_CSV: SOURCE_MP4に対応する、openfaceの結果のcsv

SOURCE_MP4: 変換前動画

TARGET_NPY: 変換後のmotionデータ

TITLE: 変換後の動画のキャプション（不要ならつけない）

OUT: 出力ファイルのパス

なお、animate_face.pyは古いバージョン。入力として、source_npy（PCAによる圧縮をする前のT, 68, 3次元のnpyデータ）、source_img（source_mp4のうち適当な１フレームを画像として出力したもの）を指定する。

一枚の画像をワーピングで動かすのがanimate_face.pyで、動画の各フレームを動かすのがanimate_face_from_csv.py

ファイル名が悪いのは見逃してください...