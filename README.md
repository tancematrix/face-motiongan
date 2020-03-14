***
# データセット作成

bvhファイルは`{スタイルの名前}_*.npy`という名前で適当な場所に置いておく。  
以下のスクリプトを実行すると、指定した場所に元のbvhファイルのコピーと、joint-positionに変換した動作のnpyファイルが生成される。
```
python make_dataset.py --dataset {ROOT_OF_BVH_FILES} --out {OUTPUT_DIRECTORY}
```
また、正規化等に使用されるSkeleton情報は、デフォルトでは先頭のbvhファイルのものが参照されるが、コマンドラインオプション`--standard_bvh`で指定して任意のファイルを参照することもできる。

### 学習用のPickleファイルの生成
各npyファイルはあくまで動作のjoint positionの情報のみを持っており、学習に必要なスタイルラベルやSpline補間曲線等の情報は適切な前処理で取り出す必要がある。  
この前処理は学習の設定に依存するため、Datasetインスタンスを生成する際に行う(core/datasets.dataset.py参照)が、学習の度に毎回行うにはやや重たい処理である。  
そこで、前処理されたデータをPickleファイルで別々に保存しておき、同様の設定で学習する際はそれを読みだすようにしている。このPickleファイルは、npyファイルと同一の階層に作成される*processed_xxx*"というディレクトリ下に保存される。


***
# 学習
モデルの学習は以下のコマンドにより実行する。
```
python train.py {PATH_TO_CONFIG_FILE}
```
特定のcheckpoint(.pth.tar)から学習を再開する場合、コマンドラインオプション`--resume`を用いて指定する。

## Configファイルについて
学習の設定は基本的に全てConfigファイルで扱う。  
Configファイルは辞書の階層構造を取り、core/utils/config.pyに従ってパースされる。  
代表的なプロパティについて説明を以下に載せる。

### モデルの指定
モデル情報はConfigファイルのmodelsプロパティに記載される。以下各プロパティの説明。

#### Generator  

| プロパティ| 説明 |
|:---:|:---|
| model | ネットワークの名前　|
| top | 最初のConv層の出力チャネル数 |
| padding_mode | 各Conv層のパディング方法 |
| kw | 各Conv層のKernelサイズ(幅) |
| w_dim | Latent Transformerの出力次元数 |
| use_z | ノイズzの生成方法 (ノイズを用いない場合None) |
| z_dim | ノイズzの次元数 |
| normalize_z | Latent Transformerの入力に対してPixelwiseNormalizationを適用するかどうか |

#### Discriminator
Generatorと同様のものは省略。

| プロパティ| 説明 |
|:---:|:---|
| norm | Normalizeレイヤを文字列で指定 |
| use_sigmoid | Real/Fakeの出力にSigmoid層を挟むどうか |


### データセットの指定
学習に用いるデータセットはtrain.datasetプロパティで指定する、以下各プロパティの説明。
| プロパティ| 説明 |
|:---:|:---|
| data_root | データセットの場所(root) |
| class_list | データセットに含まれるスタイルクラス一覧 |
| start_offset | 動作データをnpyファイルからロードする際，先頭でスキップするフレーム数 (キャリブレーション用のフレームなど) |
| control_point_interval | Spline補間を行う際のコントロールポイントの間隔(フレーム数) |
| standard_bvh | Skeleton情報を参照するbvhファイル |
| scale | データに対してかけるスケーリングの係数 |
| frame_nums | 学習に用いる1動作シーケンスのフレーム数  (これを↓のstepで割った長さがネットワークの入力長) |
| frame_step | 動作をサンプリングするフレームステップ |
| augment_fps | FPS Augmentationを行うかどうか |
| rotate | y軸を中心とした回転Augmentationを行うかどうか |
