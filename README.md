https://github.com/Sierkinhane/crnn_chinese_characters_rec

https://github.com/xiaofengShi/CHINESE-OCR

https://github.com/eragonruan/text-detection-ctpn

## INSTALL

Ubuntu18/16.04 + CUDA 8.0.61 + GeForce GTX 960M + NVIDIA Driver 430.14 + Python3.6 + Tensorflow-gpu

## ctpn

```bash
cd ctpn/utils/bbox
chmod +x make.sh
./make.sh

python ctpn/demo.py

```

test_image_folder: data/test_images

ctpn_middel_result_folder: data/middle_result

ctpn_result_folder: data/res # number_x.jpg or .jpeg

## crnn

```bash
python handle_images.py
```
handle images floder: data/images/

image<->correct number folder: crnn/to_lmdb/train.txt

handle images result folder: crnn/to_lmdb/train_images

```bash
cd crnn/to_lmdb

python tolmdb_py3.py # python to_lmdb_py2.py 
```

lmdb folder: crnn/train/lmdb

```bash
cd crnn

python crnn_main.py # train models

```
trainroot folder: crnn/to_lmdb/lmdb

valroot folder: crnn/to_lmdb/lmdb

train models result folder: crnn/expr

change params: crnn/params.py

if you load_state_dict, change crnn/params.py crnn='your pth path'

```bash
cd crnn

python test.py
```

crnn model path: crnn/trained_models/crnn_Rec_done.pth

images_path: data/res


