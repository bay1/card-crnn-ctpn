更多问题参考[More details](https://www.cnblogs.com/bay1/p/10994600.html)

# HOW TO USE

## setup

[https://github.com/bay1/card-crnn-ctpn](https://github.com/bay1/card-crnn-ctpn)

### 环境配置

Ubuntu18.04 + CUDA 8.0.61 + GeForce GTX 960M + NVIDIA Driver 430.14 + Python3.6 + Tensorflow-gpu

```
git clone https://github.com/bay1/card-crnn-ctpn.git

python3 -m virtualenv venv
source venv/bin/activate # 激活虚拟环境

pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple some-package # 安装项目依赖，指定清华源
```

### 配置warpctc-pytorch

项目中用到了[warpctc-pytorch](https://github.com/SeanNaren/warp-ctc)，需要我们手动安装

注意这里的命令需要在Python虚拟环境中执行

```
git clone https://github.com/SeanNaren/warp-ctc.gitcd warp-ctc
mkdir build; cd build
cmake ..
make
cd ../pytorch_binding
python setup.py install
```

## Run demo

下载训练完成的模型，保存到路径crnn/trained_models/

在项目目录运行

```
python run.py
```

浏览器打开本地链接：[http://127.0.0.1:5000](http://127.0.0.1:5000/)

​![image](https://upload-images.jianshu.io/upload_images/3464381-37142265a41de95d?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240) ​

## Training

ctpn使用了[text-detection-ctpn](https://github.com/eragonruan/text-detection-ctpn)中的模型，下载ckpt： [googl drive](https://drive.google.com/file/d/1HcZuB_MHqsKhKEKpfF1pEU85CYy4OlWO/view?usp=sharing) or [baidu yun](https://pan.baidu.com/s/1BNHt_9fiqRPGmEXPaxaFXw)

将ckpt模型文件保存路径：`ctpn/checkpoints_mlt/`

字符区域识别需要大量的训练数据，出于训练数据的缺失和自己训练模型性能较低的考虑，本项目使用已经通过了大量数据训练过的模型来识别文字区域，通过对文字区域宽高比的计算获取银行卡号的正确区域，从而进入下一步识别。

本项目重点训练了crnn文字识别模型
下载本地训练结果：https://pan.baidu.com/s/1TXuewaVgGzXjT0EPCj47Bg 提取码: 7we3

### 训练数据

原始图片数据路径：`data/images`

对所有的训练数据（1084张）进行数据增强处理，将每张图片拓展成80张，得到八万张左右的训练数据。

初步处理，转化为生成lmdb需要的形式。

```
python crnn/augmentation.py
python crnn/handle_images.py
python crnn/to_lmdb/to_lmdb.py -i crnn/to_lmdb/train_images -l crnn/to_lmdb/train.txt -s crnn/to_lmdb/train_lmdb/
python crnn/to_lmdb/to_lmdb.py -i crnn/to_lmdb/test_images -l crnn/to_lmdb/test.txt -s crnn/to_lmdb/test_lmdb/
python crnn/train.py
```

### 自定义参数

`ctpn/params.py`

`crnn/params.py`

crnn中的参数详解：

```
--random_sample      是否使用随机采样器对数据集进行采样, action='store_true'
--keep_ratio         设置图片保持横纵比缩放, action='store_true'
--adam               使用adma优化器, action='store_true'
--adadelta           使用adadelta优化器, action='store_true'
--saveInterval       设置多少次迭代保存一次模型
--valInterval        设置多少次迭代验证一次
--n_test_disp        每次验证显示的个数
--displayInterval    设置多少次迭代显示一次
--experiment         模型保存目录
--alphabet           设置检测分类
--crnn               选择预训练模型
--beta1            
--lr                 学习率
--niter              训练回合数
--nh                 LSTM隐藏层数
--imgW               图片宽度
--imgH               图片高度, default=32
--batchSize          设置batchSize大小, default=64
--workers            工作核数, default=2
--trainroot          训练集路径
--valroot            验证集路径
--cuda               使用GPU, action='store_true'
```

## reference

文字区域识别ctpn：[https://github.com/eragonruan/text-detection-ctpn](https://github.com/eragonruan/text-detection-ctpn)

文字字符识别crnn：[https://github.com/meijieru/crnn.pytorch](https://github.com/meijieru/crnn.pytorch)

[【OCR技术系列之五】自然场景文本检测技术综述（CTPN, SegLink, EAST）](https://www.cnblogs.com/skyfsm/p/9776611.html)

[【OCR技术系列之七】端到端不定长文字识别CRNN算法详解](https://www.cnblogs.com/skyfsm/p/10335717.html)


