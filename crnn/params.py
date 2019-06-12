# train.py params start #
random_sample = True
keep_ratio = True
adam = False
adadelta = False
saveInterval = 6
valInterval = 6
n_test_disp = 5
displayInterval = 3
experiment = 'crnn/expr'
alphabet = '0123456789'
crnn = ''
beta1 = 0.5
lr = 0.0001
niter = 10000
nh = 256  # LSTM设置隐藏层数
imgW = 120
imgH = 32
batchSize = 128  # batch size
workers = 0
trainroot = 'crnn/to_lmdb/train_lmdb'  # path to dataset
valroot = 'crnn/to_lmdb/test_lmdb'  # path to dataset
cuda = True  # enables cuda
# train.py params end #

# test.py params start #
crnn_model_path = 'crnn/trained_models/crnn_Rec_done.pth'
# test.py params end #

# handle_images.py params start #
train_images = 'crnn/to_lmdb/train_images/'
test_images = 'crnn/to_lmdb/test_images/'
original_dir = 'data/images/'  # 图片存放目录figures
train_images_labels = "crnn/to_lmdb/train.txt"
test_images_labels = "crnn/to_lmdb/test.txt"
# handle_images.py params end #
