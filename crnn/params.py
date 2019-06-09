# train.py params start #
random_sample = True
keep_ratio = True
adam = False
adadelta = False
saveInterval = 20
valInterval = 500
n_test_disp = 1
displayInterval = 3
experiment = 'crnn/expr'
alphabet = '0123456789'
crnn = ''
beta1 = 0.5
lr = 0.00005
niter = 1000
nh = 256  # LSTM设置隐藏层数
imgW = 256
imgH = 32
batchSize = 128  # batch size
workers = 0
trainroot = 'crnn/to_lmdb/lmdb'  # path to dataset
valroot = 'crnn/to_lmdb/lmdb'  # path to dataset
cuda = True  # enables cuda
# train.py params end #

# test.py params start #
crnn_model_path = 'crnn/trained_models/crnn_Rec_done.pth'
# test.py params end #

# handle_images.py params start #
lmdb_images = 'crnn/to_lmdb/train_images/'
original_dir = 'data/images/'  # 图片存放目录figures
images_labels = "crnn/to_lmdb/train.txt"
# handle_images.py params end #
