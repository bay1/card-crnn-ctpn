# train.py params start #
random_sample = True
keep_ratio = True
adam = False
adadelta = False
saveInterval = 30
valInterval = 500
n_test_disp = 5
displayInterval = 3
experiment = 'crnn/expr'
alphabet = '0123456789'
crnn = ''
beta1 = 0.5
lr = 0.0001
niter = 210
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
train_images_labels = "crnn/to_lmdb/train.txt"
test_images_labels = "crnn/to_lmdb/test.txt"
origin_train_num = 800
augmentation_train_num = 70000 # 和數據增強數目對應
original_dir = 'data/images/'  # 图片存放目录figures
augmentation_dir = 'data/generating'  # 图片存放目录figures
# handle_images.py params end #


# augmentation.py params start #
augmentation_original_dir = 'data/images'
augmentation_output = 'data/generating'
total_num = 80 # 數據增強數目
# augmentation.py params end #