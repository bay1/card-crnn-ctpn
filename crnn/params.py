import alphabets

random_sample = True
keep_ratio = True
adam = False
adadelta = False
saveInterval = 2
valInterval = 800
n_test_disp = 10
displayInterval = 5
experiment = 'crnn/expr'
alphabet = alphabets.alphabet
crnn = 'crnn/trained_models/crnn_Rec_done.pth'
beta1 =0.5
lr = 0.0001
niter = 30
nh = 256 # LSTM设置隐藏层数
imgW = 256
imgH = 32
batchSize = 128 # batch size
workers = 0
