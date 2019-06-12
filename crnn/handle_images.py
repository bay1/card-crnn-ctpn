import re
import os
import shutil

from PIL import Image

import params

def check_file_exist(file_path):
    exists = os.path.isfile(file_path)
    if exists:
        os.remove(file_path)

def check_floder_exist(floder_path):
    exists = os.path.exists(floder_path)
    if exists:
        shutil.rmtree(floder_path)
    os.mkdir(floder_path)

def OCR_lmj(img_path, file, f, result_path):
    
    image = Image.open(img_path)  # 打开图片文件
    imgry = image.convert('L')  # 转化为灰度图
    
    filename = file.split('.')[0]  # 图片名称
    new_filename = re.sub("\D", "", filename)[:-1]  # 正确文字

    line = file + " " + new_filename + "\n"
    f.writelines(line)

    # 保存图片
    imgry.save(result_path + file)

    return new_filename


def main():
    i = 0
    result_path = ''
    check_file_exist(params.train_images_labels)
    check_file_exist(params.test_images_labels)
    check_floder_exist(params.train_images)
    check_floder_exist(params.test_images)
    # 遍历figures下的png,jpg文件
    for file in os.listdir(params.original_dir):
        if file.endswith('.png') or file.endswith('.jpg'):
            if i < 1000:
                result_path = params.train_images
                f = open(params.train_images_labels, "a+")
            else:
                result_path = params.test_images
                f= open(params.test_images_labels, "a+")
            image_path = '%s/%s' % (params.original_dir, file)  # 图片路径
            recognizition = OCR_lmj(image_path, file, f, result_path)  # 图片正确结果

            print(file, recognizition)
            i += 1
            f.close()

main()
