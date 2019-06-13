import re
import os

from PIL import Image

import params
import utils


def get_lable(file, original_dir):
    """
    不同lable規則
    """
    lable = ''
    filename = file.split('.')[0]  # 图片名称
    if original_dir == params.augmentation_output:
        lable = filename.split('_')[0]
    else:
        lable = re.sub("\D", "", filename)[:-1]  # 正确文字
    return lable


def main(original_dir, i, train_num):
    result_path = ''
    # 遍历figures下的png,jpg文件
    for file in os.listdir(original_dir):
        if file.endswith('.png') or file.endswith('.jpg'):
            if i < train_num:
                result_path = params.train_images
                f = open(params.train_images_labels, "a+")
            else:
                result_path = params.test_images
                f = open(params.test_images_labels, "a+")
            image_path = '%s/%s' % (original_dir, file)  # 图片路径
            image = Image.open(image_path)  # 打开图片文件
            imgry = image.convert('L')  # 转化为灰度图
            recognizition = get_lable(file, original_dir)

            line = file + " " + recognizition + "\n"
            f.writelines(line)

            # 保存图片
            imgry.save(result_path + file)

            print(file, recognizition)
            i += 1
            f.close()
    return i


utils.check_file_exist(params.train_images_labels)
utils.check_file_exist(params.test_images_labels)
utils.check_floder_exist(params.train_images)
utils.check_floder_exist(params.test_images)

i = 0
i = main(params.original_dir, i, params.origin_train_num)
main(params.augmentation_dir, i, params.augmentation_train_num)
