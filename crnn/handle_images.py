import re
import os

from PIL import Image

import params


def OCR_lmj(img_path, file, f):
    filename = file.split('.')[0]  # 图片名称

    image = Image.open(img_path)  # 打开图片文件
    imgry = image.convert('L')  # 转化为灰度图

    new_filename = re.sub("\D", "", filename)[:-1]  # 正确文字

    new_file = new_filename + '.' + file.split('.')[1]
    print(new_file)

    line = new_file + " " + new_filename + "\n"
    f.writelines(line)

    # 保存图片
    imgry.save(params.lmdb_images + new_file)

    return new_filename


def main():
    # 识别指定文件目录下的图片
    with open(params.images_labels, "w") as f:
        # 遍历figures下的png,jpg文件
        for file in os.listdir(params.original_dir):
            if file.endswith('.png') or file.endswith('.jpg'):
                image_path = '%s/%s' % (params.original_dir, file)  # 图片路径
                recognizition = OCR_lmj(image_path, file, f)  # 图片正确结果

                print(file, recognizition)
    f.close()


main()
