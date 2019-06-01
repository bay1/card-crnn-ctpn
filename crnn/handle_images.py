from PIL import Image
import re
import os
import dataset
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
    imgry = imgry.resize((params.imgW, params.imgH))
    imgry.save('to_lmdb/train_images/' + new_file)

    return new_filename


def main():
    # 识别指定文件目录下的图片
    # 图片存放目录figures
    dir = '../data/images/'

    with open("to_lmdb/train.txt", "w") as f:
        # 遍历figures下的png,jpg文件
        for file in os.listdir(dir):
            if file.endswith('.png') or file.endswith('.jpg'):
                image_path = '%s/%s' % (dir, file)  # 图片路径

                recognizition = OCR_lmj(image_path, file, f)  # 图片正确结果

                print((file, recognizition))
    f.close()


main()
