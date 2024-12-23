import cv2
import numpy as np
import os

import PIL.Image as Image

def flip_horizontal(path):
    img = Image.open(path)
    horizontal_flip = img.transpose(Image.FLIP_LEFT_RIGHT)
    save_path = path.split('.')[0] + '_6.' + path.split('.')[1]
    horizontal_flip.save(save_path)

def flip_vertical(path):
    img = Image.open(path)
    horizontal_flip = img.transpose(Image.FLIP_TOP_BOTTOM)
    save_path = path.split('.')[0] + '_7.' + path.split('.')[1]
    horizontal_flip.save(save_path)


def gaussian(path, mean=0, var=0.3):
    img = cv2.imread(path)
    # 获取图像的形状
    h, w, c = img.shape
    # 生成高斯噪声
    sigma = var ** 0.5
    noise = np.random.normal(mean, sigma, (h, w, c)).astype(np.uint8)
    # 添加噪声
    noisy_image = cv2.add(img, noise)
    save_path = path.split('.')[0] + '_3.' + path.split('.')[1]
    cv2.imwrite(save_path, noisy_image)

def main(dir_path):
    image_path_dir = dir_path
    image_list = os.listdir(image_path_dir)
    for image in image_list:
        image_path = os.path.join(image_path_dir, image)


        gaussian(image_path)

        flip_horizontal(image_path)
        flip_vertical(image_path)

if __name__ == '__main__':
    main(dir_path=r"zibi_images/Non_Autistic")

