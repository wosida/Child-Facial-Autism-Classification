#原始数据集既有rgb也有灰度图，需更改
import cv2
import os

# 把路径中的图片读取出来，形状为[1,h,w]的改成[3,h,w]
def reshape(dir):
    paths=os.listdir(dir)
    for path in paths:
        img_path=os.path.join(dir,path)
        img=cv2.imread(img_path)
        #如果通道数为1，复制成三份
        if img.shape[2]==1:
            img=cv2.merge([img,img,img])
            cv2.imwrite(img_path,img)
        else:
            continue

if __name__ =="main":
    reshape("zibi_images/Non_Autistic")
    reshape("zibi_images/Autistic")

