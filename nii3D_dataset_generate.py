# 数据集图片：数据类型float，读取显示；uint，16位，8位保存显示；
# 数据压缩，float32转uint8；
# 数据集标签：多类别标签，灰度显示，全黑，关注根据数据类型做相应修改
from nilearn import datasets
import nibabel as nib
from nilearn import image,plotting
import os
import matplotlib.pyplot as plt
import matplotlib
from collections import Counter
#import imageio

""" img = nib.load('datasets\\Task05_Prostate\\imagesTr\\prostate_00.nii.gz')

print(img.shape)

first_img = image.index_img(img, 0)

print(first_img.shape)


for img_img in image.iter_img(img):
    # img_img is now an in-memory 3D img
    print(img_img.shape)
    plotting.plot_stat_map(img_img, 
                        #display_mode="z", 
                        #cut_coords=1,
                        colorbar=True,
                        title="hhh")
    plotting.show()
 """
import SimpleITK as sitk
from PIL import Image
import numpy as np
import imageio
import cv2

def read_img(path):
    img = sitk.ReadImage(path)
    data = sitk.GetArrayFromImage(img)
    return data

# 显示一个系列图
def show_image(data):
    number = input('数据图片系列号：')
    for i in range(data.shape[0]):
        
        # plt.imshow(data[i, :, :])##每种包对图像保存显示
        # plt.show()
        # imageio.imsave('test.png', data[5, :, :])
        # matplotlib.image.imsave('out.png', data[5, :, :])
        # cv2.imwrite("filename.png", data[i, :, :])
        # dd = cv2.imread('filename.png')
        # plt.imshow(dd)
        # plt.show()
        # cv2.imshow(dd)
        # print(Counter(dd.flatten()))
        # print(type(data))
        # print(np.max(dd),np.min(dd))
        print("Array数值检查：",np.max( data[i, :, :]),np.min( data[i, :, :]))
        img = Image.fromarray(np.uint8(data[i,:,:]/np.max(data[i,:,:])*255))#np.uinit8(lena*255)
        #查看图片
        # plt.imshow(img)
        # plt.show()
        print("Image数值检查：",np.max(img),np.min(img))
        # img.show()
        
        #img.convert('L').save('datasets\\Task05_Prostate\\tra\\47_'+str(i)+'.png',format = 'PNG')
        img.save('datasets\\Task05_Prostate\\test\\'+str(number)+'_'+str(i)+'.png',format = 'PNG')
        # 对已保存图片进行检查
        # img1 = Image.open('datasets\\Task05_Prostate\\test\\'+str(number)+'_'+str(i)+'.png')
        # plt.imshow(img1)
        # plt.show()

        #img.save('00_1.png')
        print(i)
        # data.save('datasets\\Task05_Prostate\\tra\\00\\00_'+ i +".png")
# 显示一个系列图
def show_label(data):
    ##输入数据图片系列号
    number = input('数据标签图片系列号：')
    for i in range(data.shape[0]):
        
        # plt.imshow(data[i, :, :],cmap='gray')##每种包对图像保存显示
        # plt.show()
        # imageio.imsave('test.png', data[5, :, :])
        # matplotlib.image.imsave('out.png', data[5, :, :])
        #cv2.imwrite("filename.png", data[5, :, :])
        # dd = cv2.imread('datasets\\Task05_Prostate\\val\\44_7.png',cv2.IMREAD_GRAYSCALE)
        # plt.imshow(data[i,:,:])
        # plt.show()
        # print(Counter(dd.flatten()))
        # print(type(data))
        # print(np.max(dd),np.min(dd))
        print("Array数值检查：",np.max( data[i, :, :]),np.min( data[i, :, :]),Counter(data[i, :, :].flatten()))
        
        img = data[i,:,:].astype('uint8')
        # img[img==1] = 127 ##显示可见0-255
        # img[img==2] = 255
        img = Image.fromarray(img)
        
        # print(type(img))
        #img.convert('RGB')
        #查看图片
        # plt.imshow(img)
        # plt.show()
        print("Image数值检查：",np.max(img),np.min(img))
        # img.show()
        
        #img.convert('L').save('datasets\\Task05_Prostate\\tra\\47_'+str(i)+'.png',format = 'PNG')
        img.save('datasets\\Task05_Prostate\\lab\\'+str(number)+'_'+str(i)+'.png',format = 'PNG')


        #img.save('00_1.png')
        print(i)
        # data.save('datasets\\Task05_Prostate\\tra\\00\\00_'+ i +".png")



# 单张显示
def show_img1(ori_img):
    plt.imshow(ori_img[0], cmap='gray')
    plt.show()


path = 'datasets\\Task05_Prostate\\labelsTr\\'  # 数据所在路径

for j in os.listdir(path):
    print('序列号：',j)    
    data = read_img(path+j)
    # img1 = cv2.imread("datasets\\Task05_Prostate\\tra1\\00_4.png")

# img1[img1>1.5] = 2
# img1[img1 > 0.7] = 1
# img1[img1 <= 0.7] = 0

# img1 = img1 * 255.
    # plt.imshow(img1)
    # plt.show()

    #保存label
    show_label(data)

    #读取图像4D
    # data = data[0,:,:,:]
    # show_image(data)
