import numpy as np
import cv2
"""
confusionMetric,真真假假
P\L     P    N
 
P      TP    FP
 
N      FN    TN
 
"""


class Evaluator(object):
    def __init__(self, num_class):
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class,)*2)

    def Pixel_Accuracy(self):
        # acc = (TP + TN) / (TP + TN + FP + TN)
        Acc = np.diag(self.confusion_matrix).sum() / \
            self.confusion_matrix.sum()
        return Acc

    def Pixel_Accuracy_Class(self):
        # acc = (TP) / TP + FP
        Acc = np.diag(self.confusion_matrix) / \
            self.confusion_matrix.sum(axis=1)
        Acc_class = np.nanmean(Acc)
        return Acc_class

    def Mean_Intersection_over_Union(self):
        MIoU = np.diag(self.confusion_matrix) / (
            np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
            np.diag(self.confusion_matrix))
        MIoU = np.nanmean(MIoU)
        return MIoU

    def Frequency_Weighted_Intersection_over_Union(self):
        # FWIOU =     [(TP+FN)/(TP+FP+TN+FN)] *[TP / (TP + FP + FN)]
        freq = np.sum(self.confusion_matrix, axis=1) / \
            np.sum(self.confusion_matrix)
        iu = np.diag(self.confusion_matrix) / (
            np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
            np.diag(self.confusion_matrix))

        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def _generate_matrix(self, gt_image, pre_image):
        mask = (gt_image >= 0) & (gt_image < self.num_class)
        label = self.num_class * gt_image[mask].astype('int') + pre_image[mask]
        count = np.bincount(label, minlength=self.num_class**2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    def add_batch(self, gt_image, pre_image):
        assert gt_image.shape == pre_image.shape
        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)

def meansure_pa_miou(num_class, gt_image, pre_image):
    metric = Evaluator(num_class)
    metric.add_batch(gt_image, pre_image)
    acc = metric.Pixel_Accuracy()
    mIoU = metric.Mean_Intersection_over_Union()
    print("像素准确度PA:", acc, "平均交互度mIOU:", mIoU)


if __name__ == '__main__':
    # 求miou，先求混淆矩阵，混淆矩阵的每一行再加上每一列，最后减去对角线上的值；
    # imgPredict = np.array([[0, 0, 1, 0], [1, 1, 0, 2], [2, 2, 1 ,0]])
    # imgLabel = np.array([[0, 0, 0, 1], [1, 1, 2, 2], [2, 2, 0, 0]])
    
    for i in range(35):
        imgPredict_0 = cv2.imread("Hippocampus_data/test/output00/001_"+str(i)+"/001_"+str(i)+"_0.png")
        imgPredict_1 = cv2.imread("Hippocampus_data/test/output00/001_"+str(i)+"/001_"+str(i)+"_1.png")
        imgPredict_2 = cv2.imread("Hippocampus_data/test/output00/001_"+str(i)+"/001_"+str(i)+"_2.png")
        # imgPredict_3 = cv2.imread("BrainTumour_data/test/output60/003_"+str(i)+"/003_"+str(i)+"_3.png")
        imgPredict_0[imgPredict_0 == 255] = 1 
        imgPredict_1[imgPredict_1 == 255] = 1
        imgPredict_2[imgPredict_2 == 255] = 1
        # imgPredict_3[imgPredict_3 == 255] = 1

        imgLabel = cv2.imread("/media/zihong/Cai zihong/train_data/Medical Segmentation Decathlon/Task04_Hippocampus/labels_train/001_"+str(i)+".png")

        # 设置成两类与预测图对应
        imgLabel_0 = imgLabel.copy()
        imgLabel_1 = imgLabel.copy()
        imgLabel_2 = imgLabel.copy()
        # imgLabel_3 = imgLabel.copy()

        imgLabel_0[imgLabel >= 1] = 1  # 背景1和目标0
        height, width, channels = imgLabel_0.shape # 反转
        for row in range(height):
            for list in range(width):
                for c in range(channels):
                    pv = imgLabel_0[row, list, c]
                    imgLabel_0[row, list, c] = 1 - pv

        imgLabel_1[imgLabel != 1] = 0  # 把第二类归为背景

        imgLabel_2[imgLabel != 2] = 0  # 把第一类归为背景
        imgLabel_2[imgLabel_2 == 2] = 1

        # imgLabel_3[imgLabel != 3] = 0
        # imgLabel_3[imgLabel_3 == 3] = 1
        
        print(i)

        meansure_pa_miou(2, imgLabel_0, imgPredict_0)
        meansure_pa_miou(2, imgLabel_1, imgPredict_1)
        meansure_pa_miou(2, imgLabel_2, imgPredict_2)
        # meansure_pa_miou(2, imgLabel_3, imgPredict_3)


    

