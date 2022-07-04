import torch
import cv2
import numpy as np
import random
import glob
import os

class LabelType():
    CenterOnly       = 2
    CenterInnerOuter = 4
    InnerOnly        = 5
    InnerOuter       = 10

class ImageSet:
    def __init__(self, dir, output_size, img_size):
        self.file_names = list()
        self.img_size = img_size
        self.output_size = output_size

        os.chdir(dir)
        for file_name in glob.glob("*.jpg"):
            self.file_names.append(dir + "\\" + file_name)
        print("There are %d images in %s" % (self.image_count(), dir))

    def __read_label(self, file_name, resize_k):
        """
        extract label from file name
        there are at most 11 contents in the file name:
        the date time of the file, the center and shape of the inner ellipse, and those of the outer ellipse
        :param file_name:
        :param resize_k:
        :return:
        """
        params = file_name.split('.')[0].split('_')
        if len(params) <= self.output_size:
            return None, None

        label = np.zeros(self.output_size, dtype=np.int32)
        norm_label = np.zeros(self.output_size, dtype=np.float32)
        if self.output_size == LabelType.CenterOnly:
            label[0] = int(int(params[1]) / resize_k + 0.5)
            label[1] = int(int(params[2]) / resize_k + 0.5)

            norm_label[0] = self.pixel2label(label[0])
            norm_label[1] = self.pixel2label(label[1])
        elif self.output_size == LabelType.CenterInnerOuter:
            label[0] = int(int(params[1]) / resize_k + 0.5)
            label[1] = int(int(params[2]) / resize_k + 0.5)
            label[2] = int(int(params[6]) / resize_k + 0.5)
            label[3] = int(int(params[7]) / resize_k + 0.5)

            norm_label[0] = self.pixel2label(label[0])
            norm_label[1] = self.pixel2label(label[1])
            norm_label[2] = self.pixel2label(label[2])
            norm_label[3] = self.pixel2label(label[3])
        elif self.output_size == LabelType.InnerOnly:
            label[0] = int(int(params[1]) / resize_k + 0.5)
            label[1] = int(int(params[2]) / resize_k + 0.5)
            label[2] = int(int(params[3]) / resize_k + 0.5)
            label[3] = int(int(params[4]) / resize_k + 0.5)
            label[5] = int(params[5])

            norm_label[0] = self.pixel2label(label[0])
            norm_label[1] = self.pixel2label(label[1])
            norm_label[2] = self.pixel2label(label[2])
            norm_label[3] = self.pixel2label(label[3])
            norm_label[4] = self.angle2label(label[4])
        elif self.output_size == LabelType.InnerOuter:
            label[0] = int(int(params[1]) / resize_k + 0.5)
            label[1] = int(int(params[2]) / resize_k + 0.5)
            label[2] = int(int(params[3]) / resize_k + 0.5)
            label[3] = int(int(params[4]) / resize_k + 0.5)
            label[4] = int(params[5])
            label[5] = int(int(params[6]) / resize_k + 0.5)
            label[6] = int(int(params[7]) / resize_k + 0.5)
            label[7] = int(int(params[8]) / resize_k + 0.5)
            label[8] = int(int(params[9]) / resize_k + 0.5)
            label[9] = int(params[10])

            norm_label[0] = self.pixel2label(label[0])
            norm_label[1] = self.pixel2label(label[1])
            norm_label[2] = self.pixel2label(label[2])
            norm_label[3] = self.pixel2label(label[3])
            norm_label[4] = self.angle2label(label[4])
            norm_label[5] = self.pixel2label(label[5])
            norm_label[6] = self.pixel2label(label[6])
            norm_label[7] = self.pixel2label(label[7])
            norm_label[8] = self.pixel2label(label[8])
            norm_label[9] = self.angle2label(label[9])
        return label, norm_label


    def read_labeled_image(self, file_name):
        # load image from the file and resize it
        src = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)  # read image from the file
        img = cv2.resize(src, (self.img_size, self.img_size))
        resize_k = src.shape[0] / self.img_size

        # get label
        label, norm_label = self.__read_label(file_name, resize_k)

        # normalize image
        mean = img.mean()
        std = img.std()
        norm_img = (img - mean) / std

        return img, norm_img, label, norm_label

    def diff_image(self, img):
        x = cv2.Sobel(img, cv2.CV_16S, 1, 0)
        y = cv2.Sobel(img, cv2.CV_16S, 0, 1)

        absX = cv2.convertScaleAbs(x)  # 转回uint8
        absY = cv2.convertScaleAbs(y)

        dst = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
        return dst

    def image_count(self):
        return len(self.file_names)

    def pixel2label(self, pixel):
        return pixel * 2.0 / self.img_size - 1.0

    def angle2label(self, angle):
        angle = angle % 360
        if angle >= 180:
            angle -= 180
        return angle / 90.0 - 1.0

    def label2pixel(self, label):
        return int((label + 1.0) * self.img_size)

    def label2angle(self, label):
        return int((label + 1.0) * 90.0)

    def alpha_beta_adjust(self, image, alpha, beta):
        img = image * alpha + beta
        img = np.clip(img, 0, 255)
        return img.astype(dtype=np.uint8)

    def random_alpha_beta(self, image):
        alpha = random.randint(-100, 100) * 0.005 + 1
        beta = 0
        return self.alpha_beta_adjust(image, alpha, beta)

    def add_salt_noise(self, image):
        pass

    def random_sample(self, batch_size):
        batch_img = torch.zeros(batch_size, self.img_size, self.img_size, dtype=torch.float32)
        batch_label = torch.zeros(batch_size, self.output_size, dtype=torch.float32)
        raw_images = list()

        random.seed()
        for i in range(batch_size):
            idx = random.randint(0, self.image_count() - 1)
            file_name = self.file_names[idx]
            raw, norm, raw_label, norm_label = self.read_labeled_image(file_name)

            batch_img[i] = torch.from_numpy(norm)
            batch_label[i] = torch.from_numpy(norm_label)
            raw_images.append(raw)

        return batch_img, batch_label, raw_images

    def get_sample(self, idx):
        batch_img = torch.zeros(1, self.img_size, self.img_size, dtype=torch.float32)
        batch_label = torch.zeros(1, self.output_size, dtype=torch.float32)

        file_name = self.file_names[idx]
        raw, norm, raw_label, norm_label = self.read_labeled_image(file_name)

        batch_img[0] = torch.from_numpy(norm)
        batch_label[0] = torch.from_numpy(norm_label)

        return batch_img, batch_label, raw

    def __draw_image_center_only(self, img, label):
        cv2.circle(img, (label[0], label[1]), 5, 0, 3)

    def __draw_image_center_innerouter(self, img, label):
        cv2.circle(img, (label[0], label[1]), 5, 0, 3)
        cv2.circle(img, (label[2], label[3]), 5, 0, 3)

    def __draw_image_center_inneronly(self, img, label):
        cv2.circle(img, (label[0], label[1]), 5, 0, 3)
        cv2.ellipse(img, (label[0], label[1]), (label[2], label[3]), label[4], 0, 360, 0, 2)

    def __draw_image_innerouter(self, img, label):
        cv2.circle(img, (label[0], label[1]), 5, 0, 3)
        cv2.ellipse(img, (label[0], label[1]), (label[2], label[3]), label[4], 0, 360, 0, 2)
        cv2.circle(img, (label[5], label[6]), 5, 0, 3)
        cv2.ellipse(img, (label[5], label[6]), (label[7], label[8]), label[9], 0, 360, 0, 2)

    def check_image_label(self):
        for i, file_name in enumerate(self.file_names):
            raw, norm, raw_label, norm_label = self.read_labeled_image(file_name)

            img = raw.copy()
            diff_img = self.diff_image(img)

            if self.output_size == LabelType.CenterOnly:
                self.__draw_image_center_only(img, raw_label)
            elif self.output_size == LabelType.CenterInnerOuter:
                self.__draw_image_center_innerouter(img, raw_label)
            elif self.output_size == LabelType.InnerOnly:
                self.__draw_image_center_inneronly(img, raw_label)
            elif self.output_size == LabelType.InnerOuter:
                self.__draw_image_innerouter(img, raw_label)

            cv2.imshow("image", img)
            cv2.imshow("diff image", diff_img)
            if cv2.waitKey(0) & 0xff == ord('q'):  # 按q退出
                break

if __name__ == '__main__':
    #img_set = ImageSet("D:\\AutoLoadData\\eval", output_size=5, img_size=512)
    img_set = ImageSet("D:\\inner_outer", output_size=LabelType.CenterInnerOuter, img_size=512)
    img_set.check_image_label()