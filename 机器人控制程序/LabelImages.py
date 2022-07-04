import cv2
import glob
import os
import random
import datetime
import numpy as np


def extract_filename(file_name):
    # extract file name for labels
    # there are 10 elements now in the file name:
    # the center, long and short axis, and angle of the inner ellipse
    # and those of the outer ellipse
    params = file_name.split('.')[0].split('_')
    label = (int(params[1]), int(params[2]), int(params[3]), int(params[4]), int(params[5]),
             int(params[6]), int(params[7]), int(params[8]), int(params[9]), int(params[10]))
    return label


def generate_image_by_trans(src_img, label, src_size, dst_size, random_range, count, dst_dir, fill_outside):
    if not os.path.exists(dst_dir):
        os.mkdir(dst_dir)

    # generate train images with random offset around the center of the inner ellipse
    random.seed()
    for i in range(count):
        # decide the center of the sub-image randomly around the center of the inner ellipse
        x_offset = random.randint(-random_range, random_range)
        y_offset = random.randint(-random_range, random_range)
        x_center = label[0] + x_offset
        y_center = label[1] + y_offset

        # the position of the sub-image
        src_left = int(max(0, x_center - src_size[0] / 2))
        src_top = int(max(0, y_center - src_size[1] / 2))
        src_right = int(min(src_img.shape[1], x_center + src_size[0] / 2))
        src_bottom = int(min(src_img.shape[0], y_center + src_size[1] / 2))
        src_width = src_right - src_left
        src_height = src_bottom - src_top

        # discard the uncompact sub-image
        if (src_width < src_size[0] or src_height < src_size[1]) and not fill_outside:
            continue

        sub_left = int(src_size[0] / 2 - x_center + src_left)
        sub_top = int(src_size[1] / 2 - y_center + src_top)

        sub_img = np.zeros(src_size)
        sub_img[sub_top: sub_top + src_height, sub_left: sub_left + src_width] = \
            src_img[src_top: src_bottom, src_left: src_right]

        # resize the image
        sub_img = cv2.resize(sub_img, dst_size)

        # calculate the center of the inner and outer ellipse
        inner_center_x = (src_size[0] / 2 - x_offset)
        inner_center_y = (src_size[1] / 2 - y_offset)
        outer_center_x = inner_center_x + label[5] - label[0]
        outer_center_y = inner_center_y + label[6] - label[1]

        # resize the labels
        resize_rate = (dst_size[0] / src_size[0], dst_size[1] / src_size[1])
        inner_center_x = int(inner_center_x * resize_rate[0])
        inner_center_y = int(inner_center_y * resize_rate[1])
        inner_length1 = int(label[2] * resize_rate[0])
        inner_length2 = int(label[3] * resize_rate[1])

        outer_center_x = int(outer_center_x * resize_rate[0])
        outer_center_y = int(outer_center_y * resize_rate[1])
        outer_length1 = int(label[7] * resize_rate[0])
        outer_length2 = int(label[8] * resize_rate[1])

        now = datetime.datetime.now()
        datetime_str = "\\%04d%02d%02d%02d%02d%02d" % (now.year, now.month, now.day, now.hour, now.minute, now.second)
        inner_str = "_%04d_%04d_%04d_%04d_%03d" % (inner_center_x, inner_center_y, inner_length1, inner_length2, label[4])
        outer_str = "_%04d_%04d_%04d_%04d_%03d.jpg" % (outer_center_x, outer_center_y, outer_length1, outer_length2, label[9])
        cv2.imwrite(dst_dir + datetime_str + inner_str + outer_str, sub_img)


def generate_train_samples(src_dir, dst_dir, src_size, dst_size):
    os.chdir(src_dir + "/")
    for file_name in glob.glob("*.jpg"):
        img = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)
        label = extract_filename(file_name)
        generate_image_by_trans(img, label, src_size, dst_size,
                                random_range=50, count=20, dst_dir=dst_dir, fill_outside=True)


def check_image_label(file_name):
    img = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)
    label = extract_filename(file_name)
    if len(label) >= 5:
        cv2.ellipse(img, (label[0], label[1]), (label[2], label[3]), label[4], 0, 360, 0,
                    2)
        cv2.circle(img, (label[0], label[1]), 5, 0, 3)

    if len(label) >= 10:
        cv2.ellipse(img, (label[5], label[6]), (label[7], label[8]), label[9], 0, 360, 0, 2)
        cv2.circle(img, (label[5], label[6]), 5, 0, 3)

    if img.shape[1] > 1024:
        k = 1024.0 / img.shape[1]
        img = cv2.resize(img, (int(img.shape[1] * k), int(img.shape[0] * k)))
    cv2.imshow("labeled image", img)


def image_label_test(dir):
    os.chdir(dir + "\\")
    for file_name in glob.glob("*.jpg"):
        check_image_label(file_name)
        if cv2.waitKey(0) == ord('q'):
            break


if __name__ == '__main__':
    #generate_train_samples("D:\\AutoLoadData\\1\\2", "D:\\AutoLoadData\\2", inner_ellipse=True)
    #generate_train_samples("D:\\13 加油跟踪\\小飞机图片（已做标签）", "D:\\PlaneLabeledImages", src_size=(256, 256), dst_size=(256, 256),)
    image_label_test("D:\\PlaneLabeledImages")

    #image_label_test("D:\\11.9日-小飞机照片\\标记照片")