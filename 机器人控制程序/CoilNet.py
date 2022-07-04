import cv2
import numpy as np
import math
from BaseImageNet import BaseImageNet
from ImageSet import ImageSet
from ImageSet import LabelType


class CoilNet(BaseImageNet):
    def __init__(self, img_size, output_size):
        super(CoilNet, self).__init__(regress=True, img_size=img_size, img_channels=1)
        self.add_conv_layer(channels=8, kernel_size=5, padding=2, pool_size=2)
        self.add_conv_layer(channels=16, kernel_size=5, padding=2, pool_size=2)
        self.add_conv_layer(channels=32, kernel_size=3, padding=1, pool_size=2)
        self.add_conv_layer(channels=64, kernel_size=3, padding=1, pool_size=2)
        self.add_conv_layer(channels=128, kernel_size=3, padding=1, pool_size=2)
        self.add_conv_layer(channels=256, kernel_size=3, padding=1, pool_size=2)
        self.add_linear_layer_by_divider(divider=32)
        self.add_linear_layer(output_size=output_size)

    def label2angle(self, label):
        return int((label + 1.0) * 90)

    def predict(self, batch_img):
        outputs = super().predict(batch_img)

        params = list()
        for row in range(outputs.shape[0]):
            values = np.zeros(self.output_size, np.float32)
            for col in range(self.output_size):
                # the output may contain 2 or 5 elements.
                # when only center is required, it has 2 elements for center coordination;
                # when the whole ellipse is required, it has 5 elements, center, size and angle
                if col % 5 < 4:
                    values[col] = self.label2pixel(outputs[row][col])
                else:
                    values[col] = self.label2angle(outputs[row][col])
            params.append(values)
        return params

    def train(self, train_set, eval_set, epoch_cnt, mini_batch):
        super().train(train_set, eval_set, epoch_cnt, mini_batch)

    def test_net(self, eval_set):
        for i in range(eval_set.image_count()):
            batch_img, batch_label, raw_img = eval_set.get_sample(i)
            params = self.predict(batch_img)

            cv2.circle(raw_img, (int(params[0][0]), int(params[0][1])), 2, 0, 2)
            cv2.circle(raw_img, (self.label2pixel(batch_label[0][0]), self.label2pixel(batch_label[0][1])), 2, 255, 2)

            print("-------------------------")
            print("The %d th image" % i)
            lx = self.label2pixel(batch_label[0][0])
            ly = self.label2pixel(batch_label[0][1])
            px = params[0][0]
            py = params[0][1]

            print("labeled x = %d, y = %d" % (lx, ly))
            print("predicted x = %d, y = %d" % (px, py))
            print("error = %1.1f" % math.sqrt((lx - px) * (lx - px) + (ly - py) * (ly - py)))

            if self.output_size > LabelType.CenterOnly:
                cv2.ellipse(raw_img, params[0], params[1], params[2], 0, 360, 0, 2)
            cv2.imshow("", raw_img)

            if cv2.waitKey(0) == ord('q'):
                break


if __name__ == '__main__':
    # the main is used for training and testing the neural network
    # there are basically following parameters that need to be specified
    # 1. size of the image, normally 128, 256, 512
    # 2. size of the output, normally
    #    2 (only the center, LabelType.CenterOnly),
    #    5 (center and shape of one ellipse, LabelType.InnerOnly),
    #    10 (center and shape of two ellipses, LabelType.InnerOuter)
    # 3. the direction for saving trained model, and the interval of saving
    # 4. the training and testing set
    print("Specify the model type:\n1: center only\n2: inner ellipse\n3: outer ellipse\n4: center inner outer")
    mode = int(input("Mode = "))
    output_size = LabelType.CenterOnly
    if mode == 1:
        output_size = LabelType.CenterOnly
    elif mode == 2:
        output_size = LabelType.InnerOnly
    elif mode == 3:
        output_size = LabelType.InnerOuter
    elif mode == 4:
        output_size = LabelType.CenterInnerOuter

    Net = CoilNet(img_size=256, output_size=output_size)
    Net.learn_rate = 0.00001
    Net.l2_lambda = 0.0002
    Net.dropout_rate = 0.4
    Net.save_dir = "D:\\13 加油跟踪"
    Net.save_prefix = "加油口定位"
    Net.save_interval = 10000
    Net.init_model()

    mode = input("Train(T) the model, or run(R) the model?")
    if mode == 't' or mode == 'T':
        train_set = ImageSet("D:\\PlaneLabeledImages", output_size=Net.output_size, img_size=Net.img_size)
        eval_set = ImageSet("D:\\PlaneLabeledImages", output_size=Net.output_size, img_size=Net.img_size)

        Net.train(train_set, eval_set, epoch_cnt=10000, mini_batch=64)
    else:
        Net.load_model("加油口定位2021-11-20-17-51.pt")
        eval_set = ImageSet("E:\\PlaneLabeledImages",
                            output_size=Net.output_size, img_size=Net.img_size)
        Net.test_net(eval_set)
