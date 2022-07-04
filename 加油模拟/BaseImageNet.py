import torch
import torch.nn as nn
import torch.optim as optim
import datetime
import matplotlib.pyplot as plt
import math


class MyNet(nn.Module):
    def __init__(self, img_size, regress, conv_layers, linear_layers, dropout_rate):
        super(MyNet, self).__init__()
        self.img_size = img_size
        self.regress = regress

        self.conv_layers = torch.nn.Sequential()
        for i, layer in enumerate(conv_layers):
            in_channels = layer[0]
            out_channels = layer[1]
            kernel_size = layer[2]
            padding = layer[3]
            pool_size = layer[4]

            self.conv_layers.add_module('conv%d' % i,
                                        nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                                  kernel_size=kernel_size, stride=1, padding=padding))
            self.conv_layers.add_module('conv_dropout%d' % i, nn.Dropout2d(dropout_rate))
            self.conv_layers.add_module('conv_act%d' % i, nn.ReLU())
            self.conv_layers.add_module('pool%d' % i, nn.MaxPool2d(kernel_size=pool_size, stride=pool_size, padding=0))

        self.dense_layers = torch.nn.Sequential()
        for i in range(len(linear_layers) - 1):
            in_size = linear_layers[i][0]
            out_size = linear_layers[i][1]
            self.dense_layers.add_module('linear%d' % i, nn.Linear(in_size, out_size))
            self.dense_layers.add_module('dense_dropout%d' % i, nn.Dropout2d(dropout_rate))
            self.dense_layers.add_module('dense_act%d' % i, nn.ReLU())

        self.dense_layers.add_module('output', nn.Linear(linear_layers[-1][0], linear_layers[-1][1]))
        if not regress:
            self.dense_layers.add_module('LogSoftmax', nn.LogSoftmax(dim=1))

    def forward(self, input):
        out = input.view(-1, 1, self.img_size, self.img_size)
        out = self.conv_layers(out)
        out = out.view(out.size(0), -1)
        out = self.dense_layers(out)
        return out


class BaseImageNet(object):
    def __init__(self, regress, img_size, img_channels=1):
        self.regress = regress
        self.img_size = img_size
        self.img_channel = img_channels

        self.conv_layers = list()
        self.linear_layers = list()

        self.learn_rate = 0.00001
        self.l2_lambda = 0.0002
        self.dropout_rate = 0.4

        self.save_dir = None
        self.save_prefix = None
        self.save_interval = 3000
        self.NO_CUDA = False

    def add_conv_layer(self, channels, kernel_size, padding, pool_size):
        if not self.conv_layers:
            in_channels = self.img_channel
        else:
            in_channels = self.conv_layers[-1][1]

        self.conv_layers.append((in_channels, channels, kernel_size, padding, pool_size))
        self.lst_conv_channels = channels

    def add_linear_layer(self, output_size):
        if not self.linear_layers:
            pool_size = 1
            for _, layer in enumerate(self.conv_layers):
                pool_size *= layer[4]

            pooled_img_size = int(self.img_size / pool_size)
            in_size = pooled_img_size * pooled_img_size * self.conv_layers[-1][1]
        else:
            in_size = self.linear_layers[-1][1]

        self.linear_layers.append((in_size, output_size))
        self.output_size = self.linear_layers[-1][1]

    def add_linear_layer_by_divider(self, divider):
        if not self.linear_layers:
            pool_size = 1
            for _, layer in enumerate(self.conv_layers):
                pool_size *= layer[4]

            pooled_img_size = int(self.img_size / pool_size)
            in_size = pooled_img_size * pooled_img_size * self.conv_layers[-1][1]
        else:
            in_size = self.linear_layers[-1][1]
        self.linear_layers.append((in_size, int(in_size / divider)))
        self.output_size = self.linear_layers[-1][1]

    def init_model(self):
        self.device = self.try_gpu()

        self.model = MyNet(self.img_size, self.regress, self.conv_layers, self.linear_layers, self.dropout_rate)
        self.model = self.model.to(self.device)

    def try_gpu(self):
        """If GPU is available, return torch.device as cuda:0; else return torch.device as cpu."""
        device = torch.device('cpu') if self.NO_CUDA or not torch.cuda.is_available() else torch.device('cuda')
        print("Using ", device)
        return device

    def train(self, train_set, eval_set, epoch_cnt, mini_batch, loss_func=None):
        if loss_func is None:
            loss_func = nn.MSELoss() if self.regress else nn.NLLLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learn_rate)
        self.__train_net__(num_epochs=epoch_cnt, device=self.device, optimizer=optimizer,
                           model=self.model, loss_fn=loss_func, batch_size=mini_batch, train_set=train_set,
                           eval_set=eval_set)

    def predict(self, batch_img):
        self.model.eval()
        batch_img = batch_img.to(self.device)
        outputs = self.model(batch_img)
        return outputs.cpu().detach().numpy()

    def normalize_image(self, image):
        mean = image.mean()
        std = image.std()
        norm_img = (image - mean) / std
        return norm_img

    def label2pixel(self, label):
        return int((label + 1.0) * self.img_size / 2 + 0.5)

    def save_model(self, dir):
        if self.save_prefix is None:
            path = dir + datetime.datetime.now().strftime("/%Y-%m-%d-%H-%M.pt")
        else:
            path = dir + '/' + self.save_prefix + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M.pt")
        torch.save(self.model.state_dict(), path)

    def load_model(self, file_name):
        self.model.load_state_dict(torch.load(file_name))
        self.model = self.model.to(self.device)

    def __train_net__(self, num_epochs, device, optimizer, model, loss_fn, batch_size, train_set, eval_set):
        train_curve = list()
        eval_curve = list()

        model.train()
        for epoch in range(num_epochs):
            # predict by the model
            batch_img, batch_label, _ = train_set.random_sample(batch_size)
            batch_img = batch_img.to(device)
            batch_label = batch_label.to(device)
            outputs = model(batch_img)

            # calculate the loss
            loss = loss_fn(outputs, batch_label)
            loss_in_train = loss.item()

            # L2 regularization
            l2_norm = sum(p.pow(2.0).sum() for p in model.parameters())
            loss = loss + self.l2_lambda * l2_norm

            # train the model
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # evaluate the model
            if epoch > 0 and epoch % 10 == 0:
                if eval_set != None:
                    loss_in_eval = self.eval_net(model, device, loss_fn, eval_set, batch_size)
                    train_err, eval_err = self.show_evaluation(epoch, loss_in_train, loss_in_eval)

                    train_curve.append(train_err)
                    eval_curve.append(eval_err)
                else:
                    train_err, _ = self.show_evaluation(epoch, loss_in_train, 0)
                    train_curve.append(train_err)

            if epoch > 0 and epoch % self.save_interval == 0 and self.save_dir is not None:
                self.save_model(self.save_dir)

            # save the model every certain epoches
            if epoch > 0 and epoch % self.save_interval == 0 and self.save_dir is not None:
                self.save_model(self.save_dir)

        # save the final model
        if self.save_dir is not None:
            self.save_model(self.save_dir)

        # draw training plot
        plt.plot(train_curve, color='red')
        plt.plot(eval_curve, color='blue')
        plt.show()

    def eval_net(self, model, device, loss_fn, eval_set, batch_size):
        # evaluate the current model by outputting the loss
        model.eval()

        batch_img, batch_label, _ = eval_set.random_sample(batch_size)
        batch_img = batch_img.to(device)
        batch_label = batch_label.to(device)
        outputs = model(batch_img)
        loss = loss_fn(outputs, batch_label)
        return loss.item()

    def show_evaluation(self, epoch, loss_in_train, loss_in_eval):
        if self.regress:
            train_err = math.sqrt(loss_in_train) * self.img_size / 2
            eval_err = math.sqrt(loss_in_eval) * self.img_size / 2
            print('{} Epoch {}, Training loss {}({}), Testing loss {}'.format(datetime.datetime.now(),
                                                                              epoch, train_err, loss_in_train,
                                                                              eval_err))
            return train_err, eval_err
        else:
            print('{} Epoch {}, Training loss {}, Testing loss {}'.format(datetime.datetime.now(),
                                                                              epoch, loss_in_train, loss_in_eval))
            return loss_in_train, loss_in_eval
