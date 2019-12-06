import os
import tensorflow as tf
# from tensorflow.keras import datasets, layers, models
import datetime

if __name__ == "__main__":

    data_path = os.getcwd()+'/../mnist.npz'
    print(data_path)
    

    # mnist = datasets.mnist

    # (train_images, train_labels), (test_images,
    #                                test_labels) = mnist.load_data(path=data_path)

    # # 6万张训练图片，1万张测试图片
    # train_images = train_images.reshape((60000, 28, 28, 1))
    # test_images = test_images.reshape((10000, 28, 28, 1))

    # # 像素值归一化
    # train_images = train_images / 255.0
    # test_images = test_images / 255.0

    # app = Train()
    # app.train()
