import os
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import datetime


# os.environ["CUDA_VISIBLE_DEVICES"]="-1"


class DataSource(object):
    def __init__(self):
        data_path = os.path.abspath(os.path.dirname(__file__))
        data_path += '/../mnist.npz'
        mnist = datasets.mnist

        (train_images, train_labels), (test_images,
                                       test_labels) = mnist.load_data(path=data_path)

        # 6万张训练图片，1万张测试图片
        train_images = train_images.reshape((60000, 28, 28, 1))
        test_images = test_images.reshape((10000, 28, 28, 1))

        # 像素值归一化
        train_images = train_images / 255.0
        test_images = test_images / 255.0

        self.train_images = train_images
        self.train_labels = train_labels
        self.test_images = test_images
        self.test_labels = test_labels


class UseGpu(object):
    def __init__(self):
        self.want_to_use_gpu = False
        self.gpu_memory = 570

    def this_device_use_gpu(self):
        gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
        tf.config.experimental.set_memory_growth(gpus[0], True)
        tf.config.experimental.set_virtual_device_configuration(gpus[0],
                                                                [tf.config.experimental.VirtualDeviceConfiguration(
                                                                    memory_limit=self.gpu_memory)])

    def is_device_use_gpu(self):
        if self.want_to_use_gpu:
            if tf.test.is_gpu_available():
                self.this_device_use_gpu()
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


class CNN(object):
    def __init__(self):

        UseGpu().is_device_use_gpu()

        model = models.Sequential()
        # 第1层卷积，卷积核大小为3*3，32个，28*28为待训练图片的大小
        model.add(layers.Conv2D(
            32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
        model.add(layers.MaxPooling2D((2, 2)))
        # 第2层卷积，卷积核大小为3*3，64个
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        # 第3层卷积，卷积核大小为3*3，64个
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))

        model.add(layers.Flatten())
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(10, activation='softmax'))

        self.model = model


class Train:
    def __init__(self):
        self.cnn = CNN()
        self.data = DataSource()

    def train(self):
        check_path = './point/cp-{epoch:04d}.ckpt'
        save_model_cb = tf.keras.callbacks.ModelCheckpoint(
            check_path, save_weights_only=True, verbose=1, period=5)

        self.cnn.model.compile(optimizer='adam',
                               loss='sparse_categorical_crossentropy',
                               metrics=['accuracy'])

        log_dir = "C:\\code\\cnn_Introduction\\logs\\" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

        self.cnn.model.fit(self.data.train_images, self.data.train_labels,
                           epochs=5, callbacks=[save_model_cb, tensorboard_callback])

        test_loss, test_acc = self.cnn.model.evaluate(
            self.data.test_images, self.data.test_labels)
        print("准确率: %.4f，共测试了%d张图片 " % (test_acc, len(self.data.test_labels)))


if __name__ == "__main__":
    app = Train()
    app.train()
