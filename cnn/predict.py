import tensorflow as tf
from PIL import Image
import numpy as np
import os
from mytrain import CNN
import warnings
warnings.filterwarnings('ignore')
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)
tf.config.experimental.set_virtual_device_configuration(gpus[0],
[tf.config.experimental.VirtualDeviceConfiguration(memory_limit=100)])

if __name__ == "__main__":
    path = os.getcwd()
    path += '\\..\\image\\888.png'
    img = Image.open(path).convert('L')
    flatten_img = np.reshape(img, (28, 28, 1))
    x = np.array([1 - flatten_img])
    cnn = CNN()
    cnn.model.load_weights(tf.train.latest_checkpoint('/point'))
    predicted = cnn.model.predict(x)

    print(path)
    print(predicted)
    print('这张图大概率是', np.argmax(predicted[0]))
    