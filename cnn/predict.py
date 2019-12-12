import tensorflow as tf
from PIL import Image
import numpy as np
import os
from mytrain import CNN
import warnings
warnings.filterwarnings('ignore')


if __name__ == "__main__":
    print(os.getcwd())
    path = os.path.abspath(os.path.join(os.getcwd(), '../image/888.png'))
    img = Image.open(path).convert('L')
    flatten_img = np.reshape(img, (28, 28, 1))
    x = np.array([1 - flatten_img])
    cnn = CNN()
    point_path = os.path.abspath(os.path.join(os.getcwd()+"/../point"))
    cnn.model.load_weights(tf.train.latest_checkpoint(point_path))
    predicted = cnn.model.predict(x)

    print(path)
    print(predicted)
    print('这张图大概率是', np.argmax(predicted[0]))