import tensorflow as tf
from PIL import Image
import numpy as np

from train import CNN
import warnings
warnings.filterwarnings('ignore')


if __name__ == "__main__":
    path = '/root/hand/image/888.png'
    img = Image.open(path).convert('L')
    flatten_img = np.reshape(img, (28, 28, 1))
    x = np.array([1 - flatten_img])
    cnn = CNN()
    cnn.model.load_weights(tf.train.latest_checkpoint('./checkpoint'))
    predicted = cnn.model.predict(x)

    print(path)
    print(predicted)
    print('这张图大概率是', np.argmax(predicted[0]))
    