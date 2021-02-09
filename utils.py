import numpy as np
from PIL import Image

from input import PositionEncoding
import tensorflow as tf

def load_img_PE(img_name, basis_function):
    img_name = 'image_9_img_1'
    im = Image.open(f'dataset/{img_name}.jpg')
    im2arr = np.array(im)

    testimg = im2arr
    testimg = testimg / 255.0
    H, W, C = testimg.shape

    PE = PositionEncoding(testimg, basis_function)

    return PE, (H, W, C), testimg

def pe2coord(pe, basis_function = 'sin_cos'):
    """
    Take 40-dim PE input and convert this into coordinates
    e.g. (0, 1, 0, 1, ...) -> (0, 0)
    PE: (x_sin, x_cos, y_sin, y_cos, ...) =>
    :param pe: 40-dim input
    :return: corresponding coordinates
    """
    pass

def dash2pe(dash, basis_function='sin_cos'):
    """

    :param dash: normalized coord batch ( -1~1 ) = Bx4
    :param basis_function:
    :return:   corresponding PE = Bx40
    """
    L = 10

    x = dash[:, 0]
    y = dash[:, 1]
    output = []
    for el in range(0, L):
        val = 2 ** el
        x_sin = tf.math.sin(val * x * np.pi)
        x_cos = tf.math.cos(val * x * np.pi)
        y_sin = tf.math.sin(val * y * np.pi)
        y_cos = tf.math.cos(val * y * np.pi)
        # list of Bx4 tensors (L elements)
        output.append(tf.stack([x_sin, x_cos, y_sin, y_cos], axis=-1))
    # dimension: LxBx4
    output = tf.concat(output, axis=-1)

    return output

def coord2pe(coord, shape, basis_function='sin_cos'):
    """
    :param coord: (x,y) or Bx(x,y) i.e. shape: B*2
    :param shape: (H, W)
    :param basis_function:
    :return:
    """
    H, W = shape
    # TODO 0: check whether original vector is changed or not
    coord[:, 0] = coord[:, 0] / W * 2 - 1
    coord[:, 1] = coord[:, 1] / H * 2 - 1

    return dash2pe(coord)


if __name__=='__main__':
    shape = (256, 256)
    x_i = 0
    y_i = 0
    tmp_tensor= tf.convert_to_tensor([[x_i, y_i]])

    coord2pe(tmp_tensor, shape)

