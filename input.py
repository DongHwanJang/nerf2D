import numpy as np
import tensorflow as tf

from scipy import signal, special

from random import random


class PositionEncoding(object):
    def __init__(self, image_np, basis_function):
        super().__init__()

        self.dataset_size = []

        H, W, C = image_np.shape

        self.dataset = [np.array([-1, -1, -1, -1, -1])] * W * H

        L = 10

        x_linspace = (np.linspace(0, W - 1, W) / W) * 2 - 1
        y_linspace = (np.linspace(0, H - 1, H) / H) * 2 - 1

        x_el = []
        y_el = []

        x_el_hf = []
        y_el_hf = []

        self.basis_function = basis_function

        # cache the values so you don't have to do function calls at every pixel
        for el in range(0, L):
            val = 2 ** el

            if basis_function == 'rbf':

                # Trying Random Fourier Features https://www.cs.cmu.edu/~schneide/DougalRandomFeatures_UAI2015.pdf
                # and https://gist.github.com/vvanirudh/2683295a198a688ef3c49650cada0114

                # Instead of a phase shift of pi/2, we could randomise it [-pi, pi]

                M_1 = np.random.rand(2, 2)

                phase_shift = np.random.rand(1) * np.pi

                x_1_y_1 = np.sin(val * np.matmul(M_1, np.vstack((x_linspace, y_linspace))))
                x_el.append(x_1_y_1[0, :])
                y_el.append(x_1_y_1[1, :])

                x_1_y_1 = np.sin(val * np.matmul(M_1, np.vstack((x_linspace, y_linspace))) + phase_shift)
                x_el_hf.append(x_1_y_1[0, :])
                y_el_hf.append(x_1_y_1[1, :])

            elif basis_function == 'diric':

                x = special.diric(np.pi * x_linspace, val)
                x_el.append(x)

                x = special.diric(np.pi * x_linspace + np.pi / 2.0, val)
                x_el_hf.append(x)

                y = special.diric(np.pi * y_linspace, val)
                y_el.append(y)

                y = special.diric(np.pi * y_linspace + np.pi / 2.0, val)
                y_el_hf.append(y)

            elif basis_function == 'sawtooth':
                x = signal.sawtooth(val * np.pi * x_linspace)
                x_el.append(x)

                x = signal.sawtooth(val * np.pi * x_linspace + np.pi / 2.0)
                x_el_hf.append(x)

                y = signal.sawtooth(val * np.pi * y_linspace)
                y_el.append(y)

                y = signal.sawtooth(val * np.pi * y_linspace + np.pi / 2.0)
                y_el_hf.append(y)

            elif basis_function == 'sin_cos':

                x = np.sin(val * np.pi * x_linspace)
                x_el.append(x)

                x = np.cos(val * np.pi * x_linspace)
                x_el_hf.append(x)

                y = np.sin(val * np.pi * y_linspace)
                y_el.append(y)

                y = np.cos(val * np.pi * y_linspace)
                y_el_hf.append(y)

        # TODO 3: vectorise this code!
        for y_i in range(0, H):
            for x_i in range(0, W):

                r, g, b = image_np[y_i, x_i]

                p_enc = []

                # i.e. passing raw coordinates instead of positional encoding
                if basis_function == 'raw_xy':

                    xdash = (x_i / W) * 2 - 1
                    ydash = (y_i / H) * 2 - 1
                    p_enc = [xdash, ydash]

                else:

                    for li in range(0, L):
                        p_enc.append(x_el[li][x_i])
                        p_enc.append(x_el_hf[li][x_i])

                        p_enc.append(y_el[li][y_i])
                        p_enc.append(y_el_hf[li][y_i])

                p_enc = p_enc + [x_i, y_i, r * 2 - 1, g * 2 - 1, b * 2 - 1]

                self.dataset[y_i * W + x_i] = np.array(p_enc)

        self.dataset_size = len(self.dataset)
        print('size of dataset_size = ', self.dataset_size)

        self.ind = np.arange(np.sum(self.dataset_size))
        np.random.shuffle(self.ind)

        self.batch_count = 0

    def get_batch(self, batch_size=10):

        input_vals = []
        output_vals = []
        indices_vals = []

        for i in range(batch_size):

            if self.batch_count * batch_size + i >= self.dataset_size:
                self.batch_count = 0
                np.random.shuffle(self.ind)
                print(
                    '************************************************* new shuffle *****************************************')
                # break

            p_enc = self.dataset[self.ind[self.batch_count * batch_size + i]]

            input_vals.append(p_enc[0:-5])

            r, g, b = p_enc[-3], p_enc[-2], p_enc[-1]
            x, y = p_enc[-5], p_enc[-4]

            output_vals.append([r, g, b])

            indices_vals.append([x, y])

        self.batch_count += 1
        return np.array(input_vals), np.array(output_vals), np.array(indices_vals)

    def get_test_batch(self, batch_size):
        pass


class PositionEncodingScale(object):
    def __init__(self, image_np, basis_function, scale=1.0):
        super().__init__()

        self.dataset_size = []

        org_H, org_W, C = image_np.shape
        H = int(scale * org_H)
        W = int(scale * org_W)

        self.dataset = [np.array([-1, -1, -1, -1, -1])] * W * H

        L = 10

        x_linspace = (np.linspace(0, W - 1, W) / W) * 2 - 1  # -1~0.99xx
        y_linspace = (np.linspace(0, H - 1, H) / H) * 2 - 1

        x_el = []
        y_el = []

        x_el_hf = []
        y_el_hf = []

        self.basis_function = basis_function

        # cache the values so you don't have to do function calls at every pixel
        for el in range(0, L):
            val = 2 ** el

            if basis_function == 'rbf':

                # Trying Random Fourier Features https://www.cs.cmu.edu/~schneide/DougalRandomFeatures_UAI2015.pdf
                # and https://gist.github.com/vvanirudh/2683295a198a688ef3c49650cada0114

                # Instead of a phase shift of pi/2, we could randomise it [-pi, pi]

                M_1 = np.random.rand(2, 2)

                phase_shift = np.random.rand(1) * np.pi

                x_1_y_1 = np.sin(val * np.matmul(M_1, np.vstack((x_linspace, y_linspace))))
                x_el.append(x_1_y_1[0, :])
                y_el.append(x_1_y_1[1, :])

                x_1_y_1 = np.sin(val * np.matmul(M_1, np.vstack((x_linspace, y_linspace))) + phase_shift)
                x_el_hf.append(x_1_y_1[0, :])
                y_el_hf.append(x_1_y_1[1, :])

            elif basis_function == 'diric':

                x = special.diric(np.pi * x_linspace, val)
                x_el.append(x)

                x = special.diric(np.pi * x_linspace + np.pi / 2.0, val)
                x_el_hf.append(x)

                y = special.diric(np.pi * y_linspace, val)
                y_el.append(y)

                y = special.diric(np.pi * y_linspace + np.pi / 2.0, val)
                y_el_hf.append(y)

            elif basis_function == 'sawtooth':
                x = signal.sawtooth(val * np.pi * x_linspace)
                x_el.append(x)

                x = signal.sawtooth(val * np.pi * x_linspace + np.pi / 2.0)
                x_el_hf.append(x)

                y = signal.sawtooth(val * np.pi * y_linspace)
                y_el.append(y)

                y = signal.sawtooth(val * np.pi * y_linspace + np.pi / 2.0)
                y_el_hf.append(y)

            elif basis_function == 'sin_cos':

                x = np.sin(val * np.pi * x_linspace)
                x_el.append(x)

                x = np.cos(val * np.pi * x_linspace)
                x_el_hf.append(x)

                y = np.sin(val * np.pi * y_linspace)
                y_el.append(y)

                y = np.cos(val * np.pi * y_linspace)
                y_el_hf.append(y)

        # TODO 3: vectorise this code!
        for y_i in range(0, H):
            for x_i in range(0, W):

                # r, g, b = image_np[y_i, x_i]

                p_enc = []

                # i.e. passing raw coordinates instead of positional encoding
                if basis_function == 'raw_xy':

                    xdash = (x_i / W) * 2 - 1
                    ydash = (y_i / H) * 2 - 1
                    p_enc = [xdash, ydash]

                else:

                    for li in range(0, L):
                        p_enc.append(x_el[li][x_i])
                        p_enc.append(x_el_hf[li][x_i])

                        p_enc.append(y_el[li][y_i])
                        p_enc.append(y_el_hf[li][y_i])

                p_enc = p_enc + [x_i, y_i]

                self.dataset[y_i * W + x_i] = np.array(p_enc)

        self.dataset_size = len(self.dataset)
        print('size of dataset_size = ', self.dataset_size)

        self.ind = np.arange(np.sum(self.dataset_size))
        # np.random.shuffle(self.ind)
        self.batch_count = 0

    def get_batch(self, batch_size=10):

        input_vals = []
        indices_vals = []

        for i in range(batch_size):

            if self.batch_count * batch_size + i >= self.dataset_size:
                self.batch_count = 0
                print(
                    '************************************************* new shuffle *****************************************')
                # break

            p_enc = self.dataset[self.ind[self.batch_count * batch_size + i]]

            input_vals.append(p_enc[0:-2])
            x, y = p_enc[-2], p_enc[-1]

            indices_vals.append([x, y])

        self.batch_count += 1
        return np.array(input_vals), np.array(indices_vals)

    def get_test_batch(self, batch_size):
        pass


class PositionEncodingShift(object):
    def __init__(self, image_np, basis_function, xshift=random() - .5, yshift=random() - .5):
        super().__init__()

        self.dataset_size = []

        H, W, C = image_np.shape

        self.dataset = [np.array([-1, -1, -1, -1, -1])] * W * H

        L = 10

        x_linspace = (np.linspace(0 + xshift, W + xshift - 1, W) / W) * 2 - 1  # -1~0.99xx
        y_linspace = (np.linspace(0 + yshift, H + yshift - 1, H) / H) * 2 - 1

        x_el = []
        y_el = []

        x_el_hf = []
        y_el_hf = []

        self.basis_function = basis_function

        # cache the values so you don't have to do function calls at every pixel
        for el in range(0, L):
            val = 2 ** el

            if basis_function == 'rbf':

                # Trying Random Fourier Features https://www.cs.cmu.edu/~schneide/DougalRandomFeatures_UAI2015.pdf
                # and https://gist.github.com/vvanirudh/2683295a198a688ef3c49650cada0114

                # Instead of a phase shift of pi/2, we could randomise it [-pi, pi]

                M_1 = np.random.rand(2, 2)

                phase_shift = np.random.rand(1) * np.pi

                x_1_y_1 = np.sin(val * np.matmul(M_1, np.vstack((x_linspace, y_linspace))))
                x_el.append(x_1_y_1[0, :])
                y_el.append(x_1_y_1[1, :])

                x_1_y_1 = np.sin(val * np.matmul(M_1, np.vstack((x_linspace, y_linspace))) + phase_shift)
                x_el_hf.append(x_1_y_1[0, :])
                y_el_hf.append(x_1_y_1[1, :])

            elif basis_function == 'diric':

                x = special.diric(np.pi * x_linspace, val)
                x_el.append(x)

                x = special.diric(np.pi * x_linspace + np.pi / 2.0, val)
                x_el_hf.append(x)

                y = special.diric(np.pi * y_linspace, val)
                y_el.append(y)

                y = special.diric(np.pi * y_linspace + np.pi / 2.0, val)
                y_el_hf.append(y)

            elif basis_function == 'sawtooth':
                x = signal.sawtooth(val * np.pi * x_linspace)
                x_el.append(x)

                x = signal.sawtooth(val * np.pi * x_linspace + np.pi / 2.0)
                x_el_hf.append(x)

                y = signal.sawtooth(val * np.pi * y_linspace)
                y_el.append(y)

                y = signal.sawtooth(val * np.pi * y_linspace + np.pi / 2.0)
                y_el_hf.append(y)

            elif basis_function == 'sin_cos':

                x = np.sin(val * np.pi * x_linspace)
                x_el.append(x)

                x = np.cos(val * np.pi * x_linspace)
                x_el_hf.append(x)

                y = np.sin(val * np.pi * y_linspace)
                y_el.append(y)

                y = np.cos(val * np.pi * y_linspace)
                y_el_hf.append(y)

        # vectorise the code
        p_enc_only = [np.stack(
            [np.repeat(x_el[li][None, :], H, axis=0), np.repeat(x_el_hf[li][None, :], 256, axis=0),
             np.repeat(y_el[li][:, None], W, axis=1), np.repeat(y_el_hf[li][:, None], 256, axis=1)], axis=2)
            for li in range(0, L)]  # (H, W, 4)*10
        x_y_HW = [np.stack([np.repeat(np.arange(W)[None, :], H, axis=0),
                            np.repeat(np.arange(H)[:, None], W, axis=1)], axis=2)]  # (H, W, 2)*1
        p_enc = np.concatenate(p_enc_only + x_y_HW, axis=2)  # (H,W,42)
        p_enc = np.reshape(p_enc, (H * W, 42))
        self.dataset = p_enc

        self.dataset_size = len(self.dataset)
        # print('size of dataset_size = ', self.dataset_size)

        self.ind = np.arange(np.sum(self.dataset_size))
        # np.random.shuffle(self.ind)
        self.batch_count = 0

    def get_batch(self, batch_size=10):

        input_vals = []
        output_vals = []
        indices_vals = []

        for i in range(batch_size):

            if self.batch_count * batch_size + i >= self.dataset_size:
                self.batch_count = 0
                print(
                    '************************************************* new shuffle *****************************************')
                # break

            p_enc = self.dataset[self.ind[self.batch_count * batch_size + i]]

            input_vals.append(p_enc[0:-2])
            x, y = p_enc[-2], p_enc[-1]

            indices_vals.append([x, y])

        self.batch_count += 1
        return np.array(input_vals), np.array(indices_vals)

    def get_test_batch(self, batch_size):
        pass


def resize(input_image, real_image, height, width):
    input_image = tf.image.resize(input_image, [height, width],
                                  method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    real_image = tf.image.resize(real_image, [height, width],
                                 method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    return input_image, real_image


def random_crop(input_image, real_image, IMG_HEIGHT=256, IMG_WIDTH=256):
    real_image = tf.cast(real_image, tf.float32)
    stacked_image = tf.stack([input_image, real_image], axis=0)
    cropped_image = tf.image.random_crop(
        stacked_image, size=[2, IMG_HEIGHT, IMG_WIDTH, 3])

    return cropped_image[0], cropped_image[1]


def random_jitter(input_image, real_image):
    # TODO 3: actually we don't need to resize and randomcrop using the same seed.
    #  This is just for the computational cost
    # resizing to 286 x 286 x 3
    input_image, real_image = resize(input_image, real_image, 286, 286)

    # randomly cropping to 256 x 256 x 3
    input_image, real_image = random_crop(input_image, real_image)

    # if tf.random.uniform(()) > 0.5:
    #   # random mirroring
    #   input_image = tf.image.flip_left_right(input_image)
    #   real_image = tf.image.flip_left_right(real_image)

    return input_image, real_image
