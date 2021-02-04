from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
from PIL import Image
import cv2 as cv2
from random import random
import time

from input import PositionEncoding, PositionEncodingScale, PositionEncodingShift, random_jitter
from loss import discriminator_loss, generator_loss
from network.discriminator import Discriminator

import argparse
import datetime

from network.model import build_model

default_name = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")

parser = argparse.ArgumentParser(description='Process some integers.')

parser.add_argument('--name', type=str, default=default_name,
                    help='name of the experiment')

args = parser.parse_args()

im = Image.open('dataset/cool_cows.jpg')
im2arr = np.array(im)
ckpt_path = f'load/ckpt_latset_{args.name}.ckpt'

testimg = im2arr
testimg = testimg / 255.0
H, W, C = testimg.shape

scale = 2.0
scaled_H, scaled_W = int(scale * H), int(scale * W)
_read_img = np.zeros((scaled_H, scaled_W, 3))
_read_img2 = np.zeros((H, W, 3))

PE = PositionEncoding(testimg, 'sin_cos')
PEScale = PositionEncodingScale(testimg, 'sin_cos', scale=scale)

dataset_size = PE.dataset_size

# TODO 1: calibrate lr for each optimizers.
loss_object = tf.keras.losses.MeanSquaredError()
optimizer = tfa.optimizers.AdamW(weight_decay=1e-4, lr=1e-2)
generator_optimizer = tfa.optimizers.AdamW(weight_decay=1e-4, lr=1e-2, beta_1=0.5)
discriminator_optimizer = tfa.optimizers.AdamW(weight_decay=1e-4, lr=1e-2, beta_1=0.5)
# TODO 1: compare which optimizer is better
if False:
    optimizer = tf.keras.optimizers.Adam(lr=1e-2)
    l
    generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

EPOCHS = 5

model = build_model(output_dims=3)
discriminator = Discriminator()

batch_size = 1024
decay = 0.999

count = 0
gan_step = 1
epoch_no = 0
MAX_EPOCH = 5000

save_every = 200

# TODO 2: find adequate training frequency
# gan_every = dataset_size / batch_size
# increase GAN phase frequency
# gan_every = dataset_size / batch_size
# gan_repeat = 1
# TODO 0 fix this value. this is just for debug
gan_every = 10000000

while True:

    inp_batch, inp_target, ind_vals = PE.get_batch(batch_size=batch_size)

    with tf.GradientTape() as tape, \
            tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        # GAN Phase
        if gan_step > 0 and gan_step % gan_every == 0:
            # check how long does it take to make PEShift dataset
            start_time = time.time()

            # change the scale of pixel shift, from -0.5~0.5 to -1~1.
            #  But I think, in this case, b/c of pixel shift D might notice the difference,
            #  which leads to the reason why we need to use RandomCrop for GAN training.
            xshift = (random() - 0.5)
            yshift = (random() - 0.5)
            # do fixed shift
            # xshift = 0.5
            # yshift = 0.5

            # TODO 1: Make batch for sampling operation
            PEShift = PositionEncodingShift(testimg, 'sin_cos', xshift=xshift, yshift=yshift)
            print(f'Data prepration time: {time.time() - start_time}')
            img_batch, img_ind_vals = PEShift.get_batch(batch_size=PEShift.dataset_size)
            assert PEShift.dataset_size == H * W

            output = model(img_batch, training=True)

            # vectorize the output and feed to D
            # TODO 1: get an average rather than just one pixel pick.
            reshaped_output = tf.reshape(output, [H, W, 3])  # (H*W, 3)-> (1, H, W, 3)
            gt_img = np.copy(testimg * 2 - 1)

            # random crop for the generated input and gt
            reshaped_output, gt_img = random_jitter(reshaped_output, gt_img)

            reshaped_output = tf.reshape(reshaped_output, [1, H, W, 3])
            gt_img = tf.reshape(gt_img, [1, H, W, 3])
            # make sure input & output has the same range (i.e. same normalization)
            # testimg: [-1, 1] & reshaped_output: [-inf, inf] but trained to fit in [-1, 1]

            disc_real_output = discriminator(gt_img)
            disc_generated_output = discriminator(reshaped_output)
            disc_loss = discriminator_loss(disc_real_output, disc_generated_output)
            gen_loss = generator_loss(disc_generated_output)

            # TODO 1: make an option for the 1) adversarial loss 2) feature loss

            # See the loss of D. It should be around 0.68=log(2)
            print(f"===GAN Phase: {time.time() - start_time} seconds, "
                  f"xshift: {xshift}, yshift: {yshift}, disc_loss : {disc_loss}, "
                  f"learning_rate : {generator_optimizer.learning_rate.numpy()} ===")

            # display the output from the shifted input
            img_ind_vals_int = img_ind_vals.astype('int')
            img_ind_vals_int = img_ind_vals_int[:, 1] * W + img_ind_vals_int[:, 0]

            # if count > 0 and count % save_every == 0:
            np.put(_read_img2[:, :, 0], img_ind_vals_int,
                   np.clip((output[:, 0] + 1) / 2.0, 0,
                           1))  # put output into 'ind_vals_int' idx (using np indexing function)
            np.put(_read_img2[:, :, 1], img_ind_vals_int,
                   np.clip((output[:, 1] + 1) / 2.0, 0, 1))  # Note that you don't need to sort the output!
            np.put(_read_img2[:, :, 2], img_ind_vals_int, np.clip((output[:, 2] + 1) / 2.0, 0, 1))

            fileName = './results/training_shift_evolution_' + PEShift.basis_function + f'_{int(epoch_no):04d}_{args.name}.jpg'
            save_img = np.copy(_read_img2[..., ::-1] * 255.0)
            cv2.imwrite(fileName, save_img.astype('uint8'))

            gen_gradients = gen_tape.gradient(gen_loss, model.trainable_variables)
            # check whether gen_gradients have right values from generated_input.
            # gen_gradients = gen_tape.gradient(generator_loss(gt_img), model.trainable_variables)
            disc_gradients = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

            generator_optimizer.apply_gradients(zip(gen_gradients, model.trainable_variables))
            discriminator_optimizer.apply_gradients(zip(disc_gradients, discriminator.trainable_variables))

        # L2 Phase
        output = model(inp_batch, training=True)

        loss_map = tf.sqrt(loss_object(output, inp_target))
        # to test GAN Phase only, uncomment the below code
        # loss_map = tf.convert_to_tensor(0)

        if count > 0 and count % save_every == 0:
            # TODO 0 This is for debug. fix it later.
            inp_batch, inp_target, ind_vals = PE.get_batch(batch_size=dataset_size)
            # inp_batch, ind_vals = PEScale.get_batch(batch_size=PEScale.dataset_size)
            output = model(inp_batch, training=False)

            ind_vals_int = ind_vals.astype('int')
            ind_vals_int = ind_vals_int[:, 1] * scaled_W + ind_vals_int[:, 0]

            np.put(_read_img[:, :, 0], ind_vals_int, np.clip((output[:, 0] + 1) / 2.0, 0,
                                                             1))  # put output into 'ind_vals_int' idx (using np indexing function)
            np.put(_read_img[:, :, 1], ind_vals_int,
                   np.clip((output[:, 1] + 1) / 2.0, 0, 1))  # Note that you don't need to sort the output!
            np.put(_read_img[:, :, 2], ind_vals_int, np.clip((output[:, 2] + 1) / 2.0, 0, 1))

            fileName = './results/training_sr_evolution_' + PEScale.basis_function + f'_{int(epoch_no):04d}_{args.name}.jpg'
            save_img = np.copy(_read_img[..., ::-1] * 255.0)
            cv2.imwrite(fileName, save_img.astype('uint8'))

            # run only or MAX_EPOCH epochs
            if epoch_no > MAX_EPOCH:
                break

            model.save_weights(ckpt_path)

        cv2.namedWindow('Align Example_sr', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('Align Example_sr', _read_img[..., ::-1])

        cv2.namedWindow('Align Example_shift', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('Align Example_shift', _read_img2[..., ::-1])
        key = cv2.waitKey(1)

        # Press esc or 'q' to close the image window
        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            break

        print(f'loss = {loss_map.numpy()}, learning_rate= {optimizer.learning_rate.numpy()}, '
              f'batch_no = {count}, epoch = {epoch_no}, batches_per_epoch = {batch_size}')

    # To test GAN Phase only, comment the below codes
    gradients = tape.gradient(loss_map, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    count += 1
    gan_step += 1

    if PE.batch_count == 1 and count > 1:
        # lr = float(tf.keras.backend.get_value(optimizer.lr))
        # tf.keras.backend.set_value(optimizer.lr, lr * 0.99)
        epoch_no += 1

    # TODO 2: upgrade to use multi-GPU

# TODO 3: implement the model save code

# TODO 4: port the code to the pytorch_lightning if needed
