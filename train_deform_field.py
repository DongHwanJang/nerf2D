from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
from PIL import Image
import cv2 as cv2
from random import random
import time

import utils

import argparse
import datetime

from network.model import build_model
from network.deform_field import build_deform_field

default_name = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")

parser = argparse.ArgumentParser(description='Process some integers.')

parser.add_argument('--name', type=str, default=default_name,
                    help='name of the experiment')

args = parser.parse_args()

# get ready for src_img and tgt_img
src_img_name = 'image_9_img_2'
tgt_img_name = 'image_9_img_1'

# TODO 1: debugging output_type 'coord'
input_basis_function = 'raw_xy' # input for 'deform_field' = dash
# input_basis_function = 'sin_cos' # input for 'defor_field' = sin_cos
output_type = 'coord'

src_pe, img_shape, _ = utils.load_img_PE(src_img_name, input_basis_function) # num_pixel x 43 (pe + rgb) class

# tgt_pe will not be used in the brute force way !!!
# tgt_pe, _, _ = utils.load_img_PE(tgt_img_name)
dataset_size = src_pe.dataset_size

H, W, C = img_shape

_read_img = np.zeros((H, W, 3))

# load_ckpt_path = f'load/ckpt_latset_{tgt_img_name}'
# TODO 1 change this to the above line when the debug ends!
load_ckpt_path = f'load/ckpt_latset_fractal'
save_ckpt_path = f'load/deform/ckpt_lastet_{src_img_name}_TO_{tgt_img_name}_{args.name}'

# TODO 1: calibrate lr for each optimizers.
loss_object = tf.keras.losses.MeanSquaredError()
# optimizer = tfa.optimizers.AdamW(weight_decay=1e-4, lr=1e-4)

# TODO 1: compare which optimizer is better
if True:
    optimizer = tf.keras.optimizers.Adam(lr=1e-4)

EPOCHS = 500

tgt_model = build_model(output_dims=3)

deform_field = build_deform_field(output_type=output_type)
tgt_model.load_weights(load_ckpt_path)

batch_size = 1024
decay = 0.999

count = 0
epoch_no = 0
MAX_EPOCH = 5000

save_every = 200


######## Start Training ########
while True:

    inp_batch, inp_target, ind_vals = src_pe.get_batch(batch_size=batch_size)

    with tf.GradientTape() as tape, \
            tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:

        if output_type == 'PE':
            deformed_inp_batch = deform_field(inp_batch)
        else:
            deformed_inp_batch = deform_field(inp_batch)
            deformed_inp_batch = utils.dash2pe(deformed_inp_batch)

        # L2 Phase
        # TODO 0: Use bilinear_model rather than tgt_model for the ablation!
        output = tgt_model(deformed_inp_batch, training=False)

        loss_map = tf.sqrt(loss_object(output, inp_target))

        if count > 0 and count % save_every == 0:
            inp_batch, inp_target, ind_vals = src_pe.get_batch(batch_size=dataset_size)

            if output_type == 'PE':
                deformed_inp_batch = deform_field(inp_batch)
            else:
                deformed_inp_batch = deform_field(inp_batch)
                deformed_inp_batch = utils.dash2pe(deformed_inp_batch)

                toy_deform = deform_field(np.asarray([[0,0]]))
                print(f'(H/2, W/2)->({toy_deform[0][0]},{toy_deform[0][1]})')

            output = tgt_model(deformed_inp_batch, training=False)

            ind_vals_int = ind_vals.astype('int')
            ind_vals_int = ind_vals_int[:, 1] * W + ind_vals_int[:, 0]

            np.put(_read_img[:, :, 0], ind_vals_int, np.clip((output[:, 0] + 1) / 2.0, 0,
                                                             1))  # put output into 'ind_vals_int' idx (using np indexing function)
            np.put(_read_img[:, :, 1], ind_vals_int,
                   np.clip((output[:, 1] + 1) / 2.0, 0, 1))  # Note that you don't need to sort the output!
            np.put(_read_img[:, :, 2], ind_vals_int, np.clip((output[:, 2] + 1) / 2.0, 0, 1))

            fileName = './results/training_sr_evolution_' + src_pe.basis_function + f'_{int(epoch_no):04d}_{args.name}.jpg'
            save_img = np.copy(_read_img[..., ::-1] * 255.0)
            cv2.imwrite(fileName, save_img.astype('uint8'))

            # run only or MAX_EPOCH epochs
            if epoch_no > MAX_EPOCH:
                break

            deform_field.save_weights(save_ckpt_path)

        cv2.namedWindow('Align Example_sr', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('Align Example_sr', _read_img[..., ::-1])

        key = cv2.waitKey(1)

        # Press esc or 'q' to close the image window
        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            break

        print(f'loss = {loss_map.numpy()}, learning_rate= {optimizer.learning_rate.numpy()}, '
              f'batch_no = {count}, epoch = {epoch_no}, batches_per_epoch = {batch_size}')

    gradients = tape.gradient(loss_map, deform_field.trainable_variables)
    optimizer.apply_gradients(zip(gradients, deform_field.trainable_variables))

    count += 1

    if src_pe.batch_count == 1 and count > 1:
        # lr = float(tf.keras.backend.get_value(optimizer.lr))
        # tf.keras.backend.set_value(optimizer.lr, lr * 0.99)
        epoch_no += 1

    # TODO 2: upgrade to use multi-GPU

# TODO 3: implement the model save code

# TODO 4: port the code to the pytorch_lightning if needed
