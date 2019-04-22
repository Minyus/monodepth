# Copyright UCL Business plc 2017. Patent Pending. All rights reserved.
#
# The MonoDepth Software is licensed under the terms of the UCLB ACP-A licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.
#
# For any other use of the software not covered by the UCLB ACP-A Licence,
# please contact info@uclb.com

from __future__ import absolute_import, division, print_function

# only keep warnings and errors
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='0'


import scipy.misc
import matplotlib.pyplot as plt

from monodepth_model import *

import easydict
import tensorflow as tf
import numpy as np

def post_process_disparity(disp):
    _, h, w = disp.shape
    l_disp = disp[0,:,:]
    r_disp = np.fliplr(disp[1,:,:])
    m_disp = 0.5 * (l_disp + r_disp)
    l, _ = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    l_mask = 1.0 - np.clip(20 * (l - 0.05), 0, 1)
    r_mask = np.fliplr(l_mask)
    return r_mask * l_disp + l_mask * r_disp + (1.0 - l_mask - r_mask) * m_disp


def tf_fliplr(t):
    assert len(t.shape) == 2
    t = tf.expand_dims(t, 2)
    t = tf.image.flip_left_right(t)
    t = tf.squeeze(t)
    return t
def post_process_disparity_tf(disp):
    _, h, w = disp.shape
    l_disp = disp[0,:,:]
    # r_disp = np.fliplr(disp[1,:,:])
    r_disp = tf_fliplr(disp[1, :, :])
    m_disp = 0.5 * (l_disp + r_disp)
    # l, _ = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    l, _ = tf.meshgrid(tf.linspace(0.0, 1.0, w), tf.linspace(0.0, 1.0, h))
    # l_mask = 1.0 - np.clip(20 * (l - 0.05), 0, 1)
    l_mask = 1.0 - tf.clip_by_value(20.0 * (l - 0.05), 0.0, 1.0)
    # r_mask = np.fliplr(l_mask)
    r_mask = tf_fliplr(l_mask)
    return r_mask * l_disp + l_mask * r_disp + (1.0 - l_mask - r_mask) * m_disp


def generate():

    args = easydict.EasyDict({
        'encoder': 'vgg',
        'image_path': '/home/venus/yusuke/proj/monodepth/images/frame07.png',
        'checkpoint_path': '/home/venus/yusuke/proj/monodepth/models/model_cityscapes/model_cityscapes',
        'input_height': 256,
        'input_width': 512
    })

    params = monodepth_parameters(
        encoder=args.encoder,
        height=args.input_height,
        width=args.input_width,
        batch_size=2,
        num_threads=1,
        num_epochs=1,
        do_stereo=False,
        wrap_mode="border",
        use_deconv=False,
        alpha_image_loss=0,
        disp_gradient_loss_weight=0,
        lr_loss_weight=0,
        full_summary=False)

    output_directory = os.path.dirname(args.image_path)
    output_name = os.path.splitext(os.path.basename(args.image_path))[0]

    input_image = scipy.misc.imread(args.image_path, mode="RGB")
    original_height, original_width, num_channels = input_image.shape
    input_image = scipy.misc.imresize(input_image, [args.input_height, args.input_width], interp='lanczos')
    input_image = input_image.astype(np.float32) / 255

    input_image_t = tf.placeholder(tf.float32, [args.input_height, args.input_width, 3])
    # input_image_t = tf.cast(input_image_t, dtype=tf.float32) / 255
    input_images_t = tf.stack([input_image_t, tf.image.flip_left_right(input_image_t)], 0)

    model = MonodepthModel(params, "test", input_images_t, None)
    disp_t = model.disp_left_est[0]

    disp_squeezed_t = tf.squeeze(disp_t)

    disp_pp_t = post_process_disparity_tf(disp_squeezed_t)

    # SESSION
    config = tf.ConfigProto(allow_soft_placement=True)
    sess = tf.Session(config=config)

    # SAVER
    train_saver = tf.train.Saver()

    # INIT
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    coordinator = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coordinator)

    # RESTORE
    restore_path = args.checkpoint_path.split(".")[0]
    train_saver.restore(sess, restore_path)

    pp_in_tf = False
    if not pp_in_tf:
        disp = sess.run(disp_t, feed_dict={input_image_t: input_image}) #_ (2, H, W, 1)
        print('disp.shape: ', disp.shape)

        disp_squeezed = disp.squeeze() #_ (2, H, W)
        disp_pp = post_process_disparity(disp_squeezed).astype(np.float32) #_ (H, W)
        print('disp_pp.shape: ', disp_pp.shape)

    if pp_in_tf:
        disp_pp = sess.run(disp_pp_t, feed_dict={input_image_t: input_image})  # _ (H, W)
        print('disp_pp.shape: ', disp_pp.shape)

    # np.save(os.path.join(output_directory, "{}_disp.npy".format(output_name)), disp_pp)
    disp_to_img = scipy.misc.imresize(disp_pp.squeeze(), [original_height, original_width])
    plt.imsave(os.path.join(output_directory, "{}_disp.png".format(output_name)), disp_to_img, cmap='plasma')
    print('done!')



generate()
