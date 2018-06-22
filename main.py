import tensorflow as tf
import numpy as np
from skimage.io import imread, imsave
from skimage.transform import resize
import cv2
import shutil
import os
from vgg19 import VGG19

# Height and width the images will be resized to.
im_width = 400
im_height = 400

# This is a precalculated value that must be used.
mean_pixel = np.array([123.68, 116.779, 103.939]).reshape((1,1,3))

# Clear out the results directory.
if os.path.exists('results'):
    shutil.rmtree('results')
os.mkdir('./results')

def save_image(path, img):
    save_img = img + mean_pixel
    save_img = np.clip(save_img, 0.0, 255.0)
    save_img = save_img.astype(np.uint8)

    if len(save_img.shape) == 4:
        imsave(path, save_img[0])
    else:
        imsave(path, save_img)

def preprocess_image(image):
    image = cv2.resize(image, (im_width, im_height))
    image = np.expand_dims(image, axis=0)
    image = image.astype(np.float32)
    image -= mean_pixel
    return image


# Helper function to compute the gram matrix of a tensor.
def gram_matrix(x):
    # Flatten each channel. Channel is last.
    m = tf.reshape(x, shape=[-1, x.get_shape()[-1]])
    return tf.matmul(tf.transpose(m), m)


content = imread('imgs/content_a.jpeg')
style = imread('imgs/style_a.jpeg')

# Download the weights from http://www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-19.mat
# Put them in the current directory.
net = VGG19('./imagenet-vgg-verydeep-19.mat')

im_content = preprocess_image(content)
im_style = preprocess_image(style)

# Applying some noise to the starting image normally works best, but just
# starting with the content image works well enough.
im_gen = preprocess_image(content)

# Layer names that we will compute style loss for and the weight of each layer.
use_style_layers = {
        'relu1_1': 0.2,
        'relu2_1': 0.2,
        'relu3_1': 0.2,
        'relu4_1': 0.2,
        'relu5_1': 0.2,
    }
# The layer we will use to compute the content loss. You can use more than one
# layer if you like but one layer works just fine.
use_content_layer = 'conv4_2'

# This is what we will be optimizing
generated_x = tf.Variable(im_gen, trainable = True, dtype=tf.float32)

# We fix both of these.
content_x = tf.placeholder(tf.float32, shape = im_content.shape, name='content')
style_x = tf.placeholder(tf.float32, shape = im_style.shape, name='style')

# Pass through VGG net
content_layers = net.feed_forward(content_x, scope='content')
style_layers = net.feed_forward(style_x, scope='style')
combined_layers = net.feed_forward(generated_x, scope='mixed')

# Weight of content loss
alpha = 1e-3
# Weight of style loss
beta = 1
# you only need to tune the ratio of alpha and beta so just keep beta at 1.

with tf.Session() as sess:
    # Compute content loss
    content_transformed = content_layers[use_content_layer]
    combined_transformed = combined_layers[use_content_layer]

    content_loss = tf.reduce_sum(tf.square(content_transformed -
        combined_transformed)) / 2

    # Compute the style loss
    style_loss = 0
    for feature_layer, weight in use_style_layers.items():
        style_layer = style_layers[feature_layer]
        combined_layer = combined_layers[feature_layer]

        # First dimension is the batch size which is always 1
        _, width, height, depth = style_layer.get_shape()

        N = (width * height).value
        C = depth.value

        style_g = gram_matrix(style_layer)
        combined_g = gram_matrix(combined_layer)

        E_l = tf.reduce_sum(tf.square(style_g - combined_g)) / (4 * (N ** 2) *
                (C ** 2))

        style_loss += (weight * E_l)

    # Combine losses for overall loss
    loss = (alpha * content_loss) + (beta * style_loss)

    sess.run(tf.global_variables_initializer())

    num_iters = 1000
    # You might wonder why 'BFGS'? Why not just a standard TF optimizer? Well
    # you could just use Adam optimizer and this would still work. However, I
    # found that the images look the best with this optimizer.
    # Good article on what BFGS is
    # http://aria42.com/blog/2014/12/understanding-lbfgs
    optimizer = tf.contrib.opt.ScipyOptimizerInterface(loss,
            method='L-BFGS-B', options={'maxiter': num_iters})

    i = 0
    def callback(total_loss, content_loss, style_loss):
        global i
        print('%i), total: %.2f, style: %.2f, content: %.2f' % (i, total_loss, content_loss, style_loss))
        i += 1

    optimizer.minimize(sess, feed_dict={content_x: im_content, style_x: im_style},
            fetches=[loss, style_loss, content_loss], loss_callback=callback)
    # Evaluate what the generated image tensor turned out to be.
    final_image = sess.run(generated_x)

    save_image('results/final.png', final_image)

