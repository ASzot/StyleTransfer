from keras.applications.vgg16 import VGG16
import tensorflow as tf
import numpy as np
from tqdm import tqdm
from skimage.io import imread, imsave
from skimage.transform import resize
import cv2
import shutil
import os


im_width = 400
im_height = 400
mean_values = np.array([123.68, 116.779, 103.939]).reshape((1,1,3))
shutil.rmtree('results')
os.mkdir('./results')

def save_image(path, img):
    save_img = img + mean_values
    save_img = save_img.astype(np.uint8)
    if len(save_img.shape) == 4:
        imsave(path, save_img[0])
    else:
        imsave(path, save_img)

def preprocess_image(image):
    image = cv2.resize(image, (im_width, im_height))
    image = np.expand_dims(image, axis=0)
    image = image.astype(np.float32)
    image -= mean_values
    return image


im_content_path = 'mona-lisa.jpg'
im_style_path = 'starry-night.jpg'

im_content = imread(im_content_path)
im_style = imread(im_style_path)

im_content = preprocess_image(im_content)
im_style = preprocess_image(im_style)

save_image('results/style.png', im_style)
save_image('results/content.png', im_content)

def gen_starting_image(content_image, noise_ratio = 0.6):
    noise = np.random.normal(size=(1, im_width, im_height, 3),
            scale=np.std(content_image) * 0.1)

    return (noise * noise_ratio) + (content_image * (1 - noise))

im_gen = gen_starting_image(im_content)
save_image('results/start.png', im_gen)


content_x = tf.constant(im_content, dtype=tf.float32)
style_x = tf.constant(im_style, dtype=tf.float32)

generated_x = tf.Variable(im_gen)

'''
This is what layers VGG 16 is composed of.
input_1
block1_conv1
block1_conv2
block1_pool
block2_conv1
block2_conv2
block2_pool
block3_conv1
block3_conv2
block3_conv3
block3_pool
block4_conv1
block4_conv2
block4_conv3
block4_pool
block5_conv1
block5_conv2
block5_conv3
block5_pool
flatten
fc1
fc2
predictions
'''

feature_layers = {
        'block1_conv1': 0.5,
        'block2_conv1': 1.0,
        'block3_conv1': 1.5,
        'block4_conv1': 3.0,
        'block5_conv1': 4.0,
    }


def gram_matrix(x, N, C):
    x = tf.reshape(x, [N, C])
    return tf.matmul(tf.transpose(x), x)

vgg_content = VGG16(weights='imagenet', input_tensor=content_x)
vgg_style = VGG16(weights='imagenet', input_tensor=style_x)
vgg_combined = VGG16(weights='imagenet', input_tensor=generated_x)

use_layer = 'block4_conv1'

learning_rate = 2.0
alpha = 100
beta = 5


with tf.Session() as sess:
    content_layers = dict([(layer.name, layer.output) for layer in vgg_content.layers])
    style_layers = dict([(layer.name, layer.output) for layer in vgg_style.layers])
    combined_layers = dict([(layer.name, layer.output) for layer in vgg_combined.layers])

    content_transformed = content_layers[use_layer]
    combined_transformed = combined_layers[use_layer]

    content_loss = tf.reduce_sum(tf.square(content_transformed -
        combined_transformed)) / 2


    style_loss = tf.get_variable("style_loss", dtype=tf.float32,
              initializer=tf.constant(0.0))

    for feature_layer, weight in feature_layers.items():
        style_layer = style_layers[feature_layer]
        combined_layer = combined_layers[feature_layer]

        _, width, height, C = style_layer.get_shape()

        N = width * height

        style_g = gram_matrix(style_layer, N, C)
        combined_g = gram_matrix(combined_layer, N, C)

        N = N.value
        C = C.value

        E_l = tf.reduce_sum(tf.square(style_g - combined_g)) / (4 * (N ** 2) *
                (C ** 2))

        style_loss += (weight * E_l)

    loss = (beta * content_loss) + (alpha * style_loss)

    opt = tf.train.AdamOptimizer(learning_rate).minimize(loss,
            var_list=[generated_x])


    sess.run(tf.global_variables_initializer())

    num_iters = int(1e4)
    save_interval = 100
    for update_i in tqdm(range(1, num_iters + 1)):
        loss_val, gen_im, _ = sess.run([loss, generated_x, opt], feed_dict={
            })

        print(loss_val)

        if update_i % save_interval == 0:
            save_image('results/gen_%i.png' % update_i, gen_im)









