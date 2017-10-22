import tensorflow as tf
import numpy as np
import datetime
from skimage.io import imsave
# import matplotlib.pyplot as plt

from os import listdir
from os.path import isfile, join
import math
from os import path
import utils

from dataset import Dataset


# Load MNIST data
from tensorflow.examples.tutorials.mnist import input_data


# Define the discriminator network
def discriminator(images, reuse_variables=None):
    with tf.variable_scope(tf.get_variable_scope(), reuse=reuse_variables) as scope:
        # First convolutional and pool layers
        # This finds 32 different 5 x 5 pixel features
        d_w1 = tf.get_variable('d_w1', [5, 5, nb_channel, 32], initializer=tf.truncated_normal_initializer(stddev=0.02))
        d_b1 = tf.get_variable('d_b1', [32], initializer=tf.constant_initializer(0))
        print(images)
        print(d_w1)
        d1 = tf.nn.conv2d(input=images, filter=d_w1, strides=[1, 1, 1, 1], padding='SAME')
        d1 = d1 + d_b1
        d1 = tf.nn.relu(d1)
        d1 = tf.nn.avg_pool(d1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        # Second convolutional and pool layers
        # This finds 64 different 5 x 5 pixel features
        d_w2 = tf.get_variable('d_w2', [5, 5, 32, 64], initializer=tf.truncated_normal_initializer(stddev=0.02))
        d_b2 = tf.get_variable('d_b2', [64], initializer=tf.constant_initializer(0))
        d2 = tf.nn.conv2d(input=d1, filter=d_w2, strides=[1, 1, 1, 1], padding='SAME')
        d2 = d2 + d_b2
        d2 = tf.nn.relu(d2)
        d2 = tf.nn.avg_pool(d2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        # First fully connected layer
        shape = int(np.prod(d2.get_shape()[1:]))
        d_w3 = tf.get_variable('d_w3', [shape, 1024], initializer=tf.truncated_normal_initializer(stddev=0.02))
        d_b3 = tf.get_variable('d_b3', [1024], initializer=tf.constant_initializer(0))
        d3 = tf.reshape(d2, [-1, shape])
        d3 = tf.matmul(d3, d_w3)
        d3 = d3 + d_b3
        d3 = tf.nn.relu(d3)

        # Second fully connected layer
        d_w4 = tf.get_variable('d_w4', [1024, 1], initializer=tf.truncated_normal_initializer(stddev=0.02))
        d_b4 = tf.get_variable('d_b4', [1], initializer=tf.constant_initializer(0))
        d4 = tf.matmul(d3, d_w4) + d_b4

        # d4 contains unscaled values
        return d4

# Define the generator network
def generator(z, batch_size, z_dim):
    g_w1_w = 4*image_width*image_height*nb_channel
    g_w1 = tf.get_variable('g_w1', [z_dim, g_w1_w], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.02))
    g_b1 = tf.get_variable('g_b1', [g_w1_w], initializer=tf.truncated_normal_initializer(stddev=0.02))
    g1 = tf.matmul(z, g_w1) + g_b1
    g1 = tf.reshape(g1, [-1, 2*image_width, 2*image_width, nb_channel])
    g1 = tf.contrib.layers.batch_norm(g1, epsilon=1e-5, scope='bn1')
    g1 = tf.nn.relu(g1)

    # Generate 50 features
    g_w2 = tf.get_variable('g_w2', [3, 3, nb_channel, z_dim/2], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.02))
    g_b2 = tf.get_variable('g_b2', [z_dim/2], initializer=tf.truncated_normal_initializer(stddev=0.02))
    g2 = tf.nn.conv2d(g1, g_w2, strides=[1, 2, 2, 1], padding='SAME')
    g2 = g2 + g_b2
    g2 = tf.contrib.layers.batch_norm(g2, epsilon=1e-5, scope='bn2')
    g2 = tf.nn.relu(g2)
    g2 = tf.image.resize_images(g2, [2*image_width, 2*image_width])

    # Generate 25 features
    g_w3 = tf.get_variable('g_w3', [3, 3, z_dim/2, z_dim/4], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.02))
    g_b3 = tf.get_variable('g_b3', [z_dim/4], initializer=tf.truncated_normal_initializer(stddev=0.02))
    g3 = tf.nn.conv2d(g2, g_w3, strides=[1, 2, 2, 1], padding='SAME')
    g3 = g3 + g_b3
    g3 = tf.contrib.layers.batch_norm(g3, epsilon=1e-5, scope='bn3')
    g3 = tf.nn.relu(g3)
    g3 = tf.image.resize_images(g3, [2*image_width, 2*image_width])

    # Final convolution with one output channel
    g_w4 = tf.get_variable('g_w4', [1, 1, z_dim/4, nb_channel], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.02))
    g_b4 = tf.get_variable('g_b4', [1], initializer=tf.truncated_normal_initializer(stddev=0.02))
    g4 = tf.nn.conv2d(g3, g_w4, strides=[1, 2, 2, 1], padding='SAME')
    g4 = g4 + g_b4
    g4 = tf.sigmoid(g4)
    # g4 = tf.nn.tanh(g4)

    # Dimensions of g4: batch_size x 28 x 28 x 1
    return g4


def show_result(batch_res, fname, grid_size=(8, 8), img_size=(28,28), grid_pad=5, nb_channel=3):

    big_pic = np.ones(( grid_size[0]*(img_size[0]+grid_pad),
                        grid_size[1]*(img_size[1]+grid_pad),
                        nb_channel))
    for idx, image in enumerate(batch_res):
        i = (idx // grid_size[1])*(img_size[0]+ grid_pad)
        j = (idx % grid_size[1] )*(img_size[1] + grid_pad)
        big_pic[i:i+img_size[0], j:j+img_size[1]] = image
    big_pic_reshaped = big_pic.reshape([big_pic.shape[0], big_pic.shape[1],nb_channel])

    # plt.imsave(fname, big_pic_reshaped, cmap='Greys')
    imsave(fname, big_pic_reshaped)


# mnist = input_data.read_data_sets("MNIST_data/")
dataset = Dataset.get_images('../../datasets/images/cats/64x64_rgb/')
model_name = 'gan4cats'
z_dimensions = 100
batch_size = 50
image_height = 64
image_width = 64
nb_channel = 3
img_size = image_width*image_height*nb_channel

z_placeholder = tf.placeholder(tf.float32, [None, z_dimensions], name='z_placeholder')
# z_placeholder is for feeding input noise to the generator

x_placeholder = tf.placeholder(tf.float32, shape = [None,image_height,image_width, nb_channel], name='x_placeholder')
# x_placeholder is for feeding input images to the discriminator

Gz = generator(z_placeholder, batch_size, z_dimensions)
# Gz holds the generated images

Dx = discriminator(x_placeholder)
# Dx will hold discriminator prediction probabilities
# for the real MNIST images

Dg = discriminator(Gz, reuse_variables=True)
# Dg will hold discriminator prediction probabilities for generated images

# Define losses
d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = Dx, labels = tf.ones_like(Dx)))
d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = Dg, labels = tf.zeros_like(Dg)))
g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = Dg, labels = tf.ones_like(Dg)))

# Define variable lists
tvars = tf.trainable_variables()
d_vars = [var for var in tvars if 'd_' in var.name]
g_vars = [var for var in tvars if 'g_' in var.name]

# Define the optimizers
# Train the discriminator
d_trainer_fake = tf.train.AdamOptimizer(0.0003).minimize(d_loss_fake, var_list=d_vars)
d_trainer_real = tf.train.AdamOptimizer(0.0003).minimize(d_loss_real, var_list=d_vars)

# Train the generator
g_trainer = tf.train.AdamOptimizer(0.0001).minimize(g_loss, var_list=g_vars)

# From this point forward, reuse variables
tf.get_variable_scope().reuse_variables()
saver = tf.train.Saver()
sess = tf.Session()

# Send summary statistics to TensorBoard
tf.summary.scalar('Generator_loss', g_loss)
tf.summary.scalar('Discriminator_loss_real', d_loss_real)
tf.summary.scalar('Discriminator_loss_fake', d_loss_fake)

images_for_tensorboard = generator(z_placeholder, batch_size, z_dimensions)
tf.summary.image('Generated_images', images_for_tensorboard, 5)
merged = tf.summary.merge_all()
logdir = "tensorboard/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "/"
writer = tf.summary.FileWriter(logdir, sess.graph)

checkpoint_file = "./checkpoints/"+model_name+".ckpt"

sess.run(tf.global_variables_initializer())

# # Pre-train discriminator
# for i in range(300):
#     z_batch = np.random.normal(0, 1, size=[batch_size, z_dimensions])
#     # real_image_batch = mnist.train.next_batch(batch_size)[0].reshape([batch_size, 28, 28, 1])
#     real_image_batch = dataset.next_batch(batch_size).reshape([batch_size, image_height, image_width, nb_channel])
#     _, __, dLossReal, dLossFake = sess.run([d_trainer_real, d_trainer_fake, d_loss_real, d_loss_fake],
#                                            {x_placeholder: real_image_batch, z_placeholder: z_batch})
#     print('dLossReal: '+str(dLossReal)+'  ,  dLossFake: '+str(dLossFake))

# Train generator and discriminator together
for i in range(100000):
    if path.isfile(checkpoint_file):
        saver.restore(sess, checkpoint_file)
    # real_image_batch = mnist.train.next_batch(batch_size)[0].reshape([batch_size, 28, 28, 1])
    real_image_batch = dataset.next_batch(batch_size).reshape([batch_size, image_height, image_width, nb_channel])
    z_batch = np.random.normal(0, 1, size=[batch_size, z_dimensions])

    # Train discriminator on both real and fake images
    _, __, dLossReal, dLossFake = sess.run([d_trainer_real, d_trainer_fake, d_loss_real, d_loss_fake],
                                           {x_placeholder: real_image_batch, z_placeholder: z_batch})

    # Train generator
    z_batch = np.random.normal(0, 1, size=[batch_size, z_dimensions])
    _ = sess.run(g_trainer, feed_dict={z_placeholder: z_batch})
    print(i)
    if i % 10 == 0:
        # Update TensorBoard with summary statistics
        z_batch = np.random.normal(0, 1, size=[batch_size, z_dimensions])
        summary = sess.run(merged, {z_placeholder: z_batch, x_placeholder: real_image_batch})
        writer.add_summary(summary, i)


        z_batch = np.random.normal(0, 1, size=[64, z_dimensions])
        # images = sess.run(Gz, {z_placeholder: z_batch})
        images = sess.run(Gz, {z_placeholder: z_batch})
        show_result(images, "./output/step_%s.jpg" %i, img_size=(image_height, image_width), nb_channel=nb_channel)
        save_path = saver.save(sess, checkpoint_file)
    if i % 100 == 0:
        utils.save_on_s3(model_name, save_path)
