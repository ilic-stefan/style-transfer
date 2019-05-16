import tensorflow as tf
import numpy as np
import cv2
import math

# import vgg 19
from tfvgg19 import vgg19, utils

# Load Content Image (input) #
# print that image loading has become
print("Loading Content Image...")

content_image_path = "/Users/Stefan/Desktop/sanfran.jpg"

content_img = cv2.imread(content_image_path)

# ensure that content image exists
assert content_img is not None

# set size parameters
height = 224
width = 224

# cut the content image to correct size (height by width), preserving channels
# calculate the first height index of content image window
IMHeight = (content_img.shape[0] - height) / 2
# cast as int
IMHeight = math.floor(IMHeight)

# calculate the first width index of content image window
IMWidth = (content_img.shape[1] - height) / 2
# cast as int
IMWidth = math.floor(IMWidth)

# resize content image to a height by width by 3 image
content_img = content_img[IMHeight:IMHeight+height, IMWidth:IMWidth+width, :]

# convert to doubles from 0 to 1
content_img = np.divide(content_img, 255.0)

# expand the dimensions of content-image to match later input requirement for VGG
content_img = np.expand_dims(content_img, axis=0)

# Build Computation Graph #
# construct content image tensor (placeholder)
# (note: pixel values assumed to be type uint8)
c_img = tf.placeholder(tf.float32, (1, height, width, 3))

vgg = vgg19.Vgg19()
with tf.name_scope("content_with_vgg"):
    vgg.build2(c_img)

    layer_activations = vgg.conv4_2

# Evaluate Computation Graph #
# print that computation graph is being evaluated
print("construction done.")
print("computation graph evaluating...")

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())  # initialize all variables

    layer_img = sess.run(layer_activations, feed_dict={c_img: content_img})
    # Note: vgg.conv4_2, after c_img is sent through, is a numpy array
    print(layer_img.shape)

print("done")
