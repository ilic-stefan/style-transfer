import tensorflow as tf
import numpy as np
import cv2
import math
from matplotlib import pyplot as plt
import time
# import vgg 19
from tfvgg19 import vgg19, utils

# take start time for time measurement
start_time = time.time()

# Load Content Image Data ############################################################################
# read the content image which will be fed into comp. graph
print("Retrieving Content Image...")

content_image_path = "/Users/Stefan/Desktop/sanfran.jpg"
print("Using Image at: " + content_image_path)

# read image
content_img = cv2.imread(content_image_path)

# ensure that content image exists
assert content_img is not None

# set size parameters (224, 224, 3) for VGG
height = 224
width = 224

# cut the content image to correct size (height by width), preserving channels
# calculate the first height index of content image window
IMHeight = (content_img.shape[0] - height) / 2
# cast as int to work as an array subscript
IMHeight = math.floor(IMHeight)

# calculate the first width index of content image window
IMWidth = (content_img.shape[1] - height) / 2
# cast as int
IMWidth = math.floor(IMWidth)

# resize content image to a height by width by 3 image
content_img = content_img[IMHeight:IMHeight+height, IMWidth:IMWidth+width, :]

# convert to doubles from 0 to 1
content_img = np.divide(content_img, 255.0)

# Preserve a copy of content image before expanding dimensions
# shape should be (height, width, 3)
content_img_original = content_img

# expand the dimensions of content-image to match later input requirement for VGG
content_img = np.expand_dims(content_img, axis=0)

print("Finished Reading Content Image.")
# Build the Computation Graph #############################################################################
print("Building Computation Graph...")

with tf.variable_scope("main_structure"):
    with tf.variable_scope("content_structure"):
        # construct VGG-19 Model for handling content image
        vgg = vgg19.Vgg19()

        # construct content image tensor (placeholder)
        # (note: pixel values assumed to be type uint8)
        cimg = tf.placeholder(tf.float32, (1, height, width, 3))

        # put cimg through vgg
        vgg.build2(cimg, numpy=True)

        # get feature activations at layer conv4_2
        content_activ = vgg.conv4_2

    with tf.variable_scope("developing_structure"):
        # construct VGG-19 Model for handling development image
        vgg = vgg19.Vgg19()

        # construct developing image tensor (variable, initially uniformly random)
        dimg = tf.get_variable("developing-img", [1, height, width, 3],
                               dtype=tf.float32, initializer=tf.random_uniform_initializer(0, 1))

        # put dimg through vgg
        vgg.build2(dimg, numpy=False)

        # get feature activations at layer conv4_2
        dev_activ = vgg.conv4_2

    # compute loss
    # use a mean square error between developing and content activations
    # as defined by Gatys et al. 2016 (Equation 1)
    loss = tf.reduce_sum(tf.subtract(dev_activ, content_activ) ** 2) / 2

    # specify optimizer method
    opt = tf.train.AdamOptimizer()

    # add optimization node to computation graph
    opt_operation = opt.minimize(loss)

print("Finished Building Computation Graph.")
# Evaluate the Computation Graph #############################################################################
print("Evaluating Computation Graph...")
with tf.Session() as sess:
    # initialize
    sess.run(tf.global_variables_initializer())

    # set iteration count
    iteration_count = 1000  # the number of gradient descent steps we will take

    # structure to keep track of the loss values for every iteration
    loss_history_lst = [] # make a list to hold all loss_val tensors

    # run loss minimization
    for _ in range(iteration_count):
        # do a gradient descent step
        _, loss_val = sess.run([opt_operation, loss], feed_dict={cimg: content_img})

        loss_history_lst.append(loss_val)  # append the current loss_val tensor to the list

    # convert loss list to loss tensor stack and evaluate to get numerical loss values
    loss_history_stack = tf.stack(loss_history_lst)  # make a tensor stack out of the list
    out_loss_history = sess.run(loss_history_stack)

    # evaluate developing image to get output image
    outImg = sess.run(dimg)

print("Finished Evaluating Computation Graph.")

# print time elapsed
end_time = time.time()
print("Total Time Elapsed: ", (end_time - start_time))

# Make Results ##############################################################################
print("Preparing Results.")

# Print min/max val statistics about output image
# note: should be between 0 and 1
print("outImg max:")
print(np.amax(outImg))
print("outImg min:")
print(np.amin(outImg))

# plot the loss history
iteration_history = np.arange(iteration_count)
plt.scatter(iteration_history, out_loss_history)
plt.title("loss during optimization")
plt.show()

# Display Resulting Image #
assert content_img_original.shape == (height, width, 3)
cv2.imshow("original image", content_img_original)

# convert outImg to right dimensions to be viewed
outImg = np.squeeze(outImg)
cv2.imshow("reconstructed image", outImg)
cv2.waitKey(0)

print("results done.")
