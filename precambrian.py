import tensorflow as tf
import numpy as np
import cv2
from matplotlib import pyplot as plt
import math


# Load Content Image (input) #
# print that image loading has become
print("Loading Content Image...")

content_image_path = "/Users/Stefan/Desktop/sanfran.jpg"

content_img = cv2.imread(content_image_path)

# ensure that content image exists
assert content_img is not None

# set size parameters
height = 512
width = 512

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

# show the smaller content image with openCV
# cv2.imshow("image", content_img)
# cv2.waitKey(0)
#
# print(content_img.shape)

# Build Computation Graph #

# print that computation graph is being constructed
print("loading done.")
print("computation graph constructing...")

# construct content image tensor (placeholder)
# (note: pixel values assumed to be type uint8)
cimg = tf.placeholder(tf.float32, (height, width, 3))

# construct developing image tensor (variable)
with tf.variable_scope("core"):
    dimg = tf.get_variable("developing-img", [height, width, 3],
                           dtype=tf.float32, initializer=tf.random_normal_initializer)

    # use a mean square error between dimg and cimg for loss function
    loss = tf.reduce_sum(tf.subtract(dimg, cimg)) ** 2 / (height*width)

# specify optimizer
opt = tf.train.AdamOptimizer()

# add optimization node to computation graph
opt_operation = opt.minimize(loss)

# Evaluate Computation Graph #
# print that computation graph is being evaluated
print("construction done.")
print("computation graph evaluating...")

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())  # initialize all variables

    # keep track of the loss values for every iteration
    loss_history_lst = [] # make a list to hold all loss_val tensors

    iteration_count = 10000 # the number of gradient descent steps we will take

    for _ in range(iteration_count):
        # do a gradient descent step
        _, loss_val = sess.run([opt_operation, loss], feed_dict={cimg: content_img})

        loss_history_lst.append(loss_val) # append the current loss_val tensor to the list

    loss_history_stack = tf.stack(loss_history_lst) # make a tensor stack out of the list
    # evaluate loss_history stack
    out_loss_history = sess.run(loss_history_stack)
    # evaluate developing image to get output image
    outimg = sess.run(dimg)

# plot the loss history
# iteration_history = np.arange(iteration_count)
# plt.scatter(iteration_history, out_loss_history)
# plt.title("loss during optimization")
# plt.show()

# Display Resulting Image #
cv2.imshow("reconstructed image", outimg)
cv2.waitKey(0)
