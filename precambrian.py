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

# convert to doubles from 0 to 1
content_img = np.divide(content_img, 255.0)

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
                           dtype=tf.float32, initializer=tf.random_uniform_initializer(0, 1))

    # $$$ testing out backpropagation...
    #dimg2 = tf.get_variable("test-variable-to-be-deleted", [height, width, 3],
                           #dtype=tf.float32, initializer=tf.random_uniform_initializer(0, 1))

    #tf.assign(dimg2, dimg)

    # use a mean square error between dimg and cimg for loss function
    loss = tf.reduce_sum(tf.subtract(dimg, cimg) ** 2) / (height*width)

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

    iteration_count = 4000 # the number of gradient descent steps we will take

    for _ in range(iteration_count):
        # do a gradient descent step
        _, loss_val = sess.run([opt_operation, loss], feed_dict={cimg: content_img})

        loss_history_lst.append(loss_val) # append the current loss_val tensor to the list

    loss_history_stack = tf.stack(loss_history_lst) # make a tensor stack out of the list
    # evaluate loss_history stack
    out_loss_history = sess.run(loss_history_stack)
    # evaluate developing image to get output image
    outImg = sess.run(dimg)

    print("evaluation done.")
    print("preparing results report...")

    # Get max and min statistics about outimg
    # note: should be between 0 and 1
    print("outImg max:")
    print(sess.run(tf.reduce_max(outImg)))
    print("outImg min:")
    print(sess.run(tf.reduce_min(outImg)))

# plot the loss history
iteration_history = np.arange(iteration_count)
plt.scatter(iteration_history, out_loss_history)
plt.title("loss during optimization")
plt.show()

# Display Resulting Image #
cv2.imshow("original image", content_img)
cv2.imshow("reconstructed image", outImg)
cv2.waitKey(0)

print("results done.")
