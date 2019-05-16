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

# set size parameters for all images
# (224, 224, 3) for VGG
height = 224
width = 224

# Load Content Image Data ############################################################################
# read the content image which will be fed into comp. graph
print("Retrieving Content Image...")

content_image_path = "/Users/Stefan/Desktop/sanfran.jpg"
print("Using Image at: " + content_image_path)

# read image
content_img = cv2.imread(content_image_path)

# ensure that content image exists
assert content_img is not None

# cut the content image to correct size (height by width), preserving channels
# calculate the first height index of content image window
cont_IMHeight = (content_img.shape[0] - height) / 2
# cast as int to work as an array subscript
cont_IMHeight = math.floor(cont_IMHeight)

# calculate the first width index of content image window
cont_IMWidth = (content_img.shape[1] - height) / 2
# cast as int
cont_IMWidth = math.floor(cont_IMWidth)

# resize content image to a height by width by 3 image
content_img = content_img[cont_IMHeight:cont_IMHeight + height, cont_IMWidth:cont_IMWidth + width, :]

# convert to doubles from 0 to 1
content_img = np.divide(content_img, 255.0)

# Preserve a copy of content image before expanding dimensions
# shape should be (height, width, 3)
content_img_original = content_img

# expand the dimensions of content-image to match later input requirement for VGG
content_img = np.expand_dims(content_img, axis=0)

print("Finished Reading Content Image.")

# Load Style Image Data ############################################################################
# read the style image which will be fed into comp. graph
print("Retrieving Style Image...")

style_image_path = "/Users/Stefan/Desktop/grinnell_in_fall.jpg"
print("Using Image at: " + style_image_path)

# read image
style_img = cv2.imread(style_image_path)

# ensure that style image exists
assert style_img is not None

# cut the style image to correct size (height by width), preserving channels
# calculate the first height index of content image window
style_IMHeight = (style_img.shape[0] - height) / 2
# cast as int to work as an array subscript
style_IMHeight = math.floor(style_IMHeight)

# calculate the first width index of style image window
style_IMWidth = (style_img.shape[1] - height) / 2
# cast as int
style_IMWidth = math.floor(style_IMWidth)

# resize style image to a height by width by 3 image
style_img = style_img[style_IMHeight:style_IMHeight + height, style_IMWidth:style_IMWidth + width, :]

# convert to doubles from 0 to 1
style_img = np.divide(style_img, 255.0)

# Preserve a copy of style image before expanding dimensions
# shape should be (height, width, 3)
style_img_original = style_img

# expand the dimensions of style-image to match later input requirement for VGG
style_img = np.expand_dims(style_img, axis=0)

print("Finished Reading Style Image.")
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

        # ACTIVATIONS FOR FOR CONTENT LOSS
        # get feature activations at layer conv4_2 (for content loss calculation)
        content_activ = vgg.conv4_2

    with tf.variable_scope("style_structure"):
        # construct VGG-19 Model for handling content image
        vgg = vgg19.Vgg19()

        # construct content image tensor (placeholder)
        # (note: pixel values assumed to be type uint8)
        simg = tf.placeholder(tf.float32, (1, height, width, 3))

        # put simg through vgg
        vgg.build2(simg, numpy=True)

        # GRAM MATRIX FOR STYLE LOSS
        # specify layers used and the total number of layers used
        style_layer_names = [vgg.conv1_1, vgg.conv2_1, vgg.conv3_1, vgg.conv4_1, vgg.conv5_1]
        layer_num = 5

        # preallocate for collection of gram matrices
        # with corresponding size of feature maps (M) and number of feature maps (N)
        sty_GM_lst = []

        # compute development gram matrix for all layers in turn
        for layer_name in style_layer_names: # for every layer
            # Compute attributes
            H = layer_name.get_shape().as_list()[1] # height of any feature map
            W = layer_name.get_shape().as_list()[2] # width of any feature map
            N = layer_name.get_shape().as_list()[3] # the depth, i.e. number of feature maps, N to match Gatys et al. paper
            M = H * W # the size of any feature map

            # reshape the feature maps into 1-D vectors
            layer_name_vec = tf.reshape(layer_name, (1, M, N))
            layer_name_vec_transpose = tf.transpose(layer_name_vec, perm=[0, 2, 1])

            # calculate gram matrix for this layer
            styGramMat = tf.matmul(layer_name_vec_transpose, layer_name_vec)
            # add to the collection
            sty_GM_lst.append([styGramMat, M, N])

    with tf.variable_scope("developing_structure"):
        # construct VGG-19 Model for handling development image
        vgg = vgg19.Vgg19()

        # construct developing image tensor (variable, initially uniformly random)
        dimg = tf.get_variable("developing-img", [1, height, width, 3],
                               dtype=tf.float32, initializer=tf.random_uniform_initializer(0, 1))

        # put dimg through vgg
        vgg.build2(dimg, numpy=False)

        # ACTIVATIONS FOR CONTENT LOSS
        # get feature activations at layer conv4_2 (for content loss calculation)
        dev_activ = vgg.conv4_2

        # GRAM MATRIX FOR STYLE LOSS
        # Layers used specified in style_layer_names in style_structure scope

        # preallocate for collection of gram matrices
        # with corresponding size of feature maps (M) and number of feature maps (N)
        dev_GM_lst = []

        # compute development gram matrix for all layers in turn
        for layer_name in style_layer_names: # for every layer...
            # Compute attributes
            H = layer_name.get_shape().as_list()[1] # height of any feature map
            W = layer_name.get_shape().as_list()[2] # width of any feature map
            N = layer_name.get_shape().as_list()[3] # the depth, i.e. number of feature maps, N to match Gatys paper
            M = H * W # the size of any feature map

            # reshape the feature maps into 1-D vectors
            layer_name_vec = tf.reshape(layer_name, (1, M, N))
            layer_name_vec_transpose = tf.transpose(layer_name_vec, perm=[0, 2, 1])

            # calculate gram matrix for this layer
            devGramMat = tf.matmul(layer_name_vec_transpose, layer_name_vec)
            # add to the collection
            dev_GM_lst.append([devGramMat, M, N])

    # COMPUTE CONTENT LOSS
    # use a mean square error between developing and content activations
    # as defined by Gatys et al. 2016 (Equation 1)
    content_loss = tf.reduce_sum(tf.subtract(dev_activ, content_activ) ** 2) / 2

    # COMPUTE STYLE LOSS
    # list to hold style loss for each layer
    itrm_style_losses = []
    # compute style loss for every individual layer
    for i in range(layer_num):
        layer_sty_loss = tf.reduce_sum(tf.subtract(dev_GM_lst[i][0], sty_GM_lst[i][0]) ** 2)
        layer_sty_loss = layer_sty_loss / (4 * dev_GM_lst[i][1] ** 2 * dev_GM_lst[i][2] ** 2)
        itrm_style_losses += layer_sty_loss

    # specify weights for loss value at each layer
    weights = tf.constant([1/5, 1/5, 1/5, 1/5])

    # compute final style loss
    style_loss = tf.tensordot(weights, itrm_style_losses, axes=0)

    # COMPUTE TOTAL LOSS
    # specify content vs style weighting (alpha/beta ratio of 1x10^-3)
    alpha = 1
    beta = 0.001
    loss = alpha * content_loss + beta * style_loss

    # specify optimizer method
    opt = tf.train.AdamOptimizer()

    # add optimization node to computation graph
    opt_operation = opt.minimize(content_loss)

print("Finished Building Computation Graph.")
# Evaluate the Computation Graph #############################################################################
print("Evaluating Computation Graph...")
with tf.Session() as sess:
    # initialize
    sess.run(tf.global_variables_initializer())

    # set iteration count
    iteration_count = 50  # the number of gradient descent steps we will take

    # structure to keep track of the loss values for every iteration
    loss_history_lst = [] # make a list to hold all loss_val tensors

    # run loss minimization
    for _ in range(iteration_count):
        # do a gradient descent step
        _, loss_val = sess.run([opt_operation, content_loss], feed_dict={cimg: content_img})

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
cv2.imshow("content image", content_img_original)
cv2.imshow("style image", style_img_original)

# convert outImg to right dimensions to be viewed
outImg = np.squeeze(outImg)
cv2.imshow("reconstructed image", outImg)
cv2.waitKey(0)

print("results done.")
