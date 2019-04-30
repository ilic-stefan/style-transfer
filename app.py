import numpy as np
from keras import backend
from keras.applications import vgg19
from keras.preprocessing.image import load_img, save_img, img_to_array
from scipy.optimize import fmin_l_bfgs_b
from matplotlib import pyplot as pp

# The local file paths to the content image and the style image
content_image_path = 'images\pics\gates_tower.jpg'
style_image_path = 'images\styles\starry_night.jpg'

# Set the size of the output
img_height = 512
img_width = 512


def display_to_vgg_image(img, isPath=True):
    """
    Given the path to an image, we load it and convert it for usage
    as input to the VGG-19 network
    Citation: machinelearningmastery.com/use-pre-trained-vgg-model-classify-objects-photographs/
    :param isPath: If the img parameter refers to a file path
    :param img: A path to an image file
    :return: An array representing the image after processing
    """
    if isPath:
        img = load_img(img, target_size=(img_height, img_width))

    # Convert to an array
    img = img_to_array(img)
    # Add extra dimensionality for batching in VGG-19
    img = np.expand_dims(img, axis=0)
    # Preprocess it for the VGG-19 network
    img = vgg19.preprocess_input(img)
    return img


def vgg_image_to_display(img):
    """
    Given a input image for the VGG-19 network, we convert it
    to an array representing an RGB image
    :param img: Preprocessed input image for VGG-19 network
    :return: An array representing an RGB image
    """
    # Take the added dimensionality out
    img = img.reshape((img_height, img_width, 3))

    # Add the "mean-pixel" back
    # Note: Don't know what this means
    img[:, :, 0] += 103.939
    img[:, :, 1] += 116.779
    img[:, :, 2] += 123.68

    img = img[:, :, ::-1]
    img = np.clip(img, 0, 255).astype('uint8')
    return img


# Shape: (1, Height, Width, Channels = 3)
content_tensor = backend.variable(display_to_vgg_image(content_image_path))
style_tensor = backend.variable(display_to_vgg_image(style_image_path))
recons_tensor = backend.placeholder((1, img_height, img_width, 3))

# Shape: (3, Height, Width, Channels = 3)
input_tensor = backend.concatenate([content_tensor, style_tensor, recons_tensor], axis=0)

model = vgg19.VGG19(input_tensor=input_tensor,
                    weights='imagenet', include_top=False)

layer_to_output = dict([(layer.name, layer.output) for layer in model.layers])

# Layer
desired_output = layer_to_output['block3_conv4']


def content_loss(base, recons):
    return backend.sum(backend.square(recons - base))


# Defining the Loss functions
loss = backend.variable(0.0)

# Content Loss
block5_conv2_representations = layer_to_output['block5_conv2']
features_from_content = block5_conv2_representations[0, :, :, :]
features_from_recons = block5_conv2_representations[2, :, :, :]

loss += content_loss(features_from_content, features_from_recons)

# Style Loss
# get a bunch of layer_to_output values
# compute the gram matrices and consequently the style loss

# Gradient and Loss calculation
grads = backend.gradients(loss, recons_tensor)

# Step Information contains the loss from the current iteration
# and the Gradients for the future iteration
step_information = [loss]
step_information += grads

tensor_to_loss = backend.function([recons_tensor], step_information)

our_reconstruction = display_to_vgg_image(content_image_path)
base_content_image = load_img(content_image_path, target_size=(img_height, img_width, 3))


hack_loss = None
hack_gradient = None

# recons --> 1xN vector where N = Image_Height * Image_Width * Channels (3)
def get_loss(recons):
    """

    :param recons:
    :return: A scalar loss value
    """
    recons = recons.reshape(1, img_height, img_width, 3)
    g_loss, g_grad_vals = tensor_to_loss([recons])

    # We are assigning the global variable
    global hack_loss
    global hack_gradient

    # Set the loss and gradient globally so the
    # fake get_gradient function can utilize it
    hack_loss = g_loss
    hack_gradient = g_grad_vals[:, 0, 0, 0].flatten().astype('float64')

    return hack_loss


# recons --> 1xN vector where N = Image_Height * Image_Width * Channels (3)
def get_gradients(recons):
    """

    :param recons:
    :return: A n-Dim array of Gradient values
    """
    return hack_gradient
    #recons = recons.reshape(1, img_height, img_width, 3)
    #g_loss_val, g_grad_vals = tensor_to_loss([recons])
    #return g_grad_vals[0, :, :, :].flatten().astype('float64')


# Initialize our random initial guess
our_guess = np.random.random_integers(0, 255, (img_height, img_width, 3)) - 128
our_guess = display_to_vgg_image(our_guess, False).astype('float64')

# Run the Optimization Algorithm
history = []
iterations = 10
for i in range(iterations):
    print("\nStarting iteration " + str(i+1))
    our_guess, loss, info_dict = fmin_l_bfgs_b(func=get_loss,
                      x0=our_guess.flatten(),
                      fprime=get_gradients, maxfun=20)
    history.append([loss, vgg_image_to_display(our_guess)])
    print("\nAfter " + str(i) + " iterations, the loss is: " + str(loss))


# Show the final output image
output_image = vgg_image_to_display(our_guess)
pp.figure()
pp.imshow(output_image)
pp.show()

# Plot the change in loss

"""
pp.figure()
pp.plot(loss_history)
pp.title("Loss over Time")
pp.xlabel("Iteration #")
pp.ylabel("Loss Value")
pp.show()
"""