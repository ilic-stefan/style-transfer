import app
from matplotlib import pyplot as pp

"""
    Testing out pre-processing and de-processing images
"""
# The image path
vgg_image_path = 'images/pics/chicago.jpg'
# Process and load in the image
processed_img = app.display_to_vgg_image(vgg_image_path)
# The shape should be: (1, X, Y, 3)
print(processed_img.shape)
# It should look weird
pp.figure()
pp.imshow(processed_img[0, :, :, :])
pp.show()
# Deprocess it
deprocessed_img = app.vgg_image_to_display(processed_img)
# The shape should be: (X, Y, 3)
print(deprocessed_img.shape)
# It should look normal
pp.figure()
pp.imshow(deprocessed_img)
pp.show()