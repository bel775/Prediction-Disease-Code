
from tensorflow.keras.preprocessing import image
import numpy as np
import glob

def load_x_ray_data(img_size):
  name_classes = ['NORMAL','PNEUMONIA']
  X,Y  = [], []
  for class_number, class_name in enumerate(name_classes):    # Number of directories
    for filename in glob.glob(f'./chest_xray_512/{class_name}/*.jpg'):
      im = image.load_img(filename, target_size=[img_size, img_size], color_mode = 'grayscale')
      X.append(image.img_to_array(im))
      num_class = class_number
      if(class_number == 1):
        if 'virus' in filename:
          num_class += 1
      Y.append(num_class)

  #input_shape = (img_size, img_size, 1)
  X = np.array(X)
  Y = np.array(Y)
  return X, Y

def load_Brain_Tumor_data(img_size):
  name_classes = ['glioma','meningioma','notumor','pituitary']
  X,Y  = [], []
  for class_number, class_name in enumerate(name_classes):    # Number of directories
    for filename in glob.glob(f'./BrainTumorMRI/{class_name}/*.jpg'):
      im = image.load_img(filename, target_size=[img_size, img_size], color_mode = 'grayscale')
      X.append(image.img_to_array(im))
      Y.append(class_number)

  #input_shape = (img_rows, img_cols, 1)
  X = np.array(X)
  Y = np.array(Y)

  return X, Y