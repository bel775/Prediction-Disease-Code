
from tensorflow.keras.preprocessing import image
import numpy as np
import glob
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical

####################################################
# LOAD Chaest X-Ray Data
####################################################
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

  X = np.array(X)
  Y = np.array(Y)

  return X, Y


####################################################
# LOAD Brain Tumor RMI DATA
####################################################
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

####################################################
# LOAD ISIC Data
####################################################
def load_isic_data_2(img_size):
  data = pd.read_csv('MERGED_ISIC_DATA_2019_2020.csv')
  X,Y  = [], []
  X_Unbalanced,Y_Unbalanced  = [], []
  for idx, row in data.iterrows():
    filename = f'./ISIC_Filtred/{row["image_name"]}.jpg'
    im = image.load_img(filename, target_size=[img_size, img_size], color_mode = 'grayscale')

    num = 0
    if row['diagnosis'] == 'nevus' :
      X.append(image.img_to_array(im))
      Y.append(0)
    elif row['diagnosis'] == 'melanoma':
      X.append(image.img_to_array(im))
      Y.append(1)
    else:
      X_Unbalanced.append(image.img_to_array(im))
      Y_Unbalanced.append(2)

  #input_shape = (img_rows, img_cols, 1)
  X = np.array(X)
  Y = np.array(Y)

  X_Unbalanced = np.array(X_Unbalanced)
  Y_Unbalanced = np.array(Y_Unbalanced)

  return X, Y,X_Unbalanced, Y_Unbalanced #, input_shape


def load_isic_data(img_size):
  data = pd.read_csv('MERGED_ISIC_DATA_2019_2020.csv')
  X,Y  = [], []
  X_Unbalanced,Y_Unbalanced  = [], []
  for idx, row in data.iterrows():
    filename = f'./ISIC_Filtred/{row["image_name"]}.jpg'
    im = image.load_img(filename, target_size=[img_size, img_size], color_mode = 'grayscale')

    num = 0
    if row['diagnosis'] == 'nevus' :
      X.append(image.img_to_array(im))
      Y.append(0)
    elif row['diagnosis'] == 'melanoma':
      X.append(image.img_to_array(im))
      Y.append(1)
    else:
      X_Unbalanced.append(image.img_to_array(im))
      Y_Unbalanced.append(2)

  #input_shape = (img_rows, img_cols, 1)
  X = np.array(X)
  Y = np.array(Y)

  X_Unbalanced = np.array(X_Unbalanced)
  Y_Unbalanced = np.array(Y_Unbalanced)

  return X, Y,X_Unbalanced, Y_Unbalanced #, input_shape