import skimage as sk 
import numpy as np
from skimage.filters import gaussian




def generate_small_blobs(length = 64, blob_size_fraction = 0.1,
                   n_dim = 2,
                   volume_fraction = 0.2, seed = None):
  rs = np.random.default_rng(seed)
  shape = tuple([length] * n_dim)
  mask = np.zeros(shape)
  n_pts = max(int(1. / blob_size_fraction) ** n_dim, 1)
  points = (length * rs.random((n_dim, n_pts))).astype(int)
  mask[tuple(indices for indices in points[:,:])] = 1

  mask = gaussian(mask, sigma=0.25 * length * blob_size_fraction,
                  preserve_range=False)

  threshold = np.percentile(mask, 100 * (1 - volume_fraction))
  return np.logical_not(mask < threshold)

def generate_circles_and_ellipse(ellipse = True, num_blobs = 10, img_size = 64, maj_axis=15, min_axis=5):
  img = np.zeros((img_size, img_size))
  r = np.ceil(np.sqrt(maj_axis*min_axis))
  posx = np.random.randint(r, img_size-r)
  posy = np.random.randint(r, img_size-r)
  if ellipse:
    rr, cc = sk.draw.ellipse(posy, posx, maj_axis, min_axis, shape=(img_size, img_size), rotation=np.random.randint(-15, 15)/10)
    img[rr,cc] = 1
    num_blobs-=1
  for i in range(num_blobs):
      #print(posx-r, posx+r, posy-r, posy+r)
      while 1 in img[int(posy-r):int(posy+r), int(posx-r):int(posx+r)]:
        #print(img[int(posx-r):int(posx+r), int(posy-r):int(posy+r)])
        posx = np.random.randint(r, img_size-r)
        posy = np.random.randint(r, img_size-r)
      rr, cc = sk.draw.ellipse(posy, posx, r, r, shape=(img_size, img_size))
      img[rr,cc] = 1
  # while 1 in img[int(posy-maj_axis):int(posy+maj_axis), int(posx-maj_axis):int(posx+maj_axis)]:    
  #   posx = np.random.randint(maj_axis, img_size-maj_axis)
  #   posy = np.random.randint(maj_axis, img_size-maj_axis)
  return img

def generate_new_blob_img(ellipse = True, size = 64, maj_axis = 10, min_axis = 3, num_big_blobs = 5):
    r = np.floor(np.sqrt(maj_axis*min_axis))
    x = generate_circles_and_ellipse(ellipse, num_big_blobs, size, maj_axis, min_axis)
    y = generate_small_blobs(size, blob_size_fraction = 0.06,
                      n_dim = 2, volume_fraction = 0.2)

    return np.logical_or(x,y)