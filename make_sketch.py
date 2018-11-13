import cv2
import matplotlib.pyplot as plt
import numpy as np

def dodgeV2(image, mask):
  return cv2.divide(image, 255-mask, scale=256)

def make_sketch(filename):
    """
    Create sketch for the RGB image
    """
    img = cv2.imread(filename)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gray_inv = 255 - img_gray
    img_blur = cv2.GaussianBlur(img_gray_inv, ksize=(21,21), sigmaX=0, sigmaY=0)
    img_blend = dodgeV2(img_gray, img_blur)
    return img_blend


def format_sketch(filename, size):
    sketch = make_sketch(filename)
#    sketch = sketch[20:220, 20:220]
    sketch = cv2.resize(sketch, (size, size), interpolation=cv2.INTER_AREA)
    sketch = sketch.reshape((1, size, size, 1)).transpose(0, 3, 1, 2)
    return sketch

def main_dy():
    filename = "../data/sketch_hand/155.jpg"
#    filename = "../data/output-3/watch_256.jpg"
    size = 200
    sketch = format_sketch(filename, size)
    
    sketch = sketch.transpose((2, 3, 0, 1))
    sketch = sketch.reshape( size, size)
    plt.imshow(sketch, cmap ='gray')
    plt.axis('off')
    plt.savefig("../figures/sketch_dy/sketch-155.jpg")
    
#    return sketch

if __name__ == '__main__':

    main_dy()
