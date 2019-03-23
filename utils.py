import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
import config


from IPython.display import clear_output


def displayImage(img_arr):
    plt.imshow(img_arr)




def displayImageLoop(imgs, delay=0.1, size=8):
    i = 0
    for img in imgs:
        
        clear_output(wait=True)
        fig = plt.figure(1, figsize=(size,size)); plt.clf()
        plt.title('Frame ' + str(i))
        plt.imshow(img)
        
        plt.pause(delay)
        i+=1