import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
import config
import pandas as pd
import pandas_ml as pml
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


def predictionsToDataFrame(model, x, y, dataset):
    predictions = model.predict_classes(x)
    predictions = [dataset.classes[k] for k in predictions]
    truth = [dataset.reverse_one_hot(l) for l in y]
    

    df = pd.DataFrame({'truth': truth, 'prediction': predictions})
    return df

def confusion_matrix(truth, pred):
    conf = pml.ConfusionMatrix(truth, pred)
    return conf

def display_confusion(conf):
    conf.plot()
    plt.show()
