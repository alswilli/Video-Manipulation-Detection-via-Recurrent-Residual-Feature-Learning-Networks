import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
import config
import pandas as pd
import pandas_ml as pml
from IPython.display import clear_output
from keras.callbacks import Callback
from sklearn.metrics import accuracy_score

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

class NewAccuracy(Callback):
    
    def __init__(self, x_val,y_val, dataset):
        super().__init__()
        self.x_val = x_val
        self.y_val = y_val
        self.dataset = dataset
        
    
    def on_train_begin(self, logs={}):
        self.accs = []
        
    def on_epoch_end(self, epoch, logs={}):
        true_all = []
        pred_all = []
        preds = self.model.predict(self.x_val)
        for i in range(len(preds)):
            sample = preds[i]
            args = [self.dataset.classes[p.argmax()] for p in sample]
            pred_all.extend(args)
            true_all.extend([self.dataset.reverse_one_hot(k) for k in self.y_val[i]])
        
        df = pd.DataFrame({'truth': true_all, 'pred': pred_all})
        df = df[df.truth != 'normal']
        
        acc = accuracy_score(df.truth, df.pred)
        print('Non-Normal Accuracy: {0:.4f}'.format(acc))
        
        self.accs.append(acc)
        
        return


def nonNormalAccuracy(x_val,y_val,dataset, model, batch_size=20):
        true_all = []
        pred_all = []
        curr = 0
        length = len(y_val)
        while curr < length:  
                end =  min(length, curr+batch_size)
                preds = model.predict(x_val[curr:end])
                for i in range(curr, end):
                        sample = preds[i]
                        args = [dataset.classes[p.argmax()] for p in sample]
                        pred_all.extend(args)
                        true_all.extend([dataset.reverse_one_hot(k) for k in y_val[i]])
                curr += batch_size

        df = pd.DataFrame({'truth': true_all, 'pred': pred_all})
        df = df[df.truth != 'normal']

        acc = accuracy_score(df.truth, df.pred)
        # print('Non-Normal Accuracy: {0:.4f}'.format(acc))
        return acc
