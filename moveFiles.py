import os
import pandas as pd
import glob
import random
from sklearn.model_selection import train_test_split
import shutil

def moveFiles(limit_files=None):

    path_to_all = os.path.join('data', 'UCF-101')

    # load files names
    filenames = glob.glob(os.path.join(path_to_all,'**', '*.avi'), recursive=True)


    classes = ['normal', 'insert', 'dropped', 'compressed', 'black']
    fileclasses = [random.choice(classes) for f in filenames]

    df = pd.DataFrame({'filenames': filenames, 'classes': fileclasses})
    df = df.sample(frac=1)
    if limit_files:
        df = df[:limit_files]

    
    train = df.reset_index().groupby('classes').apply(lambda x: x.sample(frac=0.8)).reset_index(drop=True).set_index('index')                  
    test = df.drop(train.index)

    trainPath = os.path.join('data', 'videos', 'train')
    testPath = os.path.join('data', 'videos', 'test')

    # Extract images
    if (os.path.isdir(trainPath)):
        shutil.rmtree(trainPath)
    os.makedirs(trainPath)

    if (os.path.isdir(testPath)):
        shutil.rmtree(testPath)
    os.makedirs(testPath)
            
    for index, row in train.iterrows():
        f = row.filenames
        videoName = f.split(os.path.sep)[-1]
        c = row.classes
        p = os.path.join(trainPath, c)
        os.makedirs(p, exist_ok = True) #if it exits do nothing
        outPath = os.path.join(p, videoName)
        shutil.copyfile(f, outPath) # copy from all to the correct folder

    for index, row in test.iterrows():
        f = row.filenames
        videoName = f.split(os.path.sep)[-1]
        c = row.classes
        p = os.path.join(testPath, c)
        os.makedirs(p, exist_ok = True) #if it exits do nothing
        outPath = os.path.join(p, videoName)
        shutil.copyfile(f, outPath) # copy from all to the correct folder

    return train, test, filenames