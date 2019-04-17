import os
import pandas as pd
import glob
import random
from sklearn.model_selection import train_test_split
import shutil

def moveFiles(trainData=None, testData=None, limit_files=config.FILE_LIMIT, classes = ['normal', 'insert', 'compressed', 'black', 'blurred']):

    path_to_all = os.path.join('data', 'UCF-101')

    if trainData is None or testData is None:
        # load files names
        filenames = glob.glob(os.path.join(path_to_all,'**', '*.avi'), recursive=True)

        fileclasses = [random.choice(classes[1:]) for f in filenames]
        fileclasses[0:10]=['normal']*10
        

        df = pd.DataFrame({'filenames': filenames, 'classes': fileclasses})
        df = df.sample(frac=1)
        if limit_files:
            df = df[:limit_files]

        
        train = df.reset_index().groupby('classes').apply(lambda x: x.sample(frac=0.8)).reset_index(drop=True).set_index('index')                  
        test = df.drop(train.index)
    else:
        filenames = glob.glob(os.path.join(path_to_all,'**', '*.avi'), recursive=True)
        train = trainData
        test = testData

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