import os
import pandas as pd
import glob
import random
from sklearn.model_selection import train_test_split
import shutil
import config

def moveFiles(trainData=None, testData=None, limit_files=config.FILE_LIMIT, classes = ['normal', 'insert', 'compressed', 'black', 'blurred']):

    path_to_all = os.path.join('data', 'UCF-101')

    if trainData is None or testData is None:
        # load files names
        filenames = glob.glob(os.path.join(path_to_all,'**', '*.avi'), recursive=True)
        fileclasses = [random.choice(classes[1:]) for f in filenames]
        
        df = pd.DataFrame({'filenames': filenames, 'classes': fileclasses})

        if limit_files:
            df = df.sample(limit_files)
        else:
            df = df.sample(frac=1)


        train = df.reset_index().groupby('classes').apply(lambda x: x.sample(frac=0.8)).reset_index(drop=True).set_index('index')                  
        test = df.drop(train.index)
        train.classes.iloc[0]='normal'
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


def randomizeNumpy():
    npz_path = os.path.join('data', 'sequences', 'npz', 'Default')
    train_files = glob.glob(os.path.join(npz_path, 'train', '*.npz'))
    test_files = glob.glob(os.path.join(npz_path, 'test', '*.npz'))

    random.shuffle(train_files)
    swap = train_files[:len(test_files)]

    for f in swap:
        #move training files into test folder
        name = os.path.basename(f)
        shutil.move(f, os.path.join(npz_path, 'test', name))
        
    for f in test_files:
        name = os.path.basename(f)
        shutil.move(f, os.path.join(npz_path, 'train', name))