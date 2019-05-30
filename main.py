import keras
from models import TestModels
import data
import moveFiles
import numpy as np
import pandas as pd
import os
from keras.models import load_model, Model
import utils
from utils import NewAccuracy


def dump():

    dataset = data.DataSet()
    dataset.classes
    experiment = '2'
    dataset.dumpNumpyFiles(seq_len_limit=20, experiment=experiment)
    return (experiment, dataset)


def train(tups):
    experiment = tups[0]
    dataset = tups[1]

    # MAIN LOOP

    k_fold = 5
    for k in range(0, k_fold):
        # Randomize train and test
        if k > 0:
            moveFiles.randomizeNumpy()
        
        # Train model
        m = TestModels(5, 'lrcn')
        outputEpochPath = os.path.join('output', 'model_checkpoints') 
        if not (os.path.isdir(outputEpochPath)):
            os.makedirs(outputEpochPath)

        filepath = os.path.join(outputEpochPath, "expLRCN{}-5,3,1-k{}.hdf5".format(experiment, k))
        checkpoint = keras.callbacks.ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

        gen = data.DataGenerator('train', batch_size = 1, useSequences=True)
        val = data.DataGenerator('test', batch_size = 1, useSequences=True)

        callbacks = [checkpoint]
        m.model.fit_generator(gen, validation_data = val, epochs=50, callbacks=callbacks) 
        
        # Evaluate and save
        path = os.path.join('output', 'model_checkpoints')
        m = load_model(os.path.join(path, "expLRCN{}-5,3,1-k{}.hdf5".format(experiment, k)))
        
        x,y,yseq = dataset.all_data_from_npz('test')
        
        outputCSVPath = os.path.join('output', 'csv') 
        if not (os.path.isdir(outputCSVPath)):
            os.makedirs(outputCSVPath)
        
        if experiment == 'standard':
            tup = utils.nonNormalAccuracy(x,yseq,dataset,m) 
            print(tup[0])
            print(tup[1])
            d = {'Non-Normal Accuracy': {'Acc': tup[0]}, 'Class Accuracy': tup[1]}
            df = pd.DataFrame(data = d)
            df.to_csv(os.path.join(outputCSVPath, "expLRCN{}-19-k{}.csv".format(experiment, k)))
        else:
            tup = utils.nonNormalAccuracy(x,yseq,dataset,m) 
            d = {'Non-Normal Accuracy': {'Acc': tup[0]}}
            df = pd.DataFrame(data = d)
            df.to_csv(os.path.join(outputCSVPath, "expLRCN{}-5,3,1-k{}.csv".format(experiment, k)))


def genAcc(experiment):
    outputCSVPath = os.path.join('output', 'csv') 
    import glob

    csv_files = glob.glob(os.path.join(outputCSVPath, '*.csv'))
    actualFiles = []
    for file in csv_files:
        if "expLRCN{}-5,3,1-k".format(experiment) in file:
            actualFiles.append(file)

    # Generate final (averaged) accuracy for current test
    if experiment == 'standard':
        acc = []
        normal = []
        black = []
        blurred = []
        insert = []
        compressed = []
        
        for file in actualFiles:
            df = pd.read_csv(file)

            acc.append(df.iloc[0,1])
            normal.append(df.iloc[5,2])
            black.append(df.iloc[1,2])
            blurred.append(df.iloc[2,2])
            insert.append(df.iloc[4,2])
            compressed.append(df.iloc[3,2])
            
        normal = np.mean(normal)
        print("normal: ", normal)
        compressed = np.mean(compressed)
        print("compressed: ", compressed)
        insert = np.mean(insert)
        print("insert: ", insert)
        blurred = np.mean(blurred)
        print("blurred: ", blurred)
        black = np.mean(black)
        print("black: ", black)
        acc = np.mean(acc)
        print("acc: ", acc)
    else:
        acc = []
        
        for file in actualFiles:
            df = pd.read_csv(file)

            acc.append(df.iloc[0,1])
            
        acc = np.mean(acc)
        print("acc: ", acc)

tups = dump()
train(tups)
genAcc(tups[0])