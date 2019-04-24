import os
import numpy as np 
import config
from keras.preprocessing.image import img_to_array, load_img
import cv2
import glob
import csv
from tqdm import tqdm
from keras.utils import to_categorical, Sequence
import h5py
import time
import shutil
import sys
from imgaug import augmenters as iaa
import random
from PIL import Image, ImageFilter
from io import BytesIO

class DataSet():

    def __init__(self):
        self.min_seq_length = config.MIN_SEQ_LENGTH
        self.max_seq_length = config.MAX_SEQ_LENGTH

        self.sequence_path = os.path.join('data', 'sequences')
        self.csv_data = self.get_csv_data()
        self.classes = self.get_classes()
        self.csv_data = self.clean_data()
        
    def get_csv_data(self):
        try:
            with open(os.path.join('data', 'data_file.csv'), 'r') as fin:
                reader = csv.reader(fin)
                data = list(reader)
            return data 
        except IOError:
            print('CSV Data File does not exist. Please run Preprocessing.extractAllVideos().')
            sys.exit()

    def get_classes(self):
        classes = []
        for item in self.csv_data:
            if item[1] not in classes:
                classes.append(item[1])
        
        classes = sorted(classes)
        return classes
    
    def clean_data(self):
        cleaned = []
        for item in self.csv_data:
            if int(item[3])>= self.min_seq_length and int(item[3])<= self.max_seq_length:
                cleaned.append(item)
        return cleaned

    def one_hot(self, class_str):
        label_encoded = self.classes.index(class_str)
        label_hot = to_categorical(label_encoded, len(self.classes))
        return label_hot

    def reverse_one_hot(self, one_hot):
        return self.classes[np.argmax(one_hot)]

    def split_train_test(self):
        train = []
        test = []
        for item in self.csv_data:
            if item[0] == 'train':
                train.append(item)
            else:
                test.append(item)
        return train, test
    
    def get_frames_for_sample(self, csv_row):
        path = os.path.join(self.sequence_path, csv_row[0], csv_row[1])
        filename = csv_row[2]
        images = sorted(glob.glob(os.path.join(path, filename+'*.jpg')))
        return images

    def all_data_from_npz(self, trainTest, folderName='Default'):
        files = glob.glob(os.path.join('data', 'sequences', 'npz', folderName, trainTest, '*.npz'))
        x,y,yseq=[],[],[]
        for f in files:
            data = np.load(f)
            x.append(data['x'])
            y.append(data['y'])
            yseq.append(data['yseq'])
        return np.array(x), np.array(y), np.array(yseq)

    def some_data_from_npz(self, trainTest, range, folderName='Default'):
        files = glob.glob(os.path.join('data', 'sequences', 'npz', folderName, trainTest, '*.npz'))
        x,y,yseq=[],[],[]
        count = 1
        for f in files:
            data = np.load(f)
            x.append(data['x'])
            y.append(data['y'])
            yseq.append(data['yseq'])
            if count == range:
                break
            count += 1
        return np.array(x), np.array(y), np.array(yseq)




    def process_image(self, image_path):
        """
        Load image from a path into array, and normalize to 0-1. 
        """
        image = load_img(image_path, target_size=(config.IMG_HEIGHT, config.IMG_WIDTH))
        img_arr = img_to_array(image)
        x = (img_arr / 255.).astype(np.float32)
        return x


    def build_image_sequence(self, frames):
        return [self.process_image(x) for x in frames]


    def dumpNumpyFiles(self, trainTest='all', seq_len_limit=config.DEFAULT_SEQ_LENGTH, folderName='Default', experimentType='standard'):
        """
        Exports sequences to .npz files in data/sequences/npz. 
        
        DataGenerator uses these files to compute batches. 

        Experiment Types:

        - Standard (1): Selects one manipluation type per video and one consecutive manipulated sequence. The size of the manipulated sequence varies. 
        - Standard Multiple (2): Selects one manipulation type per video and multiple consecutive manipulated sequences.  
        - Deluxe (4): Same as Standard but selects two or more manipulations. The size of the manipulated sequence varies.
        - Deluxe Multiple (3): Same as Standard Multiple but selects two or more manipulations. The sizes of the manipulted sequences vary.
        - Deluxe Multiple Mixed (5): Same as Deluxe Multiple but subsequences have multiple manipulations. The sizes of the manipulted sequences vary.
         
        """
        if trainTest == 'all':
            print('Exporting Train Data...')
            self.dumpNumpyFiles('train', seq_len_limit=seq_len_limit, folderName=folderName, experimentType=experimentType)
            print('Exporting Test Data')
            self.dumpNumpyFiles('test', seq_len_limit=seq_len_limit, folderName=folderName, experimentType=experimentType)
        else:
            prelimPath = os.path.join(self.sequence_path, 'npz', folderName)
            if not os.path.isdir(prelimPath):
                os.makedirs(prelimPath, exist_ok=True)

            outPath = os.path.join(prelimPath, trainTest)
            if os.path.isdir(outPath):
                shutil.rmtree(outPath)
            os.makedirs(outPath, exist_ok=True)

            train, test = self.split_train_test()
            csv_data = train if trainTest == 'train' else test

            for k in tqdm(range(len(csv_data))):
                row = csv_data[k]
                frames = self.get_frames_for_sample(row)
                sequence = self.build_image_sequence(frames)
                sequence_orig = sequence
                if seq_len_limit:
                    sequence = sequence[:seq_len_limit]
                
                vidClass = row[1]
                vidName = row[2]
                #make random frames black
                # aug_len = config.MANIPULATION_LENGTH

                if experimentType == 'Standard':
                    aug_len = config.MANIPULATION_LENGTH #Vary this length within the config.py file
                    start = np.random.randint(len(sequence)-aug_len)
                    if vidClass == 'black':
                        for i in range(start, start+aug_len):
                            #make black
                            sequence[i]=np.zeros(sequence[i].shape)
                    
                    
                    if vidClass == 'compressed':
                        # quality = np.random.randint(5,25)
                        quality = np.random.randint(1,5)
                        # compress = iaa.JpegCompression(compression=(100, 100))
                        for i in range(start, start+aug_len):
                            # sequence[i] = sequence[i]*255.
                            # sequence[i] = compress.augment_image(sequence[i].astype('uint8'))
                            # sequence[i] = (sequence[i] / 255.).astype(np.float32)
                            img = (sequence[i]*255.).astype('uint8')
                            img = Image.fromarray(img)
                            buffer = BytesIO()
                            img.save(buffer, 'JPEG', quality = quality)
                            sequence[i] = (np.array(Image.open(buffer))/255.).astype(np.float32)
                            buffer.close()
                            


                    if vidClass == 'insert':
                        pngs = glob.glob(os.path.join('data', 'pngs', '*.png'))
                        png = random.choice(pngs)
                        # obj_size = new_size = np.random.randint(25, int(0.5*config.IMG_WIDTH))
                        obj_size = new_size = np.random.randint(int(0.5*config.IMG_WIDTH), int(0.9*config.IMG_WIDTH))

                        area = (np.random.randint(config.IMG_WIDTH-new_size), np.random.randint(config.IMG_HEIGHT-new_size))  
                        for i in range(start, start+aug_len):
                            
                            
                            sequence[i] = (sequence[i]*255).astype('uint8')
                            base_image = Image.fromarray(sequence[i])
                            base_image = base_image.convert('RGBA')
                            object_image = Image.open(png)
                            object_image = object_image.convert('RGBA')

                            
                            object_image = object_image.resize((obj_size, obj_size), Image.ANTIALIAS)

                            
                            base_image.paste(object_image, area, mask = object_image)
                            base_image = base_image.convert('RGB')

                            sequence[i] = ( np.array(base_image) / 255.).astype(np.float32)
                    if vidClass == 'dropped':
                        #make sure we don't drop at the very beginning
                        start = np.random.randint(5, len(sequence)-aug_len)
                        sequence1 = sequence_orig[0:start]
                        sequence2 = sequence_orig[start+aug_len:]
                        sequence = sequence1 + sequence2
                        sequence = sequence[:seq_len_limit]
                    
                    if vidClass == 'blurred':
                        # box = (50, 100, 170, 200) #starting spots from (0,0)
                        xbox_start = np.random.randint(25, int(0.5*config.IMG_WIDTH))
                        ybox_start = np.random.randint(25, int(0.5*config.IMG_HEIGHT))
                        xbox_end = np.random.randint(xbox_start+int(0.2*config.IMG_WIDTH), xbox_start+int(0.5*config.IMG_WIDTH))
                        ybox_end = np.random.randint(ybox_start+int(0.2*config.IMG_HEIGHT), ybox_start+int(0.5*config.IMG_HEIGHT))
                        box = (xbox_start, ybox_start, xbox_end, ybox_end)
                        # region = im.crop(box)
                        # obj_size = new_size = np.random.randint(25, int(0.5*config.IMG_WIDTH))
                        # area = (np.random.randint(config.IMG_WIDTH-new_size), np.random.randint(config.IMG_HEIGHT-new_size))  
                        for i in range(start, start+aug_len):
                            
                            
                            sequence[i] = (sequence[i]*255).astype('uint8')
                            base_image = Image.fromarray(sequence[i])

                            region = base_image.crop(box)
                            region = region.filter(ImageFilter.GaussianBlur(radius=50))
                            base_image.paste(region, box)

                            sequence[i] = ( np.array(base_image) / 255.).astype(np.float32)

                elif experimentType == 'Standard Multiple':
                    # Define number of subsequences. (In what order do we define everything?)
                    numSubs = config.NUMBER_OF_SUBSEQUENCES # We could do random later but I think one table per this number is what she wants

                    # Find distance between subsequences (end and start, which must be less than the length between start to start as chosen below). 
                    # Save average distance -> do we need to find this or do we base it on avg length?

                    avgDist = config.AVG_DISTANCE_BETWEEN_SUBSEQUENCES # this will vary the start locations and max lengths!

                    # Find length of subsequences (and save average length if random)

                    avgLength = config.AVG_LENGTH_OF_SUBSEQUENCES

                    # Find start locations
                    a = np.empty(len(sequence))
                    b = np.arange(0, len(sequence), 1)
                    ind = np.arange(len(a))
                    np.put(a, ind, b) # a is now filled with numbers 0 -> len(sequence) 

                    np.random.shuffle(a) # a is now randomly shuffled -> These range numbers will have to be limited depending on length of subsequences
                    startLocs = a[0:numSubs] 

                    

                    

                    







                    aug_len = config.MANIPULATION_LENGTH #can do fixed but can also do random length
                    start = np.random.randint(len(sequence)-aug_len)
                    if vidClass == 'black':
                        for i in range(start, start+aug_len):
                            #make black
                            sequence[i]=np.zeros(sequence[i].shape)
                    
                    
                    if vidClass == 'compressed':
                        # quality = np.random.randint(5,25)
                        quality = np.random.randint(1,5)
                        # compress = iaa.JpegCompression(compression=(100, 100))
                        for i in range(start, start+aug_len):
                            # sequence[i] = sequence[i]*255.
                            # sequence[i] = compress.augment_image(sequence[i].astype('uint8'))
                            # sequence[i] = (sequence[i] / 255.).astype(np.float32)
                            img = (sequence[i]*255.).astype('uint8')
                            img = Image.fromarray(img)
                            buffer = BytesIO()
                            img.save(buffer, 'JPEG', quality = quality)
                            sequence[i] = (np.array(Image.open(buffer))/255.).astype(np.float32)
                            buffer.close()
                            


                    if vidClass == 'insert':
                        pngs = glob.glob(os.path.join('data', 'pngs', '*.png'))
                        png = random.choice(pngs)
                        # obj_size = new_size = np.random.randint(25, int(0.5*config.IMG_WIDTH))
                        obj_size = new_size = np.random.randint(int(0.5*config.IMG_WIDTH), int(0.9*config.IMG_WIDTH))

                        area = (np.random.randint(config.IMG_WIDTH-new_size), np.random.randint(config.IMG_HEIGHT-new_size))  
                        for i in range(start, start+aug_len):
                            
                            
                            sequence[i] = (sequence[i]*255).astype('uint8')
                            base_image = Image.fromarray(sequence[i])
                            base_image = base_image.convert('RGBA')
                            object_image = Image.open(png)
                            object_image = object_image.convert('RGBA')

                            
                            object_image = object_image.resize((obj_size, obj_size), Image.ANTIALIAS)

                            
                            base_image.paste(object_image, area, mask = object_image)
                            base_image = base_image.convert('RGB')

                            sequence[i] = ( np.array(base_image) / 255.).astype(np.float32)
                    if vidClass == 'dropped':
                        #make sure we don't drop at the very beginning
                        start = np.random.randint(5, len(sequence)-aug_len)
                        sequence1 = sequence_orig[0:start]
                        sequence2 = sequence_orig[start+aug_len:]
                        sequence = sequence1 + sequence2
                        sequence = sequence[:seq_len_limit]
                    
                    if vidClass == 'blurred':
                        # box = (50, 100, 170, 200) #starting spots from (0,0)
                        xbox_start = np.random.randint(25, int(0.5*config.IMG_WIDTH))
                        ybox_start = np.random.randint(25, int(0.5*config.IMG_HEIGHT))
                        xbox_end = np.random.randint(xbox_start+int(0.2*config.IMG_WIDTH), xbox_start+int(0.5*config.IMG_WIDTH))
                        ybox_end = np.random.randint(ybox_start+int(0.2*config.IMG_HEIGHT), ybox_start+int(0.5*config.IMG_HEIGHT))
                        box = (xbox_start, ybox_start, xbox_end, ybox_end)
                        # region = im.crop(box)
                        # obj_size = new_size = np.random.randint(25, int(0.5*config.IMG_WIDTH))
                        # area = (np.random.randint(config.IMG_WIDTH-new_size), np.random.randint(config.IMG_HEIGHT-new_size))  
                        for i in range(start, start+aug_len):
                            
                            
                            sequence[i] = (sequence[i]*255).astype('uint8')
                            base_image = Image.fromarray(sequence[i])

                            region = base_image.crop(box)
                            region = region.filter(ImageFilter.GaussianBlur(radius=50))
                            base_image.paste(region, box)

                            sequence[i] = ( np.array(base_image) / 255.).astype(np.float32)
                
                elif experimentType == 'Deluxe':
                    aug_len = config.MANIPULATION_LENGTH
                    start = np.random.randint(len(sequence)-aug_len)
                    if vidClass == 'black':
                        for i in range(start, start+aug_len):
                            #make black
                            sequence[i]=np.zeros(sequence[i].shape)
                    
                    
                    if vidClass == 'compressed':
                        # quality = np.random.randint(5,25)
                        quality = np.random.randint(1,5)
                        # compress = iaa.JpegCompression(compression=(100, 100))
                        for i in range(start, start+aug_len):
                            # sequence[i] = sequence[i]*255.
                            # sequence[i] = compress.augment_image(sequence[i].astype('uint8'))
                            # sequence[i] = (sequence[i] / 255.).astype(np.float32)
                            img = (sequence[i]*255.).astype('uint8')
                            img = Image.fromarray(img)
                            buffer = BytesIO()
                            img.save(buffer, 'JPEG', quality = quality)
                            sequence[i] = (np.array(Image.open(buffer))/255.).astype(np.float32)
                            buffer.close()
                            


                    if vidClass == 'insert':
                        pngs = glob.glob(os.path.join('data', 'pngs', '*.png'))
                        png = random.choice(pngs)
                        # obj_size = new_size = np.random.randint(25, int(0.5*config.IMG_WIDTH))
                        obj_size = new_size = np.random.randint(int(0.5*config.IMG_WIDTH), int(0.9*config.IMG_WIDTH))

                        area = (np.random.randint(config.IMG_WIDTH-new_size), np.random.randint(config.IMG_HEIGHT-new_size))  
                        for i in range(start, start+aug_len):
                            
                            
                            sequence[i] = (sequence[i]*255).astype('uint8')
                            base_image = Image.fromarray(sequence[i])
                            base_image = base_image.convert('RGBA')
                            object_image = Image.open(png)
                            object_image = object_image.convert('RGBA')

                            
                            object_image = object_image.resize((obj_size, obj_size), Image.ANTIALIAS)

                            
                            base_image.paste(object_image, area, mask = object_image)
                            base_image = base_image.convert('RGB')

                            sequence[i] = ( np.array(base_image) / 255.).astype(np.float32)
                    if vidClass == 'dropped':
                        #make sure we don't drop at the very beginning
                        start = np.random.randint(5, len(sequence)-aug_len)
                        sequence1 = sequence_orig[0:start]
                        sequence2 = sequence_orig[start+aug_len:]
                        sequence = sequence1 + sequence2
                        sequence = sequence[:seq_len_limit]
                    
                    if vidClass == 'blurred':
                        # box = (50, 100, 170, 200) #starting spots from (0,0)
                        xbox_start = np.random.randint(25, int(0.5*config.IMG_WIDTH))
                        ybox_start = np.random.randint(25, int(0.5*config.IMG_HEIGHT))
                        xbox_end = np.random.randint(xbox_start+int(0.2*config.IMG_WIDTH), xbox_start+int(0.5*config.IMG_WIDTH))
                        ybox_end = np.random.randint(ybox_start+int(0.2*config.IMG_HEIGHT), ybox_start+int(0.5*config.IMG_HEIGHT))
                        box = (xbox_start, ybox_start, xbox_end, ybox_end)
                        # region = im.crop(box)
                        # obj_size = new_size = np.random.randint(25, int(0.5*config.IMG_WIDTH))
                        # area = (np.random.randint(config.IMG_WIDTH-new_size), np.random.randint(config.IMG_HEIGHT-new_size))  
                        for i in range(start, start+aug_len):
                            
                            
                            sequence[i] = (sequence[i]*255).astype('uint8')
                            base_image = Image.fromarray(sequence[i])

                            region = base_image.crop(box)
                            region = region.filter(ImageFilter.GaussianBlur(radius=50))
                            base_image.paste(region, box)

                            sequence[i] = ( np.array(base_image) / 255.).astype(np.float32)

                elif experimentType == 'Deluxe Multiple':
                    aug_len = config.MANIPULATION_LENGTH
                    start = np.random.randint(len(sequence)-aug_len)
                    if vidClass == 'black':
                        for i in range(start, start+aug_len):
                            #make black
                            sequence[i]=np.zeros(sequence[i].shape)
                    
                    
                    if vidClass == 'compressed':
                        # quality = np.random.randint(5,25)
                        quality = np.random.randint(1,5)
                        # compress = iaa.JpegCompression(compression=(100, 100))
                        for i in range(start, start+aug_len):
                            # sequence[i] = sequence[i]*255.
                            # sequence[i] = compress.augment_image(sequence[i].astype('uint8'))
                            # sequence[i] = (sequence[i] / 255.).astype(np.float32)
                            img = (sequence[i]*255.).astype('uint8')
                            img = Image.fromarray(img)
                            buffer = BytesIO()
                            img.save(buffer, 'JPEG', quality = quality)
                            sequence[i] = (np.array(Image.open(buffer))/255.).astype(np.float32)
                            buffer.close()
                            


                    if vidClass == 'insert':
                        pngs = glob.glob(os.path.join('data', 'pngs', '*.png'))
                        png = random.choice(pngs)
                        # obj_size = new_size = np.random.randint(25, int(0.5*config.IMG_WIDTH))
                        obj_size = new_size = np.random.randint(int(0.5*config.IMG_WIDTH), int(0.9*config.IMG_WIDTH))

                        area = (np.random.randint(config.IMG_WIDTH-new_size), np.random.randint(config.IMG_HEIGHT-new_size))  
                        for i in range(start, start+aug_len):
                            
                            
                            sequence[i] = (sequence[i]*255).astype('uint8')
                            base_image = Image.fromarray(sequence[i])
                            base_image = base_image.convert('RGBA')
                            object_image = Image.open(png)
                            object_image = object_image.convert('RGBA')

                            
                            object_image = object_image.resize((obj_size, obj_size), Image.ANTIALIAS)

                            
                            base_image.paste(object_image, area, mask = object_image)
                            base_image = base_image.convert('RGB')

                            sequence[i] = ( np.array(base_image) / 255.).astype(np.float32)
                    if vidClass == 'dropped':
                        #make sure we don't drop at the very beginning
                        start = np.random.randint(5, len(sequence)-aug_len)
                        sequence1 = sequence_orig[0:start]
                        sequence2 = sequence_orig[start+aug_len:]
                        sequence = sequence1 + sequence2
                        sequence = sequence[:seq_len_limit]
                    
                    if vidClass == 'blurred':
                        # box = (50, 100, 170, 200) #starting spots from (0,0)
                        xbox_start = np.random.randint(25, int(0.5*config.IMG_WIDTH))
                        ybox_start = np.random.randint(25, int(0.5*config.IMG_HEIGHT))
                        xbox_end = np.random.randint(xbox_start+int(0.2*config.IMG_WIDTH), xbox_start+int(0.5*config.IMG_WIDTH))
                        ybox_end = np.random.randint(ybox_start+int(0.2*config.IMG_HEIGHT), ybox_start+int(0.5*config.IMG_HEIGHT))
                        box = (xbox_start, ybox_start, xbox_end, ybox_end)
                        # region = im.crop(box)
                        # obj_size = new_size = np.random.randint(25, int(0.5*config.IMG_WIDTH))
                        # area = (np.random.randint(config.IMG_WIDTH-new_size), np.random.randint(config.IMG_HEIGHT-new_size))  
                        for i in range(start, start+aug_len):
                            
                            
                            sequence[i] = (sequence[i]*255).astype('uint8')
                            base_image = Image.fromarray(sequence[i])

                            region = base_image.crop(box)
                            region = region.filter(ImageFilter.GaussianBlur(radius=50))
                            base_image.paste(region, box)

                            sequence[i] = ( np.array(base_image) / 255.).astype(np.float32)

                elif experimentType == 'Deluxe Multiple Mixed':
                    aug_len = config.MANIPULATION_LENGTH
                    start = np.random.randint(len(sequence)-aug_len)
                    if vidClass == 'black':
                        for i in range(start, start+aug_len):
                            #make black
                            sequence[i]=np.zeros(sequence[i].shape)
                    
                    
                    if vidClass == 'compressed':
                        # quality = np.random.randint(5,25)
                        quality = np.random.randint(1,5)
                        # compress = iaa.JpegCompression(compression=(100, 100))
                        for i in range(start, start+aug_len):
                            # sequence[i] = sequence[i]*255.
                            # sequence[i] = compress.augment_image(sequence[i].astype('uint8'))
                            # sequence[i] = (sequence[i] / 255.).astype(np.float32)
                            img = (sequence[i]*255.).astype('uint8')
                            img = Image.fromarray(img)
                            buffer = BytesIO()
                            img.save(buffer, 'JPEG', quality = quality)
                            sequence[i] = (np.array(Image.open(buffer))/255.).astype(np.float32)
                            buffer.close()
                            


                    if vidClass == 'insert':
                        pngs = glob.glob(os.path.join('data', 'pngs', '*.png'))
                        png = random.choice(pngs)
                        # obj_size = new_size = np.random.randint(25, int(0.5*config.IMG_WIDTH))
                        obj_size = new_size = np.random.randint(int(0.5*config.IMG_WIDTH), int(0.9*config.IMG_WIDTH))

                        area = (np.random.randint(config.IMG_WIDTH-new_size), np.random.randint(config.IMG_HEIGHT-new_size))  
                        for i in range(start, start+aug_len):
                            
                            
                            sequence[i] = (sequence[i]*255).astype('uint8')
                            base_image = Image.fromarray(sequence[i])
                            base_image = base_image.convert('RGBA')
                            object_image = Image.open(png)
                            object_image = object_image.convert('RGBA')

                            
                            object_image = object_image.resize((obj_size, obj_size), Image.ANTIALIAS)

                            
                            base_image.paste(object_image, area, mask = object_image)
                            base_image = base_image.convert('RGB')

                            sequence[i] = ( np.array(base_image) / 255.).astype(np.float32)
                    if vidClass == 'dropped':
                        #make sure we don't drop at the very beginning
                        start = np.random.randint(5, len(sequence)-aug_len)
                        sequence1 = sequence_orig[0:start]
                        sequence2 = sequence_orig[start+aug_len:]
                        sequence = sequence1 + sequence2
                        sequence = sequence[:seq_len_limit]
                    
                    if vidClass == 'blurred':
                        # box = (50, 100, 170, 200) #starting spots from (0,0)
                        xbox_start = np.random.randint(25, int(0.5*config.IMG_WIDTH))
                        ybox_start = np.random.randint(25, int(0.5*config.IMG_HEIGHT))
                        xbox_end = np.random.randint(xbox_start+int(0.2*config.IMG_WIDTH), xbox_start+int(0.5*config.IMG_WIDTH))
                        ybox_end = np.random.randint(ybox_start+int(0.2*config.IMG_HEIGHT), ybox_start+int(0.5*config.IMG_HEIGHT))
                        box = (xbox_start, ybox_start, xbox_end, ybox_end)
                        # region = im.crop(box)
                        # obj_size = new_size = np.random.randint(25, int(0.5*config.IMG_WIDTH))
                        # area = (np.random.randint(config.IMG_WIDTH-new_size), np.random.randint(config.IMG_HEIGHT-new_size))  
                        for i in range(start, start+aug_len):
                            
                            
                            sequence[i] = (sequence[i]*255).astype('uint8')
                            base_image = Image.fromarray(sequence[i])

                            region = base_image.crop(box)
                            region = region.filter(ImageFilter.GaussianBlur(radius=50))
                            base_image.paste(region, box)

                            sequence[i] = ( np.array(base_image) / 255.).astype(np.float32)

                y_seq = ['normal']*len(sequence)
                if not (vidClass == 'dropped'):
                    y_seq[start:start+aug_len]=[vidClass]*aug_len
                else:
                    y_seq[start]=vidClass
                y_seq = np.array([self.one_hot(k) for k in y_seq])

                np.savez_compressed(os.path.join(outPath, vidName + '-' + vidClass + '.npz'), x=np.array(sequence), y=self.one_hot(vidClass), yseq=y_seq)
        
        
class DataGenerator(Sequence):
    def __init__(self, trainTest='train', folderName='Default', useSequences=False, batch_size=1, shuffle=True, class_weights=None, filter=None):
        self.folderName = folderName
        self.trainTest = trainTest
        self.files = glob.glob(os.path.join('data', 'sequences', 'npz', self.folderName, self.trainTest, '*.npz'))
        self.data_length = len(self.files)
        self.batch_size = batch_size
        self.batch_num = 0
        self.shuffle = shuffle
        self.useSequences = useSequences
        self.class_weights = class_weights
        self.filter = filter
        
        self.on_epoch_end()
    
    def __len__(self):
        return int(np.floor(self.data_length / self.batch_size))

    def __getitem__(self, index):
        x, y = [], []
        weights = []
        batch_files = self.files[self.batch_num*self.batch_size:(self.batch_num+1)*self.batch_size]

        for f in batch_files:
            sequence = np.load(f)
            
            npy = sequence['y']
            if self.useSequences:
                seq = sequence['x']
                if self.filter == 'mean':
                    seq = np.array(seq).squeeze()
                    
                    mean = np.mean(seq, axis=0)
                    normed = np.array([k-mean for k in seq])
                    seq = np.interp(normed, (normed.min(), normed.max()), (0,1))
                    # seq = normed.clip(min=0)
                
                x.append(seq)
                y.append(sequence['yseq'])
            else:
                x.append(sequence['x'])
                y.append(npy)
            if self.class_weights:
                weights.append(self.class_weights[npy.argmax()])
        self.batch_num += 1

        
        
        if self.class_weights:
            return np.array(x), np.array(y), np.array(weights)
        else:
            return np.array(x), np.array(y)
    
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.files)
        self.batch_num = 0
   

class Preprocessing():
    def __init__(self):
        self.videos_path = os.path.join('data', 'videos')
        self.sequence_path = os.path.join('data', 'sequences')

    
    def extractAllVideos(self):
        """
        Extracts all videos in 'train' and 'test' folders into frames (location: data/sequences).

        Also generates a csv file with: train/test,class_label, videofilename, # frames in sequence
        """

        #Delete old files if they exist. 
        trainOut = os.path.join(self.sequence_path, 'train')
        testOut = os.path.join(self.sequence_path, 'test')
        if os.path.isdir(trainOut):
            shutil.rmtree(trainOut)
            os.makedirs(trainOut, exist_ok=True)
        if os.path.isdir(testOut):
            shutil.rmtree(testOut)
            os.makedirs(testOut, exist_ok=True)
        
        

        data_file = []
        folders = ['train', 'test']
        for folder in folders:
            class_folders = glob.glob(os.path.join(self.videos_path, folder, '*'))
            
            for vid_class in class_folders:
                class_files = glob.glob(os.path.join(vid_class, '*.avi'))
                
                for video_path in class_files:
                    trainTest, vidClass, vidName, filename = self.video_parts(video_path)
                    n_frames = self.extractFrames(video_path)
                    data_file.append([trainTest, vidClass, vidName, n_frames])
        
        with open(os.path.join('data', 'data_file.csv'), 'w', newline='') as fout:
            writer = csv.writer(fout)
            writer.writerows(data_file)



    def extractFrames(self, video_path):
        trainTest, vidClass, vidName, filename = self.video_parts(video_path)
        outPath = os.path.join(self.sequence_path, trainTest, vidClass)
        
        if not os.path.isdir(outPath): os.makedirs(outPath)

        vidcap = cv2.VideoCapture(video_path)
        success,image = vidcap.read()
        count = 1
        while success:
            cv2.imwrite(os.path.join(outPath,  vidName + "-%04d.jpg" % count), image)     # save frame as JPEG file      
            success,image = vidcap.read()
            count += 1
        count-=1
        return count

    def video_parts(self, video_path):
         parts = video_path.split(os.path.sep)
         filename = parts[-1]
         vidName = filename.split('.')[0]
         vidClass = parts[-2]
         trainTest = parts[-3]
         return trainTest, vidClass, vidName, filename