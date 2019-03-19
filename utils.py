import cv2
import os
vidpath = 'frames'


if not os.path.isdir(vidpath):
    os.makedirs(vidpath)




def extractFrames(video_path, output_dir):
    vidcap = cv2.VideoCapture(video_path)
    success,image = vidcap.read()
    count = 0
    while success:
        cv2.imwrite(os.path.join(output_dir, "frame%d.jpg" % count), image)     # save frame as JPEG file      
        success,image = vidcap.read()
        # print('Read a new frame: ', success)
        count += 1