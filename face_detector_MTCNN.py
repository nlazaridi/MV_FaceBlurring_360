'''
Script for face detection with several detectors for regular videos

Available detectors: MTCNN (https://github.com/timesler/facenet-pytorch)

'''


from facenet_pytorch import MTCNN
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from IPython import display
import cv2
import mmcv
import json
import sys
from os.path import join
import os
import argparse
import time


parser = argparse.ArgumentParser(description="Script for face detection in regular/360 videos with MTCNN detector")

parser.add_argument('--file_path', 
                    type = str,
                    default = "/home/fkalaganis/Desktop/python_scripts/360-processing-master/dataset/VRAG_videos/",
                    help='path where the video is stored')

parser.add_argument('--filename', 
                    type = str,
                    default= "Timelapse",
                    help='Name of the specific video example')


parser.add_argument('--blur', 
                    action='store_false',
                    help='If true then blur the video else show bboxes')

parser.add_argument('--print_frame', 
                    type = int,
                    default = 50,
                    help='How often to print the frame number')


parser.add_argument('--suffix', 
                    type = str,
                    default = ".mp4",
                    help='suffix of the video')

parser.add_argument('--suffix_bboxes', 
                    type = str,
                    default = ".json",
                    help='format of storing the bboxes')


parser.add_argument('--gpu', 
                    type = int,
                    default = 0,
                    help="GPU to be used if available")


args = parser.parse_args()


def MTCNN_detector(args):
  
  # get the args
    file_path = args.file_path
    filename = args.filename
    blur = args.blur
    print_frame = args.print_frame
    suffix = args.suffix
    suffix_bboxes = args.suffix_bboxes
    gpu = args.gpu


    device = torch.device("cuda:" + str(gpu) if torch.cuda.is_available() else 'cpu')
    print('Running on device: {}'.format(device))

    
    #read the video
    video = mmcv.VideoReader(join(file_path, filename + suffix))

    #get all the frames of the video as PLI Images!
    frames = [Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) for frame in video]
        
    #define the resolution of the detector
    det_size = frames[0].size
    print(f'The video will be processed in resolution: {det_size}')
    
    output_name = "_MTCNN_" + str(det_size)
    print(f'Name of the output filde {output_name}')
    
    results_path = join(file_path,"results",filename,filename + output_name)
    #create a folder if does not already exist
    os.makedirs(join(file_path,"results",filename),exist_ok=True)


    print(f"Working with {join(file_path, filename + suffix)}")
    print(f'Blur the video: {blur}')
    print(f'Total number of frames: {len(frames)}')
    
   # initialize the network
    mtcnn = MTCNN(keep_all=True, device=device)


    frames_tracked = []
    bboxes = {}

    start_time = time.time()
    #Loop to go through frame as image
    for i, frame in enumerate(frames):
        
        if i%print_frame==0:
            
            print('\rTracking frame: {}'.format(i + 1), end='')
        
        
        '''
        Several things are returned here: 
        -- detection boxes
        -- The detection probabilities are returned (may use that with a threshold)
        -- facial landmark detection
        boxes, probs, points = mtcnn.detect(frame, landmarks=True)
        '''
        
        # Detect faces
        boxes, _ = mtcnn.detect(frame)
        
        #append the dictionary with the bboxes
        if boxes is not None: bboxes["frame_" + str(i+1)] = boxes.tolist()
        
        # Draw faces
        frame_draw = frame.copy()
        
        #Not a nice way --> too complex for no reason
        if blur:
            mask = Image.new('L', frame_draw.size, 0)
            draw = ImageDraw.Draw(mask)
        else:
            draw = ImageDraw.Draw(frame_draw)
        
        
        # Draw the bounding boxes
        if boxes is not None:
            for box in boxes:
                
                #draw.rectangle(box.tolist(), outline=(255, 0, 0), width=6)
                draw.ellipse(box.tolist(),fill =255)
                
                #draw.rectangle(box.tolist(), fill =255) # this will fill the bboxs with a color (for example 0 is black and 255 is red etc) 

        if blur:
            blurred = frame_draw.filter(ImageFilter.GaussianBlur(52))
            frame_draw.paste(blurred, mask=mask)
            
        # Add to frame list
        #frames_tracked.append(frame_draw.resize((640, 360), Image.BILINEAR))
        # Append the list of the frames WITH Bbox in PLI image
        frames_tracked.append(frame_draw)
        
    print('\nDone')

    #save the bboxes
    with open(results_path + suffix_bboxes, 'w') as f:
        json.dump(bboxes, f)

    print(f'The bbxoes have been saved to {results_path}')

    #save the video 
    dim = frames_tracked[0].size
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    #define the same fps as the initial video
    fps = video.fps 

    print(f"Video saved with {dim} resolution and {fps} fps")

    video_tracked = cv2.VideoWriter(results_path + suffix, fourcc, fps, dim)
    for frame in frames_tracked: 
        video_tracked.write(cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR))
    video_tracked.release()


    print(f'result is stored in {results_path + suffix}')
    print("--- %s seconds ---" % (time.time() - start_time))

if __name__ == '__main__':
    MTCNN_detector(args)