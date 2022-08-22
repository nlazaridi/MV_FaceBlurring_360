'''
Script for face detection with several detectors for regular videos but tested for 360 videos

Available detectors: SCRFD (https://github.com/deepinsight/insightface/tree/master/model_zoo)

'''



# import libraries
import cv2
import insightface
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image

from PIL import Image, ImageDraw, ImageFilter
from IPython import display
import json
from os.path import join
import os
import mmcv
import numpy as np
import argparse
import time


parser = argparse.ArgumentParser(description="Script for face detection in regular/360 videos")

parser.add_argument('--file_path', 
                    type = str,
                    default = "/home/fkalaganis/Desktop/python_scripts/360-processing-master/dataset/VRAG_videos/",
                    help='path where the video is stored')

parser.add_argument('--filename', 
                    type = str,
                    default= "Video360_1080p",
                    help='Name of the specific video example')


parser.add_argument('--model_pack_name', 
                    type = str,
                    default = "buffalo_l",
                    #default = "antelope",
                    help='Name of the models used for face detection')


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
                    help="CPU to be used")


args = parser.parse_args()




def SCRFD_detector(args):


    # get the args
    file_path = args.file_path
    filename = args.filename
    model_pack_name = args.model_pack_name
    blur = args.blur
    print_frame = args.print_frame
    suffix = args.suffix
    suffix_bboxes = args.suffix_bboxes
    gpu = args.gpu


    #read the video
    video = mmcv.VideoReader(join(file_path, filename + suffix))
    
    #get all the frames of the video as PLI Images
    frames = [Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) for frame in video]

    #define the resolution of the detector
    det_size = frames[0].size

    
    print(f'The video will be processed in resolution: {det_size}')


    output_name = "_" + model_pack_name +"_" + str(det_size)
    print(f'Name of the output filde {output_name}')
    
    
    results_path = join(file_path,"results",filename,filename + output_name)
    
    #create a folder if does not already exist
    os.makedirs(join(file_path,"results",filename),exist_ok=True)

    print(f"Working with {join(file_path, filename + suffix)}")
    print(f"model used as detector: {model_pack_name}")
    print(f'Blur the video: {blur}')
    print(f'Total number of frames: {len(frames)}')

    # initialize the network
    app = FaceAnalysis(name=model_pack_name,providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    #app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    #app = insightface.model_zoo.get_model('/home/fkalaganis/anaconda3/envs/insightface/insightface/models/buffalo_l/1k3d68.onnx')
    #prepare the detector
    app.prepare(ctx_id=gpu, det_size = det_size)
    
    frames_tracked = [] #list to store all frames
    bboxes = {} #dic to store all bboxes

    start_time = time.time()
    #Loop to go through frame as image
    for i, frame in enumerate(frames):
        
        #print only in specific frames
        if i%print_frame==0:
            
            print('\rTracking frame: {}'.format(i + 1), end='')
                
        #convert the frame to regular image 
        img = np.array(frame)
        faces = app.get(img)
        
        
        boxes = []


        for f in range(len(faces)):
            boxes.append(faces[f].bbox.tolist())    
        
        
        #append the dictionary with the bboxes
        bboxes["frame_" + str(i+1)] = boxes
        

        # Draw faces
        frame_draw = frame.copy()
    
        '''
        There was an error when blurring. 
        In the final deployed version, where the blurring will be default, we can exclude the IF's below
        '''
        
        if blur:
            mask = Image.new('L', frame_draw.size, 0)
            draw = ImageDraw.Draw(mask)
        else:
            draw = ImageDraw.Draw(frame_draw)
        
        
        # Draw the bounding boxes
        if boxes is not None:
            for box in boxes:
                
                #draw.rectangle(box.tolist(), outline=(255, 0, 0), width=6)
                draw.ellipse(box,fill =255)
                
                #draw.rectangle(box.tolist(), fill =255) # this will fill the bboxs with a color (for example 0 is black and 255 is red etc) 

        if blur:
            blurred = frame_draw.filter(ImageFilter.GaussianBlur(52))
            frame_draw.paste(blurred, mask=mask)
            
        # Add to frame list
        #frames_tracked.append(frame_draw.resize((640, 360), Image.BILINEAR))
        # Append the list of the frames WITH Bbox in PLI image
        frames_tracked.append(frame_draw)
        
    print('\nDone with the video processing')

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
    
    SCRFD_detector(args)