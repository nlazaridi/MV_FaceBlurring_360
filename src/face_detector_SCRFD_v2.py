from fileinput import filename
import cv2
from os.path import join
import insightface
from insightface.app import FaceAnalysis
from PIL import Image, ImageDraw, ImageFilter
import numpy as np
from matplotlib import pyplot as plt
import argparse
import os

parser = argparse.ArgumentParser(
    description="Script for face detection in regular/360 videos"
)

parser.add_argument(
    "--file_path",
    type=str,
    default="/home/fkalaganis/Desktop/python_scripts/360-processing-master/dataset/VRAG_videos/",
    help="path where the video is stored",
)

parser.add_argument(
    "--filename",
    type=str,
    default="SAM_100_0026",
    help="Name of the specific video example",
)


parser.add_argument(
    "--model_pack_name",
    type=str,
    default="buffalo_l",
    # default = "antelope",
    help="Name of the models used for face detection",
)


parser.add_argument(
    "--blur", action="store_false", help="If true then blur the video else show bboxes"
)

parser.add_argument(
    "--print_frame", type=int, default=50, help="How often to print the frame number"
)


parser.add_argument("--suffix", type=str, default=".mp4", help="suffix of the video")

parser.add_argument(
    "--suffix_bboxes", type=str, default=".json", help="format of storing the bboxes"
)


parser.add_argument("--gpu", type=int, default=0, help="CPU to be used")


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

    videocap = cv2.VideoCapture(join(file_path, filename + suffix))
    success, image = videocap.read()
    count = 0
    det_size = (image.shape[1], image.shape[0])

    bboxes = {}  # dic to store all boxes
    frames_tracked = []
    i = 0

    fps = videocap.get(cv2.CAP_PROP_FPS)
    output_name = "_" + model_pack_name + "_" + str(det_size)
    results_path = join(file_path, "results", filename, filename + output_name)

    # create a folder if does not already exist
    os.makedirs(join(file_path, "results", filename), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_tracked = cv2.VideoWriter(results_path + suffix, fourcc, fps, det_size)
    app = FaceAnalysis(
        name=model_pack_name,
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    app.prepare(ctx_id=gpu, det_size=det_size)

    while success:
        count = count + 1
        if count % print_frame == 0:
            print(count)
        frame = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        img = np.array(frame)

        faces = model_inf(app, img)

        boxes = []
        for f in range(len(faces)):
            boxes.append(faces[f].bbox.tolist())

        # append the dictionary with the bboxes
        i = i + 1
        bboxes["frame_" + str(i + 1)] = boxes

        # Draw faces
        frame_draw = frame.copy()

        if blur:
            mask = Image.new("L", frame_draw.size, 0)
            draw = ImageDraw.Draw(mask)
        else:
            draw = ImageDraw.Draw(frame_draw)

        if boxes is not None:
            for box in boxes:
                draw.ellipse(box, fill=255)

        if blur:
            blurred = frame_draw.filter(ImageFilter.GaussianBlur(52))
            frame_draw.paste(blurred, mask=mask)

        video_tracked.write(cv2.cvtColor(np.array(frame_draw), cv2.COLOR_RGB2BGR))

        success, image = videocap.read()

    video_tracked.release()


def model_inf(model, img):
    faces = model.get(img)
    return faces


if __name__ == "__main__":

    SCRFD_detector(args)
