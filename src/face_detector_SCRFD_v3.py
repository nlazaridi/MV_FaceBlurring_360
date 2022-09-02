from fileinput import filename
import cv2
from os.path import join
import insightface

# from insightface.app import FaceAnalysis
import glob
import onnxruntime
from insightface.app import FaceAnalysis
from insightface.app.mask_renderer import DEFAULT_MP_NAME  # , ensure_available
from insightface.app.face_analysis import ensure_available
from insightface.app.face_analysis import model_zoo
from insightface.app.common import Face
from PIL import Image, ImageDraw, ImageFilter
import numpy as np
from matplotlib import pyplot as plt
import argparse
import os
import os.path as osp


# from ..utils import DEFAULT_MP_NAME, ensure_available

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


class FaceAnalysis:
    def __init__(
        self,
        name=DEFAULT_MP_NAME,
        root="~/.insightface",
        allowed_modules=None,
        **kwargs
    ):
        onnxruntime.set_default_logger_severity(3)
        self.models = {}
        self.model_dir = ensure_available("models", name, root=root)
        onnx_files = glob.glob(osp.join(self.model_dir, "*det_10g.onnx"))
        onnx_files = sorted(onnx_files)
        for onnx_file in onnx_files:
            model = model_zoo.get_model(onnx_file, **kwargs)
            if model is None:
                print("model not recognized:", onnx_file)
            elif allowed_modules is not None and model.taskname not in allowed_modules:
                print("model ignore:", onnx_file, model.taskname)
                del model
            elif model.taskname not in self.models and (
                allowed_modules is None or model.taskname in allowed_modules
            ):
                print(
                    "find model:",
                    onnx_file,
                    model.taskname,
                    model.input_shape,
                    model.input_mean,
                    model.input_std,
                )
                self.models[model.taskname] = model
            else:
                print("duplicated model task type, ignore:", onnx_file, model.taskname)
                del model
        assert "detection" in self.models
        self.det_model = self.models["detection"]

    def prepare(self, ctx_id, det_thresh=0.5, det_size=(640, 640)):
        self.det_thresh = det_thresh
        assert det_size is not None
        print("set det-size:", det_size)
        self.det_size = det_size
        for taskname, model in self.models.items():
            if taskname == "detection":
                model.prepare(ctx_id, input_size=det_size, det_thresh=det_thresh)
            else:
                model.prepare(ctx_id)

    def get(self, img, max_num=0):
        bboxes, kpss = self.det_model.detect(img, max_num=max_num, metric="default")
        if bboxes.shape[0] == 0:
            return []
        ret = []
        for i in range(bboxes.shape[0]):
            bbox = bboxes[i, 0:4]
            det_score = bboxes[i, 4]
            kps = None
            if kpss is not None:
                kps = kpss[i]
            face = Face(bbox=bbox, kps=kps, det_score=det_score)
            for taskname, model in self.models.items():
                if taskname == "detection":
                    continue
                model.get(img, face)
            ret.append(face)
        return ret


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
