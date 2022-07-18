# MV_FaceBlurring_360

A repository for 360 Face blurring. 

# Folder description

* data: We use the pre-trained models for inference. Therefore we do not use any dataset for training. The data for inference is a combination of several 360 videos provided by Vragment or online. The data are stored in the [Gdrive](https://drive.google.com/drive/folders/1cNpIMtLp9wkfduL0Bnx-9vVJIwjhW6_K)

* src: The final scripts used for face blurring

# Video processing

## MTCNN

Reference: https://arxiv.org/abs/1604.02878 



### installation
* conda create -n face_detection_py3_8 python=3.8

    For Python 3.10 was taking too long because of some dependences

* conda activate face_detection_py3_8

* facenet-pytorch: pip install facenet-pytorch (for mtcnn) 

* conda install -c conda-forge opencv 

* mmcv 1.3.15: pip3 install mmcv (if you have CUDA: conda install -c esri mmcv-full)

After this pipeline there was a mismatch between torch and torchvision (**probably cased by conda install -c seri mmcv-full**), which gave the following error:

`Couldn't load custom C++ ops. This can happen if your PyTorch and torchvision versions are incompatible, or if you had errors while compiling torchvision from source. For further information on the compatible versions, check https://github.com/pytorch/vision#installation for the compatibility matrix. Please check your PyTorch version with torch.__version__ and your torchvision version with torchvision.__version__ and verify if they are compatible, and if not please reinstall torchvision so that it matches your PyTorch install.`

So just upgrade/downgrade torchvision. For example  conda install -c pytorch torchvision==0.9.1

#### or
* Create the environment from the face_blurring_env.yaml file: conda env create -f face_blurring_env.yaml

* Activate the new environment: conda activate face_detection_py3_8

* Verify that the new environment was installed correctly: conda env list


### Experiments

Script has 2 input arguments:
* file_path: path where the video is stored
* filename: name of the specific video example

python face_detector_MTCNN.py --file_path "/home/nlazaridi/Projects/MediaVerse/360-processing/face_detection/data/VRAG_videos" --filename "Timelapse"  

