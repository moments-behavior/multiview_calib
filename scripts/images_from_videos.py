import cv2 as cv
import numpy as np
import os
import argparse
import pickle
import logging
from multiview_calib import utils
import colorsys
import concurrent.futures
from functools import partial
import matplotlib

matplotlib.use("Agg")  # Set the backend to 'Agg'
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser()
parser.add_argument("--config", "-c", type=str, required=True)
parser.add_argument("--verbose", "-v", action="store_true")

args = parser.parse_args()

config_file = args.config
root_folder = os.path.dirname(config_file)
config = utils.json_read(config_file)
vid_path = config["vid_path"]
img_path = config["img_path"]
cam_names = config["cam_ordered"]
charuco_setup = config["charuco_setup"]
output_path = img_path
if_serial = args.verbose

utils.mkdir(output_path)
utils.config_logger(os.path.join(output_path, "image_extraction.log"))

images = []


def extract_images_from_videos(vidname,cam,output_path, save_from, skip_frames=5):
    
    logging.info(f"Extracting images for camera {cam}; previous save count: {save_from}")
        
    cap = cv.VideoCapture(vidname)
    frame_count = 0
    frames_saved = 0
    images = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            # logging.info(f"End of video {vidname} for camera {cam}")
            break
            
        if frame_count % skip_frames == 0:
            frames_saved +=1
            # print(f"Saving frame {save_count} for camera {cam} to {output_path}")
            cv.imwrite(f"{output_path}/{cam}_{save_from + frames_saved + 1}.jpg", frame)
        frame_count += 1
    return frames_saved 

save_count = np.zeros(len(cam_names), dtype=int) - 1
for session_names in os.listdir(vid_path):
    print(f"Processing session: {session_names}")
    num_workers = 16
    
    for idx,cam in enumerate(cam_names):
        vidname = os.path.join(vid_path, session_names, "Cam" + cam + ".mp4")
        if not os.path.exists(vidname):
            logging.warning(f"Video file {vidname} does not exist.")
            continue
        print(f"prevously saved count for camera {cam}: {save_count[idx]}")
        num_frames_saved = extract_images_from_videos(vidname, 
                                                        cam, 
                                                        output_path, 
                                                        save_count[idx], 
                                                        skip_frames=10)

        save_count[idx] += num_frames_saved
        # exit()
        
        logging.info(f"Extracted {save_count[idx]+1} images for camera {cam} from video {session_names}")
        
    logging.info(f"Total images extracted: {save_count}")
    
    # with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
    #     partial_func = partial(
    #         extract_images_from_videos,
    #         charuco_setup=charuco_setup,
    #         output_path=output_path,
    #         verbose=False,
    #     )
        
    #     futures = [
    #         executor.submit(partial_func, cam_name, images)
    #         for cam_name, images in zip(cam_names, images_all_cams)
    #     ]
    
#     _, extension = os.path.splitext(f)
#     if extension.lower() in [".jpg", ".jpeg", ".bmp", ".tiff", ".png", ".gif"]:
#         images.append(f)
# if len(images) == 0:
#     logging.info("No images found.")