import argparse
import logging
import glob
import json
import sys
import os

from decord import VideoReader
from decord import cpu, gpu
import cv2

HOME = os.getcwd()
sys.path.append(f"{HOME}")

import src.utils.general_utils as g_utils


def main(args):
    SOURCE_VIDEO_PATH = args.source_file
    vr = VideoReader(SOURCE_VIDEO_PATH, ctx=cpu(0))

    INPUT_PATH = args.input_path
    input_path = os.path.normpath(INPUT_PATH)
    
    frames_paths = glob.glob(f"{input_path}/*/*_frames.txt")
    tracks_paths = glob.glob(f"{input_path}/*/*_ids.txt")
    coords_paths = glob.glob(f"{input_path}/*/*_coords.txt")

    source_file_names = [i.split(os.sep)[-1] for i in coords_paths]
    source_names = [i.split('_coords.txt')[0] for i in source_file_names]
    output_path = [f"{input_path}/{i}" for i in source_names]
    
    logging.info(source_names)
    logging.info(output_path)
    
    
    inds_to_drop = []
    for x in range(len(source_names)):
        
        target_path = f"{output_path[x]}/tracks"
        logging.info(target_path)
        g_utils.make_dir(target_path)
        
        with open(frames_paths[x], 'r') as f:
            frames = [int(i) for i in f.read().split()]
        
        tracks = []
        with open(tracks_paths[x], 'r') as f:
            for y, i in enumerate(f.read().split()):
                if i == 'None':
                    inds_to_drop.append(y)
                    tracks.append(None)
                else:
                    tracks.append(int(i))

        with open(coords_paths[x], 'r') as f:
            coords = [json.loads(i) for i in f.read().splitlines()]
            
        for track_x, track_id in enumerate(tracks):
            if track_id is None:
                continue
            else:
                frame_id = frames[track_x]
                logging.info("FRAME ID: %s", frame_id)
                x1, y1, x2, y2 = coords[track_x]
                save_path = f"{target_path}/{track_id}"
                os.makedirs(save_path, exist_ok=True)
                
                this_frame = vr.get_batch([frame_id]).asnumpy()[0]
                this_frame_crop = cv2.cvtColor(this_frame[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)
                cv2.imwrite(f"{save_path}/{frame_id}.png", this_frame_crop)
            

if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO, force=True,
        format='[%(asctime)s] [%(levelname)s] {%(filename)s:%(lineno)d} - %(message)s',
        handlers=[logging.FileHandler("crop_tracks.log", "w"),
                logging.StreamHandler(sys.stdout)]
    )
    
    
    parser = argparse.ArgumentParser(add_help=True)
    
    parser.add_argument("-s_file", "--source_file", help="Path to source video file",
                        required=True, type=str)
    parser.add_argument("-i_path", "--input_path", help="Path to tracks",
                        required=True, type=str)
    
    arguments = parser.parse_args()   
        
    main(arguments)
    
    logging.info('FINISHED')