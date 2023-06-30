import argparse
import logging
import json
import sys
import os

import numpy as np

HOME = os.getcwd()
sys.path.append(f"{HOME}/ByteTrack")
sys.path.append(f"{HOME}")

import src.utils.general_utils as g_utils
import src.utils.detection_utils as d_utils

def main(args):
    SOURCE_FILE = args.source_file
    SOURCE_FOLDER = args.source_folder
    OUTPUT_PATH = args.output_folder
    output_path = os.path.normpath(OUTPUT_PATH)        

    # Get file names, paths, and json.
    source_names_dict = g_utils.get_source_names(SOURCE_FILE, SOURCE_FOLDER, output_path)
    zone_coords_paths = [f"{source_names_dict['source_folder_paths'][x]}/{i}.json" for x,i in enumerate(source_names_dict["source_names"])]
    logging.info("SOURCE NAMES: %s", source_names_dict)
    logging.info("ZONE COORDS PATHS: %s", zone_coords_paths)

    for x in range(len(source_names_dict["source_names"])): 
        # Make output directory
        g_utils.make_dir(source_names_dict["output_path"][x])
        
        # Load zone coords for zone counting
        with open(zone_coords_paths[x], 'r') as f:
            zone_coords = json.load(f)
        zone_coords = np.asarray([x['coords'] for x in zone_coords.values()], dtype=int).tolist()
        logging.info("ZONE COORDS: %s", zone_coords[x])
        frame_ids, coords, tracker_ids = d_utils.detect_and_track(source_names_dict['source_video_paths'][x],
                                                     source_names_dict['output_path'][x],
                                                     zone_coords[x])
        g_utils.write_tracks(frame_ids, coords, tracker_ids,
                             source_names_dict['source_names'][x], source_names_dict['output_path'][x])
        
        
        
if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO, force=True,
        format='[%(asctime)s] [%(levelname)s] {%(filename)s:%(lineno)d} - %(message)s',
        handlers=[logging.FileHandler("track.log", "w"),
                logging.StreamHandler(sys.stdout)]
    )
    
    
    parser = argparse.ArgumentParser(add_help=True)
    
    parser.add_argument("-s_file", "--source_file", help="Path to source video file",
                        required=False, type=str)
    parser.add_argument("-s_folder", "--source_folder", help="Path to source videos folder",
                        required=False, type=str)
    parser.add_argument("-o", "--output_folder", help="path to output folder",
                        required=True, type=str)
    
    arguments = parser.parse_args()
    
    if vars(arguments)["source_file"] is not None and vars(arguments)["source_folder"] is not None:
        parser.error("Source video must be either a file or a folder path, not both.")
    
    if vars(arguments)["source_file"] is None and vars(arguments)["source_folder"] is None:
        parser.error("Please provide a source video file or folder path.")        
        
    main(arguments)
    
    logging.info('FINISHED')