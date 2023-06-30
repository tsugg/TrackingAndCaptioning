import shutil
import glob
import os


def make_dir(folder=""):
    if os.path.exists(folder):
        shutil.rmtree(folder)
        
    if not os.path.exists(folder):
        os.mkdir(folder)
       
def get_source_names(source_file, source_folder, output_folder):
    if source_file is None:
        path = os.path.normpath(source_folder)
        source_video_paths = glob.glob(f"{path}/*.mp4")
        source_video_paths = [os.path.normpath(i) for i in source_video_paths]
    else:  
        path = os.path.normpath(source_file)
        source_video_paths = [path]
        
    source_folder_paths = ['/'.join(i.split(os.sep)[:-1]) for i in source_video_paths]
    source_file_names = [i.split(os.sep)[-1] for i in source_video_paths]
    source_names = [i.split('.mp4')[0] for i in source_file_names]
    output_path = [f"{output_folder}/{i}" for i in source_names]    
        
    source_names_dict = {
        "source_video_paths": source_video_paths,
        "source_folder_paths": source_folder_paths,
        "source_file_names": source_file_names,
        "source_names": source_names,
        "output_path": output_path
    }
    return source_names_dict

def write_tracks(frame_ids, coords, tracker_ids, source_name, output_path):
    assert (len(frame_ids) == len(coords)) & (len(coords) == len(tracker_ids)), "Lists of tracks must be equal"
    frames_path = f"{output_path}/{source_name}_frames.txt"
    coords_path = f"{output_path}/{source_name}_coords.txt"
    ids_path = f"{output_path}/{source_name}_ids.txt"
    
    paths = [frames_path, coords_path, ids_path]    
    pairs = list(map(lambda x, y:(x,y), [frame_ids, coords, tracker_ids], paths))

    for x, y in pairs:
        file = open(y,'w')
        for item in x:
            file.write(str(item)+"\n")
        file.close()