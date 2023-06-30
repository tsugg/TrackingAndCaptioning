<div align="center">

# Tracking With ByteTrack
Using YOLOv8, ByteTrack, and Supervision to detect people from surveillance footage

</div>

## Requirements
* CUDA 11.7
* Python 3.10


## Set up environment
```sh
python3 -m venv venv
```

```sh
source venv/bin/activate
```

## Download Repo
```sh 
git clone https://github.com/tsugg/TrackingAndCaptioning.git
```

```sh
cd TrackingAndCaptioning
```

## Install ByteTrack from source
I forked the repo since there are some issues with dependencies in the requirements.txt
```sh 
git clone https://github.com/tsugg/ByteTrack.git
```

```sh
cd ByteTrack
```

```sh
pip install -r requirements.txt
```

```sh
python setup.py -q develop
```

## Install other dependencies
```sh
cd ..
```

```sh
pip install -r requirements.txt
```

## Usage with examples from input/
This script extracts tracked detections from the zone provided in inputs/people_car.json
```sh
python src/scripts/track.py --source_folder input/ --output_folder output/
```
This script crops all the detections and saves them into folders by track id
```sh
python src/scripts/crop_tracks.py --source_file input/people_car.mp4 --input_path output/
```