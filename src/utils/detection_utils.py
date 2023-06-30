from dataclasses import dataclass
from typing import Generator
from typing import List
import logging
import os

from yolox.tracker.byte_tracker import BYTETracker, STrack
from onemetric.cv.utils.iou import box_iou_batch
from ultralytics import YOLO
from PIL import Image
import numpy as np
import supervision
import cv2


@dataclass(frozen=True)
class BYTETrackerArgs:
    track_thresh: float = 0.25
    track_buffer: int = 30
    match_thresh: float = 0.8
    aspect_ratio_thresh: float = 3.0
    min_box_area: float = 1.0
    mot20: bool = False

# converts Detections into format that can be consumed by match_detections_with_tracks function
def detections2boxes(detections: supervision.Detections) -> np.ndarray:
    return np.hstack((
        detections.xyxy,
        detections.confidence[:, np.newaxis]
    ))


# converts List[STrack] into format that can be consumed by match_detections_with_tracks function
def tracks2boxes(tracks: List[STrack]) -> np.ndarray:
    return np.array([
        track.tlbr
        for track
        in tracks
    ], dtype=float)


# matches our bounding boxes with predictions
def match_detections_with_tracks(detections: supervision.Detections, tracks: List[STrack]) -> supervision.Detections:
    if not np.any(detections.xyxy) or len(tracks) == 0:
        return np.empty((0,))

    tracks_boxes = tracks2boxes(tracks=tracks)
    iou = box_iou_batch(tracks_boxes, detections.xyxy)
    track2detection = np.argmax(iou, axis=1)

    tracker_ids = [None] * len(detections)

    for tracker_index, detection_index in enumerate(track2detection):
        if iou[tracker_index, detection_index] != 0:
            tracker_ids[detection_index] = tracks[tracker_index].track_id

    return tracker_ids

def get_video_frames_generator(source_path: str) -> Generator[np.ndarray, None, None]:
    video = cv2.VideoCapture(source_path)
    if not video.isOpened():
        raise Exception(f"Could not open video at {source_path}")
    success, frame = video.read()
    while success:
        yield Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        success, frame = video.read()
    video.release()

def detect_and_track(source_video_path, output_path, zone_coords, model_path="yolov8s.pt"):
    TARGET_VIDEO_PATH = f"{output_path}/labels.mp4"
    target_video_path = os.path.normpath(TARGET_VIDEO_PATH)
    model = YOLO(model_path)
    model.fuse()
    CLASS_NAMES_DICT = model.model.names
    logging.info(CLASS_NAMES_DICT)
    
    colors = supervision.ColorPalette.default()
    video_info = supervision.VideoInfo.from_video_path(source_video_path)

    byte_tracker = BYTETracker(BYTETrackerArgs())
    generator = supervision.get_video_frames_generator(source_video_path)
    
    polygons = [
        np.array(zone_coords, np.int32)
    ]

    zones = [
        supervision.PolygonZone(
            polygon=polygon,
            frame_resolution_wh=video_info.resolution_wh
        )
        for polygon
        in polygons
    ]

    zone_annotators = [
        supervision.PolygonZoneAnnotator(
            zone=zone,
            color=colors.by_idx(index),
            thickness=4,
            text_thickness=4,
            text_scale=2
        )
        for index, zone
        in enumerate(zones)
    ]

    box_annotators = [
        supervision.BoxAnnotator(
            color=colors.by_idx(index),
            thickness=2,
            text_thickness=2,
            text_scale=1
            )
        for index
        in range(len(polygons))
    ]
        
    with supervision.VideoSink(target_video_path, video_info) as sink:
        coords = []
        frame_ids = []
        tracker_ids = []

        for frame_id, frame in enumerate(generator):
            try:
                results = model(frame)[0]

                detections = supervision.Detections(
                    xyxy=results.boxes.xyxy.cpu().numpy(),
                    confidence=results.boxes.conf.cpu().numpy(),
                    class_id=results.boxes.cls.cpu().numpy().astype(int)
                )

                detections = detections[(detections.confidence > .7) & ((detections.class_id == 0) | (detections.class_id == 2))]

                tracks = byte_tracker.update(
                    output_results=detections2boxes(detections=detections),
                    img_info=frame.shape,
                    img_size=frame.shape
                )

                tracker_id = match_detections_with_tracks(
                    detections=detections, tracks=tracks
                )

                detections.tracker_id = np.array(tracker_id)

                for zone, zone_annotator, box_annotator in zip(zones, zone_annotators, box_annotators):
                    mask = zone.trigger(detections=detections)
                    detections = detections[mask]
        
                    labels = [
                        f"#{tracker_id} {CLASS_NAMES_DICT[class_id]} {confidence:0.2f}"
                        for _, _, confidence, class_id, tracker_id in detections
                    ]
                    
                    for x, id in enumerate(detections.tracker_id):
                        x1, y1, x2, y2 = detections.xyxy[x].astype(int)
                        
                        frame_ids.append(frame_id)
                        coords.append([x1, y1, x2, y2])
                        tracker_ids.append(id)

                    frame = box_annotator.annotate(scene=frame, detections=detections, labels=labels)
                    frame = zone_annotator.annotate(scene=frame)

            except Exception as exception:
                logging.error("ERRORL %s", exception)
                pass
            sink.write_frame(frame)
    return frame_ids, coords, tracker_ids