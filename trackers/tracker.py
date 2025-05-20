from ultralytics import YOLO
import supervision as sv
import os
import pickle


class Tracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()  # type: ignore

    def detect_frames(self, frames):
        batch_size = 20
        detections = []

        for i in range(0, len(frames), batch_size):
            detection_batch = self.model.predict(frames[i : i + batch_size], conf=0.1)
            detections += detection_batch

        return detections

    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):

        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, "rb") as file:
                tracks = pickle.load(file)
                return tracks

        detections = self.detect_frames(frames)

        tracks = {"player": [], "referee": [], "ball": []}

        """Temporal Data
            # Frame 0
            {
                1: {"bbox": [120.5, 340.2, 145.8, 410.6]},  # Player ID 1
                2: {"bbox": [250.3, 360.1, 275.6, 430.5]},  # Player ID 2
                5: {"bbox": [410.7, 380.4, 435.9, 450.8]},  # Player ID 5
                8: {"bbox": [560.2, 390.3, 585.5, 460.7]},  # Player ID 8
            },
            # Frame 1
            {
                1: {"bbox": [122.1, 342.5, 147.4, 412.9]},  # Player ID 1 (moved slightly)
                2: {"bbox": [255.8, 365.2, 281.1, 435.6]},  # Player ID 2 (moved slightly)
                5: {"bbox": [415.4, 382.7, 440.6, 453.1]},  # Player ID 5 (moved slightly)
                8: {"bbox": [565.9, 392.6, 591.2, 463.0]},  # Player ID 8 (moved slightly)
                10: {"bbox": [620.3, 410.5, 645.6, 480.9]},  # New player appeared (ID 10)
            },
        """

        for frame_num, detection in enumerate(detections):
            cls_names = detection.names
            cls_names_inv = {v: k for k, v in cls_names.items()}

            # Convert to supervision format
            detection_supervision = sv.Detections.from_ultralytics(detection)  # type: ignore

            # Convert Goalkeeper to player object
            for object_ind, class_id in enumerate(detection_supervision.class_id):  # type: ignore
                if cls_names[class_id] == "goalkeeper":
                    detection_supervision.class_id[object_ind] = cls_names_inv["player"]  # type: ignore

            detection_with_tracks = self.tracker.update_with_detections(
                detection_supervision
            )

            tracks["player"].append({})
            tracks["referee"].append({})
            tracks["ball"].append({})

            for frame_detection in detection_with_tracks:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[2]
                track_id = frame_detection[4]

                if cls_id == cls_names_inv["player"]:
                    tracks["player"][frame_num][track_id] = {"bbox": bbox}

                if cls_id == cls_names_inv["referee"]:
                    tracks["referee"][frame_num][track_id] = {"bbox": bbox}

                # if cls_id == cls_names_inv["ball"]:
                #     tracks["ball"][frame_num][track_id] = {"bbox": bbox}

            for frame_detection in detection_supervision:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[2]

                # Instead of assigning the track_id -> Hardcode the number "1" because there only 1 ball
                if cls_id == cls_names_inv["ball"]:
                    tracks["ball"][frame_num][1] = {"bbox": bbox}

            if stub_path is not None:
                with open(stub_path, "wb") as file:
                    pickle.dump(tracks, file)

        return tracks
