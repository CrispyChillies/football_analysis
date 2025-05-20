from ultralytics import YOLO
import supervision as sv


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

    def get_object_tracks(self, frames):
        detections = self.detect_frames(frames)

        for frame_nums, detection in enumerate(detections):
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

            print(detection_supervision)
            break
