from utils import save_video, read_video
from trackers import Tracker

import torch


def main():
    video_frames = read_video("input_video/08fd33_4.mp4")

    tracker = Tracker("models/best.pt")

    tracks = tracker.get_object_tracks(video_frames)

    save_video(video_frames, "output_video/output_vide.avi")


if __name__ == "__main__":
    main()

    # model_path = "models/best.pt"
    # model_state_dict = torch.load(model_path)
    # print(model_state_dict["model"])
    # model_info_path = "models.txt"
    # with open(model_info_path, "w") as file:
    #     file.write(model_state_dict["model"])
