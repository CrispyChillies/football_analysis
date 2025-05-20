from ultralytics import YOLO

model = YOLO("yolov8l")
results = model.predict(
    "/media/aaronpham5504/New Volume/Project/FootBall_Analysis/input_video/08fd33_4.mp4",
    save=True,
)

print(results[0])
print("===================================")
for box in results[0].boxes:  # type: ignore
    print(box)
