from ultralytics import YOLO
import cv2

# Load YOLO model
model = YOLO("yolo-Weights/yolov8n.pt")

# COCO classes
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
              "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet",
              "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster",
              "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]

allowed_objects = {"cell phone", "cup", "bottle", "pen", "remote", "wallet", "book"}

def detect_objects(img):
    objects = []
    small_frame = cv2.resize(img, (320, 240))
    results = model(small_frame, stream=True)

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            class_name = classNames[cls]
            x1, y1, x2, y2 = map(int, box.xyxy[0] * 2)
            if class_name in allowed_objects:
                objects.append((class_name, (x1, y1, x2, y2)))
    return objects
