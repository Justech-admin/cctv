import cv2
import time
from playsound import playsound
from yolov8 import YOLOv8

# Load YOLOv8m ONNX model
model_path = "models/yolov8n.onnx"
yolo = YOLOv8(model_path, conf_thres=0.3, iou_thres=0.5)

# Use webcam
cap = cv2.VideoCapture(0)
cv2.namedWindow("Person Detection", cv2.WINDOW_NORMAL)

prev_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # FPS calculation
    current_time = time.time()
    fps = 1.0 / (current_time - prev_time) if current_time - prev_time > 0 else 0
    prev_time = current_time

    # Inference
    boxes, scores, class_ids = yolo(frame)
    output = yolo.draw_detections(frame)

    # Show FPS
    cv2.putText(output, f"FPS: {fps:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    

    # Show result
    cv2.imshow("Person Detection", output)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
