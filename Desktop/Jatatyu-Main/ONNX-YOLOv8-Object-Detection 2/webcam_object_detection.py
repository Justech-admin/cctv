import cv2
import os
import time
import threading
from yolov8 import YOLOv8
from threaded_camera import ThreadedCamera
from playsound import playsound


# # Create recordings directory if not exists
# os.makedirs("recordings", exist_ok=True)

# Camera source
choice = input("Select input:\n1. Webcam\n2. RTSP (Hikvision)\n> ").strip()
if choice == '1':
    camera_source = 0
elif choice == '2':
    rtsp_url = input("Enter RTSP URL: ").strip()
    if '?tcp' not in rtsp_url:
        rtsp_url += '?tcp'
    camera_source = rtsp_url
else:
    print("Invalid choice")
    exit()

# Initialize camera and model
cam = ThreadedCamera(camera_source)
model_path = "models/yolov8n.onnx"
yolov8_detector = YOLOv8(model_path, conf_thres=0.3, iou_thres=0.5)

cv2.namedWindow("Detected Objects", cv2.WINDOW_NORMAL)

prev_time = time.time()

# # Alarm function (non-blocking)
# def beep_alarm():
#     winsound.Beep(1000, 300)  # 1000Hz, 300ms

def play_beep():
    try:
        playsound("default_beep.wav", block=False)
    except Exception as e:
        print(f"Error playing beep: {e}")

# # Recorder setup
# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# recording = False
# out = None
# last_detection_time = 0
# record_duration = 5  # seconds after last detection to continue recording

while True:
    ret, frame = cam.read()
    if not ret:
        continue

    current_time = time.time()
    elapsed = current_time - prev_time
    fps = 1.0 / elapsed if elapsed > 0 else 0.0
    prev_time = current_time

    # Inference
    boxes, scores, class_ids = yolov8_detector(frame)
    combined_img = yolov8_detector.draw_detections(frame)

    # Put FPS
    cv2.putText(combined_img, f"FPS: {fps:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Human detection (class_id 0 for 'person' in COCO)
    if any(cls == 0 for cls in class_ids):
        # Alarm
        threading.Thread(target=play_beep, daemon=True).start()

        # # Start recording if not started
        # if not recording:
        #     filename = f"recordings/detection_{time.strftime('%Y%m%d_%H%M%S')}.avi"
        #     out = cv2.VideoWriter(filename, fourcc, 20.0, (frame.shape[1], frame.shape[0]))
        #     recording = True
        #     print(f"Recording started: {filename}")

        # last_detection_time = current_time

    # # Handle recording
    # if recording:
    #     out.write(frame)
    #     if current_time - last_detection_time > record_duration:
    #         print("Recording stopped.")
    #         out.release()
    #         recording = False

    # Display
    cv2.imshow("Detected Objects", combined_img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# # Cleanup
# if recording:
#     out.release()
cam.stop()
cv2.destroyAllWindows()
