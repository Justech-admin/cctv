import cv2
import threading

class ThreadedCamera:
    def __init__(self, src=0, resize_to=(640, 640)):
        self.src = src
        self.resize_to = resize_to

        # Use FFmpeg explicitly for RTSP
        if isinstance(src, str) and src.startswith("rtsp://"):
            self.cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG)
        else:
            self.cap = cv2.VideoCapture(src)

        # Drop frames by keeping buffer size small
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.ret, self.frame = self.cap.read()
        self.lock = threading.Lock()
        self.running = True

        self.thread = threading.Thread(target=self.update, daemon=True)
        self.thread.start()

    def update(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                continue
            frame = cv2.resize(frame, self.resize_to)
            with self.lock:
                self.ret, self.frame = ret, frame

    def read(self):
        with self.lock:
            return self.ret, self.frame

    def stop(self):
        self.running = False
        self.thread.join()
        self.cap.release()
