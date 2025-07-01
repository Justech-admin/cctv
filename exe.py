import sys
import cv2
import numpy as np
from ultralytics import YOLO
import time
import pygame
import os
from datetime import datetime
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                            QLabel, QPushButton, QComboBox, QLineEdit, QGroupBox, QMessageBox,
                            QGridLayout, QScrollArea)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QThread, QMutex
from PyQt5.QtGui import QImage, QPixmap

class CameraThread(QThread):
    change_pixmap_signal = pyqtSignal(int, QImage)  # camera_index, QImage
    detection_signal = pyqtSignal(int, bool)  # camera_index, detection_status
    status_signal = pyqtSignal(int, str)  # camera_index, status_message

    def __init__(self, camera_index, camera_url, parent=None):
        super().__init__(parent)
        self.camera_index = camera_index
        self.camera_url = camera_url
        self.running = True
        self.model = YOLO("yolov8n.pt")
        self.alarm_active = False
        self.last_alarm_time = 0
        self.alarm_duration = 2  # seconds
        self.alarm_cooldown = 1  # seconds
        
        # Video recording variables
        self.video_writer = None
        self.recording = False
        self.output_folder = f"detection_recordings/camera_{self.camera_index}"
        os.makedirs(self.output_folder, exist_ok=True)
        self.frame_width = 0
        self.frame_height = 0

    def run(self):
        cap = self.test_camera_connection(self.camera_url)
        if not cap:
            self.status_signal.emit(self.camera_index, "Failed to connect to camera")
            return

        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            cap.set(cv2.CAP_PROP_FPS, 15)
            
            # Get frame size from the capture
            self.frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            while self.running:
                ret, frame = cap.read()
                if not ret:
                    self.status_signal.emit(self.camera_index, "Error reading frame")
                    break

                # Run detection (only for person class - 0)
                results = self.model(frame, classes=[0], verbose=False)
                
                # Person detection flag
                person_detected = len(results[0].boxes) > 0
                self.detection_signal.emit(self.camera_index, person_detected)

                # Start/stop recording based on detection
                if person_detected:
                    if not self.recording:
                        self.start_recording()
                    self.recording = True
                else:
                    if self.recording:
                        self.stop_recording()
                    self.recording = False
                    
                # Write frame if recording
                if self.recording and self.video_writer is not None:
                    self.video_writer.write(frame)

                # Draw bounding boxes
                for box in results[0].boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                    conf = float(box.conf[0])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    label = f"Person: {conf:.2f}"
                    cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                # Alarm logic
                current_time = time.time()
                if person_detected:
                    if not self.alarm_active and (current_time - self.last_alarm_time > self.alarm_cooldown):
                        pygame.mixer.music.play(-1)  # -1 makes it loop
                        self.alarm_active = True
                        self.last_alarm_time = current_time
                    last_alarm_time = current_time
                else:
                    if self.alarm_active and (current_time - self.last_alarm_time > self.alarm_duration):
                        pygame.mixer.music.stop()
                        self.alarm_active = False

                # Add status text
                status_text = f"Cam {self.camera_index} - ALARM: {'ON' if self.alarm_active else 'OFF'}"
                status_color = (0, 0, 255) if self.alarm_active else (0, 255, 0)
                cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
                
                # Add recording status
                if self.recording:
                    cv2.putText(frame, "RECORDING", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                # Convert to QImage
                rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_image.shape
                bytes_per_line = ch * w
                qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
                self.change_pixmap_signal.emit(self.camera_index, qt_image)

        finally:
            if self.recording:
                self.stop_recording()
            cap.release()
            pygame.mixer.music.stop()

    def stop(self):
        self.running = False
        self.wait()

    def test_camera_connection(self, urls):
        if isinstance(urls, list):
            for url in urls:
                try:
                    # Clean up URL string
                    url = url.strip()
                    if url.startswith('"') and url.endswith('"'):
                        url = url[1:-1]
                    
                    # Try with FFMPEG first
                    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
                    if cap.isOpened():
                        self.status_signal.emit(self.camera_index, f"Connected to: {url}")
                        return cap
                    cap.release()
                    
                    # Fallback to default backend
                    cap = cv2.VideoCapture(url)
                    if cap.isOpened():
                        self.status_signal.emit(self.camera_index, f"Connected to: {url} (fallback)")
                        return cap
                    cap.release()
                except Exception as e:
                    self.status_signal.emit(self.camera_index, f"Connection error: {str(e)}")
                    continue
        else:
            url = urls.strip()
            if url.startswith('"') and url.endswith('"'):
                url = url[1:-1]
                
            cap = cv2.VideoCapture(url)
            if cap.isOpened():
                self.status_signal.emit(self.camera_index, "Camera connected")
                return cap
        return None
        
    def start_recording(self):
        """Initialize video writer for new recording"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(self.output_folder, f"detection_{timestamp}.mp4")
        
        # Use MP4V codec for better compatibility
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = 15  # Match the capture FPS
        
        self.video_writer = cv2.VideoWriter(output_path, fourcc, fps, (self.frame_width, self.frame_height))
        self.status_signal.emit(self.camera_index, f"Started recording: {output_path}")
        
    def stop_recording(self):
        """Finalize current recording"""
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None
            self.status_signal.emit(self.camera_index, "Recording saved")

class CameraSetupDialog(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Camera Setup")
        self.setFixedSize(800, 600)
        self.camera_count = 1
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        # Scroll area for multiple cameras
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        self.scroll_content = QWidget()
        self.scroll_layout = QVBoxLayout(self.scroll_content)

        # Camera count selection
        count_group = QGroupBox("Number of Cameras")
        count_layout = QHBoxLayout()
        self.count_combo = QComboBox()
        self.count_combo.addItems([str(i) for i in range(1, 5)])  # Support up to 4 cameras
        self.count_combo.currentIndexChanged.connect(self.update_camera_count)
        count_layout.addWidget(QLabel("Number of cameras:"))
        count_layout.addWidget(self.count_combo)
        count_layout.addStretch()
        count_group.setLayout(count_layout)
        self.scroll_layout.addWidget(count_group)

        # Camera setup widgets
        self.camera_widgets = []
        self.update_camera_count()

        scroll.setWidget(self.scroll_content)
        layout.addWidget(scroll)

        # Buttons
        button_layout = QHBoxLayout()
        self.ok_button = QPushButton("OK")
        self.cancel_button = QPushButton("Cancel")
        button_layout.addWidget(self.ok_button)
        button_layout.addWidget(self.cancel_button)

        layout.addLayout(button_layout)
        self.setLayout(layout)

        # Connect signals
        self.ok_button.clicked.connect(self.accept)
        self.cancel_button.clicked.connect(self.reject)

    def update_camera_count(self):
        new_count = int(self.count_combo.currentText())
        
        # Remove extra widgets if decreasing count
        while len(self.camera_widgets) > new_count:
            widget = self.camera_widgets.pop()
            widget.setParent(None)
            widget.deleteLater()
        
        # Add new widgets if increasing count
        while len(self.camera_widgets) < new_count:
            cam_idx = len(self.camera_widgets) + 1
            widget = self.create_camera_widget(cam_idx)
            self.camera_widgets.append(widget)
            self.scroll_layout.addWidget(widget)

    def create_camera_widget(self, cam_idx):
        group = QGroupBox(f"Camera {cam_idx}")
        layout = QVBoxLayout()

        # Camera type selection
        type_layout = QHBoxLayout()
        type_label = QLabel("Type:")
        type_combo = QComboBox()
        type_combo.addItems(["RTSP Camera (IP Camera)", "HTTP Camera (Web Server Stream)"])
        type_layout.addWidget(type_label)
        type_layout.addWidget(type_combo)
        layout.addLayout(type_layout)

        # Connection details
        ip_label = QLabel("Camera Address:")
        ip_input = QLineEdit()
        layout.addWidget(ip_label)
        layout.addWidget(ip_input)

        # Authentication
        auth_frame = QWidget()
        auth_layout = QHBoxLayout()
        username_input = QLineEdit()
        username_input.setPlaceholderText("Username")
        password_input = QLineEdit()
        password_input.setPlaceholderText("Password")
        password_input.setEchoMode(QLineEdit.Password)
        auth_layout.addWidget(username_input)
        auth_layout.addWidget(password_input)
        auth_frame.setLayout(auth_layout)
        layout.addWidget(auth_frame)

        # Port
        port_label = QLabel("Port:")
        port_input = QLineEdit()
        layout.addWidget(port_label)
        layout.addWidget(port_input)

        # Path options (for RTSP/HTTP)
        path_frame = QWidget()
        path_layout = QVBoxLayout()
        path_label = QLabel("Path:")
        path_input = QLineEdit()
        path_layout.addWidget(path_label)
        path_layout.addWidget(path_input)
        path_frame.setLayout(path_layout)
        layout.addWidget(path_frame)

        # Store references to all widgets
        group.cam_idx = cam_idx
        group.type_combo = type_combo
        group.ip_input = ip_input
        group.username_input = username_input
        group.password_input = password_input
        group.port_input = port_input
        group.path_input = path_input

        # Connect type change signal
        type_combo.currentIndexChanged.connect(lambda idx, g=group: self.update_camera_type(g))

        # Set initial state
        self.update_camera_type(group)

        group.setLayout(layout)
        return group

    def update_camera_type(self, group):
        cam_type = group.type_combo.currentIndex()
        
        if cam_type == 0:  # RTSP
            group.port_input.setText("554")
            group.path_input.setPlaceholderText("e.g. /live.sdp or /streaming/channels/1")
        else:  # HTTP
            group.port_input.setText("80")
            group.path_input.setPlaceholderText("e.g. /video_feed")

    def accept(self):
        camera_urls = []
        
        for widget in self.camera_widgets:
            cam_idx = widget.cam_idx
            cam_type = widget.type_combo.currentIndex()
            
            ip = widget.ip_input.text().strip()
            if not ip:
                QMessageBox.warning(self, "Warning", f"Please enter a camera address for Camera {cam_idx}")
                return

            username = widget.username_input.text().strip()
            password = widget.password_input.text().strip()
            port = widget.port_input.text().strip()
            path = widget.path_input.text().strip()

            if cam_type == 0:  # RTSP
                if username and password:
                    auth = f"{username}:{password}@"
                else:
                    auth = ""

                if not port:
                    port = "554"

                if path:
                    if not path.startswith("/"):
                        path = "/" + path
                    urls = [f"rtsp://{auth}{ip}:{port}{path}"]
                else:
                    urls = [
                        f"rtsp://{auth}{ip}:{port}/cam/realmonitor?channel=1&subtype=0",
                        f"rtsp://{auth}{ip}:{port}/h264/ch1/main/av_stream",
                        f"rtsp://{auth}{ip}:{port}/videoMain",
                        f"rtsp://{auth}{ip}:{port}/live/ch01",
                        f"rtsp://{auth}{ip}:{port}/live.sdp",
                        f"rtsp://{auth}{ip}:{port}/streaming/channels/1",
                        f"rtsp://{ip}:{port}/live/ch01",
                    ]
            else:  # HTTP
                if not port:
                    port = "80"
                if not path:
                    path = "/video_feed"

                urls = [
                    f"http://{ip}:{port}{path}",
                    f"http://{ip}:{port}/videostream.cgi",
                    f"http://{ip}:{port}/mjpegfeed",
                    f"http://{ip}:{port}/stream",
                    f"http://{ip}:{port}/video",
                ]

            camera_urls.append((cam_idx, urls))

        # Sort by camera index and extract just the URLs
        camera_urls.sort(key=lambda x: x[0])
        self.parent().start_detection([urls for idx, urls in camera_urls])
        self.close()

    def reject(self):
        self.close()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Multi-Camera Person Detection System")
        self.setGeometry(100, 100, 1200, 800)
   
        # Initialize pygame for sound
        pygame.mixer.init()
        try:
            pygame.mixer.music.load("alarm.wav")
        except:
            print("Warning: Could not load alarm.wav - using simple beep instead")
            beep = pygame.mixer.Sound(buffer=bytearray([127] * 1000 + [0] * 1000) * 44100)
            pygame.mixer.music = beep
        
        self.init_ui()
        
        # Video threads
        self.camera_threads = []
        self.detection_status = {}  # Track detection status per camera
        self.alarm_active = False
        self.video_labels = []
        self.status_labels = []

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QVBoxLayout()
        
        # Video display grid
        self.video_grid = QGridLayout()
        self.video_grid.setSpacing(10)
        
        # Scroll area for video grid
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_content = QWidget()
        scroll_content.setLayout(self.video_grid)
        scroll.setWidget(scroll_content)
        
        main_layout.addWidget(scroll)
        
        # Status bar
        self.status_label = QLabel("Ready")
        self.status_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(self.status_label)
        
        # Detection status
        self.detection_label = QLabel("No person detected")
        self.detection_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(self.detection_label)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        self.setup_button = QPushButton("Setup Cameras")
        self.setup_button.clicked.connect(self.show_camera_setup)
        button_layout.addWidget(self.setup_button)
        
        self.start_button = QPushButton("Start Detection")
        self.start_button.clicked.connect(self.start_detection)
        button_layout.addWidget(self.start_button)
        
        self.stop_button = QPushButton("Stop Detection")
        self.stop_button.clicked.connect(self.stop_detection)
        self.stop_button.setEnabled(False)
        button_layout.addWidget(self.stop_button)
        
        self.open_folder_button = QPushButton("Open Recordings Folder")
        self.open_folder_button.clicked.connect(self.open_recordings_folder)
        button_layout.addWidget(self.open_folder_button)
        
        main_layout.addLayout(button_layout)
        
        central_widget.setLayout(main_layout)

    def show_camera_setup(self):
        self.setup_dialog = CameraSetupDialog(self)
        self.setup_dialog.show()

    def start_detection(self, camera_urls_list=None):
        if camera_urls_list is None:
            QMessageBox.warning(self, "Warning", "Please setup cameras first")
            return
            
        # Stop any existing threads
        self.stop_detection()
        
        # Clear previous video labels
        for label in self.video_labels:
            label.setParent(None)
        for label in self.status_labels:
            label.setParent(None)
        self.video_labels = []
        self.status_labels = []
        
        # Create a grid layout for the cameras
        cols = 2  # Number of columns in the grid
        for i in range(len(camera_urls_list)):
            row = i // cols
            col = i % cols
            
            # Create video label
            label = QLabel()
            label.setAlignment(Qt.AlignCenter)
            label.setMinimumSize(640, 480)
            label.setStyleSheet("border: 2px solid black;")
            self.video_grid.addWidget(label, row, col)
            self.video_labels.append(label)
            
            # Create status label for this camera
            status_label = QLabel(f"Camera {i+1}: Not connected")
            status_label.setAlignment(Qt.AlignCenter)
            self.video_grid.addWidget(status_label, row, col, 1, 1, Qt.AlignBottom)
            self.status_labels.append(status_label)
            
            # Initialize detection status
            self.detection_status[i] = False
        
        # Create and start camera threads
        self.camera_threads = []
        for i, camera_urls in enumerate(camera_urls_list):
            thread = CameraThread(i, camera_urls)
            thread.change_pixmap_signal.connect(self.update_image)
            thread.detection_signal.connect(self.update_detection_status)
            thread.status_signal.connect(self.update_status)
            self.camera_threads.append(thread)
            thread.start()
        
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.setup_button.setEnabled(False)
        self.open_folder_button.setEnabled(True)

    def stop_detection(self):
        for thread in self.camera_threads:
            thread.stop()
        self.camera_threads = []
        
        # Stop alarm if active
        if self.alarm_active:
            pygame.mixer.music.stop()
            self.alarm_active = False
        
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.setup_button.setEnabled(True)
        self.detection_label.setText("Detection stopped")

    def update_image(self, camera_index, qt_image):
        if camera_index < len(self.video_labels):
            pixmap = QPixmap.fromImage(qt_image)
            self.video_labels[camera_index].setPixmap(pixmap.scaled(
                self.video_labels[camera_index].width(), 
                self.video_labels[camera_index].height(), 
                Qt.KeepAspectRatio))

    def update_detection_status(self, camera_index, detected):
        self.detection_status[camera_index] = detected
        
        # Update overall detection status
        any_detected = any(self.detection_status.values())
        
        if any_detected:
            self.detection_label.setText("PERSON DETECTED!")
            self.detection_label.setStyleSheet("color: red; font-weight: bold;")
            
            # Trigger alarm if not already active
            if not self.alarm_active:
                pygame.mixer.music.play(-1)  # -1 makes it loop
                self.alarm_active = True
        else:
            self.detection_label.setText("No person detected")
            self.detection_label.setStyleSheet("color: black;")
            
            # Stop alarm if active
            if self.alarm_active:
                pygame.mixer.music.stop()
                self.alarm_active = False

    def update_status(self, camera_index, message):
        if camera_index < len(self.status_labels):
            self.status_labels[camera_index].setText(f"Camera {camera_index+1}: {message}")

    def open_recordings_folder(self):
        """Open the recordings folder in the system file explorer"""
        folder_path = os.path.abspath("detection_recordings")
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            
        if sys.platform == 'win32':
            os.startfile(folder_path)
        elif sys.platform == 'darwin':
            import subprocess
            subprocess.Popen(['open', folder_path])
        else:
            import subprocess
            subprocess.Popen(['xdg-open', folder_path])

    def closeEvent(self, event):
        self.stop_detection()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
