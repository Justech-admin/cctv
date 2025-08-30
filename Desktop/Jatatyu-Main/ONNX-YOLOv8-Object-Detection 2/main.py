import sys
import os
import time
import threading
import queue
import cv2
import numpy as np
from PyQt5.QtWidgets import (QApplication, QWidget, QLabel, QLineEdit, QPushButton, 
                             QVBoxLayout, QHBoxLayout, QGridLayout, QMessageBox, 
                             QGroupBox, QSpinBox, QDoubleSpinBox, QCheckBox, QComboBox,
                             QProgressBar, QTextEdit, QTabWidget, QScrollArea, QFrame,
                             QSizePolicy, QSlider, QDialog, QDialogButtonBox, QFormLayout,
                             QMenuBar, QMenu, QMainWindow, QAction, QStackedWidget)
from PyQt5.QtCore import QTimer, QThread, pyqtSignal, Qt, QSize, QPropertyAnimation, QEasingCurve
from PyQt5.QtGui import QPixmap, QImage, QFont, QPainter, QPen, QColor, QLinearGradient, QBrush
from playsound import playsound
from yolov8 import YOLOv8

# ========== CONFIG ==========
TARGET_RESOLUTION = (640, 480)  # Larger resolution for better visibility
MAX_RECONNECT_ATTEMPTS = 5
RECONNECT_DELAY = 2
MODEL_PATH_M = "models/yolov8m.onnx"
MODEL_PATH_N = "models/yolov8n.onnx"
FPS_THRESHOLD = 15

# ========== CAMERA CONFIG DIALOG ==========
class CameraConfigDialog(QDialog):
    def __init__(self, camera_data=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Camera Configuration")
        self.setModal(True)
        self.setFixedSize(400, 300)
        self.setStyleSheet("""
            QDialog {
                background: #2d2d2d;
                border: 1px solid #444;
                border-radius: 5px;
            }
            QLabel {
                color: #eee;
                font-size: 12px;
            }
            QLineEdit, QComboBox {
                background: #333;
                border: 1px solid #444;
                border-radius: 4px;
                padding: 8px;
                color: #eee;
                font-size: 12px;
            }
            QLineEdit:focus, QComboBox:focus {
                border-color: #0078d7;
            }
            QPushButton {
                background: #0078d7;
                border: none;
                border-radius: 4px;
                padding: 8px 16px;
                color: white;
                min-width: 80px;
            }
            QPushButton:hover {
                background: #006cbe;
            }
            QPushButton:pressed {
                background: #005a9e;
            }
            QPushButton:disabled {
                background: #555;
                color: #999;
            }
        """)
        
        self.camera_data = camera_data or {}
        self.init_ui()
        
    def init_ui(self):
        layout = QFormLayout()
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)
        
        # Form fields
        self.name_input = QLineEdit(self.camera_data.get('name', ''))
        self.name_input.setPlaceholderText("Camera Name")
        
        self.ip_input = QLineEdit(self.camera_data.get('ip', ''))
        self.ip_input.setPlaceholderText("192.168.1.100")
        
        self.port_input = QLineEdit(self.camera_data.get('port', '554'))
        self.port_input.setPlaceholderText("554")
        
        self.username_input = QLineEdit(self.camera_data.get('username', ''))
        self.username_input.setPlaceholderText("admin")
        
        self.password_input = QLineEdit(self.camera_data.get('password', ''))
        self.password_input.setPlaceholderText("password")
        self.password_input.setEchoMode(QLineEdit.Password)
        
        # Add fields to form
        layout.addRow("Name:", self.name_input)
        layout.addRow("IP Address:", self.ip_input)
        layout.addRow("Port:", self.port_input)
        layout.addRow("Username:", self.username_input)
        layout.addRow("Password:", self.password_input)
        
        # Button box
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addRow(button_box)
        
        self.setLayout(layout)
    
    def get_camera_data(self):
        return {
            'name': self.name_input.text().strip(),
            'ip': self.ip_input.text().strip(),
            'port': self.port_input.text().strip() or '554',
            'username': self.username_input.text().strip(),
            'password': self.password_input.text().strip()
        }
    
    def validate_data(self):
        data = self.get_camera_data()
        return all([data['name'], data['ip'], data['username'], data['password']])

# ========== WATERMARK WIDGET ==========
class WatermarkWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(640, 480)
        
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Background
        painter.fillRect(self.rect(), QColor(30, 30, 30))
        
        # Main title
        painter.setPen(QColor(200, 200, 200))
        painter.setFont(QFont("Arial", 24, QFont.Bold))
        painter.drawText(self.rect(), Qt.AlignCenter, "JATAYU\nUNMANNED\nTECHNOLOGIES\nSYSTEM")
        
        # Subtitle
        painter.setPen(QColor(150, 150, 150))
        painter.setFont(QFont("Arial", 12))
        painter.drawText(10, self.height() - 30, "Multi-Camera Detection System")

# ========== VIDEO STREAM CLASS ==========
class VideoStream:
    def __init__(self, camera_id, rtsp_url, parent_callback=None):
        self.camera_id = camera_id
        self.rtsp_url = rtsp_url
        self.parent_callback = parent_callback
        self.cap = None
        self.frame_queue = queue.Queue(maxsize=1)
        self.stop_event = threading.Event()
        self.thread = None
        self.last_valid_frame = None
        self.reconnect_attempts = 0
        self.is_connected = False
        self.fps_history = []
        self.current_fps = 0
        self.last_frame_time = time.time()

    def start(self):
        self.thread = threading.Thread(target=self._capture_frames, daemon=True)
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        if self.thread:
            self.thread.join()
        if self.cap and self.cap.isOpened():
            self.cap.release()
        self.is_connected = False

    def _capture_frames(self):
        while not self.stop_event.is_set():
            if not self.cap or not self.cap.isOpened():
                if not self._initialize_capture():
                    time.sleep(RECONNECT_DELAY)
                    continue

            ret, frame = self.cap.read()
            if not ret:
                self._handle_capture_error()
                continue

            self.reconnect_attempts = 0
            self.is_connected = True
            
            # Calculate FPS
            current_time = time.time()
            if current_time - self.last_frame_time > 0:
                fps = 1.0 / (current_time - self.last_frame_time)
                self.fps_history.append(fps)
                if len(self.fps_history) > 10:
                    self.fps_history.pop(0)
                self.current_fps = sum(self.fps_history) / len(self.fps_history)
            self.last_frame_time = current_time

            frame = cv2.resize(frame, TARGET_RESOLUTION)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            if not self.frame_queue.empty():
                try:
                    self.frame_queue.get_nowait()
                except queue.Empty:
                    pass
            self.frame_queue.put(frame)
            self.last_valid_frame = frame

    def _initialize_capture(self):
        if self.reconnect_attempts >= MAX_RECONNECT_ATTEMPTS:
            self.is_connected = False
            return False

        self.reconnect_attempts += 1
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

        # Try default path first
        self.cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
        if self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            self.is_connected = True
            return True

        # If default fails, try fallback path
        fallback_url = self.rtsp_url.replace("/Streaming/Channels/101?tcp", "/cam/realmonitor?channel=1&subtype=0")
        self.cap = cv2.VideoCapture(fallback_url, cv2.CAP_FFMPEG)
        if self.cap.isOpened():
            self.rtsp_url = fallback_url  # Update to working fallback
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            self.is_connected = True
            return True

        self.is_connected = False
        return False


    def _handle_capture_error(self):
        self.is_connected = False
        if self.cap:
            self.cap.release()
        time.sleep(RECONNECT_DELAY)

    def get_frame(self):
        try:
            return self.frame_queue.get_nowait()
        except queue.Empty:
            return self.last_valid_frame

    def get_fps(self):
        return self.current_fps

    def is_camera_connected(self):
        return self.is_connected

# ========== DETECTION THREAD ==========
class DetectionThread(QThread):
    frame_ready = pyqtSignal(str, np.ndarray, int, float)  # camera_id, frame, person_count, fps
    connection_status = pyqtSignal(str, bool)  # camera_id, connected
    model_switched = pyqtSignal(str, str)  # camera_id, model_name

    def __init__(self, camera_configs, detection_settings):
        super().__init__()
        self.camera_configs = camera_configs
        self.detection_settings = detection_settings
        self.streams = {}
        self.yolo_models = {}
        self.current_models = {}
        self.running = False
        self.last_beep_time = 0
        self.beep_delay = 1.0

    def run(self):
        self.running = True
        
        # Initialize streams and models
        for camera_id, config in self.camera_configs.items():
            rtsp_url = f"rtsp://{config['username']}:{config['password']}@{config['ip']}:{config['port']}/Streaming/Channels/101?tcp"
            self.streams[camera_id] = VideoStream(camera_id, rtsp_url)
            self.streams[camera_id].start()
            
            # Initialize with selected model
            self.current_models[camera_id] = self.detection_settings['model_type']
            model_path = MODEL_PATH_M if self.detection_settings['model_type'] == 'yolov8m' else MODEL_PATH_N
            self.yolo_models[camera_id] = YOLOv8(model_path, 
                                               conf_thres=self.detection_settings['conf_threshold'],
                                               iou_thres=self.detection_settings['iou_threshold'])

        time.sleep(2)  # Allow streams to initialize

        while self.running:
            any_person_detected = False
            
            for camera_id, stream in self.streams.items():
                if not stream.is_camera_connected():
                    self.connection_status.emit(camera_id, False)
                    print(f"Camera {camera_id} disconnected")
                    continue
                
                self.connection_status.emit(camera_id, True)
                print(f"Camera {camera_id} connected")
                
                frame = stream.get_frame()
                if frame is None:
                    continue

                # Check if model needs to be switched based on FPS
                current_fps = stream.get_fps()
                if (current_fps < FPS_THRESHOLD and 
                    self.current_models[camera_id] == 'yolov8m' and 
                    self.detection_settings['auto_model_switch']):
                    
                    self.current_models[camera_id] = 'yolov8n'
                    self.yolo_models[camera_id] = YOLOv8(MODEL_PATH_N,
                                                       conf_thres=self.detection_settings['conf_threshold'],
                                                       iou_thres=self.detection_settings['iou_threshold'])
                    self.model_switched.emit(camera_id, 'yolov8n')
                    print(f"Switched camera {camera_id} to yolov8n due to low FPS")

                # Run detection
                boxes, scores, class_ids = self.yolo_models[camera_id](frame)
                
                person_count = 0
                display_frame = cv2.cvtColor(frame.copy(), cv2.COLOR_RGB2BGR)
                
                # Create overlay for transparent bounding boxes
                overlay = display_frame.copy()
                
                for box, score, class_id in zip(boxes, scores, class_ids):
                    if class_id == 0:  # Person class
                        person_count += 1
                        x1, y1, x2, y2 = map(int, box)
                        
                        # Draw filled rectangle with opacity
                        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), -1)
                        
                        # Draw bounding box outline
                        cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        
                        # Modern label background
                        label = f"Person {score:.2f}"
                        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                        cv2.rectangle(display_frame, (x1, y1 - label_size[1] - 10), 
                                    (x1 + label_size[0], y1), (0, 0, 255), -1)
                        cv2.putText(display_frame, label, (x1, y1 - 5),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                # Apply the overlay with transparency
                cv2.addWeighted(overlay, 0.2, display_frame, 0.8, 0, display_frame)

                if person_count > 0:
                    any_person_detected = True

                # Add compact status overlay (only showing processing FPS)
                cv2.putText(display_frame, f"FPS: {current_fps:.1f}", (10, 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                self.frame_ready.emit(camera_id, display_frame, person_count, current_fps)

            # Handle beeping
            if (any_person_detected and 
                (time.time() - self.last_beep_time) > self.beep_delay and
                self.detection_settings['sound_enabled']):
                try:
                    playsound("beep.wav", block=False)
                    self.last_beep_time = time.time()
                except Exception as e:
                    print(f"Could not play sound: {e}")

            time.sleep(0.033)  # ~30 FPS

    def stop(self):
        self.running = False
        for stream in self.streams.values():
            stream.stop()
        self.quit()
        self.wait()

# ========== TASKBAR BUTTON ==========
class TaskbarButton(QPushButton):
    def __init__(self, text, icon=None, parent=None):
        super().__init__(text, parent)
        self.setFixedHeight(40)
        self.setCheckable(True)
        self.setStyleSheet("""
            QPushButton {
                background: transparent;
                color: #ccc;
                border: none;
                border-radius: 0;
                padding: 0 15px;
                font-size: 12px;
                text-align: left;
            }
            QPushButton:hover {
                background: #3a3a3a;
                color: #fff;
            }
            QPushButton:checked {
                background: #0078d7;
                color: #fff;
            }
        """)

# ========== MAIN WINDOW CLASS ==========
class ModernCameraDetectionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("JATAYU Multi-Camera Detection System")
        self.setGeometry(100, 100, 1280, 800)
        
        # Modern styling
        self.setStyleSheet("""
            QMainWindow {
                background: #252525;
                color: #eee;
                font-family: 'Segoe UI', Arial, sans-serif;
            }
            QGroupBox {
                font-size: 14px;
                font-weight: bold;
                border: 1px solid #444;
                border-radius: 4px;
                margin-top: 10px;
                padding-top: 15px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
                color: #0078d7;
            }
            QPushButton {
                background: #0078d7;
                border: none;
                border-radius: 4px;
                padding: 8px 16px;
                color: white;
                min-width: 80px;
            }
            QPushButton:hover {
                background: #006cbe;
            }
            QPushButton:pressed {
                background: #005a9e;
            }
            QPushButton:disabled {
                background: #555;
                color: #999;
            }
            QSlider::groove:horizontal {
                border: 1px solid #444;
                height: 8px;
                background: #333;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #0078d7;
                border: 1px solid #444;
                width: 18px;
                border-radius: 9px;
            }
            QCheckBox::indicator {
                width: 18px;
                height: 18px;
                border-radius: 4px;
                border: 1px solid #444;
            }
            QCheckBox::indicator:checked {
                background: #0078d7;
            }
            QLabel {
                color: #eee;
                font-size: 12px;
            }
            QTextEdit, QLineEdit {
                background: #333;
                border: 1px solid #444;
                border-radius: 4px;
                padding: 8px;
                color: #eee;
                font-size: 12px;
            }
            QTabWidget::pane {
                border: 1px solid #444;
                background: #2d2d2d;
            }
            QTabBar::tab {
                background: #2d2d2d;
                color: #eee;
                padding: 8px 15px;
                border: 1px solid #444;
                border-bottom: none;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
            }
            QTabBar::tab:selected {
                background: #0078d7;
                color: white;
            }
            QTabBar::tab:hover {
                background: #3a3a3a;
            }
        """)
        
        self.cameras = {}  # Store camera configurations
        self.camera_widgets = {}  # Store camera display widgets
        self.detection_thread = None
        self.detection_settings = {
            'conf_threshold': 0.5,
            'iou_threshold': 0.5,
            'auto_model_switch': True,
            'sound_enabled': True,
            'model_type': 'yolov8m'  # Default model
        }
        
        self.init_ui()

    def init_ui(self):
        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        central_widget.setLayout(main_layout)
        
        # Main content area
        self.content_stack = QStackedWidget()
        main_layout.addWidget(self.content_stack, 1)
        
        # Camera display area
        self.camera_display_area = self.create_camera_display_area()
        self.content_stack.addWidget(self.camera_display_area)
        
        # Settings page
        self.settings_page = self.create_settings_page()
        self.content_stack.addWidget(self.settings_page)
        
        # Taskbar at the bottom
        self.create_taskbar()
        main_layout.addWidget(self.taskbar)
        
        # Show initial watermark
        self.show_watermark()

    def create_taskbar(self):
        self.taskbar = QWidget()
        self.taskbar.setFixedHeight(50)
        self.taskbar.setStyleSheet("background: #2d2d2d; border-top: 1px solid #444;")
        
        taskbar_layout = QHBoxLayout()
        taskbar_layout.setContentsMargins(10, 0, 10, 0)
        taskbar_layout.setSpacing(5)
        
        # Taskbar buttons
        self.dashboard_btn = TaskbarButton("Dashboard")
        self.dashboard_btn.setChecked(True)
        self.dashboard_btn.clicked.connect(lambda: self.content_stack.setCurrentIndex(0))
        
        self.settings_btn = TaskbarButton("Settings")
        self.settings_btn.clicked.connect(lambda: self.content_stack.setCurrentIndex(1))
        
        # Detection control button
        self.detect_btn = QPushButton("Start Detection")
        self.detect_btn.clicked.connect(self.toggle_detection)
        self.detect_btn.setStyleSheet("""
            QPushButton {
                background: #27ae60;
                min-width: 120px;
            }
            QPushButton:hover {
                background: #219955;
            }
            QPushButton:pressed {
                background: #1a8044;
            }
            QPushButton:disabled {
                background: #555;
            }
        """)
        
        # Status indicators
        status_layout = QHBoxLayout()
        status_layout.setSpacing(15)
        
        self.status_indicator = QLabel("● Ready")
        self.status_indicator.setStyleSheet("color: #0078d7; font-size: 12px;")
        
        self.camera_count = QLabel("Cameras: 0")
        self.camera_count.setStyleSheet("font-size: 12px;")
        
        status_layout.addWidget(self.status_indicator)
        status_layout.addWidget(self.camera_count)
        status_layout.addStretch()
        
        # Add widgets to taskbar
        taskbar_layout.addWidget(self.dashboard_btn)
        taskbar_layout.addWidget(self.settings_btn)
        taskbar_layout.addWidget(self.detect_btn)
        taskbar_layout.addLayout(status_layout)
        
        self.taskbar.setLayout(taskbar_layout)

    def create_camera_display_area(self):
        display_area = QWidget()
        self.display_layout = QGridLayout()
        display_area.setLayout(self.display_layout)
        return display_area

    def create_settings_page(self):
        settings_page = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(20, 20, 20, 20)
        
        # Detection settings group
        detection_group = QGroupBox("Detection Settings")
        detection_layout = QFormLayout()
        detection_layout.setContentsMargins(15, 15, 15, 15)
        detection_layout.setSpacing(15)
        
        # Model selection
        self.model_combo = QComboBox()
        self.model_combo.addItem("YOLOv8 Medium (yolov8m.onnx)", 'yolov8m')
        self.model_combo.addItem("YOLOv8 Nano (yolov8n.onnx)", 'yolov8n')
        self.model_combo.setCurrentIndex(0 if self.detection_settings['model_type'] == 'yolov8m' else 1)
        detection_layout.addRow("Model Type:", self.model_combo)
        
        # Confidence threshold
        self.conf_slider = QSlider(Qt.Horizontal)
        self.conf_slider.setRange(10, 100)
        self.conf_slider.setValue(50)
        self.conf_label = QLabel("0.5")
        self.conf_slider.valueChanged.connect(lambda v: self.conf_label.setText(f"{v/100:.1f}"))
        detection_layout.addRow("Confidence Threshold:", self.conf_slider)
        detection_layout.addRow("Value:", self.conf_label)
        
        # IoU threshold
        self.iou_slider = QSlider(Qt.Horizontal)
        self.iou_slider.setRange(10, 100)
        self.iou_slider.setValue(50)
        self.iou_label = QLabel("0.5")
        self.iou_slider.valueChanged.connect(lambda v: self.iou_label.setText(f"{v/100:.1f}"))
        detection_layout.addRow("IoU Threshold:", self.iou_slider)
        detection_layout.addRow("Value:", self.iou_label)
        
        # Checkboxes
        self.auto_model_cb = QCheckBox("Auto model switch (for low FPS)")
        self.auto_model_cb.setChecked(True)
        detection_layout.addRow(self.auto_model_cb)
        
        self.sound_cb = QCheckBox("Enable sound alerts")
        self.sound_cb.setChecked(True)
        detection_layout.addRow(self.sound_cb)
        
        detection_group.setLayout(detection_layout)
        layout.addWidget(detection_group)
        
        # Camera management group
        camera_group = QGroupBox("Camera Management")
        camera_layout = QVBoxLayout()
        camera_layout.setContentsMargins(15, 15, 15, 15)
        
        btn_layout = QHBoxLayout()
        add_btn = QPushButton("Add Camera")
        add_btn.clicked.connect(self.add_camera)
        remove_btn = QPushButton("Remove Camera")
        remove_btn.clicked.connect(self.remove_camera)
        
        btn_layout.addWidget(add_btn)
        btn_layout.addWidget(remove_btn)
        camera_layout.addLayout(btn_layout)
        
        camera_group.setLayout(camera_layout)
        layout.addWidget(camera_group)
        
        layout.addStretch()
        settings_page.setLayout(layout)
        return settings_page

    def show_watermark(self):
        # Clear existing widgets
        for i in reversed(range(self.display_layout.count())):
            self.display_layout.itemAt(i).widget().setParent(None)
        
        # Show watermark
        watermark = WatermarkWidget()
        self.display_layout.addWidget(watermark, 0, 0, Qt.AlignCenter)

    def add_camera(self):
        dialog = CameraConfigDialog(parent=self)
        if dialog.exec_() == QDialog.Accepted and dialog.validate_data():
            camera_data = dialog.get_camera_data()
            camera_id = f"cam_{len(self.cameras)}"
            self.cameras[camera_id] = camera_data
            self.update_camera_display()
            self.camera_count.setText(f"Cameras: {len(self.cameras)}")

    def remove_camera(self):
        if not self.cameras:
            QMessageBox.information(self, "Info", "No cameras to remove")
            return
        
        # Remove the last camera for simplicity
        camera_id = list(self.cameras.keys())[-1]
        del self.cameras[camera_id]
        self.update_camera_display()
        self.camera_count.setText(f"Cameras: {len(self.cameras)}")

    def update_camera_display(self):
        # Clear existing widgets
        for i in reversed(range(self.display_layout.count())):
            self.display_layout.itemAt(i).widget().setParent(None)
        
        self.camera_widgets.clear()
        
        if not self.cameras:
            self.show_watermark()
            return
        
        # Determine grid layout based on number of cameras (max 4)
        num_cameras = len(self.cameras)
        if num_cameras == 1:
            rows, cols = 1, 1
        elif num_cameras == 2:
            rows, cols = 1, 2
        elif num_cameras == 3:
            rows, cols = 2, 2  # One empty space
        else:  # 4 cameras
            rows, cols = 2, 2
        
        # Create camera display widgets
        for i, (camera_id, camera_data) in enumerate(self.cameras.items()):
            camera_widget = self.create_camera_widget(camera_id, camera_data)
            self.camera_widgets[camera_id] = camera_widget
            
            if num_cameras == 3 and i == 2:
                # Special case for 3 cameras - place the third one in second row first column
                self.display_layout.addWidget(camera_widget, 1, 0)
            else:
                row = i // cols
                col = i % cols
                self.display_layout.addWidget(camera_widget, row, col)

    def create_camera_widget(self, camera_id, camera_data):
        widget = QGroupBox(camera_data['name'])
        widget.setMinimumSize(640, 480)
        layout = QVBoxLayout()
        layout.setContentsMargins(5, 15, 5, 5)
        
        # Video display
        video_label = QLabel()
        video_label.setMinimumSize(640, 480)
        video_label.setStyleSheet("border: 1px solid #444; background-color: #2d2d2d;")
        video_label.setAlignment(Qt.AlignCenter)
        video_label.setText("Connecting...")
        layout.addWidget(video_label)
        
        # Status bar
        status_layout = QHBoxLayout()
        status_layout.setContentsMargins(5, 5, 5, 5)
        status_label = QLabel("●")
        status_label.setStyleSheet("color: red; font-size: 16px;")
        fps_label = QLabel("FPS: 0")
        
        status_layout.addWidget(status_label)
        status_layout.addWidget(fps_label)
        status_layout.addStretch()
        
        layout.addLayout(status_layout)
        widget.setLayout(layout)
        
        # Store references
        widget.video_label = video_label
        widget.status_label = status_label
        widget.fps_label = fps_label
        
        return widget

    def toggle_detection(self):
        if self.detection_thread and self.detection_thread.isRunning():
            self.stop_detection()
        else:
            self.start_detection()

    def start_detection(self):
        if not self.cameras:
            QMessageBox.warning(self, "Warning", "Please add at least one camera")
            return
        
        # Update detection settings
        self.detection_settings.update({
            'conf_threshold': self.conf_slider.value() / 100,
            'iou_threshold': self.iou_slider.value() / 100,
            'auto_model_switch': self.auto_model_cb.isChecked(),
            'sound_enabled': self.sound_cb.isChecked(),
            'model_type': self.model_combo.currentData()
        })
        
        # Start detection thread
        self.detection_thread = DetectionThread(self.cameras, self.detection_settings)
        self.detection_thread.frame_ready.connect(self.update_camera_frame)
        self.detection_thread.connection_status.connect(self.update_connection_status)
        self.detection_thread.model_switched.connect(self.handle_model_switch)
        self.detection_thread.start()
        
        self.detect_btn.setText("Stop Detection")
        self.detect_btn.setStyleSheet("""
            QPushButton {
                background: #e74c3c;
                min-width: 120px;
            }
            QPushButton:hover {
                background: #c0392b;
            }
            QPushButton:pressed {
                background: #992d22;
            }
        """)
        self.status_indicator.setText("● Running")
        self.status_indicator.setStyleSheet("color: #27ae60; font-size: 12px;")

    def stop_detection(self):
        if self.detection_thread:
            self.detection_thread.stop()
            self.detection_thread = None
        
        # Reset camera displays
        for camera_id, widget in self.camera_widgets.items():
            widget.video_label.setText("Stopped")
            widget.status_label.setStyleSheet("color: red; font-size: 16px;")
            widget.fps_label.setText("FPS: 0")
        
        self.detect_btn.setText("Start Detection")
        self.detect_btn.setStyleSheet("""
            QPushButton {
                background: #27ae60;
                min-width: 120px;
            }
            QPushButton:hover {
                background: #219955;
            }
            QPushButton:pressed {
                background: #1a8044;
            }
        """)
        self.status_indicator.setText("● Ready")
        self.status_indicator.setStyleSheet("color: #0078d7; font-size: 12px;")

    def update_camera_frame(self, camera_id, frame, person_count, fps):
        if camera_id not in self.camera_widgets:
            return
        
        widget = self.camera_widgets[camera_id]
        
        # Convert frame to QImage and display
        h, w, ch = frame.shape
        bytes_per_line = ch * w
        qt_image = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)
        
        widget.video_label.setPixmap(pixmap.scaled(widget.video_label.size(), Qt.KeepAspectRatio))
        widget.fps_label.setText(f"FPS: {fps:.1f}")

    def update_connection_status(self, camera_id, connected):
        if camera_id not in self.camera_widgets:
            return
        
        widget = self.camera_widgets[camera_id]
        if connected:
            widget.status_label.setStyleSheet("color: #27ae60; font-size: 16px;")
        else:
            widget.status_label.setStyleSheet("color: red; font-size: 16px;")
            widget.video_label.setText("Disconnected")

    def handle_model_switch(self, camera_id, model_name):
        if camera_id in self.camera_widgets:
            widget = self.camera_widgets[camera_id]
            widget.video_label.setText(f"Switched to {model_name}")

    def closeEvent(self, event):
        if self.detection_thread:
            self.detection_thread.stop()
        event.accept()
        
# ========== ENTRY POINT ==========
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ModernCameraDetectionApp()
    window.show()
    sys.exit(app.exec_())
