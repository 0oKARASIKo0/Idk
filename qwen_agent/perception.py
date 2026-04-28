"""
Perception module for the AI agent.
Handles camera input, microphone input, screen capture, face detection, and speech recognition.
"""
import cv2
import numpy as np
import threading
import time
import logging
from typing import Optional, Dict, Any, List
from pathlib import Path

logger = logging.getLogger(__name__)


class CameraCapture:
    """Handles camera video capture."""
    
    def __init__(self, camera_index: int = 0, width: int = 640, height: int = 480):
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.cap = None
        self.latest_frame: Optional[np.ndarray] = None
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.lock = threading.Lock()
    
    def start(self):
        """Start camera capture in background thread."""
        try:
            self.cap = cv2.VideoCapture(self.camera_index)
            if not self.cap.isOpened():
                logger.error(f"Failed to open camera {self.camera_index}")
                return False
            
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            
            self.running = True
            self.thread = threading.Thread(target=self._capture_loop, daemon=True)
            self.thread.start()
            logger.info(f"Camera {self.camera_index} started")
            return True
        except Exception as e:
            logger.error(f"Error starting camera: {e}")
            return False
    
    def _capture_loop(self):
        """Background loop to continuously capture frames."""
        while self.running:
            try:
                ret, frame = self.cap.read()
                if ret:
                    with self.lock:
                        self.latest_frame = frame
                else:
                    time.sleep(0.1)
            except Exception as e:
                logger.error(f"Error capturing frame: {e}")
                time.sleep(0.5)
    
    def get_frame(self) -> Optional[np.ndarray]:
        """Get the latest captured frame."""
        with self.lock:
            return self.latest_frame.copy() if self.latest_frame is not None else None
    
    def stop(self):
        """Stop camera capture."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)
        if self.cap:
            self.cap.release()
        logger.info("Camera stopped")


class FaceDetector:
    """Detects faces in images using OpenCV Haar cascades."""
    
    def __init__(self):
        # Try to load the cascade file
        cascade_paths = [
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml',
            cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml',
        ]
        
        self.face_cascade = None
        for path in cascade_paths:
            if Path(path).exists():
                self.face_cascade = cv2.CascadeClassifier(path)
                break
        
        if self.face_cascade is None:
            logger.warning("Face detection cascade not found, face detection disabled")
    
    def detect_faces(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Detect faces in a frame and return their locations."""
        if self.face_cascade is None or frame is None:
            return []
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        face_list = []
        for (x, y, w, h) in faces:
            face_list.append({
                'x': int(x),
                'y': int(y),
                'width': int(w),
                'height': int(h),
                'confidence': 0.8  # Placeholder confidence
            })
        
        return face_list


class ScreenCapture:
    """Captures screen content."""
    
    def __init__(self, capture_interval: float = 2.0):
        self.capture_interval = capture_interval
        self.latest_screenshot: Optional[np.ndarray] = None
        self.last_capture_time: float = 0
        self.lock = threading.Lock()
    
    def capture(self) -> Optional[np.ndarray]:
        """Capture current screen."""
        try:
            import pyautogui
            screenshot = pyautogui.screenshot()
            frame = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
            
            with self.lock:
                self.latest_screenshot = frame
                self.last_capture_time = time.time()
            
            return frame
        except Exception as e:
            logger.error(f"Error capturing screen: {e}")
            return None
    
    def get_latest(self) -> Optional[np.ndarray]:
        """Get the latest screenshot."""
        with self.lock:
            return self.latest_screenshot.copy() if self.latest_screenshot is not None else None
    
    def should_capture(self) -> bool:
        """Check if it's time to capture a new screenshot."""
        return time.time() - self.last_capture_time >= self.capture_interval


class AudioCapture:
    """Handles microphone audio capture and speech recognition."""
    
    def __init__(self, sample_rate: int = 16000, chunk_size: int = 1024, language: str = 'ru-RU'):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.language = language
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.audio_queue: List[np.ndarray] = []
        self.lock = threading.Lock()
        self.stream = None
        self.recognizer = None
        
        try:
            import speech_recognition as sr
            self.recognizer = sr.Recognizer()
            logger.info("Speech recognition initialized")
        except Exception as e:
            logger.warning(f"Speech recognition not available: {e}")
    
    def start(self):
        """Start audio capture."""
        try:
            import pyaudio
            
            self.paudio = pyaudio.PyAudio()
            self.stream = self.paudio.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size
            )
            
            self.running = True
            self.thread = threading.Thread(target=self._record_loop, daemon=True)
            self.thread.start()
            logger.info("Audio capture started")
            return True
        except Exception as e:
            logger.warning(f"Could not start audio capture: {e}")
            return False
    
    def _record_loop(self):
        """Background loop to record audio."""
        while self.running:
            try:
                if self.stream:
                    data = self.stream.read(self.chunk_size, exception_on_overflow=False)
                    audio_data = np.frombuffer(data, dtype=np.int16)
                    
                    with self.lock:
                        self.audio_queue.append(audio_data)
                        # Keep only last 5 seconds of audio
                        max_chunks = (5 * self.sample_rate) // self.chunk_size
                        if len(self.audio_queue) > max_chunks:
                            self.audio_queue = self.audio_queue[-max_chunks:]
            except Exception as e:
                logger.error(f"Error recording audio: {e}")
                time.sleep(0.5)
    
    def recognize_speech(self) -> Optional[str]:
        """Attempt to recognize speech from recent audio."""
        if not self.recognizer or not self.audio_queue:
            return None
        
        try:
            import pyaudio
            import speech_recognition as sr
            
            with self.lock:
                if len(self.audio_queue) < 10:  # Need at least some audio
                    return None
                
                # Combine recent audio chunks
                audio_data = np.concatenate(self.audio_queue[-50:])
            
            # Convert to audio data for speech recognition
            audio = sr.AudioData(
                audio_data.tobytes(),
                self.sample_rate,
                2  # 16-bit = 2 bytes
            )
            
            # Try to recognize speech
            text = self.recognizer.recognize_google(audio, language=self.language)
            
            # Clear processed audio
            with self.lock:
                self.audio_queue = []
            
            logger.info(f"Recognized speech: {text}")
            return text
            
        except sr.UnknownValueError:
            pass  # No speech detected
        except sr.RequestError as e:
            logger.warning(f"Speech recognition service error: {e}")
        except Exception as e:
            logger.error(f"Error recognizing speech: {e}")
        
        return None
    
    def stop(self):
        """Stop audio capture."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        if hasattr(self, 'paudio'):
            self.paudio.terminate()
        logger.info("Audio capture stopped")


class Perception:
    """Main perception module combining all sensors."""
    
    def __init__(self, config):
        self.config = config
        self.camera = CameraCapture(
            camera_index=config.CAMERA_INDEX,
            width=config.CAMERA_FRAME_WIDTH,
            height=config.CAMERA_FRAME_HEIGHT
        )
        self.face_detector = FaceDetector()
        self.screen = ScreenCapture(capture_interval=config.SCREEN_CAPTURE_INTERVAL)
        self.audio = AudioCapture(
            sample_rate=config.AUDIO_SAMPLE_RATE,
            chunk_size=config.AUDIO_CHUNK_SIZE,
            language=config.SPEECH_RECOGNITION_LANGUAGE
        )
        self.running = False
    
    def start(self):
        """Start all perception modules."""
        self.running = True
        self.camera.start()
        if self.config.VOICE_ENABLED:
            self.audio.start()
        logger.info("Perception system started")
    
    def stop(self):
        """Stop all perception modules."""
        self.running = False
        self.camera.stop()
        self.audio.stop()
        logger.info("Perception system stopped")
    
    def get_current_state(self) -> Dict[str, Any]:
        """Get current perception state."""
        state = {
            'timestamp': time.time(),
            'camera_frame': None,
            'faces': [],
            'screen': None,
            'speech': None
        }
        
        # Get camera frame and detect faces
        frame = self.camera.get_frame()
        if frame is not None:
            state['camera_frame'] = frame
            state['faces'] = self.face_detector.detect_faces(frame)
        
        # Capture screen if needed
        if self.screen.should_capture():
            state['screen'] = self.screen.capture()
        else:
            state['screen'] = self.screen.get_latest()
        
        # Check for speech
        if self.config.VOICE_ENABLED:
            state['speech'] = self.audio.recognize_speech()
        
        return state
