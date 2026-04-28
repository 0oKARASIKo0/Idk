"""
Perception module for the AI agent.
Handles camera input, microphone input, screen capture, face detection, speech recognition,
system audio capture, voice differentiation and emotion/intonation analysis.
"""
import cv2
import numpy as np
import threading
import time
import logging
from typing import Optional, Dict, Any, List, Tuple
from pathlib import Path
import librosa
import sounddevice as sd
from scipy import signal
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

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


class VoiceProfiler:
    """Profiles and differentiates voices using MFCC features."""
    
    def __init__(self, max_speakers: int = 5):
        self.max_speakers = max_speakers
        self.voice_profiles: Dict[str, List[np.ndarray]] = {}  # speaker_id -> list of MFCC vectors
        self.known_speakers: List[str] = ["Пользователь"]  # Default known speaker
        self.lock = threading.Lock()
        self.n_mfcc = 13  # Number of MFCC features
        
    def extract_mfcc(self, audio_data: np.ndarray, sample_rate: int) -> Optional[np.ndarray]:
        """Extract MFCC features from audio data."""
        try:
            # Normalize audio
            audio_float = audio_data.astype(np.float32) / 32768.0
            
            # Extract MFCC
            mfcc = librosa.feature.mfcc(y=audio_float, sr=sample_rate, n_mfcc=self.n_mfcc)
            
            # Average over time to get a single vector per utterance
            mfcc_mean = np.mean(mfcc, axis=1)
            
            return mfcc_mean
        except Exception as e:
            logger.error(f"Error extracting MFCC: {e}")
            return None
    
    def identify_speaker(self, audio_data: np.ndarray, sample_rate: int) -> Tuple[str, float]:
        """Identify the speaker from audio data."""
        mfcc_vector = self.extract_mfcc(audio_data, sample_rate)
        
        if mfcc_vector is None:
            return "Неизвестный", 0.0
        
        mfcc_vector = mfcc_vector.reshape(1, -1)
        
        with self.lock:
            best_match = "Неизвестный"
            best_confidence = 0.0
            
            for speaker_id, profiles in self.voice_profiles.items():
                if len(profiles) == 0:
                    continue
                
                # Calculate average profile for this speaker
                avg_profile = np.mean(profiles, axis=0).reshape(1, -1)
                
                # Calculate cosine similarity
                similarity = cosine_similarity(mfcc_vector, avg_profile)[0][0]
                
                if similarity > best_confidence:
                    best_confidence = similarity
                    best_match = speaker_id
        
        # Threshold for confidence (adjust as needed)
        if best_confidence < 0.7:
            return "Неизвестный", best_confidence
        
        return best_match, best_confidence
    
    def register_speaker(self, speaker_id: str, audio_data: np.ndarray, sample_rate: int):
        """Register or update a speaker's voice profile."""
        mfcc_vector = self.extract_mfcc(audio_data, sample_rate)
        
        if mfcc_vector is None:
            return
        
        with self.lock:
            if speaker_id not in self.voice_profiles:
                self.voice_profiles[speaker_id] = []
            
            self.voice_profiles[speaker_id].append(mfcc_vector)
            
            # Keep only last 10 samples per speaker
            if len(self.voice_profiles[speaker_id]) > 10:
                self.voice_profiles[speaker_id] = self.voice_profiles[speaker_id][-10:]
        
        logger.info(f"Registered/updated speaker profile: {speaker_id}")
    
    def add_known_speaker(self, name: str):
        """Add a known speaker name."""
        with self.lock:
            if name not in self.known_speakers:
                self.known_speakers.append(name)


class EmotionAnalyzer:
    """Analyzes emotions and intonation from audio."""
    
    def __init__(self):
        self.emotion_labels = ["нейтральный", "радостный", "грустный", "злой", "удивленный", "напряженный"]
        
    def analyze_emotion(self, audio_data: np.ndarray, sample_rate: int) -> Dict[str, Any]:
        """Analyze emotion and intonation from audio."""
        try:
            audio_float = audio_data.astype(np.float32) / 32768.0
            
            # Extract acoustic features
            zero_crossing_rate = librosa.feature.zero_crossing_rate(audio_float)[0]
            rms_energy = librosa.feature.rms(y=audio_float)[0]
            pitch, _, _ = librosa.pyin(audio_float, fmin=50, fmax=500, sr=sample_rate)
            
            # Calculate statistics
            zcr_mean = np.mean(zero_crossing_rate)
            zcr_std = np.std(zero_crossing_rate)
            energy_mean = np.mean(rms_energy)
            energy_std = np.std(rms_energy)
            pitch_mean = np.nanmean(pitch) if np.any(~np.isnan(pitch)) else 0
            pitch_std = np.nanstd(pitch) if np.any(~np.isnan(pitch)) else 0
            
            # Simple rule-based emotion detection
            emotion = "нейтральный"
            confidence = 0.5
            
            if energy_mean > 0.1 and pitch_mean > 200:
                emotion = "радостный"
                confidence = 0.7
            elif energy_mean < 0.05 and pitch_mean < 100:
                emotion = "грустный"
                confidence = 0.6
            elif energy_mean > 0.15 and zcr_mean > 0.1:
                emotion = "злой"
                confidence = 0.75
            elif energy_std > 0.08 or pitch_std > 50:
                emotion = "удивленный"
                confidence = 0.65
            elif zcr_mean > 0.12:
                emotion = "напряженный"
                confidence = 0.6
            
            # Determine intonation
            if pitch_mean > 180:
                intonation = "высокая"
            elif pitch_mean < 100:
                intonation = "низкая"
            else:
                intonation = "средняя"
            
            return {
                "emotion": emotion,
                "emotion_confidence": confidence,
                "intonation": intonation,
                "pitch_mean": pitch_mean,
                "energy_mean": energy_mean,
                "arousal": energy_mean * 10  # 0-10 scale
            }
            
        except Exception as e:
            logger.error(f"Error analyzing emotion: {e}")
            return {
                "emotion": "неизвестный",
                "emotion_confidence": 0.0,
                "intonation": "неизвестная",
                "pitch_mean": 0,
                "energy_mean": 0,
                "arousal": 0
            }


class AudioCapture:
    """Handles microphone and system audio capture with voice differentiation."""
    
    def __init__(self, sample_rate: int = 16000, chunk_size: int = 1024, language: str = 'ru-RU'):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.language = language
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.audio_queue: List[np.ndarray] = []
        self.mic_queue: List[np.ndarray] = []
        self.system_queue: List[np.ndarray] = []
        self.lock = threading.Lock()
        self.stream = None
        self.system_stream = None
        self.recognizer = None
        self.voice_profiler = VoiceProfiler()
        self.emotion_analyzer = EmotionAnalyzer()
        self.speech_active = False
        self.last_speech_time = 0
        self.speech_timeout = 2.0  # seconds
        
        try:
            import speech_recognition as sr
            self.recognizer = sr.Recognizer()
            logger.info("Speech recognition initialized")
        except Exception as e:
            logger.warning(f"Speech recognition not available: {e}")
    
    def start(self):
        """Start audio capture from microphone and system."""
        mic_started = self._start_microphone()
        system_started = self._start_system_audio()
        
        if mic_started or system_started:
            self.running = True
            self.thread = threading.Thread(target=self._process_audio_loop, daemon=True)
            self.thread.start()
            logger.info("Audio capture started (mic: {}, system: {})".format(mic_started, system_started))
            return True
        return False
    
    def _start_microphone(self) -> bool:
        """Start microphone capture."""
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
            
            logger.info("Microphone capture started")
            return True
        except Exception as e:
            logger.warning(f"Could not start microphone: {e}")
            return False
    
    def _start_system_audio(self) -> bool:
        """Start system audio capture using sounddevice."""
        try:
            # Find stereo mix or similar device
            devices = sd.query_devices()
            system_device = None
            
            for i, dev in enumerate(devices):
                if dev['max_input_channels'] > 0:
                    name_lower = dev['name'].lower()
                    if 'stereo' in name_lower or 'mix' in name_lower or 'what u hear' in name_lower:
                        system_device = i
                        logger.info(f"Found system audio device: {dev['name']}")
                        break
            
            if system_device is None:
                # Use default input device as fallback
                system_device = sd.default.device[0]
                logger.warning("Stereo Mix not found, using default input device")
            
            def system_callback(indata, frames, time_info, status):
                if status:
                    logger.warning(f"System audio status: {status}")
                with self.lock:
                    self.system_queue.append(indata.copy().flatten())
                    # Keep only last 5 seconds
                    max_chunks = (5 * self.sample_rate) // self.chunk_size
                    if len(self.system_queue) > max_chunks:
                        self.system_queue = self.system_queue[-max_chunks:]
            
            self.system_stream = sd.InputStream(
                device=system_device,
                samplerate=self.sample_rate,
                channels=1,
                callback=system_callback,
                blocksize=self.chunk_size
            )
            self.system_stream.start()
            
            logger.info("System audio capture started")
            return True
        except Exception as e:
            logger.warning(f"Could not start system audio: {e}")
            return False
    
    def _process_audio_loop(self):
        """Background loop to process audio from both sources."""
        while self.running:
            try:
                time.sleep(0.1)  # Process every 100ms
                
                # Combine audio from both sources for speech detection
                combined_audio = []
                
                with self.lock:
                    if self.mic_queue:
                        combined_audio.extend(self.mic_queue[-10:])
                    if self.system_queue:
                        combined_audio.extend(self.system_queue[-10:])
                
                if not combined_audio:
                    continue
                
                # Detect if speech is present (simple energy-based VAD)
                audio_segment = np.concatenate(combined_audio[-5:])
                energy = np.mean(np.abs(audio_segment.astype(np.float32) / 32768.0))
                
                if energy > 0.02:  # Speech threshold
                    if not self.speech_active:
                        self.speech_active = True
                        self.last_speech_time = time.time()
                        logger.debug("Speech detected")
                    else:
                        self.last_speech_time = time.time()
                else:
                    if self.speech_active and time.time() - self.last_speech_time > self.speech_timeout:
                        self.speech_active = False
                        logger.debug("Speech ended")
                
            except Exception as e:
                logger.error(f"Error in audio processing loop: {e}")
                time.sleep(0.5)
    
    def _record_loop(self):
        """Legacy record loop for microphone only (kept for compatibility)."""
        while self.running:
            try:
                if self.stream:
                    data = self.stream.read(self.chunk_size, exception_on_overflow=False)
                    audio_data = np.frombuffer(data, dtype=np.int16)
                    
                    with self.lock:
                        self.mic_queue.append(audio_data)
                        self.audio_queue.append(audio_data)
                        # Keep only last 5 seconds of audio
                        max_chunks = (5 * self.sample_rate) // self.chunk_size
                        if len(self.mic_queue) > max_chunks:
                            self.mic_queue = self.mic_queue[-max_chunks:]
                        if len(self.audio_queue) > max_chunks:
                            self.audio_queue = self.audio_queue[-max_chunks:]
            except Exception as e:
                logger.error(f"Error recording audio: {e}")
                time.sleep(0.5)
    
    def recognize_speech(self) -> Optional[Dict[str, Any]]:
        """Attempt to recognize speech from recent audio with speaker and emotion info."""
        if not self.recognizer:
            return None
        
        try:
            import pyaudio
            import speech_recognition as sr
            
            with self.lock:
                # Combine mic and system audio
                all_audio = []
                if self.mic_queue and len(self.mic_queue) >= 10:
                    all_audio.extend(self.mic_queue[-50:])
                if self.system_queue and len(self.system_queue) >= 10:
                    all_audio.extend(self.system_queue[-50:])
                
                if not all_audio:
                    return None
                
                audio_data = np.concatenate(all_audio)
            
            # Convert to audio data for speech recognition
            audio = sr.AudioData(
                audio_data.tobytes(),
                self.sample_rate,
                2  # 16-bit = 2 bytes
            )
            
            # Try to recognize speech
            text = self.recognizer.recognize_google(audio, language=self.language)
            
            if not text.strip():
                return None
            
            # Identify speaker
            speaker_id, confidence = self.voice_profiler.identify_speaker(audio_data, self.sample_rate)
            
            # Analyze emotion
            emotion_info = self.emotion_analyzer.analyze_emotion(audio_data, self.sample_rate)
            
            # Register speaker if it's likely the user
            if confidence > 0.8 and speaker_id == "Неизвестный":
                self.voice_profiler.register_speaker("Пользователь", audio_data, self.sample_rate)
                speaker_id = "Пользователь"
            
            result = {
                'text': text,
                'speaker': speaker_id,
                'speaker_confidence': confidence,
                'emotion': emotion_info['emotion'],
                'emotion_confidence': emotion_info['emotion_confidence'],
                'intonation': emotion_info['intonation'],
                'arousal': emotion_info['arousal']
            }
            
            # Clear processed audio
            with self.lock:
                self.mic_queue = []
                self.system_queue = []
                self.audio_queue = []
            
            logger.info(f"Recognized speech: '{text}' by {speaker_id} ({emotion_info['emotion']})")
            return result
            
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
        if self.system_stream:
            self.system_stream.stop()
            self.system_stream.close()
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
        
        # Check for speech with speaker and emotion info
        if self.config.VOICE_ENABLED:
            speech_result = self.audio.recognize_speech()
            if speech_result:
                state['speech'] = speech_result
        
        return state

