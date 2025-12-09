# Whisper Speech Processing

OpenAI's Whisper model is a state-of-the-art automatic speech recognition (ASR) system that can transcribe speech to text with remarkable accuracy. In this section, we'll explore how to integrate Whisper into your humanoid robot's communication system.

## Introduction to Whisper

Whisper is a general-purpose speech recognition model trained on 680,000 hours of multilingual and multitask supervised data. It demonstrates strong performance in:
- **Automatic Speech Recognition (ASR)**: Converting speech to text
- **Speech Translation**: Translating speech from one language to another
- **Language Identification**: Determining the spoken language
- **Voice Activity Detection**: Identifying when speech occurs

### Whisper Model Variants

There are several Whisper model variants with different sizes and capabilities:

- **tiny**: Fastest, smallest (39M parameters)
- **base**: Small (74M parameters)
- **small**: Medium (244M parameters)
- **medium**: Large (769M parameters)
- **large**: Largest, most accurate (1550M parameters)

For robotics applications, the choice depends on:
- **Accuracy requirements**: Larger models provide better accuracy
- **Computational resources**: Smaller models run faster with less memory
- **Latency requirements**: Real-time applications may need faster models

## Whisper Integration with Robotics

### Installation and Setup

First, install the required dependencies:

```bash
pip install openai-whisper
pip install sounddevice  # For audio input
pip install pyaudio      # Alternative audio input
pip install transformers # For LLM integration
```

### Basic Whisper Usage

```python
import whisper
import torch

# Load the Whisper model
model = whisper.load_model("small")  # Choose tiny, base, small, medium, or large

# Transcribe audio
result = model.transcribe("path/to/audio.wav")
print(result["text"])
```

### Real-time Audio Processing

For real-time speech processing in robotics, we need to capture and process audio streams:

```python
import whisper
import numpy as np
import sounddevice as sd
import queue
import threading
import time

class RealTimeWhisper:
    def __init__(self, model_size="small"):
        # Load Whisper model
        self.model = whisper.load_model(model_size)

        # Audio parameters
        self.sample_rate = 16000  # Whisper expects 16kHz
        self.chunk_duration = 1.0  # Process 1-second chunks
        self.chunk_size = int(self.sample_rate * self.chunk_duration)

        # Audio buffer
        self.audio_queue = queue.Queue()
        self.transcript_queue = queue.Queue()

        # Flags
        self.recording = False

    def audio_callback(self, indata, frames, time, status):
        """Callback for audio input"""
        if status:
            print(status)
        # Add audio data to queue
        self.audio_queue.put(indata[:, 0].copy())

    def start_recording(self):
        """Start recording audio"""
        self.recording = True

        # Start audio stream
        self.stream = sd.InputStream(
            samplerate=self.sample_rate,
            blocksize=self.chunk_size,
            channels=1,
            dtype='float32',
            callback=self.audio_callback
        )
        self.stream.start()

        # Start processing thread
        self.processing_thread = threading.Thread(target=self.process_audio)
        self.processing_thread.start()

    def stop_recording(self):
        """Stop recording audio"""
        self.recording = False
        self.stream.stop()
        self.stream.close()

    def process_audio(self):
        """Process audio chunks in a separate thread"""
        audio_buffer = np.array([])

        while self.recording:
            try:
                # Get audio chunk
                chunk = self.audio_queue.get(timeout=0.1)

                # Add to buffer
                audio_buffer = np.concatenate([audio_buffer, chunk])

                # Process when we have enough audio
                if len(audio_buffer) >= self.chunk_size:
                    # Process the audio
                    transcript = self.transcribe_chunk(audio_buffer)

                    # Add to transcript queue
                    if transcript.strip():  # Only add non-empty transcripts
                        self.transcript_queue.put(transcript)

                    # Keep remaining audio in buffer
                    audio_buffer = audio_buffer[self.chunk_size:]

            except queue.Empty:
                continue

    def transcribe_chunk(self, audio_chunk):
        """Transcribe a chunk of audio"""
        # Convert to tensor
        audio_tensor = torch.from_numpy(audio_chunk).float()

        # Transcribe
        result = self.model.transcribe(audio_tensor.numpy())

        return result["text"]

    def get_transcript(self):
        """Get next transcript from queue"""
        try:
            return self.transcript_queue.get_nowait()
        except queue.Empty:
            return None

# Usage example
whisper_robot = RealTimeWhisper(model_size="small")
whisper_robot.start_recording()

try:
    while True:
        transcript = whisper_robot.get_transcript()
        if transcript:
            print(f"Robot heard: {transcript}")
            # Process the transcript for robot actions

        time.sleep(0.1)
except KeyboardInterrupt:
    whisper_robot.stop_recording()
```

## Whisper with ROS 2 Integration

### Creating a Whisper ROS 2 Node

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from audio_common_msgs.msg import AudioData
import whisper
import torch
import numpy as np
from io import BytesIO
import wave

class WhisperNode(Node):
    def __init__(self):
        super().__init__('whisper_node')

        # Load Whisper model
        self.model_size = self.declare_parameter('model_size', 'small').get_parameter_value().string_value
        self.model = whisper.load_model(self.model_size)

        # Subscribe to audio data
        self.audio_sub = self.create_subscription(
            AudioData,
            'audio_input',
            self.audio_callback,
            10
        )

        # Publish transcriptions
        self.transcript_pub = self.create_publisher(
            String,
            'speech_transcription',
            10
        )

        self.get_logger().info(f'Whisper node initialized with {self.model_size} model')

    def audio_callback(self, msg):
        """Process incoming audio data"""
        try:
            # Convert audio data to numpy array
            audio_data = np.frombuffer(msg.data, dtype=np.int16).astype(np.float32) / 32768.0

            # Transcribe the audio
            result = self.model.transcribe(audio_data)

            # Publish transcription
            transcript_msg = String()
            transcript_msg.data = result["text"]
            self.transcript_pub.publish(transcript_msg)

            self.get_logger().info(f'Transcribed: {result["text"]}')

        except Exception as e:
            self.get_logger().error(f'Error processing audio: {str(e)}')

def main(args=None):
    rclpy.init(args=args)
    whisper_node = WhisperNode()

    try:
        rclpy.spin(whisper_node)
    except KeyboardInterrupt:
        pass
    finally:
        whisper_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Launch File for Whisper Node

```xml
<!-- launch/whisper_node.launch.py -->
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='robot_voice_interface',
            executable='whisper_node',
            name='whisper_node',
            parameters=[
                {'model_size': 'small'}  # Choose tiny, base, small, medium, or large
            ],
            remappings=[
                ('/audio_input', '/microphone/audio_raw'),
                ('/speech_transcription', '/voice_commands')
            ]
        )
    ])
```

## Advanced Whisper Features

### Language Detection and Multilingual Support

```python
def detect_language_and_transcribe(self, audio_data):
    """Detect language and transcribe accordingly"""
    # Detect language
    audio_tensor = torch.from_numpy(audio_data).float()
    mel = whisper.log_mel_spectrogram(audio_tensor)

    # Detect language
    _, probs = self.model.detect_language(mel[:1])
    detected_lang = max(probs, key=probs.get)

    # Transcribe with detected language
    result = self.model.transcribe(audio_data, language=detected_lang)

    return result["text"], detected_lang
```

### Improved Real-time Processing with VAD (Voice Activity Detection)

```python
import webrtcvad  # pip install webrtcvad

class SmartWhisperNode(Node):
    def __init__(self):
        super().__init__('smart_whisper_node')

        # Load Whisper model
        self.model = whisper.load_model("small")

        # Voice activity detection
        self.vad = webrtcvad.Vad()
        self.vad.set_mode(1)  # Aggressiveness mode (0-3)

        # Audio parameters
        self.sample_rate = 16000
        self.frame_duration = 30  # ms
        self.frame_size = int(self.sample_rate * self.frame_duration / 1000)

        # Speech detection parameters
        self.speech_buffer = []
        self.silence_threshold = 50  # frames of silence to trigger processing
        self.silence_count = 0
        self.is_speaking = False

        # ROS 2 setup
        self.audio_sub = self.create_subscription(
            AudioData,
            'audio_input',
            self.smart_audio_callback,
            10
        )

        self.transcript_pub = self.create_publisher(String, 'speech_transcription', 10)

    def smart_audio_callback(self, msg):
        """Process audio with voice activity detection"""
        # Convert to 16-bit PCM for VAD
        audio_int16 = np.frombuffer(msg.data, dtype=np.int16)

        # Process in frames
        for i in range(0, len(audio_int16), self.frame_size):
            frame = audio_int16[i:i+self.frame_size]

            # Pad frame if necessary
            if len(frame) < self.frame_size:
                frame = np.pad(frame, (0, self.frame_size - len(frame)), 'constant')

            # Check for voice activity
            is_speech = self.vad.is_speech(frame.tobytes(), self.sample_rate)

            if is_speech:
                # Add to speech buffer
                self.speech_buffer.extend(frame)
                self.silence_count = 0
                self.is_speaking = True
            else:
                # Add to silence counter
                self.silence_count += 1

                if self.is_speaking and self.silence_count > self.silence_threshold:
                    # End of speech detected, process the buffer
                    self.process_speech_buffer()
                    self.is_speaking = False

    def process_speech_buffer(self):
        """Process accumulated speech buffer"""
        if len(self.speech_buffer) > 0:
            # Convert to float32
            audio_float32 = np.array(self.speech_buffer, dtype=np.float32) / 32768.0

            # Transcribe
            result = self.model.transcribe(audio_float32)

            # Publish if we have a meaningful result
            if result["text"].strip():
                transcript_msg = String()
                transcript_msg.data = result["text"]
                self.transcript_pub.publish(transcript_msg)

                self.get_logger().info(f'Speech detected: {result["text"]}')

            # Clear buffer
            self.speech_buffer = []
```

## Performance Optimization

### Using Local Whisper Models

For better performance and privacy, use local models:

```python
# Download model to local directory
import os
from whisper import _download, _MODELS

def download_whisper_model(model_size, download_root=None):
    """Download Whisper model to local directory"""
    if download_root is None:
        download_root = os.path.expanduser("~/.cache/whisper")

    model_url = _MODELS[model_size]
    return _download(model_url, download_root, False)

# Use local model
model_path = download_whisper_model("small")
model = whisper.load_model(model_path)
```

### Quantization for Better Performance

```python
# Load quantized model for better performance
model = whisper.load_model("small", device="cuda", in_memory=True)

# Or use CPU with FP16 for better performance
model = whisper.load_model("small", device="cpu", fp16=True)
```

## Error Handling and Robustness

### Handling Different Audio Formats

```python
def process_audio_with_format_handling(self, audio_msg):
    """Handle different audio formats"""
    try:
        # Convert different sample rates to 16kHz
        audio_data = self.convert_audio_format(audio_msg)

        # Normalize audio
        audio_data = self.normalize_audio(audio_data)

        # Transcribe
        result = self.model.transcribe(audio_data)

        return result["text"]
    except Exception as e:
        self.get_logger().error(f'Audio processing error: {str(e)}')
        return ""

def convert_audio_format(self, audio_msg):
    """Convert audio to required format"""
    # Convert to numpy array
    if audio_msg.encoding == 'PCM_16':
        audio_np = np.frombuffer(audio_msg.data, dtype=np.int16).astype(np.float32) / 32768.0
    elif audio_msg.encoding == 'PCM_32':
        audio_np = np.frombuffer(audio_msg.data, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"Unsupported audio encoding: {audio_msg.encoding}")

    # Resample if necessary
    if audio_msg.rate != 16000:
        import librosa
        audio_np = librosa.resample(audio_np, orig_sr=audio_msg.rate, target_sr=16000)

    return audio_np

def normalize_audio(self, audio_data):
    """Normalize audio to prevent clipping"""
    max_val = np.max(np.abs(audio_data))
    if max_val > 1.0:
        audio_data = audio_data / max_val
    return audio_data
```

## Troubleshooting Common Issues

### 1. Audio Quality Issues
**Problem**: Poor transcription accuracy
**Solutions**:
- Use noise reduction preprocessing
- Ensure proper microphone positioning
- Check audio input levels
- Use directional microphones

### 2. Performance Issues
**Problem**: Slow processing or high latency
**Solutions**:
- Use smaller Whisper models
- Optimize audio chunk sizes
- Use GPU acceleration
- Implement audio buffering

### 3. Memory Issues
**Problem**: High memory consumption
**Solutions**:
- Use CPU instead of GPU for smaller models
- Process audio in smaller chunks
- Implement memory cleanup
- Use quantized models

## Best Practices

### 1. Audio Preprocessing
- Apply noise reduction filters
- Normalize audio levels
- Use appropriate sample rates
- Implement silence detection

### 2. Model Selection
- Choose model size based on accuracy requirements
- Consider computational constraints
- Test with domain-specific audio
- Use appropriate languages

### 3. Integration
- Implement proper error handling
- Use appropriate ROS 2 QoS settings
- Implement buffering for smooth operation
- Monitor performance metrics

## Exercise

Create a complete Whisper integration for your humanoid robot that includes:

1. Real-time audio capture from the robot's microphone
2. Whisper-based speech-to-text processing
3. Integration with ROS 2 for message passing
4. Voice activity detection to reduce processing overhead
5. Performance optimization for real-time operation
6. Error handling for various audio conditions

Test your system with various commands and evaluate the accuracy and response time.