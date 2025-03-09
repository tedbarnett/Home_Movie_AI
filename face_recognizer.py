import warnings
import whisper
import subprocess
import csv
import sys
import os
import time
import logging
import face_recognition
import cv2
import pickle
import configparser  # Add this import
from datetime import datetime
from typing import List, Tuple
from scenedetect import open_video, SceneManager
from scenedetect.detectors import ContentDetector
from sklearn.cluster import DBSCAN
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning)  # Whisper FP16 warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)  # scenedetect warnings
warnings.filterwarnings("ignore", category=FutureWarning)  # torch.load future warnings

# ----- Configuration -----
TEMP_AUDIO = "temp_audio.wav"
WHISPER_MODEL = "large"  # Choose small, medium, or large based on accuracy/performance
model = whisper.load_model(WHISPER_MODEL)

VIDEO_EXTENSIONS = (".mp4", ".wmv", ".mov", ".3gp", ".m4v")
ENCODINGS_PATH = os.path.join('setup', 'encodings.pickle')

# Read configuration from setup/config.ini
config = configparser.ConfigParser()
config.read('setup/config.ini')

VIDEO_PATH = config['DEFAULT']['VIDEO_PATH']
OUTPUT_CSV = config['DEFAULT']['OUTPUT_CSV']
FORCE_REBUILD_ENCODINGS = config['DEFAULT'].getboolean('FORCE_REBUILD_ENCODINGS', False)

# Load known face encodings
def load_known_faces(encodings_path: str):
    with open(encodings_path, 'rb') as f:
        data = pickle.load(f)
        return data["encodings"], data["names"]

# Function to create encodings.pickle
def create_encodings_pickle():
    print("Updating image encodings.pickle")
    all_encodings = []
    all_names = []

    for root, _, files in os.walk('known_faces'):
        for filename in files:
            if filename.endswith((".jpg", ".jpeg", ".png")):
                image_path = os.path.join(root, filename)
                print(f"Processing {filename}...")
                image = face_recognition.load_image_file(image_path)
                encodings = face_recognition.face_encodings(image)
                if encodings:
                    encoding = encodings[0]
                    name = os.path.splitext(filename)[0]
                    all_encodings.append(encoding)
                    all_names.append(name)

    # Cluster the encodings to group similar faces
    encodings_array = np.array(all_encodings)
    clustering = DBSCAN(eps=0.6, min_samples=1, metric='euclidean').fit(encodings_array)
    unique_encodings = []
    unique_names = []

    for cluster_id in set(clustering.labels_):
        cluster_indices = np.where(clustering.labels_ == cluster_id)[0]
        unique_encodings.append(all_encodings[cluster_indices[0]])
        unique_names.append(all_names[cluster_indices[0]])

    data = {"encodings": unique_encodings, "names": unique_names}
    with open(ENCODINGS_PATH, 'wb') as f:
        pickle.dump(data, f)

# Check if any files in the known_faces folder are newer than encodings.pickle
def check_and_update_encodings():
    if FORCE_REBUILD_ENCODINGS:
        create_encodings_pickle()
        return

    if not os.path.exists(ENCODINGS_PATH):
        create_encodings_pickle()
        return

    encodings_mtime = os.path.getmtime(ENCODINGS_PATH)
    for root, _, files in os.walk('known_faces'):
        for filename in files:
            file_path = os.path.join(root, filename)
            if os.path.getmtime(file_path) > encodings_mtime:
                create_encodings_pickle()
                return

# Run the check and update encodings if necessary
check_and_update_encodings()

# Load known face encodings
known_encodings, known_names = load_known_faces(ENCODINGS_PATH)
logging.info(f"Loaded {len(known_encodings)} unique known faces.")

# ----- Functions -----
def format_time(seconds: float) -> str:
    """Format seconds into a MM:SS string."""
    mins, secs = divmod(int(seconds), 60)
    return f"{mins}:{secs:02d}"

def detect_scenes(video_path: str) -> List[Tuple[float, float]]:
    """Detect scenes in a video and return a list of (start, end) timestamps."""
    video = open_video(video_path)
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector())
    scene_manager.detect_scenes(video)

    scenes = scene_manager.get_scene_list()

    if not scenes:
        video_duration = video.duration.get_seconds()
        return [(0.0, video_duration)]

    timestamps = [(scene[0].get_seconds(), scene[1].get_seconds()) for scene in scenes]
    return timestamps

def video_has_audio(video_path: str) -> bool:
    """Check if a video file has an audio stream."""
    cmd = f'ffprobe -i "{video_path}" -show_streams -select_streams a -loglevel error'
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return bool(result.stdout.strip())

def extract_audio(video_path: str, audio_path: str) -> None:
    """Extract audio from a video file."""
    command = f'ffmpeg -y -i "{video_path}" -q:a 0 -map a "{audio_path}"'
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        logging.error(f"ffmpeg error: {result.stderr}")
        sys.exit(1)

def transcribe_audio(audio_path: str):
    """Transcribe audio using Whisper model."""
    return model.transcribe(audio_path, fp16=True)

def assign_transcripts_to_scenes(transcript_data, scenes):
    """Assign transcript segments to corresponding scenes."""
    scene_transcripts = ['' for _ in scenes]

    for segment in transcript_data["segments"]:
        seg_start = segment['start']
        seg_end = segment['end']
        seg_text = segment['text'].strip()

        for i, (scene_start, scene_end) in enumerate(scenes):
            if seg_start >= scene_start and seg_end <= scene_end:
                scene_transcripts[i] += seg_text + " "
                break

    return scene_transcripts

def recognize_faces_in_scene(frame, known_encodings, known_names):
    """Recognize faces in a given frame."""
    rgb_frame = frame[:, :, ::-1]  # Convert BGR to RGB
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    recognized_names = set()
    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_encodings, face_encoding)
        face_distances = face_recognition.face_distance(known_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            recognized_names.add(known_names[best_match_index])

    return recognized_names

def main():
    start_time = time.time()

    if len(sys.argv) > 1:
        FOLDER_PATH = sys.argv[1]
    else:
        FOLDER_PATH = input("Enter the folder path containing videos to process: ")

    if not os.path.exists(FOLDER_PATH):
        logging.error(f"Folder '{FOLDER_PATH}' not found.")
        sys.exit(1)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    results_folder = "Results"
    os.makedirs(results_folder, exist_ok=True)
    results_csv_filename = os.path.join(results_folder, f"Scenes-{timestamp}.csv")

    video_files = [f for f in os.listdir(FOLDER_PATH) if f.lower().endswith(VIDEO_EXTENSIONS)]

    if not video_files:
        logging.warning("No video files found in the specified folder.")
        sys.exit(0)

    logging.info(f"STARTING: {len(video_files)} files in folder {FOLDER_PATH}")

    total_scenes = 0
    total_words = 0

    with open(results_csv_filename, "w", newline="", encoding="utf-8") as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["Filename", "Scene Number", "Start Time", "End Time", "Transcript", "Recognized Faces"])

        for video_file in video_files:
            VIDEO_FILE = os.path.join(FOLDER_PATH, video_file)
            file_start_time = time.time()
            logging.info(f"Processing file: {video_file}")

            scenes = detect_scenes(VIDEO_FILE)

            if video_has_audio(VIDEO_FILE):
                extract_audio(VIDEO_FILE, TEMP_AUDIO)
                transcript_data = transcribe_audio(TEMP_AUDIO)
                os.remove(TEMP_AUDIO)
                scene_transcripts = assign_transcripts_to_scenes(transcript_data, scenes)
            else:
                logging.warning(f"No audio stream in '{video_file}'. Skipping transcription.")
                scene_transcripts = ["[No audio]" for _ in scenes]

            # Open video with OpenCV
            video_capture = cv2.VideoCapture(VIDEO_FILE)
            fps = video_capture.get(cv2.CAP_PROP_FPS)

            for idx, ((start, end), text) in enumerate(zip(scenes, scene_transcripts), 1):
                mid_frame_no = int(((start + end) / 2) * fps)
                video_capture.set(cv2.CAP_PROP_POS_FRAMES, mid_frame_no)
                success, frame = video_capture.read()

                if success:
                    recognized_names = recognize_faces_in_scene(frame, known_encodings, known_names)
                    csv_writer.writerow([video_file, idx, format_time(start), format_time(end), text.strip(), ", ".join(recognized_names)])
                else:
                    csv_writer.writerow([video_file, idx, format_time(start), format_time(end), text.strip(), ""])

            video_capture.release()

            file_processing_time = time.time() - file_start_time
            file_mins, file_secs = divmod(int(file_processing_time), 60)
            file_words = sum(len(t.split()) for t in scene_transcripts)

            total_scenes += len(scenes)
            total_words += file_words

            logging.info(f"Results: {len(scenes)} scenes, {file_words} words, {file_mins:02d}:{file_secs:02d} processing time")

    total_processing_time = time.time() - start_time
    mins, secs = divmod(int(total_processing_time), 60)

    logging.info(f"All files processed. Results saved to '{results_csv_filename}'")
    logging.info(f"Whisper model = {WHISPER_MODEL}")
    logging.info(f"Total processing time: {mins:02d}:{secs:02d}")
    logging.info(f"Total # of scenes: {total_scenes}")
    logging.info(f"Total # of words transcribed: {total_words}")

if __name__ == "__main__":
    main()