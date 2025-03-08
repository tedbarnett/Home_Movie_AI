import warnings
import whisper
import subprocess
import csv
import sys
import os
import time
from datetime import datetime
from typing import List, Tuple
from scenedetect import open_video, SceneManager
from scenedetect.detectors import ContentDetector

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning)  # Whisper FP16 warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)  # scenedetect warnings
warnings.filterwarnings("ignore", category=FutureWarning)  # torch.load future warnings

# ----- Configuration -----
TEMP_AUDIO = "temp_audio.wav"
WHISPER_MODEL = "large"  # Choose small, medium, or large based on accuracy/performance

# ----- Functions -----
def format_time(seconds: float) -> str:
    mins, secs = divmod(int(seconds), 60)
    return f"{mins}:{secs:02d}"

def detect_scenes(video_path: str) -> List[Tuple[float, float]]:
    video = open_video(video_path)
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector())

    scene_manager.detect_scenes(video)
    scene_list = scene_manager.get_scene_list()

    timestamps = [(start.get_seconds(), end.get_seconds()) for start, end in scene_list]
    return timestamps

def extract_audio(video_path: str, audio_path: str) -> None:
    command = f'ffmpeg -y -i "{video_path}" -q:a 0 -map a "{audio_path}"'
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"❌ ffmpeg error: {result.stderr}")
        sys.exit(1)

def transcribe_audio(audio_path: str, model: str):
    whisper_model = whisper.load_model(model, device="cuda")
    result = whisper_model.transcribe(audio_path, word_timestamps=True)
    return result

def assign_transcripts_to_scenes(transcript_data, scenes):
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

# ----- Main Execution -----
def main():
    start_time = time.time()

    if len(sys.argv) > 1:
        FOLDER_PATH = sys.argv[1]
    else:
        FOLDER_PATH = input("Enter the folder path containing videos to process: ")

    if not os.path.exists(FOLDER_PATH):
        print(f"❌ Error: Folder '{FOLDER_PATH}' not found.")
        sys.exit(1)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    results_folder = "Results"
    os.makedirs(results_folder, exist_ok=True)
    results_csv_filename = os.path.join(results_folder, f"Scenes-{timestamp}.csv")

    video_files = [f for f in os.listdir(FOLDER_PATH) if f.lower().endswith(".mp4")]
    if not video_files:
        print("⚠️ No MP4 files found in the specified folder.")
        sys.exit(0)

    total_scenes = 0
    total_words = 0

    with open(results_csv_filename, "w", newline="", encoding="utf-8") as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["Filename", "Scene Number", "Start Time", "End Time", "Transcript"])

        for video_file in video_files:
            VIDEO_FILE = os.path.join(FOLDER_PATH, video_file)
            file_start_time = time.time()
            print(f"*** Processing file: {video_file}")

            scenes = detect_scenes(VIDEO_FILE)
            if not scenes:
                print("- Results: 0 scenes, 0 words, 00:00 processing time")
                continue

            extract_audio(VIDEO_FILE, TEMP_AUDIO)
            transcript_data = transcribe_audio(TEMP_AUDIO, WHISPER_MODEL)
            os.remove(TEMP_AUDIO)

            scene_transcripts = assign_transcripts_to_scenes(transcript_data, scenes)

            for idx, ((start, end), text) in enumerate(zip(scenes, scene_transcripts), 1):
                csv_writer.writerow([video_file, idx, format_time(start), format_time(end), text.strip()])

            file_processing_time = time.time() - file_start_time
            file_mins, file_secs = divmod(int(file_processing_time), 60)
            file_words = sum(len(t.split()) for t in scene_transcripts)

            print(f"- Results: {len(scenes)} scenes, {file_words} words, {file_mins:02d}:{file_secs:02d} processing time")

    total_processing_time = time.time() - start_time
    mins, secs = divmod(int(total_processing_time), 60)

    print(f"\n✅ All files processed. Results saved to '{results_csv_filename}'")
    print(f"Whisper model = {WHISPER_MODEL}")
    print(f"Total processing time: {mins:02d}:{secs:02d}")
    print(f"Total # of scenes: {total_scenes}")
    print(f"Total # of words transcribed: {total_words}")

if __name__ == "__main__":
    main()
