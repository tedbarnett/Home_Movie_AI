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
        VIDEO_FILE = sys.argv[1]
    else:
        VIDEO_FILE = input("Enter the filename to process (including extension, e.g., video.mp4): ")

    if not os.path.exists(VIDEO_FILE):
        print(f"❌ Error: File '{VIDEO_FILE}' not found.")
        sys.exit(1)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    results_csv_filename = f"Scenes-{timestamp}.csv"

    print(f"📹 Processing file: {VIDEO_FILE} ({os.path.getsize(VIDEO_FILE) / (1024 ** 2):.2f} MB)")

    print("🔍 Detecting scenes...")
    scenes = detect_scenes(VIDEO_FILE)
    if not scenes:
        print("⚠️ No scenes detected.")
        sys.exit(0)

    print("🎧 Extracting audio...")
    extract_audio(VIDEO_FILE, TEMP_AUDIO)

    print("🗣️ Transcribing audio with Whisper...")
    transcript_data = transcribe_audio(TEMP_AUDIO, WHISPER_MODEL)

    os.remove(TEMP_AUDIO)

    print("📝 Assigning transcripts to scenes and writing CSV...")
    scene_transcripts = assign_transcripts_to_scenes(transcript_data, scenes)

    video_filename = os.path.basename(VIDEO_FILE)

    with open(results_csv_filename, "w", newline="", encoding="utf-8") as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["Filename", "Scene Number", "Start Time", "End Time", "Transcript"])

        for idx, ((start, end), text) in enumerate(zip(scenes, scene_transcripts), 1):
            csv_writer.writerow([video_filename, idx, format_time(start), format_time(end), text.strip()])

    total_processing_time = time.time() - start_time
    mins, secs = divmod(int(total_processing_time), 60)
    total_words = sum(len(t.split()) for t in scene_transcripts)

    print(f"✅ Processing complete. Results saved to '{results_csv_filename}'")
    print(f"Whisper model = {WHISPER_MODEL}")
    print(f"Total processing time: {mins:02d}:{secs:02d}")
    print(f"Total # of scenes: {len(scenes)}")
    print(f"Total # of words transcribed: {total_words}")

if __name__ == "__main__":
    main()
