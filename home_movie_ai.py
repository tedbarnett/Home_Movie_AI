import warnings
import whisper
import subprocess
import csv
import sys
import os
from typing import List, Tuple
from scenedetect import open_video, SceneManager
from scenedetect.detectors import ContentDetector

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning)  # Whisper FP16 warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)  # scenedetect warnings
warnings.filterwarnings("ignore", category=FutureWarning)  # torch.load future warnings

# ----- Configuration -----
TEMP_AUDIO = "temp_audio.wav"
WHISPER_MODEL = "medium"  # Choose small, medium, or large based on accuracy/performance

# ----- Functions -----
def format_time(seconds: float) -> str:
    mins, secs = divmod(int(seconds), 60)
    return f"{mins}:{secs:02d}"

def detect_scenes(video_path: str) -> List[Tuple[float, float]]:
    from scenedetect import SceneManager
    from scenedetect.detectors import ContentDetector

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
    if len(sys.argv) > 1:
        VIDEO_FILE = sys.argv[1]
    else:
        VIDEO_FILE = input("Enter the filename to process (including extension, e.g., video.mp4): ")

    if not os.path.exists(VIDEO_FILE):
        print(f"❌ Error: File '{VIDEO_FILE}' not found.")
        sys.exit(1)

    video_basename = os.path.splitext(os.path.basename(VIDEO_FILE))[0]
    RESULTS_CSV = f"results-{video_basename}.csv"

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

    with open(f"results-{os.path.splitext(VIDEO_FILE)[0]}.csv", "w", newline="", encoding="utf-8") as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["Scene Number", "Start Time", "End Time", "Transcript"])

        for idx, ((start, end), text) in enumerate(zip(scenes, scene_transcripts), 1):
            csv_writer.writerow([idx, format_time(start), format_time(end), text.strip()])

    print("✅ Processing complete.")

if __name__ == "__main__":
    main()
