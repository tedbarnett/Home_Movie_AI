import os
import whisper
import subprocess
import time
import csv
import sys
from typing import List, Tuple
import warnings

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning)  # Whisper FP16 warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)  # scenedetect warnings
warnings.filterwarnings("ignore", category=FutureWarning)  # torch.load future warnings

# ----- Configuration -----
TEMP_AUDIO = "temp_audio.wav"

# Whisper model selection
WHISPER_MODEL = "medium"  # Choose small, medium, or large based on accuracy/performance


def format_time(seconds: float) -> str:
    """Formats seconds into min:sec format."""
    mins, secs = divmod(int(seconds), 60)
    return f"{mins}:{secs:02d}"


# ----- Functions -----
def detect_scenes(video_path: str) -> List[Tuple[float, float]]:
    """Detects scene boundaries and returns timestamps as (start, end)."""
    from scenedetect import open_video, SceneManager
    from scenedetect.detectors import ContentDetector

    video = open_video(video_path)
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector())

    scene_manager.detect_scenes(video)
    scene_list = scene_manager.get_scene_list()

    timestamps = [(start.get_seconds(), end.get_seconds()) for start, end in scene_list]
    return timestamps


def extract_audio(video_path: str, audio_path: str) -> None:
    """Extracts audio from video."""
    command = f'ffmpeg -i "{video_path}" -q:a 0 -map a "{audio_path}" -y'
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {result.stderr}")


def transcribe_audio(audio_path: str, model: str):
    """Transcribes audio file to text using Whisper, with timestamps."""
    whisper_model = whisper.load_model(model, device="cuda")
    result = whisper_model.transcribe(audio_path, word_timestamps=True)
    return result


def assign_transcripts_to_scenes(transcript_data, scenes):
    """Assigns transcript segments to corresponding scenes based on timestamps."""
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
if __name__ == "__main__":
    start_time = time.time()

    if len(sys.argv) > 1:
        VIDEO_FILE = sys.argv[1]
    else:
        VIDEO_FILE = input("Enter the filename to process (including extension, e.g., video.mp4): ")

    if not os.path.exists(VIDEO_FILE):
        print(f"❌ Error: File '{VIDEO_FILE}' not found.")
        exit()

    video_basename = os.path.splitext(os.path.basename(VIDEO_FILE))[0]
    RESULTS_CSV = f"results-{video_basename}.csv"

    print(f"📹 Processing file: {VIDEO_FILE} ({os.path.getsize(VIDEO_FILE) / (1024 ** 2):.2f} MB)")

    print("\n🔍 Detecting scenes...")
    scenes = detect_scenes(VIDEO_FILE)

    print("\n🎧 Extracting audio...")
    extract_audio(VIDEO_FILE, TEMP_AUDIO)

    print("\n🗣️ Transcribing audio with Whisper...")
    transcript_data = transcribe_audio(TEMP_AUDIO, WHISPER_MODEL)

    # Cleanup temporary audio file
    os.remove(TEMP_AUDIO)

    print("\n📝 Assigning transcripts to scenes and writing CSV...")
    scene_transcripts = assign_transcripts_to_scenes(transcript_data, scenes)

    # Save results to CSV with formatted times
    with open(RESULTS_CSV, "w", newline="", encoding="utf-8") as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["Scene Number", "Start Time", "End Time", "Transcript"])

        for idx, ((start, end), text) in enumerate(zip(scenes, scene_transcripts), 1):
            csv_writer.writerow([idx, format_time(start), format_time(end), text.strip()])

    elapsed_time = round(time.time() - start_time, 2)

    print(f"\n✅ Processing complete in {elapsed_time}s. Results saved in '{RESULTS_CSV}'.")
    print(f"Scenes detected: {len(scenes)}")
    print(f"Transcript length: {sum(len(t.split()) for t in scene_transcripts)} words")
