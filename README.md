# Home Movie AI

A Python script to automatically process home movies by:

- Detecting scene changes
- Extracting and transcribing audio using OpenAI's Whisper model
- Assigning transcript segments to corresponding scenes
- Generating a CSV file with scene numbers, timestamps, and transcripts

## Features
- Scene detection using [PySceneDetect](https://scenedetect.com/)
- Audio extraction with [FFmpeg](https://ffmpeg.org/)
- Speech-to-text transcription using [Whisper](https://github.com/openai/whisper)

## Requirements
- Python 3.8 or newer
- CUDA-enabled GPU recommended for performance (Whisper transcription)

## Installation
```bash
pip install scenedetect openai-whisper torch torchvision torchaudio
```

Ensure you have FFmpeg installed and accessible from the command line.

## Usage
Run the script with the video filename as a command-line argument:

```bash
python home_movie_ai.py "your-video-file.mp4"
```

Or simply:

```bash
python home_movie_ai.py
```

The script will prompt for the filename if not provided in the command.

## Output
The script generates a CSV file named `results-[your-video-file].csv` with the following columns:

- Scene Number
- Start Time (min:sec)
- End Time (min:sec)
- Transcript

## License
This project is licensed under the [MIT License](https://opensource.org/licenses/MIT). Please provide attribution when using or modifying this software.

## Attribution

Created by [Ted Barnett](https://github.com/tedbarnett). Contributions and forks are welcome!

