import face_recognition
import pickle
import os
import configparser

# Load configuration
config = configparser.ConfigParser()
config.read('config.ini')

ENCODINGS_PATH = config['DEFAULT']['ENCODINGS_PATH']
KNOWN_FACES_DIR = 'known_faces'

# Initialize lists to hold encodings and names
known_encodings = []
known_names = []

# Loop through each person in the known faces directory
for name in os.listdir(KNOWN_FACES_DIR):
    person_dir = os.path.join(KNOWN_FACES_DIR, name)
    if not os.path.isdir(person_dir):
        continue

    # Loop through each image of the person
    for filename in os.listdir(person_dir):
        filepath = os.path.join(person_dir, filename)
        print(f"Processing {filepath}")
        image = face_recognition.load_image_file(filepath)
        encodings = face_recognition.face_encodings(image)

        if encodings:
            known_encodings.append(encodings[0])
            known_names.append(name)

# Save the encodings and names to a pickle file
with open(ENCODINGS_PATH, 'wb') as f:
    pickle.dump({'encodings': known_encodings, 'names': known_names}, f)

print(f'Encodings saved to {ENCODINGS_PATH}')import torch
print("CUDA available:", torch.cuda.is_available())
print("CUDA device count:", torch.cuda.device_count())
print("CUDA device name:", torch.cuda.get_device_name(0))