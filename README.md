# Meme Detection Project

A Python project that uses MediaPipe landmarker models to detect face and hand landmarks and generate meme-style outputs.

## Project Structure

- `main.py` - Main application script.
- `models/face_landmarker.task` - Face landmark model.
- `models/hand_landmarker.task` - Hand landmark model.
- `*.png` - Output/reference images.

## Requirements

- Python 3.10+
- Dependencies used by `main.py` (install with pip)

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```bash
python main.py
```

## Notes

- Keep the `.task` files inside the `models/` folder.
- The included `.gitignore` avoids committing local environment and cache files.
