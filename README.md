# Object Tracking System

A real-time object tracking system using YOLO for object detection and a custom tracker for maintaining object identities across frames.

## Features

- Real-time object detection using YOLOv8 nano model
- Object tracking with ID assignment and persistence
- Visual tracking display with colored bounding boxes
- Camera input support

## Installation

1. Clone the repository:
```bash
git clone <your-repo-url>
cd object_tracking
```

2. Create a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the main script to start tracking objects from your camera:

```bash
python main.py
```

Press 'q' to quit the application.

## Configuration

Adjust tracking parameters in `config.py`:
- `DIST_THRESHOLD`: Maximum distance for matching detections to tracks
- `MISS_THRESHOLD`: Frames to wait before removing a lost track
- `CONFIDENCE_THRESHOLD`: Minimum confidence for detections

## Project Structure

- `main.py`: Main application entry point
- `detector.py`: YOLO-based object detection
- `tracker.py`: Object tracking logic
- `drawing.py`: Visualization utilities
- `geometry.py`: Geometric calculations
- `matching.py`: Hungarian algorithm for assignment
- `models.py`: Data structures for detections and tracks
- `config.py`: Configuration constants