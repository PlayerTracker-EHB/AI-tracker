# AI Futsal Tracker

## Overview

The **AI Futsal Tracker** is a video processing system that analyzes futsal game footage using AI techniques. It utilizes **YOLO11**, **OpenCV**, **MinIO**, and custom algorithms to extract possession statistics, generate heatmaps for each team, and produce an annotated video.

The system is designed to work with **FastAPI** as its backend. When a new video is uploaded to **MinIO**, the backend notifies the AI tracker, which downloads, processes, and returns the computed stats along with heatmaps and the processed video.

## Features

- **Player Detection & Tracking**: Uses YOLOv11 and OpenCV to detect players and the ball.
- **Possession Analysis**: Computes possession statistics for each team.
- **Heatmaps Generation**: Produces heatmaps per team to visualize movement and ball control.
- **Annotated Video Output**: Generates a video with real-time annotations of the game.
- **MinIO Integration**: Fetches videos from MinIO and uploads processed results.
- **FastAPI Communication**: Interacts with a backend system that manages video processing requests.

## Workflow

1. **Video Upload**: A futsal game video is uploaded to MinIO.
2. **Request to AI Tracker**: The backend sends a request to the AI Tracker, indicating a new video is available.
3. **Processing**:
   - The AI Tracker downloads the video from MinIO.
   - It runs object detection and tracking algorithms.
   - Possession statistics and heatmaps are computed.
   - Annotations are added to the video.
4. **Results Upload**:
   - The processed video and heatmaps are uploaded back to MinIO.
   - The AI Tracker returns the statistics and file names of the results to the backend.

## Technologies Used

- **YOLO11** – Object detection for player and ball tracking.
- **OpenCV** – Video processing and computer vision tasks.
- **MinIO** – Cloud storage for videos and processed results.
- **FastAPI** – Framework for API requests and processing coordination.
- **Python** – Core language for AI processing and backend communication.

## Installation

### Prerequisites

- Python 3.8+
- MinIO setup and credentials
- FastAPI backend configured
- Dependencies installed via `pip`

### Setup Instructions

1. Clone the repository:
   ```bash
   git clone https://github.com/yourrepo/ai-futsal-tracker.git
   cd ai-futsal-tracker
   ```
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Set up MinIO credentials in `.env`:
   ```ini
   MINIO_ENDPOINT=http://your-minio-url
   MINIO_ACCESS_KEY=your-access-key
   MINIO_SECRET_KEY=your-secret-key
   ```
5. Run the AI Tracker:
   ```bash
   python main.py
   ```

## Using Docker

Build the Docker Image

To containerize the AI Futsal Tracker, use the provided Dockerfile:

### Build the Docker image

docker build -t ai-futsal-tracker .

Run the Container

### Run the container with GPU access

docker run --gpus all -p 8000:8000 --env-file .env ai-futsal-tracker

### Dockerfile Overview

The Dockerfile is based on the PyTorch runtime image with CUDA support for GPU acceleration. It includes:

Setting environment variables for GPU access.

Installing required dependencies.

Copying application files into the container.

Running the FastAPI application with Uvicorn.

## API Endpoints

### Process Video

```http
POST /upload-video
```

**Request Body:**

```json
{
  "fileName": "match1.mp4",
  "gameId": 123
}
```

**Response:**

```json
{
  "gameId": 123,
  "videoName": "match1.mp4",
  "heatmapTeamA": "team_a_match1.png",
  "heatmapTeamB": "team_b_match1.png",
  "possessionTeamA": 55,
  "possessionTeamB": 45
}
```

## Future Improvements

- Improved stats (goals, passes, assists, interceptions)
- Advanced event detection (goals, fouls, passes)
- Improved tracking accuracy with additional AI models

⚠️ Notice

This application is computationally intensive and should be run on a system with a dedicated GPU for optimal performance. Running on a CPU may result in significantly slower processing times.
