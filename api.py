from fastapi import FastAPI, BackgroundTasks
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import requests
import minio_OPS
import subprocess
import os
import newpos
import time
from typing import Dict

bucketname_video_original = "uploaded-videos"
bucketname_video_final = "processed-videos"
bucketname_heatmap = "heatmaps"

app = FastAPI()

class UploadRequest(BaseModel):
    fileName: str
    gameId: int

# Dictionary to store progress updates
progress_updates: Dict[int, list] = {}

def event_generator(gameId: int):
    """Generator function to stream progress updates for SSE."""
    while True:
        if gameId in progress_updates and progress_updates[gameId]:
            message = progress_updates[gameId].pop(0)
            yield f"data: {message}\n\n"
        time.sleep(1)

@app.get("/events/{gameId}")
async def sse_events(gameId: int):
    """SSE endpoint to stream progress updates."""
    return StreamingResponse(event_generator(gameId), media_type="text/event-stream")

def send_progress(gameId: int, status: str, progress: int):
    """Send progress update to the SSE event stream."""
    if gameId not in progress_updates:
        progress_updates[gameId] = []
    progress_updates[gameId].append(f'{{"status": "{status}", "progress": {progress}}}')

def process_video_task(fileName: str, gameId: int):
    """Background task to process video and send progress updates."""
    start_time = time.time()

    try:
        send_progress(gameId, "Starting download", 5)
        minioclient = minio_OPS.get_minio_client()
        minio_OPS.download_file(minioclient, bucketname_video_original, fileName, f'./{fileName}')
        print(f"File {fileName} downloaded successfully")
        send_progress(gameId, "Download complete", 15)

        os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

        send_progress(gameId, "Processing video", 30)
        team_a_possession, team_b_possession, video_output_name = newpos.process_video(f'./{fileName}')

        send_progress(gameId, "Generating heatmaps", 50)
        process2 = subprocess.Popen(["python", "finalRadar.py", fileName], env=os.environ)
        process2.wait()

        base_name = os.path.splitext(fileName)[0]

        send_progress(gameId, "Uploading processed files", 70)
        minio_OPS.upload_file(minioclient, bucketname_video_final, fileName, f'./{video_output_name}')
        minio_OPS.upload_file(minioclient, bucketname_heatmap, f'team_a_{base_name}.png', "./radarFolder/team_a_with_ball_heatmap.png")
        minio_OPS.upload_file(minioclient, bucketname_heatmap, f'team_b_{base_name}.png', "./radarFolder/team_b_with_ball_heatmap.png")
        send_progress(gameId, "Upload complete", 85)

        minio_OPS.delete_file(minioclient, bucketname_video_original, fileName)
        send_progress(gameId, "Cleaning up files", 90)

        files_to_delete = [
            f"./radarFolder/team_a_with_ball_heatmap.png",
            f"./radarFolder/team_b_with_ball_heatmap.png",
            f'./{video_output_name}',
            f"./{fileName}"
        ]
        for file_path in files_to_delete:
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"Deleted: {file_path}")

        execution_time = time.time() - start_time
        print(f"Execution time: {execution_time:.2f} seconds")

        send_progress(gameId, "Sending stats", 95)
        api_url = "http://backend:3333/stats"
        payload = {
            "gameId": gameId,
            "videoName": fileName,
            "heatmapTeamA": f'team_a_{base_name}.png',
            "heatmapTeamB": f'team_b_{base_name}.png',
            "possessionTeamA": team_a_possession,
            "possessionTeamB": team_b_possession
        }
        try:
            response = requests.post(api_url, json=payload)
            print("Possession data sent:", response.json())
        except Exception as e:
            print(f"Error sending possession data: {e}")

        send_progress(gameId, "Processing complete!", 100)
    except Exception as e:
        send_progress(gameId, f"Error: {str(e)}", 0)

@app.post("/upload")
async def uploadvideo(request: UploadRequest, background_tasks: BackgroundTasks):
    """Starts video processing in the background and returns immediately."""
    background_tasks.add_task(process_video_task, request.fileName, request.gameId)
    return {"message": f"Processing for {request.fileName} started"}
