from fastapi import FastAPI, BackgroundTasks, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
import requests
import minio_OPS
import subprocess
import os
import newpos
import time

bucketname_video_original = "uploaded-videos"
bucketname_video_final = "processed-videos"
bucketname_heatmap = "heatmaps"

app = FastAPI()

class UploadRequest(BaseModel):
    fileName: str
    gameId: int

# Dictionary to store active WebSocket connections
clients = {}

@app.websocket("/ws/{gameId}")
async def websocket_endpoint(websocket: WebSocket, gameId: int):
    """WebSocket endpoint for sending real-time progress updates."""
    await websocket.accept()
    clients[gameId] = websocket  # Store the WebSocket connection
    try:
        while True:
            await websocket.receive_text()  # Keep connection open
    except WebSocketDisconnect:
        del clients[gameId]  # Remove disconnected client


async def process_video_task(fileName: str, gameId: int):
    """Background task to process video and send progress updates."""
    start_time = time.time()

    try:
        # Notify client: Starting download
        if gameId in clients:
            websocket = clients[gameId]
            await websocket.send_json({"status": "Starting download", "progress": 5})

        minioclient = minio_OPS.get_minio_client()
        minio_OPS.download_file(minioclient, bucketname_video_original, fileName, f'./{fileName}')
        print(f"File {fileName} downloaded successfully")

        # Notify client: Download complete
        if gameId in clients:
            await websocket.send_json({"status": "Download complete", "progress": 15})

        # GPU memory allocation settings
        os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

        # Notify client: Processing video
        if gameId in clients:
            await websocket.send_json({"status": "Processing video", "progress": 30})

        # Process video
        team_a_possession, team_b_possession, video_output_name = newpos.process_video(f'./{fileName}')

        # Notify client: Running finalRadar script
        if gameId in clients:
            await websocket.send_json({"status": "Generating heatmaps", "progress": 50})

        process2 = subprocess.Popen(["python", "finalRadar.py", fileName], env=os.environ)
        process2.wait()  # Ensure script finishes execution
        print("Processing completed!")

        base_name = os.path.splitext(fileName)[0]

        # Notify client: Uploading files
        if gameId in clients:
            await websocket.send_json({"status": "Uploading processed files", "progress": 70})

        # Upload processed files
        minio_OPS.upload_file(minioclient, bucketname_video_final, fileName, f'./{video_output_name}')
        minio_OPS.upload_file(minioclient, bucketname_heatmap, f'team_a_{base_name}.png', "./radarFolder/team_a_with_ball_heatmap.png")
        minio_OPS.upload_file(minioclient, bucketname_heatmap, f'team_b_{base_name}.png', "./radarFolder/team_b_with_ball_heatmap.png")

        # Notify client: Upload complete
        if gameId in clients:
            await websocket.send_json({"status": "Upload complete", "progress": 85})

        # Delete original video from MinIO
        minio_OPS.delete_file(minioclient, bucketname_video_original, fileName)

        # Notify client: Cleanup
        if gameId in clients:
            await websocket.send_json({"status": "Cleaning up files", "progress": 90})

        # Cleanup local files
        files_to_delete = [
            f"./radarFolder/team_a_with_ball_heatmap.png",
            f"./radarFolder/team_b_with_ball_heatmap.png",
            f'./{video_output_name}',
            f"./{fileName}"
        ]

        for file_path in files_to_delete:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    print(f"Deleted: {file_path}")
            except Exception as e:
                print(f"Error deleting {file_path}: {e}")

        execution_time = time.time() - start_time
        print(f"Execution time: {execution_time:.2f} seconds")

        # Notify client: Sending stats
        if gameId in clients:
            await websocket.send_json({"status": "Sending stats", "progress": 95})

        # Send processed data to backend
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

        # Notify client: Process complete
        if gameId in clients:
            await websocket.send_json({"status": "Processing complete!", "progress": 100})

    except Exception as e:
        if gameId in clients:
            await clients[gameId].send_json({"status": f"Error: {str(e)}", "progress": 0})


@app.post("/upload")
async def uploadvideo(request: UploadRequest, background_tasks: BackgroundTasks):
    """Starts video processing in the background and returns immediately."""
    background_tasks.add_task(process_video_task, request.fileName, request.gameId)
    return {"message": f"Processing for {request.fileName} started"}
