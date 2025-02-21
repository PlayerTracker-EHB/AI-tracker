from fastapi import FastAPI
from pydantic import BaseModel
import requests
import minio_OPS
import subprocess
import os
import newpos

bucketname_video_original = "videoupload"
bucketname_video_final = "videostats"
bucketname_heatmap = "heatmap"

app = FastAPI()

# ✅ Define a model for request body
class UploadRequest(BaseModel):
    fileName: str
    gameId: int


@app.post("/upload")
async def uploadvideo(request: UploadRequest):
    fileName = request.fileName
    gameId = request.gameId
    minioclient = minio_OPS.get_minio_client()
    minio_OPS.download_file(minioclient, bucketname_video_original, fileName, f'./{fileName}')
    print({"message": f"File {fileName} downloaded successfully"})
    
    # Ensure TensorFlow/PyTorch dynamically allocate memory (prevents out-of-memory issues)
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

    # Run both scripts in parallel
    team_a_possession, team_b_possession, video_output_name  = newpos.process_video(f'./{fileName}')
    process2 = subprocess.Popen(["python", "finalRadar.py", fileName], env=os.environ)


    # Wait for both scripts to complete
   
    process2.wait()

    print("Both scripts have completed!")

    base_name = os.path.splitext(fileName)[0]



    # Upload files to MinIO
    minio_OPS.upload_file(minioclient, bucketname_video_final, fileName, f'./{video_output_name}')
    minio_OPS.upload_file(minioclient, bucketname_heatmap, f'team_a_{base_name}.png', "./radarFolder/team_a_with_ball_heatmap.png")
    minio_OPS.upload_file(minioclient, bucketname_heatmap, f'team_b_{base_name}.png', "./radarFolder/team_b_with_ball_heatmap.png")

    # Delete files from MinIO
    minio_OPS.delete_file(minioclient, bucketname_video_original, fileName)

    # Delete local files after upload
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
            else:
                print(f"File not found: {file_path}")
        except Exception as e:
            print(f"Error deleting {file_path}: {e}")

    

   

     # Send possession data to the FastAPI server
    api_url = "http://127.0.0.1:8000/stats"  # Update with actual server URL if deployed
    payload = {
        "gameId" : gameId,
        "videoName": video_output_name,
        "heatmapTeamA": f'team_a_{base_name}.png',
        "heatmapTeamB": f'team_b_{base_name}.png',
        "possessionTeamA": team_a_possession,
        "possessionTeamB": team_b_possession
    }

    '''
    try:
        response = requests.post(api_url, json=payload)
        print("Possession data sent:", response.json())
    except Exception as e:
        print(f"Error sending possession data: {e}")
    '''


    return {"message": f"File {payload} processed and cleaned up successfully"}

    