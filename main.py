import subprocess
import os

# Define the video file path
video_file = "./soso-video.mp4"

# Ensure TensorFlow/PyTorch dynamically allocate memory (prevents out-of-memory issues)
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

# Run both scripts in parallel
process1 = subprocess.Popen(["python", "newpos.py", video_file], env=os.environ)
process2 = subprocess.Popen(["python", "finalRadar.py", video_file], env=os.environ)

# Wait for both scripts to complete
process1.wait()
process2.wait()

print("Both scripts have completed!")
