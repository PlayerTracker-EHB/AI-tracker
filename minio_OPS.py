from minio import Minio
import os
 
# MinIO Configuration
MINIO_ENDPOINT = "minio:9000"
ACCESS_KEY = "H4tQkzvDMOkK65LhENXb"
SECRET_KEY = "O055CIQenwTRWyT7PzuvOpHfsF88RVo6QoFLgnYS"
SECURE = False  # Set to True if using HTTPS
 
# Initialize MinIO Client
def get_minio_client():
    return Minio(MINIO_ENDPOINT, access_key=ACCESS_KEY, secret_key=SECRET_KEY, secure=SECURE)
 
# Ensure bucket exists
def ensure_bucket(minio_client, bucket_name):
    if not minio_client.bucket_exists(bucket_name):
        minio_client.make_bucket(bucket_name)
        print(f"Bucket '{bucket_name}' created.")
    else:
        print(f"Bucket '{bucket_name}' already exists.")
 
# Upload a file to MinIO
def upload_file(minio_client, bucket_name, object_name, file_path):
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' not found.")
        return
    minio_client.fput_object(bucket_name, object_name, file_path)
    print(f"Uploaded '{file_path}' as '{object_name}' to bucket '{bucket_name}'.")
 
# Download a file from MinIO
def download_file(minio_client, bucket_name, object_name, download_path):
    minio_client.fget_object(bucket_name, object_name, download_path)
    print(f"Downloaded '{object_name}' from bucket '{bucket_name}' as '{download_path}'.")
 
# Delete a file from MinIO
def delete_file(minio_client, bucket_name, object_name):
    try:
        minio_client.remove_object(bucket_name, object_name)
        print(f"Deleted '{object_name}' from bucket '{bucket_name}'.")
    except Exception as e:
        print(f"Error deleting '{object_name}': {e}")
 
# List all buckets
def list_buckets(minio_client):
    buckets = minio_client.list_buckets()
    for bucket in buckets:
        print(bucket.name, bucket.creation_date)
 
# Main Execution
if __name__ == "__main__":
    minio_client = get_minio_client()
 
#delete_file(minio_client,"uploaded-videos","test_video.mp4")
 
#upload_file(minio_client,"uploaded-videos","test_video.mp4","./demo-output_possession.mp4")
 
#download_file(minio_client,"uploaded-videos","test_video.mp4","downloaded-test-video.mp4")