import cv2
import numpy as np
import requests
from ultralytics import YOLO
import argparse

# Function to process the video
def process_video(video_filepath):
    # Load models
    player_model = YOLO("./new_trained_player_model.pt", verbose=False, show=True)
    ball_model = YOLO("./new_trained_ball_model.pt", verbose=False, show=True)

    # Open video file
    video = cv2.VideoCapture(video_filepath)
    if not video.isOpened():
        print("Error: Could not open video file!")
        exit()

    frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video.get(cv2.CAP_PROP_FPS))
    print(f"Video Loaded: {frame_width}x{frame_height} @ {fps} FPS")

    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter("output_possession.mp4", fourcc, fps, (frame_width, frame_height))

    # Global variables
    global_team_centers = None  
    global_team_labels = ["Team A", "Team B"]  
    team_possession = {"Team A": 0, "Team B": 0}  # Possession counter

    # Helper functions
    def extract_jersey_region(frame, bbox, fraction=0.5):
        x1, y1, x2, y2 = bbox
        jersey_height = int((y2 - y1) * fraction)
        return frame[y1:y1+jersey_height, x1:x2]

    def compute_avg_color_lab(image):
        if image.size == 0:
            return np.array([0, 0, 0], dtype=np.float32)
        lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        return np.mean(lab_image.reshape(-1, 3), axis=0)

    def assign_team(avg_color, team_centers):
        distances = np.linalg.norm(team_centers - avg_color, axis=1)
        return global_team_labels[np.argmin(distances)]

    def calculate_center(bbox):
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)

    # Processing loop
    frame_count = 0
    while True:
        ret, frame = video.read()
        if not ret:
            print("Processing complete! Video saved as 'output_possession.mp4'.")
            break

        frame_count += 1

        # Detect players
        player_results = player_model(frame)[0]
        ball_results = ball_model(frame)[0]

        jersey_colors, player_bboxes, detection_results = [], [], []

        if player_results.boxes is not None and len(player_results.boxes.data) > 0:
            for result in player_results.boxes.data:
                x1, y1, x2, y2, _, _ = result.cpu().numpy()
                bbox = [int(x1), int(y1), int(x2), int(y2)]
                player_bboxes.append(bbox)
                jersey_colors.append(compute_avg_color_lab(extract_jersey_region(frame, bbox, 0.5)))

        # Team assignment
        if len(jersey_colors) >= 2:
            if global_team_centers is None:
                data = np.array(jersey_colors, dtype=np.float32).reshape(-1, 3)
                _, labels, centers = cv2.kmeans(data, 2, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0), 10, cv2.KMEANS_RANDOM_CENTERS)
                global_team_centers = centers[np.argsort(centers[:, 0])] 

            for bbox, avg_color in zip(player_bboxes, jersey_colors):
                detection_results.append((bbox, assign_team(avg_color, global_team_centers)))
        else:
            detection_results = [(bbox, global_team_labels[0]) for bbox in player_bboxes]

        # Detect ball
        ball_bbox = None
        if ball_results.boxes is not None and len(ball_results.boxes.data) > 0:
            x1, y1, x2, y2, _, _ = ball_results.boxes.data[0].cpu().numpy()
            ball_bbox = [int(x1), int(y1), int(x2), int(y2)]

        # Determine possession
        if ball_bbox:
            ball_center = calculate_center(ball_bbox)
            min_distance = float("inf")
            closest_team = None

            for player_bbox, team in detection_results:
                player_center = calculate_center(player_bbox)
                distance = np.linalg.norm(np.array(player_center) - np.array(ball_center))
                if distance < min_distance:
                    min_distance = distance
                    closest_team = team

            if closest_team:
                team_possession[closest_team] += 1  # Increase possession counter

        # Calculate possession percentage in real-time
        total_frames = sum(team_possession.values())
        if total_frames > 0:
            team_a_possession = (team_possession["Team A"] / total_frames) * 100
            team_b_possession = (team_possession["Team B"] / total_frames) * 100
        else:
            team_a_possession = team_b_possession = 0

        # Draw results
        for bbox, team in detection_results:
            x1, y1, x2, y2 = bbox
            box_color = (255, 0, 0) if team == global_team_labels[0] else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
            cv2.putText(frame, team, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, box_color, 2)

        if ball_bbox:
            x1, y1, x2, y2 = ball_bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, "Ball", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Display possession percentages
        cv2.putText(frame, f"Team A Possession: {team_a_possession:.2f}%", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.putText(frame, f"Team B Possession: {team_b_possession:.2f}%", (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Write the frame
        out.write(frame)

    # Print final possession percentages
    print(f"Team A Possession: {team_a_possession:.2f}%")
    print(f"Team B Possession: {team_b_possession:.2f}%")

    # Release resources
    video.release()
    out.release()
    cv2.destroyAllWindows()

    return [team_a_possession, team_b_possession, "output_possession.mp4"]