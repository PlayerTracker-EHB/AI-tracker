import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from ultralytics import YOLO

# Load YOLO models
player_model = YOLO("./new_trained_player_model.pt")
ball_model = YOLO("./new_trained_ball_model.pt")

# Create radarFolder if it doesn't exist
os.makedirs("radarFolder", exist_ok=True)

# Function to process the video
def process_video(video_path):
    # Open video and get frame dimensions
    cap = cv2.VideoCapture(video_path)
    ret, sample_frame = cap.read()
    if not ret:
        print("Error: Could not read video file.")
        return

    frame_height, frame_width, _ = sample_frame.shape

    # Store team and ball positions
    team_a_positions = []
    team_b_positions = []
    team_a_ball_positions = []
    team_b_ball_positions = []

    # Define futsal field dimensions
    FIELD_WIDTH = 40  # Meters
    FIELD_HEIGHT = 20  # Meters

    # Function to detect objects using YOLO
    def detect_objects(model, frame):
        results = model(frame)[0]
        detections = []

        if results.boxes is not None and len(results.boxes.data) > 0:
            for result in results.boxes.data:
                x1, y1, x2, y2, _, _ = result.cpu().numpy()
                x_center = (x1 + x2) / 2
                y_center = (y1 + y2) / 2
                bbox = [int(x1), int(y1), int(x2), int(y2)]
                detections.append((x_center, y_center, bbox))
        
        return detections

    # Function to assign a player to a team using jersey color
    def assign_team(frame, bbox, team_centers):
        x1, y1, x2, y2 = bbox
        jersey_region = frame[y1:y1 + int((y2 - y1) * 0.5), x1:x2]
        avg_color = compute_avg_color_lab(jersey_region)

        distances = np.linalg.norm(team_centers - avg_color, axis=1)
        return "Team A" if np.argmin(distances) == 0 else "Team B"

    # Function to compute average color in LAB space
    def compute_avg_color_lab(image):
        if image.size == 0:
            return np.array([0, 0, 0], dtype=np.float32)
        lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        return np.mean(lab_image.reshape(-1, 3), axis=0)

    # Function to draw the futsal field
    def draw_field(ax):
        ax.set_xlim(0, FIELD_WIDTH)
        ax.set_ylim(0, FIELD_HEIGHT)
        ax.set_xlabel("Field Width (meters)")
        ax.set_ylabel("Field Height (meters)")

        # Draw field boundary (outer rectangle)
        ax.plot([0, FIELD_WIDTH, FIELD_WIDTH, 0, 0], [0, 0, FIELD_HEIGHT, FIELD_HEIGHT, 0], color="black", linewidth=4)

        # Center circle (3m radius)
        center_circle = plt.Circle((FIELD_WIDTH / 2, FIELD_HEIGHT / 2), 3, color="black", fill=False, linewidth=4)
        ax.add_patch(center_circle)

        # Mid-line
        ax.plot([FIELD_WIDTH / 2, FIELD_WIDTH / 2], [0, FIELD_HEIGHT], color="black", linewidth=4)

        # Penalty areas (6m from goal line, 10m wide)
        penalty_area_width = 3
        penalty_area_depth = 2

        # Left penalty area
        ax.plot([0, penalty_area_depth, penalty_area_depth, 0], 
                [(FIELD_HEIGHT / 2) - (penalty_area_width / 2), (FIELD_HEIGHT / 2) - (penalty_area_width / 2),
                 (FIELD_HEIGHT / 2) + (penalty_area_width / 2), (FIELD_HEIGHT / 2) + (penalty_area_width / 2)],
                color="black", linewidth=4)

        # Right penalty area
        ax.plot([FIELD_WIDTH, FIELD_WIDTH - penalty_area_depth, FIELD_WIDTH - penalty_area_depth, FIELD_WIDTH], 
                [(FIELD_HEIGHT / 2) - (penalty_area_width / 2), (FIELD_HEIGHT / 2) - (penalty_area_width / 2),
                 (FIELD_HEIGHT / 2) + (penalty_area_width / 2), (FIELD_HEIGHT / 2) + (penalty_area_width / 2)],
                color="black", linewidth=4)

        # Penalty spots (6m from goal line)
        ax.scatter([6, FIELD_WIDTH - 6], [FIELD_HEIGHT / 2, FIELD_HEIGHT / 2], color="black", s=50)
        # Double penalty spots (10m from goal line)
        ax.scatter([10, FIELD_WIDTH - 10], [FIELD_HEIGHT / 2, FIELD_HEIGHT / 2], color="black", s=30)

        # Goalposts with crossbar
        goal_width = 3
        goal_depth = 2
        goal_y_top = (FIELD_HEIGHT / 2) + (goal_width / 2)
        goal_y_bottom = (FIELD_HEIGHT / 2) - (goal_width / 2)

        # Left goal
        ax.plot([-goal_depth, 0], [goal_y_top, goal_y_top], color="red", linewidth=5)  # Crossbar
        ax.plot([-goal_depth, -goal_depth], [goal_y_top, goal_y_bottom], color="red", linewidth=5)  # Left post
        ax.plot([-goal_depth, 0], [goal_y_bottom, goal_y_bottom], color="red", linewidth=5)  # Bottom bar

        # Right goal
        ax.plot([FIELD_WIDTH, FIELD_WIDTH + goal_depth], [goal_y_top, goal_y_top], color="red", linewidth=5)  # Crossbar
        ax.plot([FIELD_WIDTH + goal_depth, FIELD_WIDTH + goal_depth], [goal_y_top, goal_y_bottom], color="red", linewidth=5)  # Right post
        ax.plot([FIELD_WIDTH, FIELD_WIDTH + goal_depth], [goal_y_bottom, goal_y_bottom], color="red", linewidth=5)  # Bottom bar

        # Free-throw arcs (dashed lines, 6m from goal)
        penalty_arc_radius = 6
        theta = np.linspace(-np.pi / 2, np.pi / 2, 100)

        left_arc_x = penalty_arc_radius * np.cos(theta)
        left_arc_y = penalty_arc_radius * np.sin(theta) + FIELD_HEIGHT / 2
        ax.plot(left_arc_x, left_arc_y, color="black", linestyle="dashed", linewidth=4)

        right_arc_x = FIELD_WIDTH - penalty_arc_radius * np.cos(theta)
        right_arc_y = penalty_arc_radius * np.sin(theta) + FIELD_HEIGHT / 2
        ax.plot(right_arc_x, right_arc_y, color="black", linestyle="dashed", linewidth=4)

        # Legends for elements
       


    # Process video frame-by-frame
    global_team_centers = None  # Stores LAB color centers for Team A and B

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        player_detections = detect_objects(player_model, frame)
        ball_detections = detect_objects(ball_model, frame)

        # Extract jersey colors for first frames to determine team clusters
        jersey_colors = []
        bboxes = []
        for x_center, y_center, bbox in player_detections:
            jersey_colors.append(compute_avg_color_lab(frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]))
            bboxes.append(bbox)

        # Initialize team color clusters
        if global_team_centers is None and len(jersey_colors) >= 2:
            data = np.array(jersey_colors, dtype=np.float32).reshape(-1, 3)
            _, _, centers = cv2.kmeans(data, 2, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0), 10, cv2.KMEANS_RANDOM_CENTERS)
            global_team_centers = centers[np.argsort(centers[:, 0])]  # Sort by brightness

        # Assign detected players to teams and store their positions
        for (x_center, y_center, bbox) in player_detections:
            if global_team_centers is not None:
                team = assign_team(frame, bbox, global_team_centers)
            else:
                team = "Team A"  # Default assignment if clustering hasn't initialized

            # Normalize positions to field dimensions
            x_scaled = (x_center / frame_width) * FIELD_WIDTH
            y_scaled = (y_center / frame_height) * FIELD_HEIGHT

            if team == "Team A":
                team_a_positions.append((x_scaled, y_scaled))
            else:
                team_b_positions.append((x_scaled, y_scaled))

        # Store detected ball positions and assign them to the nearest team
        for x_center, y_center, _ in ball_detections:
            x_scaled = (x_center / frame_width) * FIELD_WIDTH
            y_scaled = (y_center / frame_height) * FIELD_HEIGHT

            # Assign ball to the closest team (nearest detected player)
            if team_a_positions and team_b_positions:
                nearest_team = "Team A" if np.min([np.linalg.norm(np.array([x_scaled, y_scaled]) - np.array(p)) for p in team_a_positions]) < \
                                            np.min([np.linalg.norm(np.array([x_scaled, y_scaled]) - np.array(p)) for p in team_b_positions]) else "Team B"
            else:
                nearest_team = "Team A"  # Default if no players detected

            if nearest_team == "Team A":
                team_a_ball_positions.append((x_scaled, y_scaled))
            else:
                team_b_ball_positions.append((x_scaled, y_scaled))

    cap.release()

    # Plot Team A heatmap with ball positions
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.set_title("Team A Heatmap (Players & Ball) on Futsal Field")
    draw_field(ax)
    if team_a_positions:
        x_team_a, y_team_a = zip(*team_a_positions)
        ax.scatter(x_team_a, y_team_a, color="blue", alpha=0.5, label="Team A")
    if team_a_ball_positions:
        x_ball_a, y_ball_a = zip(*team_a_ball_positions)
        ax.scatter(x_ball_a, y_ball_a, color="yellow", alpha=0.8, label="Ball")
    ax.legend()
    plt.savefig("radarFolder/team_a_with_ball_heatmap.png", dpi=300)

    # Plot Team B heatmap with ball positions
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.set_title("Team B Heatmap (Players & Ball) on Futsal Field")
    draw_field(ax)
    if team_b_positions:
        x_team_b, y_team_b = zip(*team_b_positions)
        ax.scatter(x_team_b, y_team_b, color="red", alpha=0.5, label="Team B")
    if team_b_ball_positions:
        x_ball_b, y_ball_b = zip(*team_b_ball_positions)
        ax.scatter(x_ball_b, y_ball_b, color="yellow", alpha=0.8, label="Ball")
    ax.legend()
    plt.savefig("radarFolder/team_b_with_ball_heatmap.png", dpi=300)


# Main entry point
if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description="Process a video for team possession analysis")
    parser.add_argument('video_filepath', type=str, help='Path to the video file')
    args = parser.parse_args()

    # Process the video
    process_video(args.video_filepath)
