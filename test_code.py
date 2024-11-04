import cv2
import numpy as np
import mediapipe as mp
import os
from collections import defaultdict
from mediapipe.framework.formats import landmark_pb2  # Import landmark_pb2 directly

# Define hand and pose landmarks as per your specification
hand_landmarks = [
    'INDEX_FINGER_DIP', 'INDEX_FINGER_MCP', 'INDEX_FINGER_PIP', 'INDEX_FINGER_TIP',
    'MIDDLE_FINGER_DIP', 'MIDDLE_FINGER_MCP', 'MIDDLE_FINGER_PIP', 'MIDDLE_FINGER_TIP',
    'PINKY_DIP', 'PINKY_MCP', 'PINKY_PIP', 'PINKY_TIP',
    'RING_FINGER_DIP', 'RING_FINGER_MCP', 'RING_FINGER_PIP', 'RING_FINGER_TIP',
    'THUMB_CMC', 'THUMB_IP', 'THUMB_MCP', 'THUMB_TIP', 'WRIST'
]

HAND_IDENTIFIERS = [id + "_right" for id in hand_landmarks] + [id + "_left" for id in hand_landmarks]
POSE_IDENTIFIERS = ["RIGHT_SHOULDER", "LEFT_SHOULDER", "LEFT_ELBOW", "RIGHT_ELBOW"]
body_identifiers = HAND_IDENTIFIERS + POSE_IDENTIFIERS  # Total of 46 keypoints

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# Function to find the index of the first non-zero element
def find_index(array):
    for i, num in enumerate(array):
        if num != 0:
            return i
    return -1  # Return -1 if no non-zero element is found

# Function to fill in missing keypoints
def curl_skeleton(array):
    array = list(array)
    if sum(array) == 0:
        return array
    for i, location in enumerate(array):
        if location != 0:
            continue
        else:
            if i == 0 or i == len(array) - 1:
                continue
            else:
                if array[i + 1] != 0:
                    array[i] = float((array[i - 1] + array[i + 1]) / 2)
                else:
                    j = find_index(array[i + 1:])
                    if j == -1:
                        continue
                    array[i] = float(((1 + j) * array[i - 1] + array[i + 1 + j]) / (2 + j))
    return array

def process_video(video_path, save_dir):
    cap = cv2.VideoCapture(video_path)
    mp_holistic_instance = mp_holistic.Holistic(
        min_detection_confidence=0.5, min_tracking_confidence=0.5)
    # Prepare a dictionary to store keypoints
    keypoint_data = defaultdict(list)
    frame_count = 0

    with mp_holistic_instance as holistic:
        while frame_count < 101:
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(image)

            # Process right hand
            if results.right_hand_landmarks:
                for idx, landmark in enumerate(results.right_hand_landmarks.landmark):
                    keypoint_data[f"{hand_landmarks[idx]}_right_x"].append(landmark.x)
                    keypoint_data[f"{hand_landmarks[idx]}_right_y"].append(landmark.y)
            else:
                for idx in range(len(hand_landmarks)):
                    keypoint_data[f"{hand_landmarks[idx]}_right_x"].append(0)
                    keypoint_data[f"{hand_landmarks[idx]}_right_y"].append(0)

            # Process left hand
            if results.left_hand_landmarks:
                for idx, landmark in enumerate(results.left_hand_landmarks.landmark):
                    keypoint_data[f"{hand_landmarks[idx]}_left_x"].append(landmark.x)
                    keypoint_data[f"{hand_landmarks[idx]}_left_y"].append(landmark.y)
            else:
                for idx in range(len(hand_landmarks)):
                    keypoint_data[f"{hand_landmarks[idx]}_left_x"].append(0)
                    keypoint_data[f"{hand_landmarks[idx]}_left_y"].append(0)

            # Process pose landmarks (shoulders and elbows)
            if results.pose_landmarks:
                landmark_dict = {mp_holistic.PoseLandmark(idx).name: idx for idx in range(len(mp_holistic.PoseLandmark))}
                for pose_identifier in POSE_IDENTIFIERS:
                    idx = landmark_dict.get(pose_identifier, None)
                    if idx is not None:
                        landmark = results.pose_landmarks.landmark[idx]
                        keypoint_data[f"{pose_identifier}_x"].append(landmark.x)
                        keypoint_data[f"{pose_identifier}_y"].append(landmark.y)
                    else:
                        keypoint_data[f"{pose_identifier}_x"].append(0)
                        keypoint_data[f"{pose_identifier}_y"].append(0)
            else:
                for pose_identifier in POSE_IDENTIFIERS:
                    keypoint_data[f"{pose_identifier}_x"].append(0)
                    keypoint_data[f"{pose_identifier}_y"].append(0)

    cap.release()

    # Process the keypoints
    T = frame_count  # Number of frames processed
    num_keypoints = len(body_identifiers)
    keypoints_all_frames = np.empty((T, num_keypoints, 2))

    for index, identifier in enumerate(body_identifiers):
        x_key = identifier + "_x"
        y_key = identifier + "_y"
        x_array = keypoint_data.get(x_key, [0]*T)
        y_array = keypoint_data.get(y_key, [0]*T)
        data_keypoint_preprocess_x = curl_skeleton(x_array)
        data_keypoint_preprocess_y = curl_skeleton(y_array)
        keypoints_all_frames[:, index, 0] = np.asarray(data_keypoint_preprocess_x)
        keypoints_all_frames[:, index, 1] = np.asarray(data_keypoint_preprocess_y)

    # Map keypoint names to indices
    keypoint_indices = {}

    # Right hand landmarks
    for idx, landmark in enumerate(hand_landmarks):
        keypoint_indices[landmark + '_right'] = idx

    # Left hand landmarks
    for idx, landmark in enumerate(hand_landmarks):
        keypoint_indices[landmark + '_left'] = idx + len(hand_landmarks)

    # Pose landmarks
    keypoint_indices['RIGHT_SHOULDER'] = 2 * len(hand_landmarks)
    keypoint_indices['LEFT_SHOULDER'] = 2 * len(hand_landmarks) + 1
    keypoint_indices['LEFT_ELBOW'] = 2 * len(hand_landmarks) + 2
    keypoint_indices['RIGHT_ELBOW'] = 2 * len(hand_landmarks) + 3

    # Define connections based on your provided code
    # Left hand connections
    left_hand_connections = [
        (keypoint_indices['WRIST_left'], keypoint_indices['THUMB_CMC_left']),
        (keypoint_indices['THUMB_CMC_left'], keypoint_indices['THUMB_MCP_left']),
        (keypoint_indices['THUMB_MCP_left'], keypoint_indices['THUMB_IP_left']),
        (keypoint_indices['THUMB_IP_left'], keypoint_indices['THUMB_TIP_left']),

        (keypoint_indices['WRIST_left'], keypoint_indices['INDEX_FINGER_MCP_left']),
        (keypoint_indices['INDEX_FINGER_MCP_left'], keypoint_indices['INDEX_FINGER_PIP_left']),
        (keypoint_indices['INDEX_FINGER_PIP_left'], keypoint_indices['INDEX_FINGER_DIP_left']),
        (keypoint_indices['INDEX_FINGER_DIP_left'], keypoint_indices['INDEX_FINGER_TIP_left']),

        (keypoint_indices['INDEX_FINGER_MCP_left'], keypoint_indices['MIDDLE_FINGER_MCP_left']),
        (keypoint_indices['MIDDLE_FINGER_MCP_left'], keypoint_indices['MIDDLE_FINGER_PIP_left']),
        (keypoint_indices['MIDDLE_FINGER_PIP_left'], keypoint_indices['MIDDLE_FINGER_DIP_left']),
        (keypoint_indices['MIDDLE_FINGER_DIP_left'], keypoint_indices['MIDDLE_FINGER_TIP_left']),

        (keypoint_indices['MIDDLE_FINGER_MCP_left'], keypoint_indices['RING_FINGER_MCP_left']),
        (keypoint_indices['RING_FINGER_MCP_left'], keypoint_indices['RING_FINGER_PIP_left']),
        (keypoint_indices['RING_FINGER_PIP_left'], keypoint_indices['RING_FINGER_DIP_left']),
        (keypoint_indices['RING_FINGER_DIP_left'], keypoint_indices['RING_FINGER_TIP_left']),

        (keypoint_indices['WRIST_left'], keypoint_indices['PINKY_MCP_left']),
        (keypoint_indices['PINKY_MCP_left'], keypoint_indices['PINKY_PIP_left']),
        (keypoint_indices['PINKY_PIP_left'], keypoint_indices['PINKY_DIP_left']),
        (keypoint_indices['PINKY_DIP_left'], keypoint_indices['PINKY_TIP_left']),
    ]

    # Right hand connections
    right_hand_connections = [
        (keypoint_indices['WRIST_right'], keypoint_indices['THUMB_CMC_right']),
        (keypoint_indices['THUMB_CMC_right'], keypoint_indices['THUMB_MCP_right']),
        (keypoint_indices['THUMB_MCP_right'], keypoint_indices['THUMB_IP_right']),
        (keypoint_indices['THUMB_IP_right'], keypoint_indices['THUMB_TIP_right']),

        (keypoint_indices['WRIST_right'], keypoint_indices['INDEX_FINGER_MCP_right']),
        (keypoint_indices['INDEX_FINGER_MCP_right'], keypoint_indices['INDEX_FINGER_PIP_right']),
        (keypoint_indices['INDEX_FINGER_PIP_right'], keypoint_indices['INDEX_FINGER_DIP_right']),
        (keypoint_indices['INDEX_FINGER_DIP_right'], keypoint_indices['INDEX_FINGER_TIP_right']),

        (keypoint_indices['INDEX_FINGER_MCP_right'], keypoint_indices['MIDDLE_FINGER_MCP_right']),
        (keypoint_indices['MIDDLE_FINGER_MCP_right'], keypoint_indices['MIDDLE_FINGER_PIP_right']),
        (keypoint_indices['MIDDLE_FINGER_PIP_right'], keypoint_indices['MIDDLE_FINGER_DIP_right']),
        (keypoint_indices['MIDDLE_FINGER_DIP_right'], keypoint_indices['MIDDLE_FINGER_TIP_right']),

        (keypoint_indices['MIDDLE_FINGER_MCP_right'], keypoint_indices['RING_FINGER_MCP_right']),
        (keypoint_indices['RING_FINGER_MCP_right'], keypoint_indices['RING_FINGER_PIP_right']),
        (keypoint_indices['RING_FINGER_PIP_right'], keypoint_indices['RING_FINGER_DIP_right']),
        (keypoint_indices['RING_FINGER_DIP_right'], keypoint_indices['RING_FINGER_TIP_right']),

        (keypoint_indices['WRIST_right'], keypoint_indices['PINKY_MCP_right']),
        (keypoint_indices['PINKY_MCP_right'], keypoint_indices['PINKY_PIP_right']),
        (keypoint_indices['PINKY_PIP_right'], keypoint_indices['PINKY_DIP_right']),
        (keypoint_indices['PINKY_DIP_right'], keypoint_indices['PINKY_TIP_right']),
    ]

    # Pose connections
    pose_connections = [
        (keypoint_indices['RIGHT_SHOULDER'], keypoint_indices['RIGHT_ELBOW']),
        (keypoint_indices['RIGHT_ELBOW'], keypoint_indices['WRIST_right']),
        (keypoint_indices['RIGHT_SHOULDER'], keypoint_indices['LEFT_SHOULDER']),
        (keypoint_indices['LEFT_SHOULDER'], keypoint_indices['LEFT_ELBOW']),
        (keypoint_indices['LEFT_ELBOW'], keypoint_indices['WRIST_left']),
    ]

    # Combine all connections
    all_connections = left_hand_connections + right_hand_connections + pose_connections

    # Draw the keypoints on black background and save images
    os.makedirs(save_dir, exist_ok=True)
    image_size = (240, 320, 3)  # Height x Width x Channels

    for idx in range(T):
        black_image = np.zeros(image_size, dtype=np.uint8)
        keypoints = keypoints_all_frames[idx]

        # Create a list of landmarks
        all_landmarks_list = []
        for i in range(num_keypoints):
            x = keypoints[i, 0]
            y = keypoints[i, 1]
            all_landmarks_list.append(
                landmark_pb2.NormalizedLandmark(x=x, y=y)
            )

        all_landmarks = landmark_pb2.NormalizedLandmarkList(landmark=all_landmarks_list)

        # Draw landmarks with custom connections
        mp_drawing.draw_landmarks(
            black_image,
            all_landmarks,
            all_connections,
            mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2))

        # Save image
        output_file = os.path.join(save_dir, f"frame_{idx:05d}.png")
        cv2.imwrite(output_file, black_image)

if __name__ == "__main__":
    # video_path = '/home/trdung/Documents/VSLRecog/1-200 videos/videos/01_Co-Hien_1-100_1-2-3_0108___center_device02_signer01_center_ord1_2.mp4'
    video_path = '/home/trdung/Documents/VSLRecog/1-200 videos/videos/01_Co-Hien_1-100_1-2-3_0108___left_device01_signer01_left_ord1_2.mp4'
    # video_path = '/home/trdung/Documents/VSLRecog/1-200 videos/videos/01_Co-Hien_1-100_1-2-3_0108___right_device03_signer01_right_ord1_2.mp4'
    # save_directory = "/home/trdung/Documents/VSLRecog/Paper/inputKPcenter"
    save_directory = "/home/trdung/Documents/VSLRecog/Paper/inputKPleft"
    # save_directory = "/home/trdung/Documents/VSLRecog/Paper/inputKPright"
    process_video(video_path, save_directory)
    print("Processing completed.")


