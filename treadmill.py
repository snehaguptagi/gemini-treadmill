import cv2
import google.generativeai as genai
import json
import os
import time
import mediapipe as mp
import numpy as np

# --- Configuration ---
# Set your video source: 0 for webcam, or "path/to/your_treadmill_video.mp4"
VIDEO_SOURCE = "your_treadmill_video.mp4" 

# Gemini API configuration
# It's HIGHLY recommended to set this as an environment variable (export GOOGLE_API_KEY="YOUR_KEY")
API_KEY = os.environ.get("GOOGLE_API_KEY")
if not API_KEY:
    print("Error: GOOGLE_API_KEY environment variable not set.")
    print("Please set it using: export GOOGLE_API_KEY='YOUR_API_KEY_HERE' (Linux/macOS)")
    print("Or: set GOOGLE_API_KEY='YOUR_API_KEY_HERE' (Windows Cmd)")
    exit()

genai.configure(api_key=API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash')  # current multimodal model (gemini-pro-vision was retired)

# Frame processing rate for Gemini (send one image every X seconds)
GEMINI_QUERY_INTERVAL = 2.0 # seconds

# --- MediaPipe Pose Setup ---
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1, # 0, 1, or 2 (higher is more accurate but slower)
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# --- Global variables for display ---
current_gemini_feedback = "Waiting for analysis..."
last_gemini_query_time = 0

# --- Helper Functions for Pose Analysis ---
def calculate_angle(a, b, c):
    """Calculates the angle between three points (a, b, c) where b is the vertex."""
    a = np.array(a) # First point (e.g., shoulder)
    b = np.array(b) # Mid point (e.g., elbow)
    c = np.array(c) # End point (e.g., wrist)

    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(np.degrees(radians))

    if angle > 180.0:
        angle = 360 - angle
    return angle

def get_landmark_coords(landmarks, landmark_enum, frame_width, frame_height):
    """Safely gets pixel coordinates for a landmark if visible."""
    if not landmarks or not landmarks.landmark:
        return None
    
    lm = landmarks.landmark[landmark_enum]
    if lm.visibility > 0.6: # Adjust visibility threshold as needed
        return [int(lm.x * frame_width), int(lm.y * frame_height)]
    return None

def extract_treadmill_metrics(pose_landmarks, frame_width, frame_height):
    """Extracts relevant metrics for treadmill form analysis."""
    metrics = {}
    
    if not pose_landmarks:
        return metrics

    landmarks_dict = {
        lm.name: get_landmark_coords(pose_landmarks, getattr(mp_pose.PoseLandmark, lm.name), frame_width, frame_height)
        for lm in mp_pose.PoseLandmark
    }

    # Example: Knee Angles
    right_hip = landmarks_dict.get('RIGHT_HIP')
    right_knee = landmarks_dict.get('RIGHT_KNEE')
    right_ankle = landmarks_dict.get('RIGHT_ANKLE')
    left_hip = landmarks_dict.get('LEFT_HIP')
    left_knee = landmarks_dict.get('LEFT_KNEE')
    left_ankle = landmarks_dict.get('LEFT_ANKLE')

    if all([right_hip, right_knee, right_ankle]):
        metrics['right_knee_angle'] = round(calculate_angle(right_hip, right_knee, right_ankle), 2)
    if all([left_hip, left_knee, left_ankle]):
        metrics['left_knee_angle'] = round(calculate_angle(left_hip, left_knee, left_ankle), 2)

    # Example: Hip/Shoulder/Ankle alignment (simple vertical alignment check)
    # Check for leaning forward/backward - approximate using average x-coordinates
    nose = landmarks_dict.get('NOSE')
    if all([nose, right_hip, left_hip]):
        avg_hip_x = (right_hip[0] + left_hip[0]) / 2 if right_hip and left_hip else None
        if avg_hip_x:
            # If nose is significantly forward of hips, might indicate leaning
            # This requires calibration to what "straight" looks like for your setup
            metrics['nose_hip_horizontal_diff'] = round(nose[0] - avg_hip_x, 2)
            
    # Example: Arm swing (approximate using elbow-shoulder-wrist angle)
    right_shoulder = landmarks_dict.get('RIGHT_SHOULDER')
    right_elbow = landmarks_dict.get('RIGHT_ELBOW')
    right_wrist = landmarks_dict.get('RIGHT_WRIST')
    if all([right_shoulder, right_elbow, right_wrist]):
        metrics['right_elbow_angle'] = round(calculate_angle(right_shoulder, right_elbow, right_wrist), 2)
    
    left_shoulder = landmarks_dict.get('LEFT_SHOULDER')
    left_elbow = landmarks_dict.get('LEFT_ELBOW')
    left_wrist = landmarks_dict.get('LEFT_WRIST')
    if all([left_shoulder, left_elbow, left_wrist]):
        metrics['left_elbow_angle'] = round(calculate_angle(left_shoulder, left_elbow, left_wrist), 2)

    # You can add many more metrics:
    # - Stride length estimation (hip-to-hip or ankle-to-ankle distance over time)
    # - Cadence (counting foot strikes)
    # - Vertical oscillation (tracking vertical movement of hips/head over time)
    # - Foot strike (harder without direct view of foot, but ankle position can hint)

    return metrics

# --- Main Video Processing Loop ---
cap = cv2.VideoCapture(VIDEO_SOURCE)

if not cap.isOpened():
    print(f"Error: Could not open video source {VIDEO_SOURCE}.")
    exit()

cv2.namedWindow('Treadmill Form Analyzer', cv2.WINDOW_NORMAL)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame horizontally for selfie-view, if using webcam
    # if VIDEO_SOURCE == 0:
    #     frame = cv2.flip(frame, 1)

    h, w, c = frame.shape

    # Convert the frame to RGB for MediaPipe
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame for pose landmarks
    results = pose.process(frame_rgb)

    treadmill_metrics = {}
    if results.pose_landmarks:
        # Draw landmarks on the frame
        mp_drawing.draw_landmarks(
            frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2), # Green landmarks
            connection_drawing_spec=mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2) # Blue connections
        )
        
        # Extract treadmill-specific metrics
        treadmill_metrics = extract_treadmill_metrics(results.pose_landmarks, w, h)
    
    current_time = time.time()

    # --- Send to Gemini API conditionally ---
    if current_time - last_gemini_query_time >= GEMINI_QUERY_INTERVAL:
        if results.pose_landmarks: # Only query if a person is detected
            # Encode the frame as JPEG bytes for Gemini's inline-image payload
            _, buffer = cv2.imencode('.jpg', frame)
            image_bytes = buffer.tobytes()

            # --- New Prompt for Treadmill Analysis ---
            prompt_text = (
                "Analyze the running/walking form of the person in the attached image, who is on a treadmill. "
                "Consider their posture, arm swing, leg mechanics, and overall balance. "
                "Here are some calculated pose metrics from the current frame: "
                f"{json.dumps(treadmill_metrics, indent=2)}\n\n"
                "Output a JSON object with the following fields, indicating `true` if the issue is detected, "
                "`false` otherwise, and provide a `risk_level` (low, medium, high) based on overall form:\n"
                "- `overstriding`: Is the foot landing too far in front of the body's center of mass?\n"
                "- `heel_striking`: Is the heel making initial contact with the ground?\n"
                "- `leaning_forward_excessively`: Is the upper body leaning too far forward or backward?\n"
                "- `holding_rails`: Are they gripping the treadmill rails?\n"
                "- `poor_arm_swing`: Is the arm swing too wide, too low, or non-existent?\n"
                "- `hip_drop`: Is one hip dropping significantly more than the other?\n"
                "- `overall_feedback`: A brief textual summary of their form and the most critical actionable tip.\n"
                "- `risk_level`: Overall injury risk (low, medium, high).\n"
                "Ensure the output is a valid JSON object ONLY. Do not include any other text before or after the JSON."
            )

            contents = [
                {"mime_type": "image/jpeg", "data": image_bytes},
                prompt_text,
            ]

            print(f"[{time.strftime('%H:%M:%S')}] Sending query to Gemini...")
            try:
                response = model.generate_content(contents)
                gemini_raw_text = response.text.strip()
                # Attempt to parse as JSON
                try:
                    # Clean the response to ensure it's pure JSON if Gemini adds markdown
                    if gemini_raw_text.startswith("```json") and gemini_raw_text.endswith("```"):
                        gemini_raw_text = gemini_raw_text[7:-3].strip()
                    
                    parsed_feedback = json.loads(gemini_raw_text)
                    
                    # Update global feedback for display
                    current_gemini_feedback = (
                        f"Overstriding: {parsed_feedback.get('overstriding', 'N/A')}\n"
                        f"Heel Striking: {parsed_feedback.get('heel_striking', 'N/A')}\n"
                        f"Leaning Fwd: {parsed_feedback.get('leaning_forward_excessively', 'N/A')}\n"
                        f"Holding Rails: {parsed_feedback.get('holding_rails', 'N/A')}\n"
                        f"Poor Arm Swing: {parsed_feedback.get('poor_arm_swing', 'N/A')}\n"
                        f"Hip Drop: {parsed_feedback.get('hip_drop', 'N/A')}\n"
                        f"Risk Level: {parsed_feedback.get('risk_level', 'N/A').upper()}\n"
                        f"Feedback: {parsed_feedback.get('overall_feedback', 'No specific feedback.')}"
                    )
                    print("Parsed Gemini Feedback:")
                    print(current_gemini_feedback)

                except json.JSONDecodeError as jde:
                    current_gemini_feedback = f"Gemini response not valid JSON: {jde}\nRaw: {gemini_raw_text[:200]}..."
                    print(current_gemini_feedback)
                
            except Exception as e:
                current_gemini_feedback = f"Error calling Gemini API: {e}"
                print(current_gemini_feedback)

            last_gemini_query_time = current_time
        else:
            current_gemini_feedback = "No person detected. Waiting for pose..."
            # print("No pose detected, skipping Gemini query.")

    # --- Visualization ---
    # Display MediaPipe Pose data (metrics)
    y_offset_metrics = 30
    cv2.putText(frame, f"Right Knee: {treadmill_metrics.get('right_knee_angle', 'N/A')} deg", (w - 250, y_offset_metrics), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1, cv2.LINE_AA)
    cv2.putText(frame, f"Left Knee: {treadmill_metrics.get('left_knee_angle', 'N/A')} deg", (w - 250, y_offset_metrics + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1, cv2.LINE_AA)
    cv2.putText(frame, f"Right Elbow: {treadmill_metrics.get('right_elbow_angle', 'N/A')} deg", (w - 250, y_offset_metrics + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1, cv2.LINE_AA)
    cv2.putText(frame, f"Left Elbow: {treadmill_metrics.get('left_elbow_angle', 'N/A')} deg", (w - 250, y_offset_metrics + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1, cv2.LINE_AA)
    # Add more metric displays here

    # Display Gemini's overall feedback
    y_offset_feedback = h - 150 # Start near the bottom
    for i, line in enumerate(current_gemini_feedback.split('\n')):
        color = (0, 255, 255) # Yellow for general feedback
        if "Risk Level: HIGH" in line:
            color = (0, 0, 255) # Red for high risk
        elif "Risk Level: MEDIUM" in line:
            color = (0, 165, 255) # Orange for medium risk
        
        cv2.putText(frame, line, (10, y_offset_feedback + (i * 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1, cv2.LINE_AA)
    
    # Optional: Draw borders based on risk level
    if "Risk Level: HIGH" in current_gemini_feedback:
        cv2.rectangle(frame, (0, 0), (w, h), (0, 0, 255), 5) # Red border
    elif "Risk Level: MEDIUM" in current_gemini_feedback:
        cv2.rectangle(frame, (0, 0), (w, h), (0, 165, 255), 5) # Orange border
    elif "Risk Level: LOW" in current_gemini_feedback:
        cv2.rectangle(frame, (0, 0), (w, h), (0, 255, 0), 5) # Green border


    cv2.imshow('Treadmill Form Analyzer', frame)

    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- Cleanup ---
cap.release()
cv2.destroyAllWindows()
pose.close() # Release MediaPipe resources