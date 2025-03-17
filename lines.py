import cv2
import numpy as np
import requests
import time

"""
PSUDOCODE:
Initialize video capture and variables (prev_left_line, prev_right_line, lane_change_active, timestamp)
Define log_action(action) to send action data to Flask

Define region_of_interest(img, vertices) to mask ROI
Define draw_lines(img, lines) to draw detected lane lines
Define extrapolate_lines(lines, img_shape, prev_line) to fit lane lines
Define enforce_lane_distance(left_line, right_line) to ensure valid lane spacing
Define overlay_arrow(frame, direction) to display turn direction
Define compute_center_line(left_line, right_line) to find lane midpoint

Define detect_lanes(frame, prev_left, prev_right, lane_change, timestamp):
    Convert frame to grayscale, blur, and detect edges
    Apply ROI mask and Hough transform to detect lane lines
    Categorize lines into left and right based on slope
    Extrapolate left and right lanes, ensuring valid distance
    Compute center line and update lane change state
    Draw lane lines and determine turn direction
    Log direction and overlay arrow
    Return processed frame and updated lane data

Define generate_video_frames():
    Open video and process frames in loop:
        Detect lanes and encode frame to JPEG
        Yield frame for streaming
        Determine and log direction based on midline
    Release video resources

If script runs as main, call generate_video_frames()
"""

def log_action(action):
    """Send log data to Flask app"""
    try:
        requests.post('http://127.0.0.1:5000/log_action', json={"action": action})
    except Exception as e:
        print(f"Error logging action: {e}")

def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, vertices, 255)
    return cv2.bitwise_and(img, mask)

def draw_lines(img, lines, color=(0, 0, 255), thickness=10):
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                y1 = max(y1, int(img.shape[0] * 0.75))
                y2 = max(y2, int(img.shape[0] * 0.75))
                cv2.line(img, (x1, y1), (x2, y2), color, thickness)

def extrapolate_lines(lines, img_shape, prev_line):
    if not lines and prev_line:
        return prev_line  # Extend previous line if no new detection
    
    x_coords, y_coords = [], []
    for x1, y1, x2, y2 in lines:
        x_coords.extend([x1, x2])
        y_coords.extend([y1, y2])
    
    if len(x_coords) > 1 and len(y_coords) > 1:
        poly = np.polyfit(y_coords, x_coords, deg=1)
        y1, y2 = img_shape[0], int(img_shape[0] * 0.8)
        x1, x2 = int(np.polyval(poly, y1)), int(np.polyval(poly, y2))
        return (x1, y1, x2, y2)
    return prev_line

def enforce_lane_distance(left_line, right_line, min_dist=200, max_dist=500):
    if left_line and right_line:
        left_x = (left_line[0] + left_line[2]) // 2
        right_x = (right_line[0] + right_line[2]) // 2
        distance = abs(right_x - left_x)
        
        if distance < min_dist or distance > max_dist:
            return None, None
    return left_line, right_line

def overlay_arrow(frame, direction):
    arrow = cv2.imread("static/arrow.png", cv2.IMREAD_UNCHANGED)
    if arrow is None:
        return frame
    
    arrow = cv2.resize(arrow, (50, 50))
    if direction == "left":
        arrow = cv2.rotate(arrow, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif direction == "right":
        arrow = cv2.rotate(arrow, cv2.ROTATE_90_CLOCKWISE)
    
    h, w = arrow.shape[:2]
    roi = frame[20:20+h, 20:20+w]
    mask = arrow[:, :, 3] / 255.0
    for c in range(3):
        roi[:, :, c] = (1 - mask) * roi[:, :, c] + mask * arrow[:, :, c]
    
    frame[20:20+h, 20:20+w] = roi
    return frame

def compute_center_line(left_line, right_line):
    if left_line is None or right_line is None:
        return None 
    
    x1_left, y1_left, x2_left, y2_left = left_line
    x1_right, y1_right, x2_right, y2_right = right_line
    
    x1_center = (x1_left + x1_right) // 2
    y1_center = (y1_left + y1_right) // 2
    x2_center = (x2_left + x2_right) // 2
    y2_center = (y2_left + y2_right) // 2
    
    return [x1_center, y1_center, x2_center, y2_center]

def detect_lanes(frame, prev_left_line, prev_right_line, lane_change_active, timestamp):
    height, width = frame.shape[:2]
    
    roi_vertices = np.array([[(120, height), (width - 120, height), (width // 2, height // 2 + 50)]], np.int32)
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    masked_edges = region_of_interest(edges, roi_vertices)
    
    lines = cv2.HoughLinesP(masked_edges, 2, np.pi / 180, 100, np.array([]), minLineLength=40, maxLineGap=50)
    left_lines, right_lines = [], []
    
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            slope = (y2 - y1) / (x2 - x1 + 1e-6)
            if slope < -0.5:
                left_lines.append((x1, y1, x2, y2))
            elif slope > 0.5:
                right_lines.append((x1, y1, x2, y2))
    
    left_line = extrapolate_lines(left_lines, frame.shape, prev_left_line)
    right_line = extrapolate_lines(right_lines, frame.shape, prev_right_line)
    
    left_line, right_line = enforce_lane_distance(left_line, right_line)
    center_line = compute_center_line(left_line, right_line)
    
    if left_line and right_line:
        lane_change_active = False
    elif not left_line or not right_line:
        lane_change_active = True
    
    time.sleep(0.02)  # Adjust speed to match main video
    
    line_img = np.zeros_like(frame)
    if not lane_change_active:
        if left_line:
            draw_lines(line_img, [[left_line]], color=(0, 0, 255), thickness=10)
        if right_line:
            draw_lines(line_img, [[right_line]], color=(0, 0, 255), thickness=10)
        if center_line:
            draw_lines(line_img, [[center_line]], color=(255, 0, 0), thickness=10)
    
    result = cv2.addWeighted(frame, 0.8, line_img, 1, 0)
    direction = "straight"
    
    if left_line and right_line and not lane_change_active:
        left_x, right_x = np.mean([left_line[0], left_line[2]]), np.mean([right_line[0], right_line[2]])
        mid_x = int((left_x + right_x) / 2)
        if mid_x < width // 2 - 30:
            direction = "left"
        elif mid_x > width // 2 + 30:
            direction = "right"
    
    log_action(direction)
    result = overlay_arrow(result, direction)
    return result, left_line, right_line, lane_change_active


def generate_video_frames():
    """Generator function to yield processed frames"""
    video_path = "static/Main.MP4"
    cap = cv2.VideoCapture(video_path)

    prev_left_line, prev_right_line = None, None
    lane_change_active = False
    timestamp = 0
    fps = cap.get(cv2.CAP_PROP_FPS)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        lanes_frame, prev_left_line, prev_right_line, lane_change_active = detect_lanes(
            frame, prev_left_line, prev_right_line, lane_change_active, timestamp
        )

        # Encode frame to JPEG format
        _, buffer = cv2.imencode('.jpg', lanes_frame)
        frame_bytes = buffer.tobytes()

        # Yield the frame for streaming
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

        # Log detected direction
        direction = "straight"
        if prev_left_line and prev_right_line:
            left_x = (prev_left_line[0] + prev_left_line[2]) // 2
            right_x = (prev_right_line[0] + prev_right_line[2]) // 2
            mid_x = (left_x + right_x) // 2
            if mid_x < frame.shape[1] // 2 - 30:
                direction = "left"
            elif mid_x > frame.shape[1] // 2 + 30:
                direction = "right"

        log_action(direction)
        timestamp += 1 / fps

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    generate_video_frames()  
