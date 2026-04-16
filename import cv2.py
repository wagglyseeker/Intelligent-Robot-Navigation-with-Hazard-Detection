import cv2
import cv2.aruco as aruco
import numpy as np
import heapq
from ultralytics import YOLO

# --- Constants ---
ARENA_WIDTH, ARENA_HEIGHT = 800, 600
BLUE_ROAD_LOWER = np.array([100, 0, 0])
BLUE_ROAD_UPPER = np.array([255, 70, 70])

# --- ArUco Setup ---
cap = cv2.VideoCapture(1)
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
parameters = aruco.DetectorParameters()
detector = aruco.ArucoDetector(aruco_dict, parameters)

# --- YOLO Model ---
yolo_model = YOLO("C:/Users/waggly/Downloads/ArUco-markers-with-OpenCV-and-Python-main/runs/detect/train9/weights/last.pt")

# Reference markers (known world coordinates)
reference_markers = {
    0: (0, 0),
    4: (4, 0),
    2: (0, 3),
    3: (4, 3)
}

ROBOT_ID = 5
GOAL_ID = 7
robot_pos = None
goal_pos = None
hazard_positions = set()
visited_hazards = set()

# --- Helper Functions ---
def euclidean_distance(a, b):
    return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

def get_blue_road_mask(frame):
    mask = cv2.inRange(frame, BLUE_ROAD_LOWER, BLUE_ROAD_UPPER)
    return cv2.bitwise_and(frame, frame, mask=mask)

def get_homography(corners, ids):
    if ids is None:
        return None
    detected_points = []
    world_points = []
    for i in range(len(ids)):
        marker_id = ids[i][0]
        if marker_id in reference_markers:
            c = corners[i][0]
            marker_center = (int(c[:, 0].mean()), int(c[:, 1].mean()))
            detected_points.append(marker_center)
            world_points.append(reference_markers[marker_id])
    if len(detected_points) >= 4:
        detected_points = np.array(detected_points, dtype=np.float32)
        world_points = np.array(world_points, dtype=np.float32)
        H, _ = cv2.findHomography(detected_points, world_points)
        return H
    return None

def transform_position(H, point):
    if H is None:
        return None
    point_array = np.array([[point]], dtype=np.float32)
    transformed = cv2.perspectiveTransform(point_array, H)
    return (float(transformed[0,0,0]), float(transformed[0,0,1]))

def a_star(start, goal, hazards, road_mask):
    if start is None or goal is None:
        return None
    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    g_score = {start: 0}
    f_score = {start: euclidean_distance(start, goal)}
    while open_set:
        _, current = heapq.heappop(open_set)
        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            return path[::-1]
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            neighbor = (current[0] + dx, current[1] + dy)
            if not (0 <= neighbor[0] < ARENA_WIDTH and 0 <= neighbor[1] < ARENA_HEIGHT):
                continue
            if road_mask[neighbor[1], neighbor[0]].sum() == 0:
                continue
            if neighbor in hazards and neighbor not in visited_hazards and neighbor != goal:
                continue
            tentative_g_score = g_score[current] + 1
            if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = tentative_g_score + euclidean_distance(neighbor, goal)
                heapq.heappush(open_set, (f_score[neighbor], neighbor))
    return None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    road_mask = get_blue_road_mask(frame)
    corners, ids, _ = detector.detectMarkers(frame)

    if ids is not None:
        aruco.drawDetectedMarkers(frame, corners, ids)
        H = get_homography(corners, ids)
        for i in range(len(ids)):
            marker_id = ids[i][0]
            c = corners[i][0]
            pixel_pos = (int(c[:, 0].mean()), int(c[:, 1].mean()))
            if H is not None:
                world_pos = transform_position(H, pixel_pos)
                if world_pos is not None and marker_id == ROBOT_ID:
                    print(f"Bot World Position: {world_pos}")
            if marker_id == ROBOT_ID:
                robot_pos = pixel_pos
            elif marker_id == GOAL_ID:
                goal_pos = pixel_pos

    # YOLO detection
    results = yolo_model(frame)
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = yolo_model.names[int(box.cls[0])]
            if label.lower().startswith("hazard"):
                cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                hazard_positions.add((cx, cy))
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    if robot_pos and goal_pos:
        unvisited_hazards = hazard_positions - visited_hazards
        if unvisited_hazards:
            closest = min(unvisited_hazards, key=lambda h: euclidean_distance(robot_pos, h))
            path = a_star(robot_pos, closest, hazard_positions, road_mask)
            if path:
                for i in range(len(path) - 1):
                    cv2.line(frame, path[i], path[i + 1], (0, 255, 0), 2)
                if euclidean_distance(robot_pos, closest) < 30:
                    visited_hazards.add(closest)
        else:
            path = a_star(robot_pos, goal_pos, hazard_positions, road_mask)
            if path:
                for i in range(len(path) - 1):
                    cv2.line(frame, path[i], path[i + 1], (0, 0, 255), 2)

    cv2.putText(frame, f"Hazards left: {len(hazard_positions - visited_hazards)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.imshow("YOLO + ArUco Hazard Navigation", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
