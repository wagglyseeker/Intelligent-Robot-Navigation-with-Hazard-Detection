
import cv2
import cv2.aruco as aruco
import numpy as np
import heapq
import json
import time
import websocket
import threading
from pyproj import Transformer
from collections import deque

# --- Constants ---
ARENA_WIDTH, ARENA_HEIGHT = 800, 600
BLUE_ROAD_LOWER = np.array([100, 0, 0])  
BLUE_ROAD_UPPER = np.array([255, 70, 70])  
ROBOT_ID = 5
GOAL_ID = 7
HAZARD_IDS = set(range(8, 108))  
ESP32_IP = "192.168.1.100" 
WS_URL = f"ws://{ESP32_IP}:81"

# --- ArUco Setup ---
cap = cv2.VideoCapture(0)
time.sleep(2)  
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
parameters = aruco.DetectorParameters()

# Reference markers (known world coordinates)
reference_markers = {
    0: (26.220378189, 78.200431565),  # Bottom-left (Origin)
    4: (26.227192381, 78.208915356),  # Bottom-right
    2: (26.225281723, 78.195509754),  # Top-left
    3: (26.232129848, 78.204001524)   # Top-right
}

# Coordinate transformer
transformer = Transformer.from_crs("EPSG:32643", "EPSG:4326", always_xy=True)
geojson_path = "bot_position.geojson"

# --- Navigation Variables ---
robot_pos = None
goal_pos = None
hazard_positions = set()
visited_hazards = set()
path = []
robot_state = "STOP" 

# --- WebSocket Setup ---
ws = None
ws_connected = False

def connect_websocket():
    global ws, ws_connected
    while True:
        try:
            print("Connecting to WebSocket...")
            ws = websocket.WebSocket()
            ws.connect(WS_URL)
            ws_connected = True
            print("WebSocket connected successfully")
            break
        except Exception as e:
            print(f"WebSocket connection failed: {e}")
            time.sleep(2)

# Start WebSocket connection thread
ws_thread = threading.Thread(target=connect_websocket)
ws_thread.daemon = True
ws_thread.start()

def send_command(cmd):
    if not ws_connected:
        print("WebSocket not connected!")
        return False
    try:
        ws.send(cmd)
        print(f"Command sent: {cmd}")
        return True
    except Exception as e:
        print(f"Error sending command: {e}")
        ws_connected = False
        connect_websocket()
        return False

def perform_180_turn():
    send_command("TURN180")
    time.sleep(1.5)  

def determine_command(robot_pos, path):
    if not path or len(path) < 2:
        return "STOP"
    

    target = path[1]
    dx = target[0] - robot_pos[0]
    dy = target[1] - robot_pos[1]
    

    norm = np.sqrt(dx**2 + dy**2)
    dx, dy = dx/norm, dy/norm
    

    if abs(dx) > abs(dy):
        if dx > 0.5:
            return "RIGHT"
        else:
            return "LEFT"
    else:
        if dy > 0.5:
            return "FORWARD"
        else:
            perform_180_turn()
            return "FORWARD"


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

def update_geojson(marker_id, lat, lon):
    geojson_data = {
        "type": "FeatureCollection",
        "features": [{
            "type": "Feature",
            "properties": {"marker_id": marker_id},
            "geometry": {"type": "Point", "coordinates": [lon, lat]}
        }]
    }
    with open(geojson_path, "w") as f:
        json.dump(geojson_data, f)
    print(f"Updated GeoJSON: Marker {marker_id} at ({lat:.6f}, {lon:.6f})")

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
        
        if euclidean_distance(current, goal) < 10:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            return path[::-1]
        
        for dx, dy in [(0,1),(1,0),(0,-1),(-1,0),(1,1),(1,-1),(-1,1),(-1,-1)]:
            neighbor = (current[0] + dx, current[1] + dy)
            
            if not (0 <= neighbor[0] < ARENA_WIDTH and 0 <= neighbor[1] < ARENA_HEIGHT):
                continue
                
            if road_mask[neighbor[1], neighbor[0]].sum() == 0:
                continue
                
            if neighbor in hazards and neighbor not in visited_hazards and neighbor != goal:
                continue
                
            move_cost = 1 if abs(dx) + abs(dy) == 1 else 1.414
            tentative_g_score = g_score[current] + move_cost
            
            if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = tentative_g_score + euclidean_distance(neighbor, goal)
                heapq.heappush(open_set, (f_score[neighbor], neighbor))
    return None

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        road_mask = get_blue_road_mask(frame)
        corners, ids, _ = aruco.detectMarkers(frame, aruco_dict, parameters=parameters)
        
        if ids is not None:
            frame = aruco.drawDetectedMarkers(frame, corners, ids)
            H = get_homography(corners, ids)
            
            for i in range(len(ids)):
                marker_id = ids[i][0]
                c = corners[i][0]
                pixel_pos = (int(c[:, 0].mean()), int(c[:, 1].mean()))
                
                if H is not None:
                    world_pos = transform_position(H, pixel_pos)
                    if world_pos is not None:
                        lat, lon = transformer.transform(world_pos[0], world_pos[1])
                        
                        if marker_id == ROBOT_ID:
                            robot_pos = pixel_pos
                            print(f"Bot Position: {world_pos} | GPS: ({lat:.6f}, {lon:.6f})")
                            update_geojson(marker_id, lat, lon)
                        elif marker_id == GOAL_ID:
                            goal_pos = pixel_pos
                        elif marker_id in HAZARD_IDS:
                            hazard_positions.add(pixel_pos)
        
    
        if robot_pos and goal_pos:
            unvisited_hazards = hazard_positions - visited_hazards
            
            if unvisited_hazards:
                closest_hazard = min(unvisited_hazards, key=lambda h: euclidean_distance(robot_pos, h))
                path = a_star(robot_pos, closest_hazard, hazard_positions, road_mask)
                
                for hazard in unvisited_hazards:
                    if hazard != closest_hazard:
                        hazard_path = a_star(robot_pos, hazard, hazard_positions, road_mask)
                        if hazard_path:
                            for i in range(len(hazard_path) - 1):
                                cv2.line(frame, hazard_path[i], hazard_path[i+1], (0, 180, 0), 1)
                
                if path:
                    for i in range(len(path) - 1):
                        cv2.line(frame, path[i], path[i+1], (0, 255, 0), 2)
                    
                    cmd = determine_command(robot_pos, path)
                    send_command(cmd)
                    robot_state = "MOVING"
                    
                    if euclidean_distance(robot_pos, closest_hazard) < 30:
                        visited_hazards.add(closest_hazard)
                        send_command("STOP")
                        time.sleep(1)
            else:
                path = a_star(robot_pos, goal_pos, hazard_positions, road_mask)
                if path:
                    for i in range(len(path) - 1):
                        cv2.line(frame, path[i], path[i+1], (0, 0, 255), 2)
                    
                    cmd = determine_command(robot_pos, path)
                    send_command(cmd)
                    robot_state = "MOVING"
                    
                    if euclidean_distance(robot_pos, goal_pos) < 30:
                        send_command("STOP")
                        robot_state = "REACHED_GOAL"
                        print("Goal reached!")
        
        status_text = f"State: {robot_state} | Hazards left: {len(hazard_positions - visited_hazards)}"
        cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow("Blue Road Mask", road_mask)
        cv2.imshow("Vanguard Navigation", frame)
        
        if cv2.waitKey(1) == ord('q'):
            break

finally:
    send_command("STOP")
    if ws_connected:
        ws.close()
    cap.release()
    cv2.destroyAllWindows()
