import cv2
import numpy as np
import heapq
import random
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle, Circle

# --- Simulation Constants ---
ARENA_WIDTH, ARENA_HEIGHT = 800, 600
MARKER_SIZE = 40
ROBOT_SIZE = 30
HAZARD_SIZE = 25
GOAL_SIZE = 30

# Reference markers positions (same as your real setup)
REFERENCE_MARKERS = {
    0: (50, 50),
    4: (ARENA_WIDTH - 50, 50),
    2: (50, ARENA_HEIGHT - 50),
    3: (ARENA_WIDTH - 50, ARENA_HEIGHT - 50)
}

# --- Simulation Setup ---
class Simulation:
    def __init__(self):
        self.fig, self.ax = plt.subplots(figsize=(10, 7))
        self.ax.set_xlim(0, ARENA_WIDTH)
        self.ax.set_ylim(0, ARENA_HEIGHT)
        self.ax.set_aspect('equal')
        self.ax.invert_yaxis()  # To match image coordinates
        
        # Create arena elements
        self.create_arena()
        self.create_road()
        self.create_markers()
        self.create_hazards()
        self.create_robot_and_goal()
        
        # Navigation variables
        self.hazard_positions = set((hazard.center[0], hazard.center[1]) for hazard in self.hazards)
        self.visited_hazards = set()
        self.path = []
        self.current_target = None
        self.status_text = self.ax.text(10, 20, "", fontsize=12)
        
        # Animation
        self.ani = FuncAnimation(self.fig, self.update, frames=200, 
                                interval=100, blit=False)
    
    def create_arena(self):
        """Create the arena boundary"""
        border = Rectangle((0, 0), ARENA_WIDTH, ARENA_HEIGHT, 
                          linewidth=2, edgecolor='black', facecolor='none')
        self.ax.add_patch(border)
    
    def create_road(self):
        """Create a blue road network (simplified as a main path with branches)"""
        # Main horizontal road
        road_width = 60
        self.road_mask = np.zeros((ARENA_HEIGHT, ARENA_WIDTH), dtype=np.uint8)
        
        # Horizontal main road
        cv2.rectangle(self.road_mask, 
                     (0, ARENA_HEIGHT//2 - road_width//2),
                     (ARENA_WIDTH, ARENA_HEIGHT//2 + road_width//2),
                     255, -1)
        
        # Vertical roads
        cv2.rectangle(self.road_mask, 
                     (ARENA_WIDTH//4, ARENA_HEIGHT//2 - road_width//2),
                     (ARENA_WIDTH//4 + road_width, ARENA_HEIGHT),
                     255, -1)
        
        cv2.rectangle(self.road_mask, 
                     (3*ARENA_WIDTH//4, 0),
                     (3*ARENA_WIDTH//4 + road_width, ARENA_HEIGHT//2 + road_width//2),
                     255, -1)
        
        # Draw the road visualization
        road_viz = np.zeros((ARENA_HEIGHT, ARENA_WIDTH, 3), dtype=np.uint8)
        road_viz[self.road_mask == 255] = [255, 150, 0]  # Blue color
        self.ax.imshow(road_viz, alpha=0.3, extent=[0, ARENA_WIDTH, ARENA_HEIGHT, 0])
    
    def create_markers(self):
        """Create ArUco markers at reference positions"""
        self.markers = []
        for marker_id, (x, y) in REFERENCE_MARKERS.items():
            marker = Rectangle((x - MARKER_SIZE//2, y - MARKER_SIZE//2), 
                              MARKER_SIZE, MARKER_SIZE,
                              facecolor='white', edgecolor='black')
            self.ax.add_patch(marker)
            self.ax.text(x, y, str(marker_id), ha='center', va='center')
            self.markers.append(marker)
    
    def create_hazards(self, num_hazards=5):
        """Create hazards at random positions along roads"""
        self.hazards = []
        
        # Find road pixels where hazards can be placed
        road_pixels = np.argwhere(self.road_mask == 255)
        selected_indices = np.random.choice(len(road_pixels), num_hazards, replace=False)
        
        for idx in selected_indices:
            x, y = road_pixels[idx][1], road_pixels[idx][0]
            hazard = Circle((x, y), HAZARD_SIZE//2, 
                          facecolor='red', edgecolor='darkred')
            self.ax.add_patch(hazard)
            self.hazards.append(hazard)
    
    def create_robot_and_goal(self):
        """Create robot and goal markers"""
        # Place robot near first reference marker
        robot_x, robot_y = REFERENCE_MARKERS[0]
        robot_x += 100
        self.robot = Circle((robot_x, robot_y), ROBOT_SIZE//2, 
                           facecolor='green', edgecolor='darkgreen')
        self.ax.add_patch(self.robot)
        self.robot_pos = (robot_x, robot_y)
        
        # Place goal near last reference marker
        goal_x, goal_y = REFERENCE_MARKERS[3]
        goal_x -= 100
        self.goal = Circle((goal_x, goal_y), GOAL_SIZE//2, 
                          facecolor='blue', edgecolor='darkblue')
        self.ax.add_patch(self.goal)
        self.goal_pos = (goal_x, goal_y)
    
    def euclidean(self, a, b):
        return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)
    
    def a_star(self, start, goal, hazards):
        """A* pathfinding algorithm (same as your real code)"""
        if start is None or goal is None:
            return None
            
        start = tuple(map(int, start))
        goal = tuple(map(int, goal))
        open_set = [(0, start)]
        came_from = {}
        g_score = {start: 0}
        f_score = {start: self.euclidean(start, goal)}

        while open_set:
            _, current = heapq.heappop(open_set)
            if self.euclidean(current, goal) < 5:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.reverse()
                return path
                
            for dx, dy in [(0,1),(1,0),(0,-1),(-1,0)]:
                neighbor = (current[0] + dx, current[1] + dy)
                if not (0 <= neighbor[0] < ARENA_WIDTH and 0 <= neighbor[1] < ARENA_HEIGHT):
                    continue
                if self.road_mask[neighbor[1], neighbor[0]] == 0:
                    continue
                if neighbor in hazards and neighbor not in self.visited_hazards:
                    continue
                    
                tentative = g_score[current] + 1
                if neighbor not in g_score or tentative < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative
                    f_score[neighbor] = tentative + self.euclidean(neighbor, goal)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
        return None
    
    def update(self, frame):
        """Update simulation state"""
        # Clear previous path
        while len(self.ax.lines) > 0:
            self.ax.lines[0].remove()
        
        # Navigation logic (same as your real code)
        remaining_hazards = self.hazard_positions - self.visited_hazards
        
        if remaining_hazards:
            # Find closest unvisited hazard
            self.current_target = min(remaining_hazards, 
                                     key=lambda h: self.euclidean(self.robot_pos, h))
            
            # Calculate path to hazard
            self.path = self.a_star(self.robot_pos, self.current_target, self.hazard_positions)
            
            if self.path:
                # Draw path to hazard (green)
                x_vals, y_vals = zip(*self.path)
                self.ax.plot(x_vals, y_vals, 'g-', linewidth=2)
                
                # Move robot along path
                if len(self.path) > 1:
                    next_pos = self.path[1]
                    self.robot.center = next_pos
                    self.robot_pos = next_pos
                    
                    # Check if reached hazard
                    if self.euclidean(self.robot_pos, self.current_target) < 10:
                        self.visited_hazards.add(self.current_target)
                        # Change hazard color to indicate visited
                        for hazard in self.hazards:
                            if (hazard.center[0], hazard.center[1]) == self.current_target:
                                hazard.set_facecolor('purple')
                                break
        else:
            # All hazards visited, go to goal
            self.current_target = self.goal_pos
            self.path = self.a_star(self.robot_pos, self.goal_pos, self.hazard_positions)
            
            if self.path:
                # Draw path to goal (blue)
                x_vals, y_vals = zip(*self.path)
                self.ax.plot(x_vals, y_vals, 'b-', linewidth=2)
                
                # Move robot along path
                if len(self.path) > 1:
                    next_pos = self.path[1]
                    self.robot.center = next_pos
                    self.robot_pos = next_pos
        
        # Update status text
        self.status_text.set_text(
            f"Hazards left: {len(remaining_hazards)} | "
            f"Visited: {len(self.visited_hazards)} | "
            f"Target: {'Hazard' if remaining_hazards else 'Goal'}"
        )
        
        return [self.robot, self.status_text] + self.hazards

# Run the simulation
sim = Simulation()
plt.title("ArUco Navigation Simulation")
plt.tight_layout()
plt.show()