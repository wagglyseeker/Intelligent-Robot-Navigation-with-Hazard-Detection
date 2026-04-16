# Autonomous Navigation using ArUco Fiducial Markers and Hazard-Aware Path Planning

An intelligent autonomous navigation system that detects hazards, plans optimal paths, and navigates a mobile robot using computer vision and real-time path planning algorithms.

This project integrates *ArUco-based localization, **YOLOv8 hazard detection, and **A path planning** to create a robust, real-time navigation pipeline.

---

##  Overview

This system enables a robotic platform to autonomously navigate an environment while detecting hazards and avoiding obstacles.

The navigation pipeline consists of:

1. Real-time vision processing
2. Hazard detection using deep learning
3. Path planning using A* algorithm
4. PID-based motion control
5. Real-time robot communication

The system dynamically updates paths when hazards are detected, ensuring safe and efficient navigation.

---

##  Key Features

- Real-time *hazard detection* using YOLOv8
- Accurate *robot localisation* using ArUco fiducial markers
- Shortest-path planning using *A algorithm**
- Dynamic *path re-planning* for new hazards
- Real-time robot communication via *WebSockets*
- PID-based motion control for smooth navigation
- Homography-based coordinate transformation

---

##  System Architecture

The system consists of two main modules:

### Computational Module

Responsible for:

- Vision processing
- ArUco detection
- Hazard detection
- Path planning
- Command generation

Key components:

- OpenCV-based ArUco detection
- YOLOv8 hazard detection model
- A* path planning engine
- Homography-based coordinate mapping

### Hardware Module

Responsible for:

- Receiving navigation commands
- Motion execution
- Sensor feedback
- Real-time movement

Key components:

- Wheeled robotic platform
- Tracking sensors
- Motor driver
- PID controller
- Wireless communication module

---

## Methodology

### Step 1 — ArUco-Based Localization

- ArUco markers define coordinate reference points.
- Pixel coordinates are converted into real-world coordinates using homography transformation.
- Robot and goal positions are detected using marker IDs.

### Step 2 — Hazard Detection

- Each frame is processed using a *YOLOv8 model*.
- Hazards are detected using bounding boxes.
- Hazard positions are mapped to world coordinates.

### Step 3 — Path Planning

- Environment converted into grid-based nodes.
- A* algorithm computes the shortest path.
- Dynamic replanning occurs when new hazards appear.

### Step 4 — Motion Control

- PID controller adjusts motor speed.
- Direction commands sent using WebSockets.
- Robot follows the computed path.

---

## Technologies Used

- Python
- OpenCV
- YOLOv8
- Computer Vision
- A* Algorithm
- PID Control
- WebSocket Communication
- Robotics Systems

---

## Performance Highlights

- Real-time hazard detection using YOLOv8
- Sub-centimeter localization accuracy
- Dynamic obstacle avoidance
- Real-time navigation updates
- Efficient path computation using A* (O(n log n))
