# A.M.I. – Augmented Maze Interactive

A.M.I. (Augmented Maze Interactive) is an augmented reality game that transforms a hand-drawn maze into an interactive 3D labyrinth.  
By combining computer vision, camera calibration, and 3D modeling, the project bridges the physical and digital worlds to create an immersive AR experience.

This project was developed as a **final-year major project** during engineering studies in **Imaging, Modeling, and Computer Science** at CPE Lyon.

---

## Project Objective

The goal of A.M.I. is to allow a player to:
- Draw a maze on paper,
- Automatically detect and interpret the drawing,
- Generate a corresponding 3D maze in real time,
- Interact with the maze through augmented reality using a physical marker.

A ball evolves inside the maze, influenced by gravity and user interactions, while the maze itself can be manipulated in the real world.

---

## Global Architecture

The project is divided into three main components:

1. **Computer Vision – Maze Detection**
2. **Camera Calibration & Coordinate Systems**
3. **3D Reconstruction & Gameplay (Unity)**

Each component communicates through structured JSON files.

---

## 1. Maze Detection (Computer Vision)

**Input:** A photo of a hand-drawn maze  
**Output:** A structured description of the maze geometry

Key steps:
- Image preprocessing and binarization
- Contour detection and line approximation
- Detection of maze walls with color-based behaviors:
  - **Black walls**: static walls
  - **Red walls**: moving walls
- Detection of special elements (start/end circles)
- Fusion of line segments and noise filtering
- Normalization of coordinates
- Export of maze structure to JSON

Technologies:
- Python
- OpenCV
- Image processing (contours, morphology, Hough transform)
- Custom geometric post-processing

---

## 2. Camera Calibration & AR Alignment

To ensure consistency between the real world and the virtual environment:
- Two cameras are calibrated using **ArUco markers**
- Extrinsic parameters (rotation & translation) are computed
- Coordinate transformations between:
  - World reference
  - Object reference
  - Unity reference system
- A Kalman filter is used to smooth motion and reduce jitter

This enables accurate alignment between physical movements and the virtual maze.

Technologies:
- OpenCV
- Pose estimation (`solvePnP`)
- Geometric transformations
- Kalman filtering

---

## 3. 3D Reconstruction & Gameplay (Unity)

Using the detected maze data:
- A 3D labyrinth is generated dynamically in Unity
- Walls, holes, start/end points are instantiated from JSON
- Physics-based ball movement with gravity
- Visual effects (particles, trails, animations)
- Interactive gameplay logic (win/lose conditions)

Key features:
- Real-time AR interaction
- Dynamic wall behaviors
- Object collection
- Visual feedback and animations

Technologies:
- Unity (C#)
- Physics engine
- Particle systems
- Real-time rendering

---

## Technologies & Tools

- **Python** – Computer vision & data processing  
- **OpenCV** – Image analysis & camera calibration  
- **Unity (C#)** – 3D rendering & gameplay  
- **JSON** – Data exchange between modules  
- **Git/GitHub** – Version control  

---

## Results

- Functional augmented reality game
- Real-time conversion from drawing to playable 3D maze
- Robust interaction between physical and virtual environments
- Modular architecture allowing future extensions

---

## Possible Improvements

- Curved walls using spline-based modeling
- Additional wall behaviors (fragile walls, portals)
- Adaptive difficulty levels
- Improved robustness to varied lighting and drawing styles
- Enhanced user interface and menus

---

## Author

Juliette Carpentier, Florian Davaux, Lucas Valladon
Engineering graduates – Imaging, Modeling & Computer Science  
