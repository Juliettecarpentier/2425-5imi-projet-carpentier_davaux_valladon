import numpy as np
import cv2 as cv
from cv2 import aruco

################################################################################
# Détection des caméras
################################################################################

def list_and_display_cameras():
    """
    Teste les indices des caméras pour trouver celles qui sont connectées
    et affiche leur flux vidéo.
    """
    available_cameras = []
    
    for cam_id in range(10):  # Tester les indices de 0 à 9
        cap = cv.VideoCapture(cam_id) 
        if cap.isOpened():
            print(f"Camera found with ID: {cam_id}")
            available_cameras.append(cam_id)
            
            cap.release()
            cv.destroyAllWindows()
        else:
            continue

    return available_cameras

################################################################################
# Configuration
################################################################################

def configure_system(cameras):
    """
    Configure les paramètres nécessaires au fonctionnement du programme.
    """
    config = {

        # Paramètres Aruco tags
        "ARUCO_DICT": aruco.getPredefinedDictionary(aruco.DICT_4X4_50),
        "ARUCO_PARAMETERS": aruco.DetectorParameters(),
        "ARUCO_SIZE_mm": 41,
        "ARUCO_SPACING_mm": 14,

        # Paramètres intrinsèques webcam
        "sensor_mm": np.array([3.58, 2.685]),
        "focal_mm": 4,
        "resolution": np.array([1280, 960]),
        "distortion": np.zeros((4, 1)),

        # Webcam
        "id_cam1": cameras[0],
    }

    # Calcul du centre et de la matrice intrinsèque de la caméra
    resolution = config["resolution"]
    center = (resolution[0] / 2, resolution[1] / 2)
    focal_mm = config["focal_mm"]
    sensor_mm = config["sensor_mm"]

    config["m_cam"] = np.array([
        [focal_mm * resolution[0] / sensor_mm[0], 0, center[0]],
        [0, focal_mm * resolution[1] / sensor_mm[1], center[1]],
        [0, 0, 1]
    ], dtype="double")

    return config

################################################################################
# Initialisation
################################################################################

def initialize_cameras(config):
    """
    Initialise les caméras avec les paramètres spécifiés.
    """
    print('opening camera ', config["id_cam1"])
    cap1 = cv.VideoCapture(config["id_cam1"])

    resolution = config["resolution"]    
    cap1.set(cv.CAP_PROP_FRAME_WIDTH, resolution[0])
    cap1.set(cv.CAP_PROP_FRAME_HEIGHT, resolution[1])

    return cap1

def initialize_calibration_points(nb_colonnes, nb_lignes, taille_tag, espacement):
    """
    Définit les positions globales des tags ArUco dans un repère monde global.
    """
    positions = {}
    id_tag = 0
    for ligne in range(nb_lignes):
        for colonne in range(nb_colonnes):
            # Coordonnées globales du centre du tag
            x = colonne * (taille_tag + espacement)
            y = ligne * (taille_tag + espacement)
            z = 0  # Plan XY (z=0)
            
            # Coordonnées des 4 coins dans le repère monde
            coin_hg = np.array([x, y, z])
            coin_hd = np.array([x + taille_tag, y, z])
            coin_bd = np.array([x + taille_tag, y + taille_tag, z])
            coin_bg = np.array([x, y + taille_tag, z])
            
            positions[id_tag] = np.array([coin_hg,coin_hd, coin_bd, coin_bg], dtype=np.float32)
            id_tag += 1

    return positions


################################################################################
# Toolbox
################################################################################

def detect_aruco_tags(frame, config):
    """
    Detects ArUco markers in the given frame.
    """
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    corners, ids, _ = aruco.detectMarkers(
        gray,
        config["ARUCO_DICT"],
        parameters=config["ARUCO_PARAMETERS"]
    )
    return corners, ids

def calibrate_camera(config, calibration_points, corners, ids):
    """
    Calibre une image de la caméra à l'aide des positions dans le repère monde
    des coins des tags ArUco.
    """
    ret, rvec, tvec = None, None, None
    object_points = []
    image_points = []
    for i, marker_id in enumerate(ids.flatten()):
        if marker_id in calibration_points:
            for j in range(4):
                object_points.append(calibration_points[marker_id][j])
                image_points.append(corners[i][0][j])    
    object_points = np.array(object_points, dtype=np.float32).reshape(-1, 3)
    image_points = np.array(image_points, dtype=np.float32).reshape(-1, 2)

    if len(corners) > 0:
        ret, rvec, tvec = cv.solvePnP(object_points, image_points, config["m_cam"], config["distortion"])

    return ret, rvec, tvec

################################################################################
# Main Loop
################################################################################

def main_loop(cap, config, calibration_points):
    """
    Boucle principale pour capturer les images et envoyer les paramètres via UDP.
    """
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:

            # Détection des tags ArUco
            marker_corners, marker_IDs = detect_aruco_tags(frame, config)

            # Affichage des tags
            for ids, corners in zip(marker_IDs, marker_corners):
                cv.polylines(frame, [corners.astype(np.int32)], True, (0, 255, 255), 4, cv.LINE_AA)
                corners = corners.reshape(4, 2)
                corners = corners.astype(int)
                tag_center = corners.mean(axis=0).astype(int)
                tag_center[0] -= 20
                cv.putText(
                    frame,
                    f"id: {ids[0]}",
                    tuple(tag_center),
                    cv.FONT_HERSHEY_PLAIN,
                    1.3,
                    (200, 100, 0),
                    2,
                    cv.LINE_AA,
                )

            # Calibrage de la caméra
            ret, rvec, tvec = calibrate_camera(config, calibration_points, marker_corners, marker_IDs)

            if ret:
                # Affichage des points de calibration sur l'image en les projetant
                for id in marker_IDs.flatten():
                    if id in calibration_points:
                        projected, _ = cv.projectPoints(calibration_points[id], rvec, tvec, config["m_cam"], config["distortion"])
                        for point in projected:
                            cv.circle(frame, tuple(point.ravel().astype(int)), 5, (0, 0, 255), -1)
            
            # Affichage du flux vidéo
            cv.imshow("frame", frame)
        
        # Exit on 'q' key press
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cv.destroyAllWindows()


################################################################################
# Programme principal
################################################################################

if __name__ == "__main__":
    # Trouver les caméras disponibles
    cameras = list_and_display_cameras()
    if not cameras:
        print("Aucune caméra n'est connectée.")
    
    # Configuration du système
    config = configure_system(cameras)
    # Initialisation des caméras
    cap = initialize_cameras(config)
    # Initialisation des points de calibration
    calibration_points = initialize_calibration_points(5, 7, config["ARUCO_SIZE_mm"], config["ARUCO_SPACING_mm"])
    print("Configuration terminée.")

    # Boucle principale
    try:
        main_loop(cap, config, calibration_points)
    finally:
        cap.release()
        cv.destroyAllWindows()