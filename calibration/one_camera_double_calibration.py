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
        "ARUCO_DICT_world": aruco.getPredefinedDictionary(aruco.DICT_4X4_50),
        "ARUCO_PARAMETERS_world": aruco.DetectorParameters(),
        "ARUCO_SIZE_mm_world": 41,
        "ARUCO_SPACING_mm_world": 14,
        "ARUCO_ID_LIST": list(range(35)),
        "ARUCO_DICT_object": aruco.getPredefinedDictionary(aruco.DICT_6X6_50),
        "ARUCO_PARAMETERS_object": aruco.DetectorParameters(),
        "ARUCO_SIZE_mm_object": 29,
        "ARUCO_SPACING_mm_object": 10,

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
            
            positions[id_tag] = np.array([coin_hg, coin_hd, coin_bd, coin_bg], dtype=np.float32)
            id_tag += 1

    return positions


################################################################################
# Toolbox
################################################################################

def detect_aruco_tags(frame, dict, parameters):
    """
    Detects ArUco markers in the given frame.
    """
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    corners, ids, _ = aruco.detectMarkers(
        gray,
        dict,
        parameters=parameters
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

def rtvec_to_matrix(rvec, tvec):
    """
    Convertit les vecteurs de rotation et de translation en matrice de transformation.
    """
    rvec = np.asarray(rvec)
    tvec = np.asarray(tvec)

    R, _ = cv.Rodrigues(rvec)
    T = np.hstack((R, tvec.reshape(-1, 1)))
    T = np.vstack((T, [0, 0, 0, 1]))
    return T

def matrix_to_rtvec(matrix):
    """
    Convertit une matrice de transformation en vecteurs de rotation et de translation.
    """
    rvec, _ = cv.Rodrigues(matrix[:3, :3])
    tvec = matrix[:3, 3]
    return rvec, tvec

def object_points_to_world_points(object_points, rvec_object, tvec_object, rvec_world, tvec_world):
    """
    Convertit les points de l'objet en points du repère monde.
    
    Parameters:
        object_points (dict): Dictionnaire des points dans le repère de l'objet.
        rvec_object (np.ndarray): Vecteur de rotation du repère de l'objet.
        tvec_object (np.ndarray): Vecteur de translation du repère de l'objet.
        rvec_world (np.ndarray): Vecteur de rotation du repère monde.
        tvec_world (np.ndarray): Vecteur de translation du repère monde.
    
    Returns:
        dict: Points transformés dans le repère monde.
    """
    world_points = {}

    # Calcul des matrices extrinsèques
    T_object = rtvec_to_matrix(rvec_object, tvec_object)
    T_world = rtvec_to_matrix(rvec_world, tvec_world)

    # Calcul de la matrice de transformation
    T_object_to_world = np.linalg.pinv(T_world) @ T_object
    print(f"\n Matrice de transformation : \n {T_object_to_world}")

    # Passage des points de l'objet au repère monde
    for id_object, points_object in object_points.items():
        tag_points = []
        for point_object in points_object:
            # Ajout d'une coordonnée homogène
            point_object = np.append(point_object, 1)
            # Transformation des points
            point_world = T_object_to_world @ point_object
            # Retour à des coordonnées cartésiennes
            point_world = point_world[:3] / point_world[3]
            # Ajout du point transformé
            tag_points.append(point_world)
        world_points[id_object] = np.array(tag_points, dtype=np.float32)

    return world_points


################################################################################
# Main Loop
################################################################################

def main_loop(cap, config, world_calibration_points, object_calibration_points):
    """
    Boucle principale pour capturer les images et envoyer les paramètres via UDP.
    """
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:

            # Détection des tags ArUco
            marker_corners_world, marker_IDs_world = detect_aruco_tags(frame, config["ARUCO_DICT_world"], config["ARUCO_PARAMETERS_world"])
            marker_corners_object, marker_IDs_object = detect_aruco_tags(frame, config["ARUCO_DICT_object"], config["ARUCO_PARAMETERS_object"])

            # Affichage des tags
            for ids, corners in zip(marker_IDs_world, marker_corners_world):
                cv.polylines(frame, [corners.astype(np.int32)], True, (255,0,0), 4, cv.LINE_AA)
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
            
            for ids, corners in zip(marker_IDs_object, marker_corners_object):
                cv.polylines(frame, [corners.astype(np.int32)], True, (0,255,255), 4, cv.LINE_AA)
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
            ret_world, rvec_world, tvec_world = calibrate_camera(config, world_calibration_points, marker_corners_world, marker_IDs_world)
            ret_object, rvec_object, tvec_object = calibrate_camera(config, object_calibration_points, marker_corners_object, marker_IDs_object)

            if ret_world and ret_object:
                object_world_points = object_points_to_world_points(object_calibration_points, rvec_object, tvec_object, rvec_world, tvec_world)

                print(f"\n Points de calibration monde : \n {world_calibration_points}")
                print("\n\n\n")
                print(f"\n Points de calibration objet : \n {object_calibration_points}")
                print("\n\n\n")
                print(f"\n Points de calibration objet dans le repère monde : \n {object_world_points}")


                # Affichage des points de calibrage sur l'image en les projetant
                for id_world, id_object in zip(marker_IDs_world.flatten(), marker_IDs_object.flatten()):

                    # Affichage des points de calibrage
                    if id_world in config["ARUCO_ID_LIST"]:
                        projected_world, _ = cv.projectPoints(world_calibration_points[id_world], rvec_world, tvec_world, config["m_cam"], config["distortion"])
                        for point_world in projected_world:
                            cv.circle(frame, tuple(point_world.ravel().astype(int)), 5, (0, 0, 255), -1)
                    if id_object in config["ARUCO_ID_LIST"]:
                        projected_object, _ = cv.projectPoints(object_calibration_points[id_object], rvec_object, tvec_object, config["m_cam"], config["distortion"])
                        projected_object_world, _ = cv.projectPoints(object_world_points[id_object], rvec_world, tvec_world, config["m_cam"], config["distortion"])
                        for point_object, point_object_world in zip(projected_object, projected_object_world):
                            cv.circle(frame, tuple(point_object.ravel().astype(int)), 5, (255, 0, 0), -1)
                            cv.circle(frame, tuple(point_object_world.ravel().astype(int)), 2, (0, 255, 0), -1)

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
    world_calibration_points = initialize_calibration_points(5, 7, config["ARUCO_SIZE_mm_world"], config["ARUCO_SPACING_mm_world"])
    object_calibration_points = initialize_calibration_points(5, 7, config["ARUCO_SIZE_mm_object"], config["ARUCO_SPACING_mm_object"])
    print("Configuration terminée.")

    # Boucle principale
    try:
        main_loop(cap, config, world_calibration_points, object_calibration_points)
    finally:
        cap.release()
        cv.destroyAllWindows()