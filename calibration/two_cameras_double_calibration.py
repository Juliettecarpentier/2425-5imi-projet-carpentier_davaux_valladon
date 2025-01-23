import numpy as np
import cv2 as cv
from cv2 import aruco
import socket
import json

################################################################################
# Filtre de Kalman
################################################################################

def initialize_kalman_filter():
    """
    Initialise un filtre de Kalman pour suivre les paramètres extrinsèques (rvec, tvec).
    """
    kalman = cv.KalmanFilter(12, 12)  # 12 états (rvec et tvec)*2, 12 mesures
    kalman.measurementMatrix = np.eye(12, dtype=np.float32)
    kalman.transitionMatrix = np.eye(12, dtype=np.float32)
    kalman.processNoiseCov = np.eye(12, dtype=np.float32) * 1e-3
    kalman.measurementNoiseCov = np.eye(12, dtype=np.float32) * 5e-2
    kalman.errorCovPost = np.eye(12, dtype=np.float32)
    kalman.statePost = np.zeros(12, dtype=np.float32) * 0.5
    return kalman

def update_kalman_filter(kalman, rvec1, tvec1, rvec2, tvec2):
    """
    Met à jour le filtre de Kalman avec les nouvelles observations de rvec et tvec.
    """
    # Convertir les données d'entrée en un vecteur de mesure
    measurement = np.concatenate((rvec1.flatten(), tvec1.flatten(), rvec2.flatten(), tvec2.flatten()), axis=0).astype(np.float32)

    # Prédiction
    kalman.predict()

    # Correction
    kalman.correct(measurement)

    # Récupérer les valeurs corrigées
    rvec1 = kalman.statePost[:3].reshape(3, 1)
    tvec1 = kalman.statePost[3:6].reshape(3, 1)
    rvec2 = kalman.statePost[6:9].reshape(3, 1)
    tvec2 = kalman.statePost[9:12].reshape(3, 1)

    return rvec1, tvec1, rvec2, tvec2

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
        "id_cam2": cameras[1],

        # Serveur
        "UDP_IP": "127.0.0.1",
        "UDP_PORT": 5065,
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
    print('opening camera ', config["id_cam2"])
    cap2 = cv.VideoCapture(config["id_cam2"])

    resolution = config["resolution"]
    cap1.set(cv.CAP_PROP_FRAME_WIDTH, resolution[0])
    cap1.set(cv.CAP_PROP_FRAME_HEIGHT, resolution[1])
    cap2.set(cv.CAP_PROP_FRAME_WIDTH, resolution[0])
    cap2.set(cv.CAP_PROP_FRAME_HEIGHT, resolution[1])

    return cap1, cap2

def setup_udp_server(config):
    """
    Configure le serveur UDP pour la communication.
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    return sock

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
            z = ligne * (taille_tag + espacement)
            y = 0  # Plan XZ (y=0)
            
            # Coordonnées des 4 coins dans le repère monde
            coin_hg = np.array([x, y, z])
            coin_hd = np.array([x + taille_tag, y, z])
            coin_bd = np.array([x + taille_tag, y, z + taille_tag])
            coin_bg = np.array([x, y, z + taille_tag])
            
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

def get_transformation_matrix(r1, t1, r2, t2):
    """
    Calcule la matrice de transformation entre deux caméras.
    """
    rm1, _ = cv.Rodrigues(r1)
    rm2, _ = cv.Rodrigues(r2)
    rm12 = np.dot(rm2, rm1.T)
    r12, _ = cv.Rodrigues(rm12)
    t12 = t2 - np.dot(rm12, t1)
    return r12, t12


def compute_camera2_from_camera1(r1, t1, r12, t12):
    """
    Calcule les paramètres extrinsèques de la caméra 2 en fonction de la caméra 1.
    """
    rm1, _ = cv.Rodrigues(r1)
    rm12, _ = cv.Rodrigues(r12)
    r2, _ = cv.Rodrigues(np.dot(rm12, rm1))
    t2 = np.dot(rm12, t1) + t12
    return r2, t2

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
    tvec = matrix[:3, 3].reshape(3, 1)
    return rvec, tvec

def object_world_transformation(rvec_object, tvec_object, rvec_world, tvec_world):
    """
    Calcule le rvec et vec de changement de repère.
    """
    r_obj, t_obj = None, None

    T_object = rtvec_to_matrix(rvec_object, tvec_object)
    T_world = rtvec_to_matrix(rvec_world, tvec_world)
    T_object_to_world = np.linalg.pinv(T_world) @ T_object
    r_obj, t_obj = matrix_to_rtvec(T_object_to_world)

    return r_obj, t_obj

def display_results(frame, points, rvec, tvec, m_cam, distortion, color=(0, 0, 255)):
    """
    Affiche les résultats de la calibration.
    """
    img_points, _ = cv.projectPoints(points, rvec, tvec, m_cam, distortion)
    img_points = img_points.reshape(-1, 2).astype(int)

    for point in img_points:
        cv.circle(frame, tuple(point), 5, color, -1)
    

################################################################################
# Main Loop
################################################################################

def calibrate_cameras(cap1, cap2, config, calibration_points):
    """
    Effectue la calibration des deux caméras et calcule la matrice de transformation.
    """
    calibrated = False
    r12, t12 = None, None

    while not calibrated and cap1.isOpened() and cap2.isOpened():
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()

        if ret1 and ret2:
            # Détection des tags ArUco
            marker_corners1, marker_IDs1 = detect_aruco_tags(frame1, config["ARUCO_DICT_world"], config["ARUCO_PARAMETERS_world"])
            marker_corners2, marker_IDs2 = detect_aruco_tags(frame2, config["ARUCO_DICT_world"], config["ARUCO_PARAMETERS_world"])

            ret1, r1, t1 = calibrate_camera(config, calibration_points, marker_corners1, marker_IDs1)
            ret2, r2, t2 = calibrate_camera(config, calibration_points, marker_corners2, marker_IDs2)

            if ret1 and ret2:
                r12, t12 = get_transformation_matrix(r1, t1, r2, t2)
                calibrated = True

                # Affichage des résultats
                for id in marker_IDs1.flatten():
                    # Affichage des points de calibrage
                    if id in config["ARUCO_ID_LIST"]:
                        display_results(frame1, calibration_points[id], r1, t1, config["m_cam"], config["distortion"])
                        display_results(frame2, calibration_points[id], r2, t2, config["m_cam"], config["distortion"])

                        frame3 = frame2
                        cv.imshow('Computed', frame3)

        if cv.waitKey(1) == 27:
            break

    return r12, t12

def main_loop(cap1, config, world_calibration_points, object_calibration_points, r12, t12, sock):
    """
    Boucle principale pour capturer les images et envoyer les paramètres via UDP.
    """
    while cap1.isOpened():
        ret, frame = cap1.read()
        if ret:
            # Détection des tags ArUco
            marker_corners_world, marker_IDs_world = detect_aruco_tags(frame, config["ARUCO_DICT_world"], config["ARUCO_PARAMETERS_world"])
            marker_corners_object, marker_IDs_object = detect_aruco_tags(frame, config["ARUCO_DICT_object"], config["ARUCO_PARAMETERS_object"])
            if len(marker_corners_world) < 4 or len(marker_corners_object) < 4:
                continue

            # Calibration des caméras
            ret1_world, rvec1_world, tvec1_world = calibrate_camera(config, world_calibration_points, marker_corners_world, marker_IDs_world)
            ret1_object, rvec1_object, tvec1_object = calibrate_camera(config, object_calibration_points, marker_corners_object, marker_IDs_object)
            
            if ret1_world and ret1_object:
                rvec1_object, tvec1_object, rvec1_world, tvec1_world = update_kalman_filter(kalman, rvec1_object, tvec1_object, rvec1_world, tvec1_world)  
                r_obj, t_obj = object_world_transformation(rvec1_object, tvec1_object, rvec1_world, tvec1_world)

                # Calcul des paramètres extrinsèques de la caméra 2
                r2, t2 = compute_camera2_from_camera1(rvec1_world, tvec1_world, r12, t12)
                rm, _ = cv.Rodrigues(r2)

                # Construction du message JSON
                message = json.dumps({
                    'C': config["id_cam2"],
                    'M': config["m_cam"].reshape(-1).tolist(),
                    'R': r2.T.tolist()[0],
                    'T': t2.T.tolist()[0],
                    'F': rm[:, 2].tolist(),
                    'U': rm[:, 1].T.tolist(),
                    'R_obj': r_obj.T.tolist()[0],
                    'T_obj': t_obj.T.tolist()[0],
                })
                sock.sendto(message.encode(), (config["UDP_IP"], config["UDP_PORT"]))

            cv.imshow('Camera 1', frame)

        if cv.waitKey(1) == 27:
            break


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

    # Initialisation du filtre de Kalman
    kalman = initialize_kalman_filter()
    
    # Initialisation des caméras
    cap1, cap2 = initialize_cameras(config)
    # Initialisation des points de calibration
    world_calibration_points = initialize_calibration_points(5, 7, config["ARUCO_SIZE_mm_world"], config["ARUCO_SPACING_mm_world"])
    object_calibration_points = initialize_calibration_points(5, 7, config["ARUCO_SIZE_mm_object"], config["ARUCO_SPACING_mm_object"])
    # Initialisation du serveur UDP
    sock = setup_udp_server(config)
    print("Configuration terminée.")

    # Boucle principale
    try:
        r12, t12 = calibrate_cameras(cap1, cap2, config, world_calibration_points)
        cap2.release()
        main_loop(cap1, config, world_calibration_points, object_calibration_points, r12, t12, sock)
    finally:
        cap1.release()
        cv.destroyAllWindows()