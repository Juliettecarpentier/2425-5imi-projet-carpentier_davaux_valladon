import cv2
import numpy as np
import matplotlib.pyplot as plt
import json
from PIL import Image
import copy

################## IMAGE INIT ##################
def image_to_gray(image):
    In = cv2.imread(image)
    I = cv2.cvtColor(In, cv2.COLOR_RGB2GRAY)
    return I

############# COLOR IMAGE EXTRACTION #############
def get_black_image(image, s=128):

    I = image_to_gray(image)
    _, I_binary = cv2.threshold(I, 128, 255, cv2.THRESH_BINARY)

    return I_binary

def get_rgb_image(image, s):
    # on souhaite obtenir une image pour chaque canal de couleur (rouge, vert, bleu) et les binariser

    # Lecture de l'image avec correction d'orientation
    image_no = Image.open(image)
    In = np.array(image_no)

    # on récupère les canaux de couleur
    R, G, B = cv2.split(In)

    # Suppression de l'influence des autres canaux
    def isolate_color_channel(main, other1, other2):
        return np.where((main > other1) & (main > other2), main, 0)
    
    R_m, G_m, B_m = [isolate_color_channel(c1, c2, c3) for c1, c2, c3 in [(R, G, B), (G, R, B), (B, R, G)]]
    
    # Binarisation des canaux de couleur
    Rb, Gb, Bb = [cv2.threshold(channel, s, 255, cv2.THRESH_BINARY)[1] for channel in [R_m, G_m, B_m]]
    
    return Rb, Gb, Bb

############## LINES SELECTION ##############
def merge_segments(segments, n, distance_threshold, n_min):
    
    # si des segments détectés sont très proches les uns des autres, on les fusionne
    merged_segments = segments.copy()
    for i, seg_i in enumerate(segments[:-1]):
        for j, seg_j in enumerate(segments[i + 1:], start=i + 1):
            
            if should_merge_segments(seg_i, seg_j, n, distance_threshold, n_min):
                shorter_idx = i if segment_length(seg_i) < segment_length(seg_j) else j
                merged_segments[shorter_idx] = None

    return [seg for seg in merged_segments if seg]

def should_merge_segments(seg1, seg2, n, distance_threshold, n_min):
    
    # on crée n points pour chaque segment
    points1 = np.linspace(seg1[0], seg1[1], n)
    points2 = np.linspace(seg2[0], seg2[1], n)
    
    # on calcule la distance entre les points des deux segments
    distances = np.linalg.norm(points1[:, None] - points2, axis=2).flatten()

    # on renvoie True si le nombre de points proches est supérieur à n_min
    return (distances < distance_threshold).sum() > n_min

def segment_length(segment):
    return np.linalg.norm(np.array(segment[1]) - np.array(segment[0]))

def delete_small_segments(segments, l):
    # on supprime les segments qui sont trop courts
    deleted_segments = segments.copy()
    for i, segment in enumerate(segments):
        
        longueur = segment_length(segment)

        # si la longueur est inférieure à l, on supprime le segment
        if longueur < l:
            deleted_segments[i] = None
    
    deleted_segments = [segment for segment in deleted_segments if segment is not None]

    return deleted_segments

def merge_points(segments, distance_between_points):
    # on souhaite fusionner les points qui sont très proches les uns des autres
    # Convertir les segments en listes pour permettre les modifications
    segments = [list(segment) for segment in segments]
    merged_segments = segments.copy()

    for i, seg_i in enumerate(segments[:-1]):
        for j, seg_j in enumerate(segments[i + 1:], start=i + 1):

            distance_1 = segment_length([seg_i[0], seg_j[0]])
            distance_2 = segment_length([seg_i[1], seg_j[1]])
            distance_3 = segment_length([seg_i[0], seg_j[1]])
            distance_4 = segment_length([seg_i[1], seg_j[0]])

            if distance_1 < distance_between_points:
                # on fait la moyen des points
                new_coord = (int((seg_i[0][0] + seg_j[0][0]) / 2), int((seg_i[0][1] + seg_j[0][1]) / 2))
                merged_segments[i][0] = new_coord
                merged_segments[j][0] = new_coord
            if distance_2 < distance_between_points:
                new_coord = (int((seg_i[1][0] + seg_j[1][0]) / 2), int((seg_i[1][1] + seg_j[1][1]) / 2))
                merged_segments[i][1] = new_coord
                merged_segments[j][1] = new_coord
            if distance_3 < distance_between_points:
                new_coord = (int((seg_i[0][0] + seg_j[1][0]) / 2), int((seg_i[0][1] + seg_j[1][1]) / 2))
                merged_segments[i][0] = new_coord
                merged_segments[j][1] = new_coord
            if distance_4 < distance_between_points:
                new_coord = (int((seg_i[1][0] + seg_j[0][0]) / 2), int((seg_i[1][1] + seg_j[0][1]) / 2))
                merged_segments[i][1] = new_coord
                merged_segments[j][0] = new_coord
    
    return merged_segments

def merge_points_mixed(segments, distance_between_points):
    # on souhaite fusionner les points qui sont très proches les uns des autres
    # Convertir les segments en listes pour permettre les modifications
    segments = [list(segment) for segment in segments]
    segments = copy.deepcopy(segments)
    merged_segments = copy.deepcopy(segments)

    # Convertir les tuples en listes pour permettre les modifications
    for i in range(len(segments)):
        for j in range(len(segments[i])):
            segments[i][j] = list(segments[i][j])
            merged_segments[i][j] = list(merged_segments[i][j])

    for i, seg_i in enumerate(segments[:-1]):
        for j, seg_j in enumerate(segments[i + 1:], start=i + 1):
            distance_1 = segment_length([seg_i[0][0], seg_j[0][0]])
            distance_2 = segment_length([seg_i[0][1], seg_j[0][1]])
            distance_3 = segment_length([seg_i[0][0], seg_j[0][1]])
            distance_4 = segment_length([seg_i[0][1], seg_j[0][0]])

            if distance_1 < distance_between_points:
                new_coord = (int((seg_i[0][0][0] + seg_j[0][0][0]) / 2), int((seg_i[0][0][1] + seg_j[0][0][1]) / 2))
                merged_segments[i][0][0] = new_coord
                merged_segments[j][0][0] = new_coord
            if distance_2 < distance_between_points:
                new_coord = (int((seg_i[0][1][0] + seg_j[0][1][0]) / 2), int((seg_i[0][1][1] + seg_j[0][1][1]) / 2))
                merged_segments[i][0][1] = new_coord
                merged_segments[j][0][1] = new_coord
            if distance_3 < distance_between_points:
                new_coord = (int((seg_i[0][0][0] + seg_j[0][1][0]) / 2), int((seg_i[0][0][1] + seg_j[0][1][1]) / 2))
                merged_segments[i][0][0] = new_coord
                merged_segments[j][0][1] = new_coord
            if distance_4 < distance_between_points:
                new_coord = (int((seg_i[0][1][0] + seg_j[0][0][0]) / 2), int((seg_i[0][1][1] + seg_j[0][0][1]) / 2))
                merged_segments[i][0][1] = new_coord
                merged_segments[j][0][0] = new_coord

    for segment in merged_segments:
        # Fusionner les lettres si le deuxième élément est une liste de caractères
        if len(segment) > 1 and isinstance(segment[1], list):
            segment[1] = ''.join(segment[1])  

    return merged_segments

def select_lines(image, segments, l=20, n=15, distance_between_points=10, n_min=7, distance_between_summit=10):

    # on supprime les segments qui sont trop courts
    segments = delete_small_segments(segments, l)

    # on fusionne les sommets puis les segments qui sont très proches les uns des autres
    segments = merge_points(segments, distance_between_summit)
    segments = merge_segments(segments, n, distance_between_points, n_min)

    return segments

################## LINES AND CIRCLES DETECTION ##################
def increase_exposure(image, gamma=1.5):
    # Augmente l'exposition d'une image en ajustant le gamma.
    # Vérification des types
    if not isinstance(image, np.ndarray):
        raise ValueError("L'entrée doit être une image sous forme de tableau NumPy.")
    
    # Construction de la table de conversion gamma 
    # gamma modifie la luminosité de l'image
    inv_gamma = 1.0 / gamma
    lookup_table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype("uint8")
    
    # Application de la correction gamma
    # la fonction LUT applique la table de conversion à l'image
    return cv2.LUT(image, lookup_table)

def region_growing(I, kernel_size=5):
    
    # Ouverture morphologique pour créer la graine
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    seed = cv2.morphologyEx(I, cv2.MORPH_OPEN, kernel)
    
    seeds = np.argwhere(seed > 0) # Obtenir les pixels non-nuls comme graines
    
    # Croissance de région
    height, width = I.shape
    visited = np.zeros_like(I, dtype=bool)
    output = np.zeros_like(I, dtype=np.uint8)

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)] # Définir les directions pour le parcours
    
    # Parcours des graines
    for sx, sy in seeds:
        if visited[sx, sy]:
            continue
        
        # Initialisation de la file pour la croissance
        queue = [(sx, sy)]
        visited[sx, sy] = True
        
        while queue:
            x, y = queue.pop(0)
            output[x, y] = 255  # Marque la région comme visitée
            
            # Explore les voisins
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if 0 <= nx < height and 0 <= ny < width:  # Vérifie les limites
                    if not visited[nx, ny] and I[nx, ny] > 0:  # Croissance conditionnelle
                        visited[nx, ny] = True
                        queue.append((nx, ny))
    
    return output

def detect_lines(image, binary, e=0.001):

    # Détection des contours (findContours renvoie une liste de contours et une liste de hiérarchie)
    contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Parcourir les contours détectés
    segments = []
    for contour in contours:
        # Approximation des contours pour obtenir des segments droits
        epsilon = e * cv2.arcLength(contour, True)         # Epsilon est la précision de l'approximation
        approx = cv2.approxPolyDP(contour, epsilon, True)  # Approximation des contours par des segments droits

        # Extraction des segments (chaque segment est défini par deux points)
        for i in range(len(approx)):
            x1, y1 = approx[i][0]
            x2, y2 = approx[(i + 1) % len(approx)][0]
            segments.append(((x1, y1), (x2, y2)))

    return segments

def detect_circles(image, segments, s=128):
    # on souhaite récupérer les coordonnées du centre des cercles

    # on récupère le canal bleu
    _, _, Bb = get_rgb_image(image, s)

    # érosion pour éliminer le bruit
    kernel = np.ones((5, 5), np.uint8)
    Bb = cv2.erode(Bb, kernel, iterations=1)

    # on détecte les cercles
    dp = 1          # inverse du ratio de résolution de l'accumulateur
    minDist = 30    # distance minimale entre les centres des cercles
    param1 = 20     # seuil pour la détection des bords des cercles (plus il est grand, moins il y a de cercles détectés)
    param2 = 0.1    # seuil pour la détection des centres des cercles (plus il est petit, moins il y a de cercles détectés)
    minRadius = 10  # rayon minimal
    maxRadius = 60  # rayon maximal
    circles = cv2.HoughCircles(Bb, cv2.HOUGH_GRADIENT_ALT, dp, minDist, param1=param1, param2=param2, minRadius=minRadius, maxRadius=maxRadius)

    # on vérifie si des cercles ont été détectés
    if circles is None:
        circles = default_circles(image, segments)
    else:
        circles = circles[0, :, :2]
        circles = np.unique(circles, axis=0)
        if len(circles) < 2:
            circles = default_circles(image, segments)
        if len(circles) > 2:
            circles = circles[:2]

    return circles

def default_circles(image, segments):
    # on défini 2 cercles dans les angles du rectangle englobant
    # un peu décalé pour ne pas être confondu avec les segments

    # on récupère les coordonnées du rectangle englobant
    x_min, y_min, x_max, y_max = find_englobing_rectangle(image, segments)

    # on défini les coordonnées des cercles
    C = [(x_min + 10, y_min + 10), (x_max - 10, y_max - 10)]

    return C

################## NORMALIZATION AND EXTRACTION ##################
def find_englobing_square(image, segments):
    
    width, height = image.shape[1], image.shape[0]

    # dans segment, on supprime les point trop proche du bord de l'image (10 pixels)
    segments = [segment for segment in segments if all(10 < x < width - 10 and 10 < y < height - 10 for x, y in segment)]

    # Initialisation des limites à partir des segments
    x_min, y_min, x_max, y_max = find_englobing_rectangle(image, segments)

    # Calcul des dimensions
    width = x_max - x_min
    height = y_max - y_min

    # Ajustement pour obtenir un carré
    if width > height:
        padding = (width - height) // 2
        y_min -= padding
        y_max += padding
    else:
        padding = (height - width) // 2
        x_min -= padding
        x_max += padding

    # Conversion en entiers et retour des coordonnées
    return int(x_min), int(y_min), int(x_max), int(y_max)

def find_englobing_rectangle(image, segments):

    # Récupération des dimensions de l'image
    height, width = image.shape[0], image.shape[1]

    if len(segments) == 0:
        return 0, 0, width, height
    
    x_min = min(min(segment[0][0], segment[1][0]) for segment in segments)
    x_max = max(max(segment[0][0], segment[1][0]) for segment in segments)
    y_min = min(min(segment[0][1], segment[1][1]) for segment in segments)
    y_max = max(max(segment[0][1], segment[1][1]) for segment in segments)

    # Retour des coordonnées du rectangle
    return (x_min, y_min, x_max, y_max)

def normalize_points(image, segments_black, segments_red, circles):

    x_min, y_min, x_max, y_max = find_englobing_square(image, segments_black)

    # on supprime les points qui ne sont pas dans le carré englobant
    segments_black = [segment for segment in segments_black if all(x_min < x < x_max and y_min < y < y_max for x, y in segment)]
    segments_red = [segment for segment in segments_red if all(x_min < x < x_max and y_min < y < y_max for x, y in segment)]
    circles = [circle for circle in circles if x_min < circle[0] < x_max and y_min < circle[1] < y_max]

    def norm(x, y):
        # Normalise un point (x, y) pour le ramener dans [-1, 1] selon les limites du carré englobant
        x_norm = 2 * (x - x_min) / (x_max - x_min) - 1
        y_norm = 2 * (y - y_min) / (y_max - y_min) - 1
        return x_norm, y_norm

    # Normalisation des segments noirs
    L_black = [tuple(norm(*point) for point in segment) for segment in segments_black]
    
    # Normalisation des segments rouges
    L_red = [tuple(norm(*point) for point in segment) for segment in segments_red]
    
    # Normalisation des cercles (si présents)
    if circles is not None and len(circles) > 0:
        C = [norm(x, y) for x, y in circles]
    else:
        C = []

    return L_black, L_red, C

def add_objects(image, segments_black, segments_red, circles):

    # On veut mettre 3 objets dans le labyrinthe là où il n'y a pas de segments ni de cercles
    x_min, y_min, x_max, y_max = find_englobing_rectangle(image, segments_black)

    # On récupère les coordonnées des segments et des cercles
    segments = segments_black + segments_red
    objects = []

    # On crée 3 objets
    for _ in range(3):
        # On tire aléatoirement les coordonnées de l'objet
        x = np.random.uniform(x_min + 0.3, x_max - 0.3)
        y = np.random.uniform(y_min + 0.3, y_max - 0.3)

        val = True
        while val:
            val = False
            for seg in segments:
                # on met plusieurs points dans le segment 
                pts = np.linspace(seg[0], seg[1], 10)
                for pt in pts:
                    # on vérifie si l'objet n'est pas dans un segment
                    if segment_length([[x, y], pt]) < 0.2:
                        x = np.random.uniform(x_min + 0.3, x_max - 0.3)
                        y = np.random.uniform(y_min + 0.3, y_max - 0.3)
                        val = True
                        break

        objects.append((x, y))

    return objects

def reconstruire_labyrinthe_from_normalize(image, L_black, L_red, C, objects):

    # Création d'une image blanche de la taille de l'image
    width, height = image.shape[1], image.shape[0]
    image_segments = np.ones((height, width, 3), np.uint8) * 255

    # Affichage des segments fusionnés
    for segment in L_black:
        x1, y1 = segment[0]
        x2, y2 = segment[1]
        cv2.line(image_segments, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 0), 2)

    for segment in L_red:
        x1, y1 = segment[0]
        x2, y2 = segment[1]
        cv2.line(image_segments, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)


    # Affichage des cercles (on les représente par des points)
    for circle in C:
        x, y = circle
        cv2.circle(image_segments, (int(x), int(y)), 10, (0, 255, 0), 2)

    for obj in objects:
        x, y = obj

        # passer les coordonnées de [-1;1] à [0;height] et [0;width]
        x = (x + 1) * width / 2
        y = (y + 1) * height / 2

        cv2.circle(image_segments, (int(x), int(y)), 10, (255, 0, 0), 2)
    
    # Affichage de l'image
    plt.figure()
    plt.imshow(cv2.cvtColor(image_segments, cv2.COLOR_BGR2RGB))
    plt.title("Segments et Cercles Normalisés")
    plt.show()

def export_data_to_json(image, segments_black, segments_red, circles, objects, output_file):

    # Construction des données JSON
    data = {
        "input": [],
        "sides": [],
        "objects": [],
        "walls": []
    }

    def add_wall(segment, color, section):
        # Ajoute un mur au fichier JSON
        
        if len(segment) != 2:
            print(f"Segment invalide ignoré : {segment}")
            return
        
        # (x1, y1), (x2, y2) = segment
        x1, y1 = map(float, segment[0])
        x2, y2 = map(float, segment[1])
        wall = {
            "points": [
                {"x": x1, "y": 0, "z": y1},
                {"x": x2, "y": 0, "z": y2}
            ],
            "color": color
        }
        data[section].append(wall)

    # Ajout des segments noirs et rouges
    for segment in segments_black:
        add_wall(segment, "black", "walls")
    for segment in segments_red:
        add_wall(segment, "red", "walls")


    x_min, y_min, x_max, y_max = find_englobing_rectangle(image, segments_black)
    add_wall([(x_min, y_min), (x_max, y_min)], "black", "sides")
    add_wall([(x_max, y_min), (x_max, y_max)], "black", "sides")
    add_wall([(x_max, y_max), (x_min, y_max)], "black", "sides")
    add_wall([(x_min, y_max), (x_min, y_min)], "black", "sides")    


    for circle in circles:
        # Vérifiez que le cercle a bien un point
        if len(circle) != 2:
            print(f"Cercle invalide ignoré : {circle}")
            continue
        x, y = map(float, circle)
        input = {
            "points": [
                {"x": x, "y": 0, "z": y}
            ],
        }
        data["input"].append(input)

    for obj in objects:
        # Vérifiez que l'objet a bien un point
        if len(obj) != 2:
            print(f"Objet invalide ignoré : {obj}")
            continue
        x, y = map(float, obj)
        object = {
            "points": [
                {"x": x, "y": 0, "z": y}
            ],
        }
        data["objects"].append(object)

    # Écriture des données dans un fichier JSON
    with open(output_file, "w") as json_file:
        json.dump(data, json_file, indent=4)

    print(f"Fichier JSON exporté avec succès : {output_file}")
