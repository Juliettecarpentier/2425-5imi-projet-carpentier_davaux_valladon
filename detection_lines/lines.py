import lines_functions as lf
import cv2

############## Paramètres ################
image_path = 'images_test/labyrinthe/papier/labyrinthe_9.jpg'  # chemin de l'image

# paramètres pour la détection des lignes
e = 0.01                       # paramètre d'approximation des contours en droites
s_black = 245                  # seuil de binarisation pour les lignes noires
s_red = 150                    # seuil de binarisation pour les lignes rouges
s_circle = 100                 # seuil de binarisation pour les cercles

# paramètres pour la sélection des lignes
l = 30                               # longueur minimale d'un segment pour être détecté
n = 15                               # nombre de points calculé entre deux points d'un segment
distance_between_points_black = 10   # distance maximale entre deux points des segments pour les considérer comme proches (noir)
distance_between_points_red = 40     # distance maximale entre deux points des segments pour les considérer comme proches (rouge)
n_min = 6                            # nombre minimal de points proches pour que deux segments soient fusionnés
distance_between_summits_black = 20  # distance maximale entre deux sommets pour les fusionner (noir)
distance_between_summits_red = 30    # distance maximale entre deux sommets pour les fusionner (rouge)


image = cv2.imread(image_path)

############## Détection des lignes et cercles ################
binary_black = lf.get_black_image(image_path, s_black)
L_black = lf.detect_lines(image, binary_black, e)

exposed_image = lf.increase_exposure(image, gamma=10.0)

# enregistrer l'image exposée
cv2.imwrite("images_test/labyrinthe/papier/labyrinthe_exposed.jpg", exposed_image)
image_exposed_path = "images_test/labyrinthe/papier/labyrinthe_exposed.jpg"

binary_red, _, _ = lf.get_rgb_image(image_exposed_path, s_red)
binary = lf.region_growing(binary_red, 12)
L_red = lf.detect_lines(image, binary, e)

C = lf.detect_circles(image_path, L_black, s_circle)

L_red = lf.select_lines(image, L_red, l, n, distance_between_points_red, n_min, distance_between_summits_red)
L_black = lf.select_lines(image, L_black, l, n, distance_between_points_black, n_min, distance_between_summits_black)

# Rassembler les segments noirs et rouges en conservant l'information de couleur
L_combined = []
for segment in L_black:
    L_combined.append((segment, 'black'))
for segment in L_red:
    L_combined.append((segment, 'red'))

L = lf.merge_points_mixed(L_combined, distance_between_summits_black)

# on re sépare les rouge et noir
L_black = []
L_red = []
for segment in L:
    if segment[1] == 'black':
        L_black.append(segment[0])
    else:
        L_red.append(segment[0])


############## Normalisation des points ################
L_black_norm, L_red_norm, C_norm = lf.normalize_points(image, L_black, L_red, C)

########### Ajout des objets ################
obj_norm = lf.add_objects(image, L_black_norm, L_red_norm, C_norm)

############## Reconstruction du labyrinthe ################
lf.reconstruire_labyrinthe_from_normalize(image, L_black, L_red, C, obj_norm)

############## Exportation des segments en json ################
lf.export_data_to_json(image, L_black_norm, L_red_norm, C_norm, obj_norm, 'labyrinthe.json')

