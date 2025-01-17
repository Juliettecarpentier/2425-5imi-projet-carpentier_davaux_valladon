import cv2 as cv
from cv2 import aruco
import numpy as np

def create_aruco_board(num_tags_x, num_tags_y, dictionary, margin_cm=1):
    """
    Crée une image d'une feuille A4 avec une grille ArUco centrée
    et des marges définies en centimètres.
    
    Parameters:
        - num_tags_x: Nombre de tags en largeur.
        - num_tags_y: Nombre de tags en hauteur.
        - dictionary: Dictionnaire ArUco à utiliser.
        - margin_cm: Taille des marges autour de la grille (en cm).
    """
    # A4 sheet dimensions in meters
    a4_width_m = 0.210
    a4_height_m = 0.297

    # Conversion des dimensions en pouces
    dpi = 300  # Dots per inch (résolution)
    margin_inch = margin_cm / 2.54  # Convertir la marge de cm en pouces

    # Taille de l'image en pixels
    img_width = int(a4_width_m * 39.3701 * dpi)
    img_height = int(a4_height_m * 39.3701 * dpi)

    # Taille des marges en pixels
    margin_px = int(margin_inch * dpi)

    # Taille des tags et de leur espacement en mètres
    tag_size = 0.03  # 3 cm
    tag_spacing = 0.01  # 1 cm

    # Taille totale de la grille (en pixels)
    tag_size_px = int(tag_size * 39.3701 * dpi)
    tag_spacing_px = int(tag_spacing * 39.3701 * dpi)

    # Créer la planche ArUco
    board = aruco.GridBoard(
        (num_tags_x,
        num_tags_y),
        tag_size,
        tag_spacing,
        aruco.getPredefinedDictionary(dictionary)
    )

    # Générer une image pour la grille uniquement
    grid_img_size = (
        num_tags_x * tag_size_px + (num_tags_x - 1) * tag_spacing_px,
        num_tags_y * tag_size_px + (num_tags_y - 1) * tag_spacing_px,
    )
    grid_img = board.generateImage(grid_img_size)

    # Créer une image blanche de la taille A4
    img = 255 * np.ones((img_height, img_width), dtype=np.uint8)

    # Placer la grille centrée avec des marges
    start_x = margin_px
    start_y = margin_px

    # Calculer la fin de la zone de la grille
    end_x = start_x + grid_img.shape[1]
    end_y = start_y + grid_img.shape[0]

    # Copier la grille dans l'image blanche
    img[start_y:end_y, start_x:end_x] = grid_img

    print(f"Generated image with margins and size: {img.shape}")

    # Afficher l'image
    cv.imshow('Aruco Board', img)
    cv.waitKey(0)  # Attendre une touche pour fermer
    cv.destroyAllWindows()

    # Sauvegarder l'image
    cv.imwrite('aruco_board.png', img)

if __name__ == "__main__":
    try:
        create_aruco_board(5, 7, aruco.DICT_6X6_50, margin_cm=1)
    finally:
        cv.destroyAllWindows()
