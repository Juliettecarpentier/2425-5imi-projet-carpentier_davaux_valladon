using UnityEngine;
using System.IO;
using System.Collections;
using System.Collections.Generic;

public class WallInstance : MonoBehaviour
{
    public GameObject plane; 
    public GameObject cylindre;

    void Start()
    {
        // Recupérer le scale du plan
        float scale = plane.transform.localScale.x/2;

        // Récupération du composant WallGenerator
        WallGenerator generator = GetComponent<WallGenerator>();

        // Chemin du fichier JSON
        string filePath = Path.Combine(Application.dataPath, "Scripts/labyrinthe.json");

        // Lecture du fichier JSON
        string jsonContent = File.ReadAllText(filePath);
        WallData wallData = JsonUtility.FromJson<WallData>(jsonContent);

        // Murs exterieurs lis les sides du fichier JSON
        List<Vector3> coords = new List<Vector3>();
        List<Vector3> corners = new List<Vector3>();
        List<string> colors = new List<string>();

        Vector3 planePosition = plane.transform.position;

        // On parcourt les murs exterieurs 
        foreach (var wall in wallData.sides)
        {
            colors.Add(wall.color);
            // Add the first point of the wall to the corners list
            var firstPoint = wall.points[0];
            Vector3 transformedCorner = new Vector3(firstPoint.x * scale, firstPoint.y * scale, firstPoint.z * scale) + planePosition;
            corners.Add(transformedCorner);

            // Add all points to coords
            foreach (var point in wall.points)
            {
                Vector3 transformedPoint = new Vector3(point.x * scale, point.y * scale, point.z * scale) + planePosition;
                coords.Add(transformedPoint);
            }
        }

        // On resize le plan pour qu'il englobe tous les murs
        // Calcul des dimensions et du centre
        float minX = Mathf.Min(corners[0].x, corners[1].x, corners[2].x, corners[3].x);
        float maxX = Mathf.Max(corners[0].x, corners[1].x, corners[2].x, corners[3].x);
        float minZ = Mathf.Min(corners[0].z, corners[1].z, corners[2].z, corners[3].z);
        float maxZ = Mathf.Max(corners[0].z, corners[1].z, corners[2].z, corners[3].z);

        float width = maxX - minX; // Largeur sur X
        float height = maxZ - minZ; // Hauteur sur Z

        Vector3 center = new Vector3((minX + maxX) / 2, planePosition.y, (minZ + maxZ) / 2);

        // Ajuster la position et la taille du plan
        plane.transform.position = center;
        plane.transform.localScale = new Vector3(width, 0.12f, height); // La scale Y reste inchangée

        // On parcourt l'interieur du labyrinthe
        foreach (var wall in wallData.walls)
        {
            colors.Add(wall.color);
            foreach (var point in wall.points)
            {
                coords.Add(new Vector3(point.x * scale, point.y * scale, point.z * scale) + planePosition);
            }
        }

        // On appelle la fonction GenerateWallsFromCoordinates du WallGenerator pour générer les murs
        generator.GenerateWallsFromCoordinates(coords.ToArray(), colors.ToArray());
    }
}

[System.Serializable]
public class Point
{
    public float x;
    public float y;
    public float z;
}

[System.Serializable]
public class InputData
{
    public List<Point> points;
}

[System.Serializable]
public class WallData
{
    public List<InputData> input;
    public List<Sides> sides;
    public List<Wall> walls;
}
[System.Serializable]
public class Sides
{
    public List<Point> points;
    public string color;
}

[System.Serializable]
public class Wall
{
    public List<Point> points;
    public string color;
}