using UnityEngine;
using System.IO;
using System.Collections;

public class Ball : MonoBehaviour
{
    public GameObject plane; 

    void Start()
    {
        // On rend la balle invisible tant que l'animation des murs n'est pas terminée
        Renderer ballRenderer = GetComponent<Renderer>();
        ballRenderer.enabled = false;

        float scale = plane.transform.localScale.x/2;

        // Chemin du fichier JSON
        string filePath = Path.Combine(Application.dataPath, "Scripts/labyrinthe.json");

        // Lecture du fichier JSON
        string jsonContent = File.ReadAllText(filePath);
        WallData wallData = JsonUtility.FromJson<WallData>(jsonContent);

        // On récupère le premier point du fichier json pour positionner la balle
        var point = wallData.input[0].points[0];
        Vector3 targetPosition = new Vector3(point.x * scale, 0.4f, point.z * scale);
        StartCoroutine(DropTheBall(targetPosition, ballRenderer));
    }

    private IEnumerator DropTheBall(Vector3 targetPosition, Renderer ballRenderer)
    {
        // On attend la fin de l'animation des murs
        yield return new WaitForSeconds(3f);
        ballRenderer.enabled = true;
        // Positionner la balle à une hauteur initiale pour qu'elle tombe
        transform.localPosition = new Vector3(targetPosition.x, 5f, targetPosition.z);
    }
}