using UnityEngine;
using System.Collections;
using System.Collections.Generic;

public class WallGenerator : MonoBehaviour
{
    // Prefab du mur à instancier
    public GameObject wallPrefab;
    public GameObject parentPlane;
    public GameObject roof;
    public GameObject ball;
    public GameObject hole;
    public GameObject particlePrefab; 
    public GameObject cylinder;
    private float duration = 3f; 
    private List<GameObject> walls = new List<GameObject>();

    void Start()
    {
        // On rend le mur de base (prefab) invisible
        Renderer prefabRenderer = wallPrefab.GetComponent<Renderer>();
        prefabRenderer.enabled = false;

        // On rend le cylindre de base (prefab) invisible
        Renderer cylinderRenderer = cylinder.GetComponent<Renderer>();
        cylinderRenderer.enabled = false;

        Renderer dustRenderer = particlePrefab.GetComponent<Renderer>();
        dustRenderer.enabled = false; 
    }

    public void GenerateWallsFromCoordinates(Vector3[] coordinates, string[] colors)
    {
        for (int i = 0; i < coordinates.Length; i += 2)
        {
            GenerateWall(coordinates[i], coordinates[i + 1], colors[i/2]);
        }
    }


    // Fonction pour générer un mur entre deux points
    public void GenerateWall(Vector3 start, Vector3 end, string color)
    {
        // On calcule la position du mur qui va être instancié (selon x et z)
        Vector3 position = (start + end) / 2;
        
        // On s'assure que le mur est bien à la hauteur du prefab (selon y)
        position.y = wallPrefab.transform.position.y;

        // On calcule la direction du mur qui va être instancié
        Vector3 direction = end - start;

        // On calcule la longueur du mur qui va être instancié
        float length = direction.magnitude;

        // On calcule la rotation du mur qui va être instancié
        Quaternion rotation = Quaternion.LookRotation(direction);

        // Instanciation du mur
        GameObject wall = Instantiate(wallPrefab);

        // On instancie aussi deux cylindres à mettre à chaque extrémité du mur
        GameObject cylinder1 = Instantiate(cylinder);
        GameObject cylinder2 = Instantiate(cylinder);

        // On place les cylindres à chaque extrémité du mur
        cylinder1.transform.position = new Vector3(start.x, start.y+cylinder1.transform.localPosition.y, start.z);
        cylinder2.transform.position = new Vector3(end.x, end.y+cylinder1.transform.localPosition.y, end.z);

        cylinder1.transform.parent = parentPlane.transform;
        cylinder2.transform.parent = parentPlane.transform;

        walls.Add(wall);

        // Positionner le mur à la position calculée
        wall.transform.position = position;
        // Aligner le mur avec la direction donnée
        wall.transform.rotation = rotation;

        // Ajuster l'échelle pour correspondre à la distance entre les points
        Vector3 originalScale = wallPrefab.transform.localScale; // Échelle initiale du prefab

        // On ajuste la longueur du mur
        wall.transform.localScale = new Vector3(originalScale.x, originalScale.y, length);

        // Définir le mur instancié comme enfant du plan
        wall.transform.parent = parentPlane.transform;

        // Fonction pour animer les murs
        StartCoroutine(AnimateWall(wall, cylinder1, cylinder2, wall.transform.position, cylinder1.transform.position, cylinder2.transform.position, duration));
        // Fonction pour générer les particules
        GenerateParticles(start, end, length, rotation);

        // Si la couleur est rouge on fait monter et descendre le mur en boucle
        if (color == "red")
        {
            StartCoroutine(RiseAndFall(cylinder1, 1f, duration));
            StartCoroutine(RiseAndFall(cylinder2, 1f, duration));
            StartCoroutine(RiseAndFall(wall, 1f, duration));
        }
    }

    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////// ANIMATIONS MURS & PLAFOND ///////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    private IEnumerator AnimateWall(GameObject wall, GameObject cylinder1, GameObject cylinder2, Vector3 targetPosition, Vector3 targetPositionCylinder1, Vector3 targetPositionCylinder2, float duration)
    {
        // Convertir les positions en espace local
        Vector3 localTargetPosition = parentPlane.transform.InverseTransformPoint(targetPosition);
        Vector3 localStartPosition = localTargetPosition + new Vector3(0, -7f, 0);

        // Coordonnées en espace local des cylindres
        Vector3 localTargetPositionCylinder1 = parentPlane.transform.InverseTransformPoint(targetPositionCylinder1);
        Vector3 localStartPositionCylinder1 = localTargetPositionCylinder1 + new Vector3(0, -7f, 0);

        Vector3 localTargetPositionCylinder2 = parentPlane.transform.InverseTransformPoint(targetPositionCylinder2);
        Vector3 localStartPositionCylinder2 = localTargetPositionCylinder2 + new Vector3(0, -7f, 0);

        float elapsedTime = 0;
        float shakeMagnitude = 0.006f; // Magnitude du tremblement

        while (elapsedTime < duration)
        {
            Vector3 randomShake = new Vector3(
                Random.Range(-shakeMagnitude, shakeMagnitude),
                Random.Range(-shakeMagnitude, shakeMagnitude),
                Random.Range(-shakeMagnitude, shakeMagnitude)
            ); // Calcul d'un tremblement aléatoire

            // Interpolation linéaire entre la position de départ et la position cible
            wall.transform.localPosition = Vector3.Lerp(localStartPosition, localTargetPosition, elapsedTime / duration) + randomShake;
            cylinder1.transform.localPosition = Vector3.Lerp(localStartPositionCylinder1, localTargetPositionCylinder1, elapsedTime / duration) + randomShake;
            cylinder2.transform.localPosition = Vector3.Lerp(localStartPositionCylinder2, localTargetPositionCylinder2, elapsedTime / duration) + randomShake;
            elapsedTime += Time.deltaTime;
            yield return null; // Attendre la prochaine frame
        }

        wall.transform.localPosition = localTargetPosition;
        cylinder1.transform.localPosition = localTargetPositionCylinder1;
        cylinder2.transform.localPosition = localTargetPositionCylinder2;
    }

    private IEnumerator RiseAndFall(GameObject wall, float speed, float waitTime)
    {
        yield return new WaitForSeconds(waitTime);
        Vector3 originalLocalPosition = wall.transform.localPosition;
        Vector3 targetLocalPosition = new Vector3(originalLocalPosition.x, originalLocalPosition.y - 10f, originalLocalPosition.z);
        float elapsedTime = 0;

        while (true)
        {
            // Interpolation linéaire pour descendre
            wall.transform.localPosition = Vector3.Lerp(originalLocalPosition, targetLocalPosition, Mathf.PingPong(elapsedTime * speed, 1));
            elapsedTime += Time.deltaTime;
            yield return null; // Attendre la prochaine frame
        }
    }
    
    //////////////////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////// PARTICULES ///////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////
    private void GenerateParticles(Vector3 start, Vector3 end, float length, Quaternion direction)
    {
        Quaternion particleRotation = direction * Quaternion.Euler(0, 90, 0);

        // Calcul du nombre de particules à instancier
        int numberOfParticles = Mathf.FloorToInt(length);

        if (numberOfParticles == 0)
        {
            // Calcul de la position centrale
            Vector3 centerPosition = Vector3.Lerp(start, end, 0.5f);
            centerPosition.y = parentPlane.transform.position.y + 0.1f; // Légèrement au-dessus du plan

            // Lancer la coroutine pour instancier les particules avec un délai
            StartCoroutine(InstantiateParticlesWithDelay(centerPosition, particleRotation, parentPlane.transform));
        }
        else
        {
            for (int i = 0; i <= numberOfParticles; i++)
            {
                // Calcul de la position de chaque particule
                Vector3 position = Vector3.Lerp(start, end, i / (float)numberOfParticles);
                position.y = parentPlane.transform.position.y + 0.1f; // Légèrement au-dessus du plan

                // Lancer la coroutine pour instancier les particules avec un délai
                StartCoroutine(InstantiateParticlesWithDelay(position, particleRotation, parentPlane.transform));
            }
        }
    }

    private IEnumerator InstantiateParticlesWithDelay(Vector3 position, Quaternion rotation, Transform parent)
    {
        yield return new WaitForSeconds(0.4f);

        Renderer dustRenderer = particlePrefab.GetComponent<Renderer>();
        dustRenderer.enabled = true;

        GameObject particles = Instantiate(particlePrefab, position, rotation);

        particles.transform.localScale = new Vector3(0.7f, 0.7f, 0.7f);

        // Rotation de 90° autour de l'axe x
        particles.transform.Rotate(-90, 0, 0);

        // Fixer les particules comme enfant du plan pour qu'elles ne suivent pas le mur
        particles.transform.parent = parent;

        // Ajuster la rotation et l'échelle des particules
        particles.transform.localScale = new Vector3(1f, 1f, 1f);
    }

}