using UnityEngine;
using System.Collections;

// Classe pour animer le drapeau 
public class Cylinder : MonoBehaviour
{
    public GameObject flag;
    void Start()
    {
        Renderer cylinderRenderer = gameObject.GetComponent<Renderer>();
        // On récupère le renderer du drapeau
        Renderer cylinderChildRenderer = flag.GetComponent<Renderer>();

        // On n'affiche pas le cylindre et le drapeau tant que l'animation des murs n'est pas terminée
        cylinderRenderer.enabled = false;
        cylinderChildRenderer.enabled = false;
        
        Vector3 localtargetPosition = gameObject.transform.localPosition;
        StartCoroutine(AnimateFlag(localtargetPosition, 1f, cylinderRenderer, cylinderChildRenderer));
    }

    private IEnumerator AnimateFlag(Vector3 targetPosition, float duration, Renderer cylinderRenderer, Renderer cylinderChildRenderer)
    {
        yield return new WaitForSeconds(3.5f);
        cylinderRenderer.enabled = true;
        cylinderChildRenderer.enabled = true;
        Vector3 startPosition = targetPosition + new Vector3(0, -30f, 0);
        float elapsedTime = 0;

        // On fait sortir le drapeau du plan par en dessous
        while (elapsedTime < duration)
        {
            // Interpolation linéaire entre la position de départ et la position cible
            gameObject.transform.localPosition = Vector3.Lerp(startPosition, targetPosition, elapsedTime / duration);
            elapsedTime += Time.deltaTime;
            yield return null; // Attendre la prochaine frame
        }
    }
}