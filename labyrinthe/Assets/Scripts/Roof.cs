using UnityEngine;
using System.IO;
using System.Collections;

public class Roof : MonoBehaviour
{
    
    void Start()
    {
        Renderer roofRenderer = GetComponent<Renderer>();
        Collider roofCollider = GetComponent<Collider>();

        // On rend le toit invisible tant que l'animation des murs n'est pas terminée
        // On rend aussi le collider inactif pour que la balle ne rebondisse pas dessus quand elle tombe au début   
        roofCollider.enabled = false;
        roofRenderer.enabled = false;
        StartCoroutine(PlaceTheRoof(roofRenderer, roofCollider));
    }

    // Fonction pour rendre le toit visible après que tous les éléments soient placés
    private IEnumerator PlaceTheRoof(Renderer roofRenderer, Collider roofCollider)
    {
        yield return new WaitForSeconds(4.5f);
        roofRenderer.enabled = true;
        roofCollider.enabled = true;
    }
}
