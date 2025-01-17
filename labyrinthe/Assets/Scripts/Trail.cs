using UnityEngine;
using System.Collections;
public class Trail : MonoBehaviour
{
    public Transform ball;          
    public GameObject ballObject;   
    public ParticleSystem trailParticles; 
    private Vector3 lastPosition;

    void Start()
    {
        // Tant que l'animation des murs n'est pas terminée, on ne veut pas afficher la traînée
        Renderer trailRenderer = trailParticles.GetComponent<Renderer>();
        trailRenderer.enabled = false;
        lastPosition = ball.position;
    }

    void Update()
    {
        StartCoroutine(Wait());

        // On récupère la direction de la balle pour orienter la traînée
        Vector3 direction = (ball.position - lastPosition).normalized;
        transform.position = ball.position;

        Vector3 ballDirection = ballObject.GetComponent<Rigidbody>().linearVelocity;
        float speed = ballDirection.magnitude;

        // On oriente la traînée dans la direction de la balle 
        if (ballDirection != Vector3.zero)
        {
            transform.rotation = Quaternion.LookRotation(-ballDirection.normalized);
        }

        // On ajuste le nombre de particules de la traînée en fonction de la vitesse de la balle
        var emission = trailParticles.emission;
        emission.rateOverTime = speed * 20;

        lastPosition = ball.position;
    }

    private IEnumerator Wait()
    {
        // On attend la fin de l'animation des murs
        yield return new WaitForSeconds(3f);
        Renderer trailRenderer = trailParticles.GetComponent<Renderer>();
        trailRenderer.enabled = true;
    }
}