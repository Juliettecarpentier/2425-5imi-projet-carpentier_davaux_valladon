using UnityEngine;
using System.IO;
using System.Collections.Generic;

public class Orientation : MonoBehaviour
{
    public float rotationSpeed = 100f;
    
    // Update est appelée une fois par frame
    void Update()
    {
        Vector3 rotationPoint = transform.position;

        if (Input.GetKey(KeyCode.A)) // Q pour Q sur AZERTY
        {
            transform.RotateAround(rotationPoint, Vector3.forward, rotationSpeed * Time.deltaTime);
        }
        else if (Input.GetKey(KeyCode.D)) // D pour D sur AZERTY
        {
            transform.RotateAround(rotationPoint, Vector3.forward, -rotationSpeed * Time.deltaTime);
        }
        else if (Input.GetKey(KeyCode.W)) // W pour Z sur AZERTY
        {
            transform.RotateAround(rotationPoint, Vector3.right, rotationSpeed * Time.deltaTime);
        }
        else if (Input.GetKey(KeyCode.S))
        {
            transform.RotateAround(rotationPoint, Vector3.right, -rotationSpeed * Time.deltaTime);
        }
    }
}
    