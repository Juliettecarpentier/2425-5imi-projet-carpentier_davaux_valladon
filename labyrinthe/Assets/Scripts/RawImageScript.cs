using UnityEngine;
using UnityEngine.UI;
public class RawImageScript : MonoBehaviour
{
    void Start()
    {
        WebCamDevice[] devices = WebCamTexture.devices;
        WebCamTexture webcamtexture = new WebCamTexture(devices[2].name);
        GetComponent<RawImage>().material.mainTexture = webcamtexture;
        webcamtexture.Play();
    }
}
