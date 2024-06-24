using System.Collections;
using System.Collections.Generic;
using UnityEngine;

/// <summary>
/// Class Wall: to display the wall or not
/// </summary>
public class Wall : MonoBehaviour
{
    /// <summary>
    /// get the object's MeshRenderer
    /// </summary>
    private void Start()
    {
        GetComponent<MeshRenderer>().enabled = false;
    }

    /// <summary>
    /// display the wall if drone enter in the wall collider
    /// </summary>
    /// <param name="other">Collider: collider of the object that has entered the wall collider</param>
    private void OnTriggerEnter(Collider other)
    {
        if(other.tag == "Drone")
        {
            GetComponent<MeshRenderer>().enabled = true;
        }
    }

    /// <summary>
    /// display the wall if drone exit in the wall collider
    /// </summary>
    /// <param name="other">Collider: collider of the object that has entered the wall collider</param>
    private void OnTriggerExit(Collider other)
    {
        if (other.tag == "Drone")
        {
            GetComponent<MeshRenderer>().enabled = false;
        }
    }
}
