using System;

/// <summary>
/// Struct Data: structure of data received
/// </summary>
[Serializable]
public struct Data
{
    /// <value>
    /// bool isInstantiate : true: instantiated drone, false: non-instantiated drone
    /// int nbDrone : number of drones present in the scene (target not taken into account)
    /// int size : size of the simulation zone
    /// int id : drone identifier
    /// float posX : drone's X coordinate
    /// float posY : drone's Y coordinate
    /// float posZ : drone's Z coordinate
    /// string str : message to correctly process the data sent
    /// </value>
    public bool isInstantiate;
    public int ndDrone;
    public int size;
    public int id;
    public float posX;
    public float posY;
    public float posZ;
    public string str;
}
