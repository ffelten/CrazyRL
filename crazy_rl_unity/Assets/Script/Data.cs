using System;

/// <summary>
/// Struct Data: structure of data received
/// </summary>
[Serializable]
public struct Data
{
    /// <value>
    /// int nbDrones : number of drones present in the scene (target not taken into account)
    /// int size : size of the simulation zone
    /// int id : drone identifier
    /// float posX : drone's X coordinate
    /// float posY : drone's Y coordinate
    /// float posZ : drone's Z coordinate
    /// string type : message to correctly process the data sent (init -> set up env, Drone -> update drone, Target -> update target)
    /// </value>
    public int nbDrones;
    public int size;
    public int id;
    public float posX;
    public float posY;
    public float posZ;
    public string type;
}
