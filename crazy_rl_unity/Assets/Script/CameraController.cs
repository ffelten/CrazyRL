using UnityEngine;

/// <summary>
/// Class Camera Controller : Control camera position
/// </summary>

public class CameraController : MonoBehaviour
{
    /// <value>
    /// float size: Size of the area sides, -1 -> data not recovered
    /// float margin: extra space for better vision
    /// Game Object obj: Game Object Empty, is the parent of drones (in the hierarchy of unity)
    /// Vector3 posA: coordinates of position A
    /// Vector3 posB: coordinates of position B
    /// Vector3 posC: coordinates of position C
    /// Vector3 posD: coordinates of position D
    /// Vector3 poFace: Coordinated to face the drones
    /// Vector3 posBack: Coordinated to be behind drones
    /// Vector3 posLeft: Coordinate to the left of the drones
    /// Vector3 posRight: Coordinate to the right of the drones
    /// GameObject dropDownView: Dropdown to select the view before starting the simulation ->  retrieves the position requested by the user
    /// float zoomSpeed: Camera zoom speed, zoom triggered by mouse wheel
    /// </value>
    public float size = -1;
    public float margin = 8;
    public GameObject obj;
    public Vector3 posA;
    public Vector3 posB;
    public Vector3 posC;
    public Vector3 posD;
    public Vector3 posFace;
    public Vector3 posBack;
    public Vector3 posLeft;
    public Vector3 posRight;
    public GameObject dropDownView;
    public float zoomSpeed = 100;

    public GameObject test;

    /// <summary>
    /// calculates the possible positions for the camera, taking into account size and margin.
    /// position (0,0,0) is at the center of the cube with dimensions size+margin.
    /// </summary>
    public void CalculPos()
    {
        //float pos = size / 2 + margin / 2;
        float pos = size + margin;
        posA = new Vector3(pos, pos, -pos);
        posB = new Vector3(pos, pos, pos);
        posC = new Vector3(-pos, pos, pos);
        posD = new Vector3(-pos, pos, -pos);
        posFace = new Vector3(0, 0, -pos);
        posBack = new Vector3(0, 0, pos);
        posLeft = new Vector3(-pos, 0, 0);
        posRight = new Vector3(pos, 0, 0);

        ChangePos(dropDownView.GetComponent<Menu>().posInit);
    }

    /// <summary>
    /// sets the camera to the position, requested in the parameters
    /// options: "A" -> positions the camera at cube vertex A         C_________________B
    ///        : "B" -> positions the camera at cube vertex B         /                /|
    ///        : "C" -> positions the camera at cube vertex C        /                / |
    ///        : "D" -> positions the camera at cube vertex D      D/________________/ A|
    ///                                                            |                 |  |
    ///                                                            |                 |  |
    ///                                                            |                 |  /
    ///                                                            |                 | /
    ///                                                            |________________ |/
    ///        : "F" -> positions the camera on the front of the cube
    ///        : "Back" -> positions the camera on the back of the cube
    ///        : "R" -> positions the camera on the right-hand side of the cube
    ///        : "L" -> positions the camera on the left-hand side of the cube
    /// </summary>
    /// <param name="Angle"> string : requested position</param>
    public void ChangePos(string Angle)
    {
        switch (Angle)
        {
            case "A":
                transform.position = posA;
                break;
            case "B":
                transform.position = posB;
                break;
            case "C":
                transform.position = posC;
                break;
            case "D":
                transform.position = posD;
                break;
            case "F":
                transform.position = posFace;
                break;
            case "Back":
                transform.position = posBack;
                break;
            case "L":
                transform.position = posLeft;
                break;
            case "R":
                transform.position = posRight;
                break;
        }
        Vector3 pos = new Vector3(0, 0, 0);
        //points the camera at the drones
        if (obj.transform.childCount != 0)
        {
            for (int i = 0; i < obj.transform.childCount; i++)
            {
                pos.x += obj.transform.GetChild(i).transform.position.x;
                pos.y += obj.transform.GetChild(i).transform.position.y;
                pos.z += obj.transform.GetChild(i).transform.position.z;
            }
            pos.x /= obj.transform.childCount;
            pos.y /= obj.transform.childCount;
            pos.z /= obj.transform.childCount;
            test.transform.position = pos;
            transform.LookAt(pos);
        }
    }

    /// <summary>
    /// manages camera zoom
    /// </summary>
    private void Update()
    {
        if (obj.transform.childCount != 0)
        {
            if (Input.mouseScrollDelta.y > 0)
                GetComponent<Camera>().fieldOfView -= zoomSpeed * Time.deltaTime;
            else if (Input.mouseScrollDelta.y < 0)
                GetComponent<Camera>().fieldOfView += zoomSpeed * Time.deltaTime;
        }
    }
}
