using System.Collections.Generic;
using UnityEngine;

/// <summary>
/// class Drone : each drone belongs to this class; it contains the data (positions...) of the drone and uses them to move it around
/// </summary>
public class Drone : MonoBehaviour
{
    /// <summary>
    /// float precision : precision of movement, desired position +/- precision
    /// Vector3 wantedPosition : position where the drone should go
    /// bool isFinish : true: the drone flew to all the positions in its position list, false: the drone did not fly to all the positions in the list.
    /// bool isSaved : true: all data has been saved during the first execution, false: there is still data to be saved (value modified by the client script).
    /// int idPos : position table index, used to navigate through the table
    /// bool isPos : is used in client.cs, true: the drone is in the desired position, false: the drone is not in the desired position
    /// List<float> lisPos : list containing the coordinates of the drone (x, y, z, x1, y1, z1, .... , xn, yn, zn)
    /// float spinSpeed : Rotor speed
    /// </summary>
    public float precision = 0.5f;
    public Vector3 wantedPosition;
    public bool isFinish = false;
    public bool isSaved = false;
    public int idPos = 0;
    public bool isPos = false;
    public List<float> lisPos;
    public float spinSpeed = 2000.0f;

    /// <summary>
    /// Update is called once per frame
    /// moves and animates the drone
    /// </summary>
    void Update()
    {
        //rotors animation
        for (int i = 1; i <= 4; i++)
            SpinRotor(transform.GetChild(i));

        if (isSaved) //all the data is saved
        {
            if (!isFinish) //drone didn't do all the positions
            {
                //drone movement
                transform.position = Vector3.Lerp(
                    transform.position,
                    wantedPosition,
                    Time.deltaTime
                );

                //checks whether the drone is more or less in the desired position
                if (
                    (
                        transform.position.x <= wantedPosition.x + precision
                        && transform.position.x >= wantedPosition.x - precision
                    )
                    && (
                        transform.position.y <= wantedPosition.y + precision
                        && transform.position.y >= wantedPosition.y - precision
                    )
                    && (
                        transform.position.z <= wantedPosition.z + precision
                        && transform.position.z >= wantedPosition.z - precision
                    )
                )
                {
                    //advances in the position list
                    idPos += 3;

                    if (idPos == lisPos.Count) //all the positions have been taken
                    {
                        isFinish = true;
                    }
                    else //change requested position
                    {
                        wantedPosition = new Vector3(
                            lisPos[idPos],
                            lisPos[idPos + 1],
                            lisPos[idPos + 2]
                        );
                    }
                }
            }
        }
    }

    /// <summary>
    /// turns the rotors
    /// </summary>
    /// <param name="rotor"> transform : transform of the object to be toured </param>
    private void SpinRotor(Transform rotor)
    {
        rotor.Rotate(0, 0, spinSpeed * Time.deltaTime);
    }

    /// <summary>
    /// reset all the drone's attributes (except listPos) to be able to run the simulation again
    /// </summary>
    public void ResetDrone()
    {
        transform.position = new Vector3(lisPos[0], lisPos[1], lisPos[2]);
        wantedPosition = new Vector3(lisPos[0], lisPos[1], lisPos[2]);
        idPos = 0;
        isFinish = false;
        isSaved = false;
    }

    /// <summary>
    /// restarts the simulation
    /// </summary>
    public void RestartDrone()
    {
        isSaved = true;
    }
}
