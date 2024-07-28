using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Threading;
using AsyncIO;
using NetMQ;
using NetMQ.Sockets;
using TMPro;
using UnityEngine;
using UnityEngine.UI;

/// <summary>
/// class Receiver : receives data
/// </summary>
public class Receiver
{
    /// <summary>
    /// readonly Thread receiveThread : thread containing all the data received
    /// bool running : true: client connect to server, false: client disconnected from server
    /// bool messageReceived : true: message received, false: no message received
    /// TimeSpan timeout : waiting time for the reception of a message, if the time is exceeded the client disconnects from the server
    /// bool isButtonReset :indicates whether button reset can be activated or not
    /// </summary>
    private readonly Thread receiveThread;
    public bool running;
    public bool messageReceived = true;
    TimeSpan timeout = new TimeSpan(0, 0, 5);
    public bool isButtonReset = false;

    /// <summary>
    /// constructor
    /// makes the connection with the server, sends and receives messages which it saves in a thread
    /// </summary>
    /// <param name="ip"> string : ip to connect to the server</param>
    /// <param name="port"> string : port to connect to the server</param>
    public Receiver(string ip, string port)
    {
        receiveThread = new Thread(
            (object callback) =>
            {
                using (var socket = new RequestSocket())
                {
                    socket.Connect("tcp://" + ip + ":" + port);
                    Debug.Log("connecting to the server");

                    while (running)
                    {
                        string message = "";

                        //send then receive a message
                        if (socket.TrySendFrameEmpty()) //succeeds in sending the message
                        {
                            //receives a message
                            messageReceived = socket.TryReceiveFrameString(timeout, out message);
                        }

                        //Checks if it has received a message within 5 seconds
                        if (messageReceived) //message received
                        {
                            //converts the message into data
                            Data data = JsonUtility.FromJson<Data>(message);
                            ((Action<Data>)callback)(data);
                            isButtonReset = false;
                        }
                        else //message not received
                        {
                            //stop the connection
                            isButtonReset = true;
                            Debug.Log("Stop client");
                            Stop();
                            NetMQConfig.Cleanup();
                        }
                    }
                }
            }
        );
    }

    /// <summary>
    /// starts the thread
    /// </summary>
    /// <param name="callback"> Action<Data> : action on data</param>
    public void Start(Action<Data> callback)
    {
        running = true;
        receiveThread.Start(callback);
    }

    /// <summary>
    /// stop the thread
    /// </summary>
    public void Stop()
    {
        running = false;
        receiveThread.Join();
    }
}

/// <summary>
/// class client: manages the entire simulation, interprets messages, gives instructions to the drones and activates the reset and restart buttons
/// </summary>
public class Client : MonoBehaviour
{
    /// <summary>
    /// readonly ConcurrentQueue<Action> runOnMainThread : Thread allowing to retrieve the following data
    /// Receiver receiver : Receiver to connect and communicate with the server
    /// Action action : Action to transfer the data into the variable 'data'
    /// bool isConnect : true: Client connected to the server, false: Client disconnected from the server
    /// </summary>
    private readonly ConcurrentQueue<Action> runOnMainThread = new ConcurrentQueue<Action>();
    private Receiver receiver;
    private Action action;
    private bool isConnect = false;

    /// <summary>
    /// string ip : IP for the server connection
    /// string port : port for the server connection
    /// TMP_InputField inputIp : InputField to modify IP
    /// TMP_InputField inputPort : InputField to modify port
    /// </summary>
    public string ip = "127.0.0.1";
    public string port = "5555";
    public TMP_InputField inputIp;
    public TMP_InputField inputPort;

    /// <summary>
    /// GameObject parent : GameObject empty will contain all the created GameObjects
    /// GameObject[] drones : list of drones
    /// GameObject[] targets : list of Target
    /// GameObject prefab_drone : Prefab copied to create a drone
    /// GameObject prefab_target : Prefab copied to create a target
    /// int nbDrones : Number of drones without the target
    /// int nbTargets : number of targets
    /// bool isInvoke : indicates whether to continue retrieving data. true: data table is up to date , false : data table is not up to date
    /// float speed : speed of drone
    /// TMP_InputField inputSpeed : InputField to modify speed
    /// int nbDrobesScene : number of drone in scene
    /// </summary>
    public GameObject parent;
    public GameObject[] drones;
    public GameObject[] targets;
    public GameObject prefab_drone;
    public GameObject prefab_target;
    private int nbDrones;
    private int nbTargets;
    private bool isInvoke = false;
    public TMP_InputField inputSpeed;
    public int speed = 10;
    int nbDronesScene = 0;

    /// <summary>
    /// Data data : Data being processed
    /// Data[] tabData : Data table containing the next position of each drone
    /// int idTabData : Index to navigate through the data table
    /// </summary>
    private Data data;
    public Data[] tabData;
    private int idTabData = -1;

    /// <summary>
    /// GameObject mainCamera : Camera of the scene
    /// GameObject connectionPanel : Connection Panel
    /// Button buttonRestart : Restart button
    /// Button buttonReset : Reset Button
    /// </summary>
    public GameObject mainCamera;
    public GameObject connectionPanel;
    public Button buttonRestart;
    public Button buttonReset;

    /// <summary>
    /// connects the client to the server using the connection button
    /// </summary>
    public void Connection()
    {
        ForceDotNet.Force(); // If you have multiple sockets in the following threads

        //modifies the ip address and/or port
        if (inputIp.text != "")
            ip = inputIp.text;
        if (inputPort.text != "")
            port = inputPort.text;

        //create the receiver and launches the message collector
        receiver = new Receiver(ip, port);
        receiver.Start(
            (Data d) =>
                runOnMainThread.Enqueue(() =>
                {
                    data = d;
                })
        );
        isConnect = true;

        //disables the connection panel
        connectionPanel.SetActive(false);
    }

    public void Update()
    {
        if (isConnect) // connected to the server
        {
            if (data.type == "init")
            {
                nbDrones = data.nbDrones;
                nbTargets = data.nbTargets;
                mainCamera.GetComponent<CameraController>().size = data.size;
            }
            if (nbDronesScene <= nbDrones + nbTargets) // Drones aren't instantiated
            {
                if (!runOnMainThread.IsEmpty) //data still to be processed
                {
                    if (runOnMainThread.TryDequeue(out action))
                    {
                        action.Invoke();
                        Initiate(data);
                        nbDronesScene += 1;
                    }
                }
            }
            else //Drones aren't instantiated
            {
                if (tabData.Length != drones.Length + targets.Length)
                {
                    tabData = new Data[drones.Length + targets.Length];
                    //calculate and positions the camera in the right place
                    mainCamera.GetComponent<CameraController>().CalculPos();
                }

                if (idTabData < tabData.Length - 1)
                {
                    if (!isInvoke)
                    {
                        idTabData++;
                        tabData[idTabData] = data;

                        if (data.type == "Drone")
                            SaveData(data, ref drones[data.id].GetComponent<Drone>().lisPos);
                        else if (data.type == "Target")
                            SaveData(data, ref targets[data.id].GetComponent<Drone>().lisPos);

                        isInvoke = true;
                    }

                    if (!runOnMainThread.IsEmpty) //data still to be processed
                    {
                        if (runOnMainThread.TryDequeue(out action))
                        {
                            action.Invoke();
                            isInvoke = false;
                        }
                    }
                }
                else
                {
                    for (int j = 0; j < drones.Length + targets.Length; j++)
                    {
                        if (tabData[j].type == "Drone")
                        {
                            Movement(tabData[j], drones[tabData[j].id].transform);
                            VerificationPos(tabData[j], drones[tabData[j].id]);
                        }
                        else if (tabData[j].type == "Target")
                        {
                            Movement(tabData[j], targets[tabData[j].id].transform);
                            VerificationPos(tabData[j], targets[tabData[j].id]);
                        }
                    }
                    if (IsChangeTabData())
                    {
                        idTabData = -1;
                    }
                }
            }

            if (receiver.isButtonReset)
            {
                foreach (GameObject dr in drones)
                {
                    dr.GetComponent<Drone>().isFinish = true;
                }
                foreach (GameObject tar in targets)
                    tar.GetComponent<Drone>().isFinish = true;

                buttonReset.interactable = true;
                receiver.isButtonReset = false;
                isConnect = false;
            }
            else
            {
                buttonReset.interactable = false;
            }
        }
        else // not connected to the server
        {
            if (IsRestart()) //can restart the simulation
            {
                buttonReset.interactable = true;
            }
            else //can't restart the simulation
                buttonReset.interactable = false;

            foreach (GameObject dr in drones)
            {
                speed = int.Parse(inputSpeed.text);
                dr.GetComponent<Drone>().speed = speed;
            }

            foreach (GameObject tar in targets)
            {
                speed = int.Parse(inputSpeed.text);
                tar.GetComponent<Drone>().speed = speed;
            }
        }
    }

    /// <summary>
    /// create and position the drone in the right place
    /// </summary>
    /// <param name="d">Data: data being processed </param>
    void Initiate(Data d)
    {
        if (d.type == "Drone") //treatment for a drone
        {
            if (drones.Length != nbDrones) //uninitialised table
            {
                //initialisation of the drone array and recovery of the zone size
                drones = new GameObject[nbDrones];
            }
            //positions the drone in the right place and makes it a child of the parent GameObject in the unity hierarchy
            Vector3 posInit = new Vector3(d.posX, d.posY, d.posZ);
            Quaternion qua = new Quaternion(0, 0, 0, 0);
            drones[d.id] = Instantiate(prefab_drone, posInit, qua, parent.transform);

            //save the details for the next simulation if necessary
            SaveData(d, ref drones[d.id].GetComponent<Drone>().lisPos);
        }
        else if (d.type == "Target") //target treatment
        {
            if (targets.Length != nbTargets) //uninitialised table
            {
                //initialisation of the target array and recovery of the zone size
                targets = new GameObject[nbTargets];
            }
            //positions the target in the right place and makes it a child of the parent GameObject in the unity hierarchy
            Vector3 posInit = new Vector3(d.posX, d.posY, d.posZ);
            Quaternion qua = new Quaternion(0, 0, 0, 0);
            Debug.Log(d.id);
            targets[d.id] = Instantiate(prefab_target, posInit, qua, parent.transform);

            //save the details for the next simulation if necessary
            SaveData(d, ref targets[d.id].GetComponent<Drone>().lisPos);
        }
    }

    /// <summary>
    /// moves the drone to the desired position (one movement per frame)
    /// </summary>
    /// <param name="d">Data : data containing coordinates</param>
    /// <param name="transform"> Tansform : transform of the object to be moved</param>
    void Movement(Data d, Transform transform)
    {
        speed = int.Parse(inputSpeed.text);
        transform.position = Vector3.Lerp(
            transform.position,
            new Vector3(d.posX, d.posY, d.posZ),
            Time.deltaTime * speed
        );
    }

    /// <summary>
    /// checks whether the drone is more or less where it needs to be. Sets the drone's isPos to true if this is the case, otherwise it remains at false.
    /// </summary>
    /// <param name="d"> Data: data containing coordinates </param>
    /// <param name="obj"> Game Object: drones</param>
    void VerificationPos(Data d, GameObject obj)
    {
        float precision = obj.GetComponent<Drone>().precision;
        if (
            (
                obj.transform.position.x <= d.posX + precision
                && obj.transform.position.x >= d.posX - precision
            )
            && (
                obj.transform.position.y <= d.posY + precision
                && obj.transform.position.y >= d.posY - precision
            )
            && (
                obj.transform.position.z <= d.posZ + precision
                && obj.transform.position.z >= d.posZ - precision
            )
        )
        {
            obj.GetComponent<Drone>().isPos = true;
        }
        else
        {
            obj.GetComponent<Drone>().isPos = false;
        }
    }

    /// <summary>
    /// check if all the drone have finished their move to be able to move on to the next move
    /// </summary>
    /// <returns>true: data can changed, false: data cannot changed</returns>
    bool IsChangeTabData()
    {
        for (int k = 0; k < nbDrones; k++)
        {
            if (drones[k].GetComponent<Drone>().isPos == false)
                return false;
        }
        for (int k = 0; k < nbTargets; k++)
            if (targets[k].GetComponent<Drone>().isPos == false)
                return false;

        return true;
    }

    /// <summary>
    /// saves positions (data) in the drone's position list
    /// </summary>
    /// <param name="d">Data: data containing coordinates</param>
    /// <param name="list">List<float> list of drone positions (coordinates) </param>
    public void SaveData(Data d, ref List<float> list)
    {
        list.Add(d.posX);
        list.Add(d.posY);
        list.Add(d.posZ);
    }

    /// <summary>
    /// reset the drone attributes to be able to redo the simulation
    /// disable the reset button and enable the restart button
    /// </summary>
    public void ResetSimulation()
    {
        buttonReset.interactable = false;
        foreach (GameObject dr in drones)
        {
            dr.GetComponent<Drone>().ResetDrone();
        }
        foreach (GameObject tar in targets)
            tar.GetComponent<Drone>().ResetDrone();
        buttonRestart.interactable = true;
    }

    /// <summary>
    /// restarts the simulation et and disable the restart button
    /// </summary>
    public void RestartSimulation()
    {
        buttonRestart.interactable = false;
        foreach (GameObject dr in drones)
        {
            dr.GetComponent<Drone>().RestartDrone();
        }
        foreach (GameObject tar in targets)
            tar.GetComponent<Drone>().RestartDrone();
    }

    /// <summary>
    /// Check that all the drones have completed their course to see if a restart is possible.
    /// </summary>
    /// <returns>bool: true : restart possible, false: restart impossible</returns>
    bool IsRestart()
    {
        foreach (GameObject dr in drones)
        {
            if (!dr.GetComponent<Drone>().isFinish)
                return false;
        }
        foreach (GameObject tar in targets)
            if (!tar.GetComponent<Drone>().isFinish)
                return false;
        return true;
    }

    /// <summary>
    /// stop the connection to the server if necessary, and destroy all created objects
    /// </summary>
    private void OnDestroy()
    {
        if (receiver.running)
        {
            receiver.Stop();
            NetMQConfig.Cleanup();
        }
        for (int i = 0; i < drones.Length; i++)
        {
            Destroy(drones[i]);
        }
        for (int i = 0; i < targets.Length; i++)
        {
            Destroy(targets[i]);
        }
    }

    /// <summary>
    /// quit simulator and stop connection
    /// </summary>
    public void Quit()
    {
        if (receiver.running)
        {
            receiver.Stop();
            NetMQConfig.Cleanup();
        }
        Application.Quit();
    }
}
