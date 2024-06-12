using System.Collections.Generic;
using UnityEngine;
using TMPro;
using UnityEngine.UI;

/// <summary>
/// Class Menu: Manages user interaction with the main menu + open/close the setting menu
/// </summary>

public class Menu : MonoBehaviour
{
    /// <value>
    /// Game Object optionMenu : it's panel UI object "Option menu" with all settings
    /// Game Object buttonOpenMenu : it's button UI object for open the option menu
    /// String posInit : camera position selected before starting simulation, default is “A” to position the camera at cube vertex A
    /// Game Object TerrainNature : it's the terrain object whith the nature
    /// Game Object TerrainIndustrial : it's the terrain object whith the the industrial cuty
    /// TMP_Dropdown resolutionDropdown : dropdown to select the resolution of the simulation window 
    /// List<string> resolutionDropdownOption : list of possible resolutions, resolution dropdown option 
    /// Resolution[] resolutions : resolutions table
    /// TMP_Dropdown resolutionDropdown2 : dropdown (of the option menu) to select the resolution of the simulation window
    /// </value>
    public GameObject optionMenu;
    public GameObject buttonOpenMenu;
    public string posInit = "A";
    public GameObject naturalTerrain;
    public GameObject industrialTerrain;
    public TMP_Dropdown resolutionDropdown;
    List<string> resolutionDropdownOption = new List<string> { };
    Resolution[] resolutions;
    public TMP_Dropdown resolutionDropdown2;
    public Sprite[] tabSpriteView;
    public Image imageView;


    /// <summary>
    /// retrieves all possible resolutions and adds them to resolution dropdowns
    /// </summary>
    private void Start()
    {
        resolutions = Screen.resolutions;
        resolutionDropdown.ClearOptions();
        resolutionDropdown2.ClearOptions();
        foreach (var res in resolutions)
        {
            resolutionDropdownOption.Add(res.width.ToString() + "x" + res.height.ToString());
        }
        resolutionDropdown.AddOptions(resolutionDropdownOption);
        resolutionDropdown2.AddOptions(resolutionDropdownOption);
    }

    /// <summary>
    /// change la résolution de la fenetre par rapport au choix de l'utilsateur 
    /// </summary>
    /// <param name="selecctedIndex"> int : resolution number select by user</param>
    public void ChangeResolution(int selecctedIndex)
    {
        Debug.Log(selecctedIndex);
        Screen.SetResolution(resolutions[selecctedIndex].width, resolutions[selecctedIndex].height, false);
    }

    /// <summary>
    /// displays the option menu
    /// </summary>
    public void OpenMenu()
    {
        optionMenu.SetActive(true);
        buttonOpenMenu.SetActive(false);
    }

    /// <summary>
    /// remove the option menu
    /// </summary>
    public void CloseMenu()
    {
        optionMenu.SetActive(false);
        buttonOpenMenu.SetActive(true);
    }

    /// <summary>
    /// selects the view according to the user's choices and change the sprite of icon
    /// options: 0 -> "A" -> positions the camera at cube vertex A         C_________________B
    ///        : 1 -> "B" -> positions the camera at cube vertex B         /                /| 
    ///        : 2 -> "C" -> positions the camera at cube vertex C        /                / |
    ///        : 3 -> "D" -> positions the camera at cube vertex D      D/________________/ A|
    ///                                                                 |                 |  |
    ///                                                                 |                 |  |
    ///                                                                 |                 |  /
    ///                                                                 |                 | /
    ///                                                                 |________________ |/
    ///        : 6 -> "F" -> positions the camera on the front of the cube
    ///        : 7 -> "Back" -> positions the camera on the back of the cube
    ///        : 5 -> "R" -> positions the camera on the right-hand side of the cube
    ///        : 4 -> "L" -> positions the camera on the left-hand side of the cube   
    /// </summary>
    /// <param name="index"> int : view number selected by user </param>
    public void DropdownSample(int index)
    {
        switch (index)
        {
            case 0: 
                posInit = "A";
                imageView.sprite = tabSpriteView[0];
                break;
            case 1:
                posInit = "B";
                imageView.sprite = tabSpriteView[1];
                break;
            case 2:
                posInit = "C";
                imageView.sprite = tabSpriteView[2];
                break;
            case 3:
                posInit = "D";
                imageView.sprite = tabSpriteView[3];
                break;
            case 4:
                posInit = "L";
                imageView.sprite = tabSpriteView[4];
                break;
            case 5:
                posInit = "R";
                imageView.sprite = tabSpriteView[5];
                break;
            case 6:
                posInit = "F";
                imageView.sprite = tabSpriteView[6];
                break;
            case 7:
                posInit = "Back";
                imageView.sprite = tabSpriteView[7];
                break;

        }
    }


    /// <summary>
    /// selects the terrain according to the user's choices
    /// options: 0 -> Natural Terrain
    ///        : 1 -> Industrial Terrain
    /// </summary>
    /// <param name="index"> int : terrain number selected by user </param>
    public void DropdownSampleTerrain(int index)
    {
        switch (index)
        {
            case 0:
                naturalTerrain.SetActive(true);
                industrialTerrain.SetActive(false);
                break;
            case 1:
                naturalTerrain.SetActive(false);
                industrialTerrain.SetActive(true);
                break;
            default:
                naturalTerrain.SetActive(true);
                industrialTerrain.SetActive(false);
                break;
        }
    }

    /// <summary>
    /// quit simulator
    /// </summary>
    public void Quit()
    {
        Application.Quit();
    }
}