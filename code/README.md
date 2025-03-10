   
## Configuration Parameters

Below are the parameters that can be manually modified to configure the model. These parameters allow you to set up the anthropometric characteristics and control options for the URDF model:
<div align="center">
   
| Parameter        | Description                                                                                                                                 |  
|:----------------:|:-------------------------------------------------------------------------------------------------------------------------------------------:|
| H                | Total height [m]                                                                                             |
| m                | Total body mass [Kg]                                                                                         |
| Neck [X]         | Diameter of the neck [m]                                                                                                                        |
| UpperTrunk [X]   | Depth of the upper trunk [m]                                                                                                                  |
| LowerTrunk [X]   | Depth of the lower trunk [m]                                                                                                                    |
| Pelvis [X]       | Depth of the pelvis [m]                                                                                                                      |
| Shoulder [X, Z]  | Width of the shoulder [m]                                                                                                                       |
| UpperArm [X, Z]  | Diameter of the upper arm [m]                                                                                                                   |
| ForeArm [X, Z]   | Diameter of the fore arm [m]                                                                                                                 |
| Hand [Z]         | Height of the hand [m]                                                                                                                          |
| Hand [X]         | Width of the hand [m]                                                                                                                         |
| UpperLeg [X, Y]  | Diameter of the upper leg [m]                                                                                                                 |
| LowerLeg  [X, Y] | Diameter of the lower leg [m]                                                                                                                 |

</div>

## Additional Options

These options provide further customization for the model's consistency check, movement type, and visualization settings:
<div align="center">
   
| Option                              | Value               | Description                                                  |
|:-----------------------------------:|:-------------------:|:-------------------------------------------------------------|
| OPT_CHECK_CONSISTENCY_MODEL         | `True` or `False`   | Check the consistency of the model                         |
| OPT_VISUALIZZATION_MODEL            | `True` or `False`   | Visualize the movement                                      |
| OPT_VISUALIZZATION_MEASUREOFCONTROL | `True` or `False`   | Visualize the measure of control                           |
| OPT_COLOR_LINK_MESH                 | `[R, G, B, alpha]`  | Defines the RGBA color of the link mesh for visualization, with alpha transparency factor   |
| OPT_COLOR_MUSCLE_MESH               | `[R, G, B, alpha]`  | Defines the RGBA color of the muscle mesh for visualization, with alpha transparency factor  |

</div>


