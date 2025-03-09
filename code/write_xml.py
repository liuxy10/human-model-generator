import xml.etree.ElementTree as ET
from urdf_parser_py.urdf import URDF
from scipy.spatial.transform import Rotation as R
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from src import *
from config import *
def scale_torque_related_params(cfg, scale):
    print("Scaling torque related parameters by", scale)
    cfg.jkp *= scale
    cfg.jkd  *= scale
    cfg.torque_lim *= scale
    return cfg



def calculate_humanoid_height(model_path):
    # Load the model
    model = mujoco.MjModel.from_xml_path(model_path)
    # write the model in local coordinate
    data = mujoco.MjData(model)
    
    # Reset the simulation to default pose
    mujoco.mj_resetData(model, data)
    
    # reset the hip joint z direction to +- 0.3 rad
    data.qpos[6] = -0.3
    data.qpos[12] = 0.3
    # print(data.qpos)
    # Find the highest and lowest points in the model
    # First, forward kinematics to update body positions
    mujoco.mj_forward(model, data)
    
    # Initialize min and max height values
    min_height = float('inf')
    max_height = float('-inf')
    
    # Check all geoms to find the highest and lowest points
    for i in range(model.ngeom):
        # Get geom position
        geom_pos = data.geom_xpos[i]
        
        # For sphere and capsule geoms, consider their size
        if model.geom_type[i] == mujoco.mjtGeom.mjGEOM_SPHERE:
            geom_height = geom_pos[2]
            geom_size = model.geom_size[i, 0]  # Radius for sphere
            min_height = min(min_height, geom_height - geom_size)
            max_height = max(max_height, geom_height + geom_size)
        elif model.geom_type[i] == mujoco.mjtGeom.mjGEOM_CAPSULE:
            geom_height = geom_pos[2]
            geom_radius = model.geom_size[i, 0]
            min_height = min(min_height, geom_height - geom_radius)
            max_height = max(max_height, geom_height + geom_radius)
        elif model.geom_type[i] == mujoco.mjtGeom.mjGEOM_BOX:
            geom_height = geom_pos[2]
            geom_halfsize = model.geom_size[i, 2]  # z-dimension half-size
            min_height = min(min_height, geom_height - geom_halfsize)
            max_height = max(max_height, geom_height + geom_halfsize)
        else:
            # For other geom types, just use the center position
            min_height = min(min_height, geom_pos[2])
            max_height = max(max_height, geom_pos[2])
    
    # Calculate total height
    total_height = max_height - min_height
    
    return total_height

def process_body(body, linkMass, name_map):
    name = body.get("name")
    if name in name_map:
        if not isinstance(name_map[name], tuple):
            n = name_map[name]
            if n.startswith("Left"):
                n = n[4:]
            elif n.startswith("Right"):
                n = n[5:]
            mass = linkMass[n + "_mass"]
            print(f"mass of {n}:", mass)
            
        else:
            mass = 0 
            for n in name_map[name]:
                if n.startswith("Left"):
                    n = n[4:]
                elif n.startswith("Right"):
                    n = n[5:]
                mass += linkMass[n + "_mass"]
            print(f"mass of {n}:", mass)
        # ET.SubElement(body, "inertial", pos=pos, mass=mass)
        geom = body.find("geom")
        if geom is not None:
            geom.set("mass", str(mass))
        else:
            print(f"Warning: 'geom' element not found in body '{name}'")
        # print("at:", body.get("name"), "children:", [c.get("name") for c in body.findall("body")])

def assign_mass_inertia( m, linkMass, input_mujoco_xml, output_mujoco_xml):
    name_map = {
        "root": "Pelvis",
        "lfemur": "LeftUpperLeg",
        "ltibia": "LeftLowerLeg",
        "lfoot": ("LeftFoot", "LeftToe"),
        "rfemur": "RightUpperLeg",
        "rtibia": "RightLowerLeg",
        "rfoot": ("RightFoot", "RightToe"),
        "upperback": "UpperTrunk",
        "thorax": "LowerTrunk",
        "lowerneck": ("Head", "Neck"),
        "lclavicle": "LeftShoulder",
        "lhumerus": "LeftUpperArm",
        "lradius": "LeftForeArm",
        "lwrist": "LeftHand",
        "rclavicle": "RightShoulder",
        "rhumerus": "RightUpperArm",
        "rradius": "RightForeArm",
        "rwrist": "RightHand"
    }
    
    
    linkMass = scaleMass(m, linkMass)
    
    print("lowerTrunk:", linkMass["LowerTrunk_mass"])
    tree = ET.parse(input_mujoco_xml)
    root = tree.getroot()
    
    for body in root.findall(".//body"):
        process_body(body, linkMass, name_map)
    # print()    
    tree.write(output_mujoco_xml)

def convert_urdf_to_mujoco(urdf_path, output_xml_path):
    try:
        robot = URDF.from_xml_file(urdf_path)
    except Exception as e:
        print(f"Error parsing URDF: {e}")
        return

    mujoco = ET.Element("mujoco", model=robot.name)
    worldbody = ET.SubElement(mujoco, "worldbody")

    def process_link(link, parent_element):
        body = ET.SubElement(parent_element, "body", name=link.name)

        if link.visual and link.visual.origin:
            origin = link.visual.origin
            body.set("pos", " ".join(map(str, origin.xyz)))
            body.set("quat", " ".join(map(str, R.from_euler('xyz', origin.rotation, degrees=True))))

        if link.visual and link.visual.geometry:
            geometry = link.visual.geometry
            if geometry.box:
                size = geometry.box.size
                ET.SubElement(body, "geom", name=f"{link.name}_geom", type="box", size=" ".join(map(str, [s / 2.0 for s in size])))
            elif geometry.cylinder:
                radius = geometry.cylinder.radius
                length = geometry.cylinder.length
                ET.SubElement(body, "geom", name=f"{link.name}_geom", type="cylinder", size=f"{radius} {length / 2.0}")
            elif geometry.sphere:
                radius = geometry.sphere.radius
                ET.SubElement(body, "geom", name=f"{link.name}_geom", type="sphere", size=str(radius))
            elif geometry.mesh:
                print(f"Warning: Mesh geometry not fully supported for link {link.name}")

        if link.inertial:
            inertia = link.inertial
            body.set("mass", str(inertia.mass))
            ET.SubElement(body, "inertia", ixx=str(inertia.inertia.ixx), iyy=str(inertia.inertia.iyy), izz=str(inertia.inertia.izz), ixy=str(inertia.inertia.ixy), ixz=str(inertia.inertia.ixz), iyz=str(inertia.inertia.iyz))

        for child_joint in robot.child_map[link.name]:
            child_link = robot.child_map[link.name][child_joint][0]
            process_link(robot.link_map[child_link], body)

    root_link = None
    for link in robot.links:
        if link.name not in robot.parent_map:
            root_link = link
            break

    if root_link:
        process_link(root_link, worldbody)
    else:
        print("Error: Could not find root link.")
        return

    actuators = ET.SubElement(mujoco, "actuator")
    for joint in robot.joints:
        if joint.type == 'revolute' or joint.type == 'prismatic':
            ET.SubElement(actuators, "motor", name=joint.name, joint=joint.name, gear="1")

    tree = ET.ElementTree(mujoco)
    ET.indent(tree, space="  ", level=0)
    tree.write(output_xml_path, encoding="utf-8", xml_declaration=True)

if __name__ == "__main__":
    urdf_path = "code/models/humanModels/test.urdf"
    output_xml_path = "code/models/humanModels/test_converted.xml"
    
    xml_input = "models/humanModels/mocap_v2.xml"
    xml_output = "models/humanModels/mocap_v2_converted.xml"
    H = 1.75
    m = 75
    assign_mass_inertia(H, m, linkDimensions, linkMass, xml_input, xml_output)
    
    
