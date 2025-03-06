import os
import xml.etree.ElementTree as ET
import sys
import idyntree.bindings as iDynTree
from urdfModifiers.utils import *
__file__ = "/root/Github/human-model-generator/code/test.ipynb"
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

import mujoco
import mujoco_py
import numpy as np

def write_urdf_with_changes(urdf_file_path):
    tree = ET.parse(urdf_file_path)
    root = tree.getroot()

# Find or create the <mujoco> element
    mujoco_element = root.find('mujoco')
    if mujoco_element is None:
        mujoco_element = ET.SubElement(root, 'mujoco')


    # Create the <compiler> element
        compiler = ET.SubElement(mujoco_element, "compiler")
        compiler.set("balanceinertia", "true")
        compiler.set("discardvisual", "true")
        compiler.set("boundmass", "1e-6")
        compiler.set("boundinertia", "1e-6")
        compiler.set("autolimits", "true")

# find a link named world, if not found, create one
    world_link = root.find(".//link[@name='world']") 
    if world_link is None:
        world_link = ET.Element('link')
        world_link.set('name', 'world')
        root.insert(0, world_link)
# find a joint named world_to_pelvis, if not found, create one
    world_to_pelvis_joint = root.find(".//joint[@name='world_to_pelvis']")
    if world_to_pelvis_joint is None:
        world_to_pelvis_joint = ET.Element('joint')
        world_to_pelvis_joint.set('name', 'world_to_pelvis')
        world_to_pelvis_joint.set('type', 'fixed')
        root.insert(-1, world_to_pelvis_joint)
        world_link_name = world_link.attrib['name']
        pelvis_link_name = 'Pelvis'
        parent_element = ET.SubElement(world_to_pelvis_joint, 'parent')
        parent_element.set('link', world_link_name)
        child_element = ET.SubElement(world_to_pelvis_joint, 'child')
        child_element.set('link', pelvis_link_name)
        origin_element = ET.SubElement(world_to_pelvis_joint, 'origin')
        origin_element.set('xyz', '0 0 0')
        origin_element.set('rpy', '0 0 0')
    # print(world_to_pelvis_joint, world_link)
    
    tree.write(urdf_file_path)


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

def scale_humanoid_model(model_path, scaled_model_path, height, weight = None):
    height_original = calculate_humanoid_height(model_path)
    scale = height / height_original
    print(f"Scaling the model size to {height} meters")
    
    # Load the model
    tree = ET.parse(model_path)
    root = tree.getroot()
    
    # Find the worldbody element
    worldbody = root.find('worldbody')
    
    def scale_entity(parent, scale):
        
        # Scale the model
        for child in parent:
            # print(child.tag)
            if child.tag == 'geom':
                # Scale the size attribute
                size = child.attrib['size']
                size = [float(x) for x in size.split(' ')]
                size = [x * scale for x in size]
                child.set('size', ' '.join(str(x) for x in size))
                # fromto
                fromto = child.attrib.get('fromto')
                if fromto:
                    fromto = [float(x) for x in fromto.split(' ')]
                    fromto = [x * scale for x in fromto]
                    child.set('fromto', ' '.join(str(x) for x in fromto))
                pos = child.attrib.get('pos')
                if pos:
                    pos = [float(x) for x in pos.split(' ')]
                    pos = [x * scale for x in pos]
                    child.set('pos', ' '.join(str(x) for x in pos))
                scale_entity(child, scale)
            elif child.tag == 'body':
                # Scale the pos attribute
                pos = child.attrib['pos']
                pos = [float(x) for x in pos.split(' ')]
                pos = [x * scale for x in pos]
                child.set('pos', ' '.join(str(x) for x in pos))
                scale_entity(child, scale)
            elif child.tag == 'joint':
                # Scale the pos attribute
                pos = child.attrib['pos']
                pos = [float(x) for x in pos.split(' ')]
                pos = [x * scale for x in pos]
                child.set('pos', ' '.join(str(x) for x in pos))
                scale_entity(child, scale)
                
    # Scale the model
    scale_entity(worldbody, scale)       
    
    # reset the total mass
    compiler = tree.find('compiler')
    # print(f"compiler: {compiler.attrib}")
    if weight is not None:
        if type(weight) is float or type(weight) is int:
            compiler.set('settotalmass', str(weight)) # set the total mass
            print(f"total mass: {weight} kg")
        elif type(weight) is dict:
            assign_inertial_to_bodies(worldbody, weight)
                
        # print(f"total mass: {weight} kg")
    # compiler.attrib.pop('autolimits') #remove the autolimits attribute
    
    # Save the scaled model
    
    print(f"Saving the scaled model to {scaled_model_path}")
    tree.write(scaled_model_path)

def assign_inertial_to_bodies(parent, mass_dict):
    # Scale the model
    for child in parent:
        if child.tag == 'body':
            # Scale the pos attribute
            body_name = child.attrib['name']
            if body_name in mass_dict:
                mass = mass_dict[body_name]
                # child.set('mass', str(mass))
                mass_element = child.find('mass')
                if mass_element is None:
                    mass_element = ET.SubElement(child, 'mass')
                    mass.set('value', str(mass))
            assign_inertial_to_bodies(child, mass_dict)
        else:
            pass
    
    
  
def print_body_info_table(model):
    sum_mass = np.sum(model.body_mass)
    print("Body Information")
    print("Body Name          | Body Mass ")
    print("-------------------|-------------------------")
    for i in range(model.nbody):
        # print(f"{model.body_id2name(i):<19}{model.body_mass[i]:<19}")
        print(f"Body {i}: {model.body(i).name:<19} | {model.body_mass[i]:<19}")
    print(f"Total Mass: {sum_mass:.4f}")

def remove_autolimits_attribute(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    compiler = root.find('compiler')
    if 'autolimits' in compiler.attrib:
        compiler.attrib.pop('autolimits')
    tree.write(xml_file)
