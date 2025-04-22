import tkinter as tk
from tkinter import font, scrolledtext, filedialog, simpledialog
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import Image
import mcschematic as mcs
from time import time,sleep
import subprocess
import threading
#import multiprocessing
import trimesh
from collections import defaultdict
import gc

import pyembree
import embreex
import embreex.rtcore

# delay between print()'s for updatingte progress percentage:
delta_T = 0.25



##################################################################################
### BLOCK NAME TRANSLATION STUFF #################################################
##################################################################################

colors = ["red","orange","yellow","lime","green","cyan","light_blue","blue","purple","magenta","white","light_gray","gray","black","brown","pink"]
# blocks where the stair variant exists in vanilla:
stairs_exist = ["granite","andesite","limestone","light_limestone","large_granite_brick","large_andesite_masonry","large_light_limestone_brick","polished_deepslate","blackstone","polished_blackstone","limestone_cobble","limestone_brick","end_stone_brick","clay_brick","sandstone","mossy_limestone_cobble","marble_brick","red_sandstone_cobble","oak_wood_plank","birch_wood_plank","spruce_wood_plank","rough_palm_wood_plank","acacia_wood_plank","dark_oak_wood_plank","warped_planks","crimson_shell_planks"]
# blocks where the full block variant exists in vanilla (only need to put those that are not already listed in conquest_to_vanilla):
full_block_exists = stairs_exist + ["grass_block","loamy_dirt","peridotite","old_rusted_metal_tile","gray_clay","iron_block","end_stone","obsidian","crying_obsidian","snow","prismarine","bundled_hay","umbre_mudstone","carbonite_paneling","magma","old_rusted_metal_block","white_concrete","concrete_wall"]
full_block_exists = full_block_exists + [color+"_wool" for color in colors]
full_block_exists = full_block_exists + [color+"_stained_glass" for color in colors]
# all blocks with DIFFERENT names in vanilla/conquest:
conquest_to_vanilla = {"limestone":"stone",
                       "light_limestone":"diorite",
                       "large_granite_brick":"polished_granite",
                       "large_andesite_masonry":"polished_andesite",
                       "large_light_limestone_brick":"polished_diorite",
                       "limestone_cobble":"cobblestone",
                       "limestone_brick":"stone_bricks",
                       "end_stone_brick":"end_stone_bricks",
                       "clay_brick":"bricks",
                       "mossy_limestone_cobble":"mossy_cobblestone",
                       "marble_brick":"quartz_block",
                       "red_sandstone_cobble":"red_sandstone",
                       "oak_wood_plank":"oak_planks",
                       "birch_wood_plank":"birch_planks",
                       "spruce_wood_plank":"spruce_planks",
                       "rough_palm_wood_plank":"jungle_planks",
                       "acacia_wood_plank":"acacia_planks",
                       "dark_oak_wood_plank":"dark_oak_planks",
                       "warped_planks":"warped_planks",
                       "crimson_shell_planks":"crimson_planks",
                       "loamy_dirt":"dirt",
                       "peridotite":"bedrock",
                       "old_rusted_metal_tile":"nether_bricks",
                       "gray_clay":"clay",
                       "snow":"snow_block",
                       "bundled_hay":"hay_block",
                       "umbre_mudstone":"terracotta",
                       "carbonite_paneling":"purpur_block",
                       "magma":"magma_block",
                       "old_rusted_metal_block":"red_nether_bricks",
                       "concrete_wall":"light_gray_concrete",
                       "mossy_limestone_brick":"mossy_stone_bricks",
                       "oak_log":"oak_wood",
                       "birch_log":"birch_wood",
                       "dark_spruce_log":"spruce_wood",
                       "palm_log":"jungle_wood",
                       "acacia_log":"acacia_wood",
                       "dark_oak_log":"dark_oak_wood",
                       "oak_wood_beam":"stripped_oak_wood",
                       "birch_wood_beam":"stripped_birch_wood",
                       "spruce_wood_beam":"stripped_spruce_wood",
                       "palm_wood_beam":"stripped_jungle_wood",
                       "acacia_wood_beam":"stripped_acacia_wood",
                       "dark_oak_wood_beam":"stripped_dark_oak_wood",
                       "mangrove_wood_beam":"stripped_mangrove_wood",
                       "cherry_wood_beam":"stripped_cherry_wood",
                       "bamboo_wood_beam":"stripped_bamboo_block",
                       "green_bamboo_wood_beam":"bamboo_block",
                       "light_mudstone":"white_terracotta",
                       "worn_light_gray_plaster":"light_gray_terracotta",
                       "gray_cave_silt":"gray_terracotta",
                       "black_hardened_clay":"black_terracotta",
                       "brown_mudstone":"brown_terracotta",
                       "pink_clay":"pink_terracotta",
                       "worn_magenta_plaster":"magenta_terracotta",
                       "worn_purple_plaster":"purple_terracotta",
                       "blue_clay_beaver_tail_tile":"blue_terracotta",
                       "dirty_blue_clay_beaver_tail_tile":"light_blue_terracotta",
                       "old_slate_roof_tile":"cyan_terracotta",
                       "overgrown_green_clay_shingle":"lime_terracotta",
                       "green_clay_shingle":"green_terracotta",
                       "yellow_mudstone":"yellow_terracotta",
                       "orange_mudstone":"orange_terracotta",
                       "red_mudstone":"red_terracotta"
                      }
# add the ones where the name changed
full_block_exists = full_block_exists +[name for name in conquest_to_vanilla if name not in full_block_exists]
# add the ones that have the same name
conquest_to_vanilla.update({name:name for name in full_block_exists if name not in conquest_to_vanilla}) #{color+"_wool":color+"_wool" for color in colors})
conquest_to_vanilla_stairs = conquest_to_vanilla.copy()
conquest_to_vanilla_stairs["limestone_brick"] = "stone_brick"
conquest_to_vanilla_stairs["clay_brick"] = "brick"
conquest_to_vanilla_stairs["end_stone_brick"] = "end_stone_brick"
conquest_to_vanilla_stairs["marble_brick"] = "quartz_stairs"
conquest_to_vanilla_stairs["oak_wood_plank"] = "oak_stairs"
conquest_to_vanilla_stairs["birch_wood_plank"] = "birch_stairs"
conquest_to_vanilla_stairs["spruce_wood_plank"] = "spruce_stairs"
conquest_to_vanilla_stairs["rough_palm_wood_plank"] = "jungle_stairs"
conquest_to_vanilla_stairs["acacia_wood_plank"] = "acacia_stairs"
conquest_to_vanilla_stairs["dark_oak_wood_plank"] = "dark_oak_stairs"
conquest_to_vanilla_stairs["warped_planks"] = "warped_stairs"
conquest_to_vanilla_stairs["crimson_shell_planks"] = "crimson_stairs"





##################################################################################
### GROUP THEORY STUFF ###########################################################
##################################################################################

def concat_perms(p1,p2):
    return tuple(p2[i] for i in p1)

allowed_perms = {(0,1,2,3,4,5):"id",(3,1,2,0,4,5):"x", (0,4,2,3,1,5):"y", (0,1,5,3,4,2):"z", (1,0,2,4,3,5):"xy", (0,2,1,3,5,4):"yz", (2,1,0,5,4,3):"zx", (4,3,2,1,0,5):"yx", (0,5,4,3,2,1):"zy", (5,1,3,2,4,0):"xz"}
find_flips={(0,1,2,3,4,5):tuple()}
for a in allowed_perms:
    for b in allowed_perms:
        for c in allowed_perms:
            permutation = concat_perms(concat_perms(a,b),c)
            flip_list = (allowed_perms[a],allowed_perms[b],allowed_perms[c])
            flip_list = tuple(flip for flip in flip_list if flip!="id")
            if len(find_flips.get(permutation,(1,2,3,4)))>len(flip_list):
                find_flips[permutation] = flip_list

facing_perms = {"Down":(0,1,2,3,4,5),"Up":(0,4,5,3,1,2),"North":(0,2,4,3,5,1),"East":(2,3,4,5,0,1),"South":(3,5,4,0,2,1),"West":(5,0,4,2,3,1)}





##################################################################################
### UTILITY FUNCTIONS ############################################################
##################################################################################

#def safe_eval(seconds, s):
#    """Run eval until timeout in seconds reached."""
#    with multiprocessing.Pool(processes=2) as pool:
#        result = pool.apply_async(eval, [s])
#        try:
#            return float(result.get(timeout=seconds)),True
#        except multiprocessing.TimeoutError:
#            return None,False
#        except Exception as e:
#            return None,None

## non-safe version:
def safe_eval(seconds, s):
    """non safe version."""
    try:
        return float(eval(s)),True
    except Exception as e:
        return None,None

def load_image( infilename ) :
    img = Image.open( infilename )
    img.load()
    data = np.asarray( img, dtype="int32" )
    return data

def save_image( npdata, outfilename ) :
    img = Image.fromarray( np.asarray( np.clip(npdata,0,255), dtype="uint8"), "L" )
    img.save( outfilename )

def distance(hm1,hm2): # difference between 2 heightmaps
    return sum(sum(abs(hm1-hm2)))

def rotate90(matrix):
    return np.flip(matrix.T,0)
def rotate180(matrix):
    return rotate90(rotate90(matrix))
def rotate270(matrix):
    return rotate90(rotate90(rotate90(matrix)))

def array_hm_to_int(matrix,facing="d"):
    n,m = matrix.shape
    k=0
    shape=0
    for i in range(n):
        for j in range(m):
            n = int(matrix[i][j]+0.5)
            if facing=="d":
                shape += ((1<<n)-1)<<8*k
            elif facing=="u":
                shape += ((1<<n)-1)<<(8*k+8-n)
            k+=1;
    return shape





##################################################################################
### BLOCK SHAPES #################################################################
##################################################################################

H = 1 # 8*H is the number of voxels used per block (in each direction, so 512 voxels per block)

FULL         = np.ones((8*H,8*H),int)*H
THREEQUARTER = np.ones((6*H,6*H),int)*H
HALF         = np.ones((4*H,4*H),int)*H
QUARTER      = np.ones((2*H,2*H),int)*H
EIGHTH       = np.ones((1*H,1*H),int)*H
ROW = np.ones((H,8*H),int)*H
COL = np.ones((8*H,H),int)*H

default_full = FULL*8
default_slab = FULL
default_vertical_slab = [np.block([[8*ROW]]*1+[[0*ROW]]*7),
                         np.block([[8*ROW]]*2+[[0*ROW]]*6),
                         np.block([[8*ROW]]*4+[[0*ROW]]*4),
                         np.block([[8*ROW]]*6+[[0*ROW]]*2)]
default_stairs = [np.block([[8*HALF,4*HALF],
                            [4*HALF,4*HALF]]),
                  np.block([[8*HALF,8*HALF],
                            [4*HALF,4*HALF]]),
                  np.block([[8*HALF,8*HALF],
                            [8*HALF,4*HALF]])]
default_quarter_slab = [np.block([[2*ROW]]*2+[[0*ROW]]*6),
                        np.block([[4*ROW]]*4+[[0*ROW]]*4),
                        np.block([[6*ROW]]*6+[[0*ROW]]*2)]
default_vertical_quarter = [np.pad(8*EIGHTH,(0,7*H)),
                            np.pad(8*QUARTER,(0,6*H)),
                            np.pad(8*HALF,(0,4*H)),
                            np.pad(8*THREEQUARTER,(0,2*H))]
default_vertical_corner = [default_full-default_vertical_quarter[i] for i in range(1,4)] + [np.maximum(np.pad(8*ROW,((7*H,0),(0,0))),np.pad(8*COL,((0,0),(7*H,0))))]
default_corner_slab = np.block([[4*HALF,4*HALF],
                                [4*HALF,0*HALF]])
default_vertical_corner_slab = [np.block([[8*HALF,4*HALF],
                                          [0*HALF,0*HALF]]),
                                np.block([[4*HALF,8*HALF],
                                          [0*HALF,0*HALF]])]
default_eighth_slab = np.block([[4*HALF,0*HALF],
                                [0*HALF,0*HALF]])

stair_shapes = ["outer_left","straight","inner_left"]
hinges = ["left","right"]


# AIR and FULL block
facing_down = [(0*FULL, "air", ""),
               (8*FULL, "", "")]
# SLAB
facing_down += [(default_slab*i,"_slab",f"[layers={i}]") for i in range(1,8)]
# VERTICAL SLAB
facing_down += [(default_vertical_slab[i],"_vertical_slab",f"[layer={i+1},facing=east]") for i in range(4)]
facing_down += [(rotate90(default_vertical_slab[i]),"_vertical_slab",f"[layer={i+1},facing=north]") for i in range(4)]
facing_down += [(rotate180(default_vertical_slab[i]),"_vertical_slab",f"[layer={i+1},facing=west]") for i in range(4)]
facing_down += [(rotate270(default_vertical_slab[i]),"_vertical_slab",f"[layer={i+1},facing=south]") for i in range(4)]
# STAIRS
facing_down += [(default_stairs[i],"_stairs",f"[shape={stair_shapes[i]},facing=west]") for i in range(3)]
facing_down += [(rotate90(default_stairs[i]),"_stairs",f"[shape={stair_shapes[i]},facing=south]") for i in range(3)]
facing_down += [(rotate180(default_stairs[i]),"_stairs",f"[shape={stair_shapes[i]},facing=east]") for i in range(3)]
facing_down += [(rotate270(default_stairs[i]),"_stairs",f"[shape={stair_shapes[i]},facing=north]") for i in range(3)]
# QUARTER SLAB
facing_down += [(default_quarter_slab[i],"_quarter_slab",f"[layer={i+1},facing=east]") for i in range(3)]
facing_down += [(rotate90(default_quarter_slab[i]),"_quarter_slab",f"[layer={i+1},facing=north]") for i in range(3)]
facing_down += [(rotate180(default_quarter_slab[i]),"_quarter_slab",f"[layer={i+1},facing=west]") for i in range(3)]
facing_down += [(rotate270(default_quarter_slab[i]),"_quarter_slab",f"[layer={i+1},facing=south]") for i in range(3)]
# VERTICAL QUARTER
facing_down += [(default_vertical_quarter[i],"_vertical_quarter",f"[layer={i+1},facing=east]") for i in range(4)]
facing_down += [(rotate90(default_vertical_quarter[i]),"_vertical_quarter",f"[layer={i+1},facing=north]") for i in range(4)]
facing_down += [(rotate180(default_vertical_quarter[i]),"_vertical_quarter",f"[layer={i+1},facing=west]") for i in range(4)]
facing_down += [(rotate270(default_vertical_quarter[i]),"_vertical_quarter",f"[layer={i+1},facing=south]") for i in range(4)]
# VERTICAL CORNER
facing_down += [(default_vertical_corner[3-i],"_vertical_corner",f"[layer={i+1},facing=west]") for i in range(4)]
facing_down += [(rotate90(default_vertical_corner[3-i]),"_vertical_corner",f"[layer={i+1},facing=south]") for i in range(4)]
facing_down += [(rotate180(default_vertical_corner[3-i]),"_vertical_corner",f"[layer={i+1},facing=east]") for i in range(4)]
facing_down += [(rotate270(default_vertical_corner[3-i]),"_vertical_corner",f"[layer={i+1},facing=north]") for i in range(4)]
# CORNER SLAB
facing_down += [(default_corner_slab,"_corner_slab","[facing=east]")]
facing_down += [(rotate90(default_corner_slab),"_corner_slab","[facing=north]")]
facing_down += [(rotate180(default_corner_slab),"_corner_slab","[facing=west]")]
facing_down += [(rotate270(default_corner_slab),"_corner_slab","[facing=south]")]
# VERTICAL CORNER SLAB
facing_down += [(default_vertical_corner_slab[i],"_vertical_corner_slab",f"[hinge={hinges[i]},facing=east]") for i in range(2)]
facing_down += [(rotate90(default_vertical_corner_slab[i]),"_vertical_corner_slab",f"[hinge={hinges[i]},facing=north]") for i in range(2)]
facing_down += [(rotate180(default_vertical_corner_slab[i]),"_vertical_corner_slab",f"[hinge={hinges[1-i]},facing=west]") for i in range(2)]
facing_down += [(rotate270(default_vertical_corner_slab[i]),"_vertical_corner_slab",f"[hinge={hinges[1-i]},facing=south]") for i in range(2)]
# EIGHTH SLAB
facing_down += [(default_eighth_slab,"_eighth_slab",f"[facing=east]")]
facing_down += [(rotate90(default_eighth_slab),"_eighth_slab",f"[facing=north]")]
facing_down += [(rotate180(default_eighth_slab),"_eighth_slab",f"[facing=west]")]
facing_down += [(rotate270(default_eighth_slab),"_eighth_slab",f"[facing=south]")]


# AIR and FULL block
facing_up = []
# SLAB
facing_up += [(default_slab*i,"_slab",f"[layers={i},type=top]") for i in range(1,8)]
# STAIRS
facing_up += [(default_stairs[i],"_stairs",f"[shape={stair_shapes[i]},facing=west,half=top]") for i in range(3)]
facing_up += [(rotate90(default_stairs[i]),"_stairs",f"[shape={stair_shapes[i]},facing=south,half=top]") for i in range(3)]
facing_up += [(rotate180(default_stairs[i]),"_stairs",f"[shape={stair_shapes[i]},facing=east,half=top]") for i in range(3)]
facing_up += [(rotate270(default_stairs[i]),"_stairs",f"[shape={stair_shapes[i]},facing=north,half=top]") for i in range(3)]
# QUARTER SLAB
facing_up += [(default_quarter_slab[i],"_quarter_slab",f"[layer={i+1},facing=east,type=top]") for i in range(3)]
facing_up += [(rotate90(default_quarter_slab[i]),"_quarter_slab",f"[layer={i+1},facing=north,type=top]") for i in range(3)]
facing_up += [(rotate180(default_quarter_slab[i]),"_quarter_slab",f"[layer={i+1},facing=west,type=top]") for i in range(3)]
facing_up += [(rotate270(default_quarter_slab[i]),"_quarter_slab",f"[layer={i+1},facing=south,type=top]") for i in range(3)]
# CORNER SLAB
facing_up += [(default_corner_slab,"_corner_slab","[facing=east,type=top]")]
facing_up += [(rotate90(default_corner_slab),"_corner_slab","[facing=north,type=top]")]
facing_up += [(rotate180(default_corner_slab),"_corner_slab","[facing=west,type=top]")]
facing_up += [(rotate270(default_corner_slab),"_corner_slab","[facing=south,type=top]")]
# VERTICAL CORNER SLAB
facing_up += [(default_vertical_corner_slab[i],"_vertical_corner_slab",f"[hinge={hinges[i]},facing=east,type=top]") for i in range(2)]
facing_up += [(rotate90(default_vertical_corner_slab[i]),"_vertical_corner_slab",f"[hinge={hinges[i]},facing=north,type=top]") for i in range(2)]
facing_up += [(rotate180(default_vertical_corner_slab[i]),"_vertical_corner_slab",f"[hinge={hinges[1-i]},facing=west,type=top]") for i in range(2)]
facing_up += [(rotate270(default_vertical_corner_slab[i]),"_vertical_corner_slab",f"[hinge={hinges[1-i]},facing=south,type=top]") for i in range(2)]
# EIGHTH SLAB
facing_up += [(default_eighth_slab,"_eighth_slab",f"[facing=east,type=top]")]
facing_up += [(rotate90(default_eighth_slab),"_eighth_slab",f"[facing=north,type=top]")]
facing_up += [(rotate180(default_eighth_slab),"_eighth_slab",f"[facing=west,type=top]")]
facing_up += [(rotate270(default_eighth_slab),"_eighth_slab",f"[facing=south,type=top]")]

## global variable containing ALL allowed block shapes as big ints
block_models  = [(array_hm_to_int(block[0],'d'), block[1], block[2]) for block in facing_down]
block_models += [(array_hm_to_int(block[0],'u'), block[1], block[2]) for block in facing_up]





##################################################################################
### IMAGE --> NP ARRAY --> VOXELMAP --> SCHEMATIC ################################
##################################################################################

def voxelmap_from_mesh(mesh, sizes, invert, force_dir='automatic'):
    res_x,res_y,res_z = sizes

    # Optional: Ensure it's watertight
    watertight = mesh.is_watertight
    print("Model is watertight: ", watertight)

    total_total_time = time()

    tmp=time()
    print("Starting voxelization process...",end="")
    # Calculating bounding box
    bounds_min, bounds_max = mesh.bounds
    # Creating voxel grid
    grid_size = np.array([res_x,res_y,res_z]) * 8
    voxel_size = (bounds_max - bounds_min) / (grid_size-1)
    # Generating ray origins and directions
    x = np.linspace(bounds_min[0], bounds_max[0], grid_size[0])
    y = np.linspace(bounds_min[1], bounds_max[1], grid_size[1])
    z = np.linspace(bounds_min[2], bounds_max[2], grid_size[2])
    x_indices = np.arange(0, grid_size[0], 1)
    y_indices = np.arange(0, grid_size[1], 1)
    z_indices = np.arange(0, grid_size[2], 1)
    x_start = bounds_min[0] - voxel_size[0]  # slightly outside the mesh
    #x_end = bounds_max[0] + voxel_size[0]
    x_dir = np.array([1, 0, 0])
    y_start = bounds_min[1] - voxel_size[1]  # slightly outside the mesh
    #y_end = bounds_max[1] + voxel_size[1]
    y_dir = np.array([0, 1, 0])
    z_start = bounds_min[2] - voxel_size[2]  # slightly outside the mesh
    #z_end = bounds_max[2] + voxel_size[2]
    z_dir = np.array([0, 0, 1])
    origins_x = np.array(np.meshgrid(x_start, y, z, indexing='ij')).reshape(3, -1).T
    origins_indices_x = np.array(np.meshgrid(0, y_indices, z_indices, indexing='ij')).reshape(3, -1).T
    directions_x = np.tile(x_dir, (origins_x.shape[0], 1))
    origins_y = np.array(np.meshgrid(x, y_start, z, indexing='ij')).reshape(3, -1).T
    origins_indices_y = np.array(np.meshgrid(x_indices, 0, z_indices, indexing='ij')).reshape(3, -1).T
    directions_y = np.tile(y_dir, (origins_y.shape[0], 1))
    origins_z = np.array(np.meshgrid(x, y, z_start, indexing='ij')).reshape(3, -1).T
    origins_indices_z = np.array(np.meshgrid(x_indices, y_indices, 0, indexing='ij')).reshape(3, -1).T
    directions_z = np.tile(z_dir, (origins_z.shape[0], 1))
    if force_dir in ["x","y","z"]:
        ray_direction = force_dir
        watertight = True
    elif (watertight or force_dir=="fastest") and force_dir!="all":
        watertight=True
        ray_direction = "xyz"[np.argmin([len(origins_x),len(origins_y),len(origins_z)])]
    else:
        watertight=False
        ray_direction = "x-, y- and z"
    print(f" DONE ({round(time()-tmp,2)}s)")

    tmp=time()
    print(f"Performing ray-mesh intersections (using rays in {ray_direction}-direction)...",end="")

    ray_intersector = trimesh.ray.ray_pyembree.RayMeshIntersector(mesh)

    if ray_direction=="x" or not watertight:
        locations_x, index_ray_x, _ = ray_intersector.intersects_location(origins_x, directions_x, multiple_hits=True)
        if invert:
            voxel_grid_x = np.ones(tuple(grid_size), dtype=bool)
        else:
            voxel_grid_x = np.zeros(tuple(grid_size), dtype=bool)
    if ray_direction=="y" or not watertight:
        locations_y, index_ray_y, _ = ray_intersector.intersects_location(origins_y, directions_y, multiple_hits=True)
        if invert:
            voxel_grid_y = np.ones(tuple(grid_size), dtype=bool)
        else:
            voxel_grid_y = np.zeros(tuple(grid_size), dtype=bool)
    if ray_direction=="z" or not watertight:
        locations_z, index_ray_z, _ = ray_intersector.intersects_location(origins_z, directions_z, multiple_hits=True)
        if invert:
            voxel_grid_z = np.ones(tuple(grid_size), dtype=bool)
        else:
            voxel_grid_z = np.zeros(tuple(grid_size), dtype=bool)
    print(f" DONE ({round(time()-tmp,2)}s)")

    tmp=time()
    print("Collecting ray hits...",end="")
    if ray_direction=="x" or not watertight:
        ray_hit_dict_x = defaultdict(list)
        for loc, idx in zip(locations_x, index_ray_x):
            ray_hit_dict_x[idx].append(loc)
        del locations_x
        del index_ray_x
    if ray_direction=="y" or not watertight:
        ray_hit_dict_y = defaultdict(list)
        for loc, idx in zip(locations_y, index_ray_y):
            ray_hit_dict_y[idx].append(loc)
        del locations_y
        del index_ray_y
    if ray_direction=="z" or not watertight:
        ray_hit_dict_z = defaultdict(list)
        for loc, idx in zip(locations_z, index_ray_z):
            ray_hit_dict_z[idx].append(loc)
        del locations_z
        del index_ray_z
    gc.collect()
    print(f" DONE ({round(time()-tmp,2)}s)")


    count = 0
    print("Processing intersections...")
    total_time = time()
    time1=0
    time2=0
    timer=0
    if   ray_direction=="x":
        total = len(ray_hit_dict_x)
    elif ray_direction=="y":
        total = len(ray_hit_dict_y)
    elif ray_direction=="z":
        total = len(ray_hit_dict_z)
    else:
        total = len(ray_hit_dict_x) + len(ray_hit_dict_y) + len(ray_hit_dict_z)
    if ray_direction=="x" or not watertight:
        for i, hits in ray_hit_dict_x.items():
            if time()-timer>delta_T:
                timer=time()
                print(f"\r  progress     : {int(count/total*1000+.5)/10}%           ",end="")
            count+=1

            ray_hits = np.array(hits)

            tmp=time()
            if len(ray_hits) < 2:
                continue
            ray_hits = ray_hits[np.argsort(ray_hits[:, 0]),0]
            time1 += time()-tmp

            for j in range(0, len(ray_hits)-1, 2):

                tmp=time()
                start, end = ray_hits[j], ray_hits[j+1]
                y_idx = origins_indices_x[i][1]
                z_idx = origins_indices_x[i][2]
                x_idxs = np.arange(np.ceil((start - bounds_min[0]) / voxel_size[0]), (end - bounds_min[0]) / voxel_size[0], 1).astype(int)
                x_idxs = np.clip(x_idxs, 0, grid_size[0] - 1)

                voxel_grid_x[x_idxs, y_idx, (grid_size[2]-1) - z_idx] = (not invert)
                time2 += time()-tmp
        voxel_grid = voxel_grid_x
        del ray_hit_dict_x
        gc.collect()
    if ray_direction=="y" or not watertight:
        for i, hits in ray_hit_dict_y.items():
            if time()-timer>delta_T:
                timer=time()
                print(f"\r  progress     : {int(count/total*1000+.5)/10}%           ",end="")
            count+=1

            ray_hits = np.array(hits)

            tmp=time()
            if len(ray_hits) < 2:
                continue
            ray_hits = ray_hits[np.argsort(ray_hits[:, 1]),1]
            time1 += time()-tmp

            for j in range(0, len(ray_hits)-1, 2):

                tmp=time()
                start, end = ray_hits[j], ray_hits[j+1]
                x_idx = origins_indices_y[i][0]
                z_idx = origins_indices_y[i][2]
                y_idxs = np.arange(np.ceil((start - bounds_min[1]) / voxel_size[1]), (end - bounds_min[1]) / voxel_size[1], 1).astype(int)
                y_idxs = np.clip(y_idxs, 0, grid_size[1] - 1)

                voxel_grid_y[x_idx, y_idxs, (grid_size[2]-1) - z_idx] = (not invert)
                time2 += time()-tmp
        voxel_grid = voxel_grid_y
        del ray_hit_dict_y
        gc.collect()
    if ray_direction=="z" or not watertight:
        for i, hits in ray_hit_dict_z.items():
            if time()-timer>delta_T:
                timer=time()
                print(f"\r  progress     : {int(count/total*1000+.5)/10}%           ",end="")
            count+=1

            ray_hits = np.array(hits)

            tmp=time()
            if len(ray_hits) < 2:
                continue
            ray_hits = ray_hits[np.argsort(ray_hits[:, 2]),2]
            time1 += time()-tmp

            for j in range(0, len(ray_hits)-1, 2):

                tmp=time()
                start, end = ray_hits[j], ray_hits[j+1]
                x_idx = origins_indices_z[i][0]
                y_idx = origins_indices_z[i][1]
                z_idxs = np.arange(np.ceil((start - bounds_min[2]) / voxel_size[2]), (end - bounds_min[2]) / voxel_size[2], 1).astype(int)
                z_idxs = np.clip(z_idxs, 0, grid_size[2] - 1)

                voxel_grid_z[x_idx, y_idx, (grid_size[2]-1) - z_idxs] = (not invert)
                time2 += time()-tmp
        voxel_grid = voxel_grid_z
        del ray_hit_dict_z
        gc.collect()

    total_time = time() - total_time
    print(f"\r  progress : 100%              \n  total    : {round(total_time,2)}s"
         +f"\n    sorting ray-hits : {round(time1,2)}s ({int(time1/total_time*1000+.5)/10}%)"
         +f"\n    set values       : {round(time2,2)}s ({int(time2/total_time*1000+.5)/10}%)"
         +f"\n    other            : {round(total_time-time1-time2,2)}s ({int((total_time-time1-time2)/total_time*1000+.5)/10}%)")

    if not watertight:
        tmp=time()
        # we want:
        # voxel_grid = (voxel_grid_z & (voxel_grid_x|voxel_grid_y)) | (voxel_grid_x&voxel_grid_y)
        # but to not run out of memory, we do:
        print("Error correction... (0%)",end="")
        voxel_grid = np.logical_or(voxel_grid_x,voxel_grid_y)
        print("\rError correction... (25%)",end="")
        voxel_grid = np.logical_and(voxel_grid,voxel_grid_z)
        del voxel_grid_z
        gc.collect()
        print("\rError correction... (50%)",end="")
        voxel_grid_x = np.logical_and(voxel_grid_x,voxel_grid_y)
        del voxel_grid_y
        gc.collect()
        print("\rError correction... (75%)",end="")
        voxel_grid = np.logical_or(voxel_grid,voxel_grid_x)
        del voxel_grid_x
        gc.collect()
        print(f"\rError correction... DONE ({round(time()-tmp,2)}s)                ")

    vm = {}
    print("Building voxelmap:")
    total_time = time()
    time0=0
    time1=0
    time2=0
    count = 0
    total = res_x*res_y*res_z
    N_full_block = (1<<(8**3))-1

    tmp=time()
    print("Packing binary array to uint8...",end="")
    packed = np.packbits(voxel_grid, axis=1, bitorder='little')
    print(f" DONE ({round(time() - tmp,2)}s)")
    tmp=time()
    print("Reshaping array...",end="")
    flat_blocks = packed.reshape(res_x, 8, res_y, res_z, 8).transpose(0, 2, 3, 1, 4).reshape(res_x, res_y, res_z, 64)
    print(f" DONE ({round(time() - tmp,2)}s)")
    tmp=time()
    print("Packing uint8 array to 512-bit integer array...",end="")
    byte_array_512bit = flat_blocks.view(np.uint64).dot(1 << (np.arange(8).astype(object) * 64))
    print(f" DONE ({round(time() - tmp,2)}s)")

    print("Writing voxelmap...",end="")
    gc.collect()
    timer=0
    for x in range(res_x):
        for y in range(res_y):
            for z in range(res_z):
                if time()-timer>delta_T:
                    timer=time()
                    print(f"\rWriting voxelmap... {int(count/total*1000+.5)/10}%           ",end="")
                count+=1

                N = byte_array_512bit[x][y][z]

                #if N!=0 and N!=N_full_block:
                if N==N_full_block:
                    N=-1 #smaller!
                #else:
                 #   N=0
                if N!=0:
                    vm[(x,z,y)] = N

    total_time = time() - total_time
    print(f"\rWriting voxelmap... DONE ({round(total_time,2)}s)                ",end="")

    total_total_time = time()-total_total_time
    print(f"\nTOTAL TIME (for getting the voxelmap): {round(total_total_time,2)}s")

    return vm


def voxelmap_from_array(hm, depth_multiplier=1/32, scale=8, noair=True):
    total_time = time()
    # scale = pixel per block = 8, 4, 2 or 1 (or you'll have to increase H)
    print(f"building voxelmap...",end="\n")

    hm = hm * depth_multiplier
    xmax, zmax = hm.shape
    xmax //= scale
    zmax //= scale
    ymin = int(hm.min())//scale
    ymax = int(hm.max())//scale
    N_full_block = (1<<((H*8)**3))-1
    empty_block = np.zeros((8*H,8*H),int)
    full_block  = 8*np.ones((8*H,8*H),int)*H
    eps = 0.1
    plt.imshow(hm,cmap="gray")

    ##setup schematic
    schem = mcs.MCSchematic()

    ## actual calculation:
    cut_y_slice_time = 0
    fullblock_time = 0
    clamp_time   = 0
    convert_time = 0
    compare_time = 0
    timer=0
    count=0
    factor = (8*H)//scale
    if scale>8*H or (8*H)%scale!=0:
        raise IOError(f"invalid scale, must divide {8*H}")
    total = xmax*zmax#*(ymax-ymin)
    voxelmap = {}    #{(i,j,k):0 for i in range(ymax) for j in range(zmax) for k in range(xmax)}
    shift_lut_64 = np.tile(np.arange(0,64,8, dtype=np.uint64)[:, None], (1, 8))
    shift_lut_8  = np.arange(0,512,64, dtype=object)
    for x in range(xmax):
        for z in range(zmax):
            if time()-timer>delta_T:
                timer=time()
                print(f"\r  progress : {int(count/total*1000+.5)/10}%           ",end="")
            count += 1
            # xz_hm is always (8*H)x(8*H):
            tmp = time()
            if scale==8*H:
                xz_hm = hm[scale*x:scale*(x+1),scale*z:scale*(z+1)]
            else:
                xz_hm = np.kron(hm[scale*x:scale*(x+1),scale*z:scale*(z+1)], np.ones((factor,factor)))*factor

            ymin_local = int(xz_hm.min())//(8*H)
            ymax_local = int(xz_hm.max())//(8*H)+1

            cut_y_slice_time += time()-tmp
            tmp = time()
            for y in range(ymin_local-ymin):
                voxelmap[(x,z,y)] = -1
                #block_hm = np.minimum(np.maximum(xz_hm-8*H*(y+ymin),0),8*H)
            fullblock_time += time()-tmp
            for y in range(ymin_local-ymin,ymax_local-ymin):
                tmp = time()
                block_hm = np.minimum(np.maximum(xz_hm-8*H*(y+ymin),0),8*H)
                clamp_time += time()-tmp

                tmp = time()
                if (block_hm==empty_block).all():
                    if not noair:
                        voxelmap[(x,z,y)] = 0
                elif (block_hm==full_block).all():
                    voxelmap[(x,z,y)] = -1
                else:

                    #shape = sum(((1<<int(block_hm[i][j]+0.5))-1)<<8*H*(8*H*i+j) for i in range(8*H) for j in range(8*H))

                    exponents = (block_hm.T + 0.5).astype(np.uint64)
                    subsubsums = (1<<exponents)-1
                    subsums = np.sum(subsubsums << shift_lut_64, axis=0).astype(object)
                    #shape = sum((int(subsums[i])<<(64*i)) for i in range(8))
                    shape = np.sum(subsums << shift_lut_8, axis=0)

                    if shape==N_full_block:
                        voxelmap[(x,z,y)] = -1
                    elif shape>0 or not noair:
                        voxelmap[(x,z,y)] = shape

                convert_time += time()-tmp


    total_time = time() - total_time
    print(f"\r  progress : 100%           \n  total    : {round(total_time,2)}s" +
                                        f"\n    dividing heightmap into blocks   : {round(cut_y_slice_time+clamp_time,2)}s ({round((cut_y_slice_time+clamp_time)/total_time*100,1)}%)" +
                                        f"\n    saving full blocks               : {round(fullblock_time,2)}s ({round(fullblock_time/total_time*100,1)}%)" +
                                        f"\n    saving shapes of non-full blocks : {round(convert_time,2)}s ({round(convert_time/total_time*100,1)}%)" +
                                        f"\n    other                            : {round(total_time-cut_y_slice_time-fullblock_time-clamp_time-convert_time,2)}s ({round((total_time-cut_y_slice_time-fullblock_time-clamp_time-convert_time)/total_time*100,1)}%)")

    return voxelmap

def schematic_from_voxelmap(voxelmap,output_filename="my_schematic", material="limestone", allowed_block_shapes=block_models):
    ## setting up variables:
    scale = H*8
    N = (1<<(scale**3))-1

    models = ["","_stairs","_slab","_vertical_slab","_quarter_slab","_vertical_quarter","_eighth_slab","_vertical_corner","_corner_slab","_vertical_corner_slab"]
    prefix = {model:"conquest:"+material for model in models}
    prefix["air"] = "minecraft:"
    if material in full_block_exists:
        prefix[""]        = "minecraft:" + conquest_to_vanilla[material]
    if material in stairs_exist:
        prefix["_stairs"] = "minecraft:" + conquest_to_vanilla_stairs[material]

    ##setup schematic
    schem = mcs.MCSchematic()

    ## actual calculation:
    total_time = time()
    clamp_time   = 0
    compare_time = 0
    setblock_time = 0
    timer=0
    count=0
    total = len(voxelmap)

    print(f"building schematic...",end="\n")
    for (x,z,y) in voxelmap:
        count+=1
        if time()-timer>delta_T:
            timer = time()
            print(f"\r  progress  : {int(count/total*1000+.5)/10}%            ",end="")

        block_hm_int = voxelmap[(x,z,y)]

        tmp = time()
        if block_hm_int==0: # air
            block_model,block_states = "air",""
        elif block_hm_int in [-1,N]: # full block
            block_model,block_states = "",""
        else:
            shape,block_model,block_states = min(allowed_block_shapes, key = lambda block : (block[0]^block_hm_int).bit_count())
        compare_time += time()-tmp

        tmp = time()
        schem.setBlock((x, y, -z), prefix[block_model] + block_model + block_states)
        setblock_time += time()-tmp

    ## save schematic and print times
    tmp = time()
    schem.save("", output_filename, mcs.Version.JE_1_20_1)
    saving_time = time() - tmp

    total_time = time() - total_time
    print(f"\r  progress : 100%               \n  total    : {round(total_time,2)}s" +
                                            f"\n    choosing closest block model : {round(compare_time,2)}s ({round(compare_time/total_time*100,1)}%)" +
                                            f"\n    saving blocks to schematic   : {round(setblock_time,2)}s ({round(setblock_time/total_time*100,1)}%)" +
                                            f"\n    saving schematic file        : {round(saving_time,2)}s ({round(saving_time/total_time*100,1)}%)" +
                                            f"\n    other                        : {round(total_time-compare_time-setblock_time-saving_time,2)}s ({round((total_time-compare_time-setblock_time-saving_time)/total_time*100,1)}%)")





##################################################################################
### FUNCTIONS ON VOXELMAPS #######################################################
##################################################################################

def inverted_vm(voxelmap):
    total_time = time()
    scale = H*8
    N = (1<<(scale**3))-1
    new_voxelmap = {}
    print(f"inverting voxelmap...",end="\n")
    timer=0
    count=0
    total=len(voxelmap)
    ymax= -1000000
    ymin={}

    tmp = time()
    for (x,z,y) in voxelmap:
        ymax = max(ymax,y+1)
        ymin[(x,z)] = max(ymin.get((x,z),y+1),y+1)
        if time()-timer>delta_T:
            timer=time()
            print(f"\r  progress : {int(count/total*1000+.5)/10}%              ",end="")
        count+=1
        new_voxelmap[(x,z,y)] = N^voxelmap[(x,z,y)]
    invert_time = time()-tmp
    tmp = time()
    for (x,z) in ymin:
        for y in range(ymin[(x,z)],ymax):
            new_voxelmap[(x,z,y)] = -1
    fullblock_time = time()-tmp
    total_time = time() - total_time
    print(f"\r  progress : 100%               \n  total    : {round(total_time,2)}s" +
                                            f"\n    inverting block shapes : {round(invert_time,2)}s ({round(invert_time/total_time*100,1)}%)" +
                                            f"\n    placing full blocks    : {round(fullblock_time,2)}s ({round(fullblock_time/total_time*100,1)}%)" +
                                            f"\n    other                  : {round(total_time-invert_time-fullblock_time,2)}s ({round((total_time-invert_time-fullblock_time)/total_time*100,1)}%)")

    return new_voxelmap


def flip(voxelmap,axis):
    total_time = time()
    new_voxelmap={}
    timer=0
    count=0
    total=len(voxelmap)
    match axis:
        case "y":
            constants = [(   k,      7-k,  ((1<<(64*8))-1)//255            ) for k in range(4)]
        case "z":
            constants = [( 8*k,  8*(7-k),  255*((1<<(64*8))-1)//((1<<64)-1)) for k in range(4)]
        case "x":
            constants = [(64*k, 64*(7-k), (1<<64)-1                        ) for k in range(4)]
        case "xy":
            constants = [(k,   64*k, (((1<<(8*8))-1)//((1<<8)-1))*(((1<<(65*(8-k)))-1)//((1<<65)-1))) for k in range(1,8)]
        case "zy":
            constants = [(k,   8*k,  (((1<<(64*8))-1)//((1<<64)-1))*(((1<<(9*(8-k)))-1)//((1<<9)-1))) for k in range(1,8)]
        case "xz":
            constants = [(8*k, 64*k, 255*(((1<<(72*(8-k)))-1)//((1<<72)-1))                         ) for k in range(1,8)]
        case "yx":
            constants = [(k,   64*(7-k)+7, (((1<<(8*8))-1)//((1<<8)-1))*((((1<<(63*(k+1)))-1))//((1<<63)-1))) for k in range(7)]
        case "yz":
            constants = [(k,   8*(7-k)+7,  (((1<<(64*8))-1)//((1<<64)-1))*((((1<<(7*(k+1)))-1))//((1<<7)-1))) for k in range(7)]
        case "zx":
            constants = [(8*k, 64*(7-k)+7, (((1<<(1*8))-1)//((1<<1)-1))*((((1<<(56*(k+1)))-1))//((1<<56)-1))) for k in range(7)]
        case _:
            raise IOError("invalid axis")

    print(f"flipping in {axis}-direction...",end="\n")
    for (x,z,y) in voxelmap:
        if time()-timer>delta_T:
            timer=time()
            print(f"\r  progress : {int(count/total*1000+.5)/10}%           ",end="")
        count+=1
        N = voxelmap[(x,z,y)]
        if N>0: # dont do anything if N=0 (air) or N=-1 (full block)
            for (n,k,const) in constants:
                b  = ((N>>n)^(N>>k)) & const
                N ^= ((b<<n)|(b<<k))
        match axis:
            case "y":
                new_voxelmap[( x, z,-y)] = N
            case "z":
                new_voxelmap[( x,-z, y)] = N
            case "x":
                new_voxelmap[(-x, z, y)] = N
            case "xy":
                new_voxelmap[( y, z, x)] = N
            case "zy":
                new_voxelmap[( x, y, z)] = N
            case "xz":
                new_voxelmap[( z, x, y)] = N
            case "yx":
                new_voxelmap[(-y, z,-x)] = N
            case "yz":
                new_voxelmap[( x,-y,-z)] = N
            case "zx":
                new_voxelmap[(-z,-x, y)] = N
    total_time = time() - total_time
    print(f"\r  progress : 100%           \n  time     : {round(total_time,2)}s")
    return new_voxelmap

def transform_vm(voxelmap,matrix):

    correction = np.diag([1,1,-1])
    matrix = correction.dot(matrix).dot(correction)
    inverse_matrix = np.linalg.inv(matrix)

    total_time = time()

    search_max = np.max(matrix,axis=0)
    search_min = np.min(matrix,axis=0)
    search_box = [np.array([x,y,z]).astype(np.int32) for x in range(int(0.5+ search_min[0]-1), int(0.5+ search_max[0]+1)) for y in range(int(0.5+ search_min[1]-1), int(0.5+ search_max[1]+1)) for z in range(int(0.5+ search_min[2]-1), int(0.5+ search_max[2]+1))]

    print(f"mapping by the given linear transformation:")
    tmp = time()
    new_voxelmap={}
    image_xyz=set()

    N_full_block = (1<<(8**3))-1
    count=0
    total = len(voxelmap)
    tmp = time()
    print(f"computing rough shape...",end="")
    timer = 0
    for (x,z,y) in voxelmap:
        if time()-timer>delta_T:
            timer=time()
            print(f"\rcomputing rough shape... {int(count/total*1000+.5)/10}%               ",end="")
        count+=1

        if voxelmap[(x,z,y)]!=0: # dont do anything if N=0 (air)
            transformed_xyz = matrix.dot(np.array([x,y,z])).round().astype(np.int32)
            for delta_xyz in search_box:
                image_xyz.add(tuple(transformed_xyz + delta_xyz))
    print(f"\rcomputing rough shape... DONE ({round(time()-tmp,2)}s)               ")

    # invert matrix and setup arrays:
    tmp = time()
    print(f"computing pre-image coordinates...",end="")
    image_xyz = np.array([list(a) for a in image_xyz])
    coordinate_weights = np.array([64,1,8]).astype(np.int32)
    i=0
    matrix_dot_dxyz_eighths = []
    weights_dot_dxyz = []
    for dx in range(8):
        for dy in range(8):
            for dz in range(8):
                curr_dxyz = np.array([dx,dy,dz])
                matrix_dot_dxyz_eighths.append(inverse_matrix.dot(curr_dxyz)/8)
                weights_dot_dxyz.append(coordinate_weights.dot(curr_dxyz))
                i += 1
    matrix_dot_dxyz_eighths = np.array(matrix_dot_dxyz_eighths)
    weights_dot_dxyz        = np.array(weights_dot_dxyz)

    matrix_dot_image_xyz = inverse_matrix.dot(image_xyz.T).T
    print(f"\rcomputing pre-image coordinates... DONE ({round(time()-tmp,2)}s)               ")

    # voxels tested for recognizing full blocks / air blocks (massive speedup! But might be inaccurate for 'weird' shapes.)
    test_corners = [0,7,56,63,0+64*7,7+64*7,56+64*7,63+64*7]
    others = [i for i in range(512) if i not in test_corners]

    total = image_xyz.shape[0]
    tmp = time()
    print(f"computing exact shape...",end="")
    timer = 0
    for count in range(total):
        if time()-timer>delta_T:
            timer=time()
            print(f"\rcomputing exact shape... {int(count/total*1000+.5)/10}%               ",end="")
        curr_xyz = image_xyz[count]
        x,y,z = curr_xyz

        N = 0
        preimage = matrix_dot_image_xyz[count] + matrix_dot_dxyz_eighths

        pre_xyz  = np.floor(preimage).astype(np.int32)
        pre_dxyz = np.floor(8*(preimage - pre_xyz)).astype(np.int32)
        shift    = (coordinate_weights*pre_dxyz).sum(axis=1).astype(object)
        pre_xyz = pre_xyz.astype(object)

        tally = 0
        for i in test_corners:
            preimage_voxel_value = (1 & (voxelmap.get((pre_xyz[i][0],pre_xyz[i][2],pre_xyz[i][1]),0)>>shift[i]))
            N |= (1<<int(weights_dot_dxyz[i])) * preimage_voxel_value
            tally += preimage_voxel_value
        if tally==0: # assume air block
            new_voxelmap[(x,z,y)] = 0
            continue
        if tally==len(test_corners): # assume full block
            new_voxelmap[(x,z,y)] = -1
            continue
        for i in others:
            preimage_voxel_value = (1 & (voxelmap.get((pre_xyz[i][0],pre_xyz[i][2],pre_xyz[i][1]),0)>>shift[i]))
            N |= (1<<int(weights_dot_dxyz[i])) * preimage_voxel_value

        if N == N_full_block:
            N = -1
        new_voxelmap[(x,z,y)] = N
    print(f"\rcomputing exact shape... DONE ({round(time()-tmp,2)}s)               ")

    total_time = time() - total_time
    print(f"TOTAL TIME (for the linear transform): {round(total_time,2)}s")
    return new_voxelmap



##################################################################################
### TKINTER STUFF ################################################################
##################################################################################

class RedirectText:
    def __init__(self, widget):
        self.widget = widget

    def write(self, text):

        self.widget.configure(state ='normal')
        if len(text)>0:
            if text[0]=='\r':

                last_line_index = self.widget.index("end-1c linestart")  # Start of the last line
                self.widget.delete(last_line_index, "end-1c")  # Delete last line

                text = text[1:]
        self.widget.insert(tk.END, text)
        self.widget.see(tk.END)
        self.widget.configure(state ='disabled')

    def flush(self):
        pass  # Required for compatibility with sys.stdout

class ImageToSchematic:
    def __init__(self, root, fsize=16, button_fsize=8):
        self.root = root
        self.root.title("Image to Schematic")
        self.fsize = fsize
        self.button_fsize = button_fsize
        self.export_process_running = False

        ## Set window size ##
        self.root.geometry("1920x1080") #1280x720 #800x600  # Width x Height in pixels
        self.root.resizable(True, True) # Disable window resizing

        self.font_family = "latin modern roman"


        ## Configure tags and set up fonts ##
        self.font_bold = font.Font(family=self.font_family, weight="bold", size=self.button_fsize)
        self.font_italic = font.Font(family=self.font_family, slant="italic", size=self.button_fsize)
        self.font_underlined = font.Font(family=self.font_family, underline=1, size=self.button_fsize)
        self.font_strikethrough = font.Font(family=self.font_family, overstrike=1, size=self.button_fsize)
        self.font_normal = font.Font(family=self.font_family, size=self.button_fsize)
        self.font_normal_text = font.Font(family=self.font_family, size=self.fsize)

        orig_font = font.nametofont("TkFixedFont")
        self.font_monospaced = font.Font(**orig_font.configure())
        self.font_monospaced.configure(size=self.fsize)


        ### BUTTONS ON LEFT SIDE ##################################################################################################

        ## FRAME ##

        button_frame_side = tk.Frame(root,bg="green")
        button_frame_side.place(relx=0,rely=0,relheight=0.5,relwidth=0.5)
        for i in range(4):
            button_frame_side.columnconfigure(i, weight = 1, uniform="1")
        button_frame_side.rowconfigure(0, weight = 1, uniform="2")
        for i in range(1,4):
            button_frame_side.rowconfigure(i, weight = 3, uniform="2")


        ## CONTENT ##

        # Create sliders to control font size
        slider = tk.Scale(
            button_frame_side,
            from_=4,
            to=48,
            orient="horizontal",
            command=self.update_font_size_button,
        )
        slider.grid(row=0,column=0,sticky="nsew")
        slider.set(self.button_fsize)

        self.default_bg_color = slider.cget("background")

        label = tk.Label(button_frame_side, text="GUI Text",font=self.font_normal,anchor="w")
        label.grid(row=0,column=1,sticky="nsew")

        slider_text = tk.Scale(
            button_frame_side,
            from_=4,
            to=32,
            orient="horizontal",
            command=self.update_font_size_text,
            bg = self.default_bg_color
        )
        slider_text.grid(row=0,column=2,sticky="nsew")
        slider_text.set(self.fsize)

        label = tk.Label(button_frame_side, text="Log Text",font=self.font_normal,anchor="w")
        label.grid(row=0,column=3,sticky="nsew")

        # Create import image button
        import_hm_button = tk.Button(button_frame_side, font=self.font_bold, text="Import Heightmap", command=self.import_image, bg="light goldenrod", fg="red")
        import_hm_button.grid(row=1,column=0,columnspan=2,sticky="nsew")

        # Create import obj button
        import_obj_button = tk.Button(button_frame_side, font=self.font_bold, text="Import .obj", command=self.import_obj, bg="light goldenrod", fg="red")
        import_obj_button.grid(row=1,column=2,columnspan=2,sticky="nsew")

        # Create flip/rotate buttons
        flipLR_button = tk.Button(button_frame_side, font=self.font_bold, text="\u2194", command=self.flipLR)
        flipLR_button.grid(row=2,column=0,sticky="nsew")

        flipUD_button = tk.Button(button_frame_side, font=self.font_bold, text="\u2195", command=self.flipUD)
        flipUD_button.grid(row=2,column=1,sticky="nsew")

        rotate_pos90_button = tk.Button(button_frame_side, font=self.font_bold, text="90°", command=self.rotate_pos90)
        rotate_pos90_button.grid(row=2,column=2,sticky="nsew")

        rotate_neg90_button = tk.Button(button_frame_side, font=self.font_bold, text="-90°", command=self.rotate_neg90)
        rotate_neg90_button.grid(row=2,column=3,sticky="nsew")

        # Dropdown menu to select facing and invert button
        facing_col = "blue"
        self.facing = tk.StringVar()
        self.facing.set("Down")  # Default value
        predefined_values = ["Down","Up","North","East","South","West"]
        param_label = tk.Label(button_frame_side, font=self.font_bold, text="Facing ")
        param_label.grid(row=3,column=0,sticky="nsew")
        param_menu = tk.OptionMenu(button_frame_side, self.facing, *predefined_values, command=self.set_parameter_facing)
        param_menu.grid(row=3,column=1,sticky="nsew")
        param_menu.config(font=self.font_bold,fg=facing_col)
        menu = self.root.nametowidget(param_menu.menuname)  # Get menu widget.
        menu.config(font=self.font_bold,fg=facing_col)  # Set the dropdown menu's font

        self.orientation_setting = [flipLR_button, flipUD_button, rotate_pos90_button, rotate_neg90_button, param_label, param_menu]

        #invert_button = tk.Button(self.root, font=self.font_bold, text="Invert", command=self.invert)
        #invert_button.place(relx=1/4,rely=7/20,relheight=3/20,relwidth=1/4)
        invert_button = tk.Button(button_frame_side, font=self.font_bold, text="Invert", command=self.invert)
        invert_button.grid(row=3, column=2, sticky="nsew")

        matrix_button = tk.Button(button_frame_side, font=self.font_bold, text="Matrix", command=self.set_trafo_mat)
        matrix_button.grid(row=3, column=3, sticky="nsew")


        ### HEIGHTMAP DISPLAY #####################################################################################################

        ## FRAME ##
        small_weight = 1
        big_weight = 3
        image_frame = tk.Frame(root, bg="gray")
        image_frame.place(relx=0,rely=0.5,relheight=0.5,relwidth=0.5)
        for i in [1,3]:
            image_frame.columnconfigure(i, weight = big_weight, uniform="3")
            image_frame.rowconfigure(i, weight = big_weight, uniform="4")
        for i in [0,2,4]:
            image_frame.columnconfigure(i, weight = small_weight, uniform="3")
            image_frame.rowconfigure(i, weight = small_weight, uniform="4")

        ## CONTENT ##

        # Create N/S/E/W/U/D direction labels
        NSEWUPcol = "blue"
        self.labelU = tk.Label(image_frame, text="N",font=self.font_bold, fg=NSEWUPcol)
        self.labelU.grid(row=0,column=2,sticky="nsew")
        self.labelD = tk.Label(image_frame, text="S",font=self.font_bold, fg=NSEWUPcol)
        self.labelD.grid(row=4,column=2,sticky="nsew")
        self.labelL = tk.Label(image_frame, text="W",font=self.font_bold, fg=NSEWUPcol)
        self.labelL.grid(row=2,column=0,sticky="nsew")
        self.labelR = tk.Label(image_frame, text="E",font=self.font_bold, fg=NSEWUPcol)
        self.labelR.grid(row=2,column=4,sticky="nsew")

        #Create canvas to "plt.imshow" the imported image
        self.heightmap_image = None
        self.heightmap_image_original = None
        self.fig, self.ax = plt.subplots()
        plt.axis('off')
        self.ax.text(0.5, 0.5, "No Image Loaded", ha='center', va='center', fontsize=12)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        small_weight, big_weight = small_weight/(2*(2*big_weight+3*small_weight)), big_weight/(2*(2*big_weight+3*small_weight))
        self.canvas.get_tk_widget().place(relx=small_weight,rely=0.5+small_weight,relheight=2*big_weight+small_weight,relwidth=2*big_weight+small_weight)


        ### BUTTONS ON RIGHT SIDE #################################################################################################

        ## FRAME ##

        button_frame = tk.Frame(root,bg="gray")
        button_frame.place(relx=0.5,rely=0,relheight=0.5,relwidth=0.5)
        for i in range(2):
            button_frame.columnconfigure(i, weight = 1, uniform="4")
        for i in range(6):
            button_frame.rowconfigure(i, weight = 1, uniform="5")

        ## CONTENT ##

        scale_col = "green"
        self.scale = tk.StringVar()
        self.scale.set("8")  # Default value
        predefined_values_scale = ["8", "4", "2", "1"]
        self.param_label_scale = tk.Label(button_frame, font=self.font_bold, text="Pixels per block")
        self.param_label_scale.grid(row=0,column=0,sticky="nsew")
        self.param_menu_scale = tk.OptionMenu(button_frame, self.scale, *predefined_values_scale, command=self.set_parameter_scale)
        self.param_menu_scale.grid(row=0,column=1,sticky="nsew")
        self.param_menu_scale.config(font=self.font_bold,fg=scale_col)
        menu_scale = self.root.nametowidget(self.param_menu_scale.menuname)  # Get menu widget.
        menu_scale.config(font=self.font_bold,fg=scale_col)  # Set the dropdown menu's font

        self.param_label_depth = tk.Label(button_frame, font=self.font_bold, text="Depth Multiplier")
        self.param_label_depth.grid(row=1,column=0,sticky="nsew")

        self.yscale = tk.StringVar()
        self.yscale.set("1/32")
        yscale_entry = tk.Entry(button_frame, textvariable=self.yscale,font=self.font_bold)
        yscale_entry.grid(row=1,column=1,sticky="nsew")
        self.yscale.trace_add("write", lambda *args: self.set_parameter_yscale())

        # display of dimensions
        self.dim_label = tk.Label(button_frame, font=self.font_bold, text="Size: ???")
        self.dim_label.grid(row=2,column=0,columnspan=2,sticky="nsew")

        param_label_schem = tk.Label(button_frame, font=self.font_bold, text="Schematic name")
        param_label_schem.grid(row=3,column=0,sticky="nsew")

        self.out_filename = tk.StringVar()
        self.out_filename.set("???")
        filename_entry = tk.Entry(button_frame, textvariable=self.out_filename,font=self.font_bold)
        filename_entry.grid(row=3,column=1,sticky="nsew")

        param_label_material = tk.Label(button_frame, font=self.font_bold, text="Material")
        param_label_material.grid(row=4,column=0,sticky="nsew")

        self.material = tk.StringVar()
        self.material.set("limestone")
        material_entry = tk.Entry(button_frame, textvariable=self.material,font=self.font_bold)
        material_entry.grid(row=4,column=1,sticky="nsew")

        # set-allowed-blocks button
        set_allowed_button = tk.Button(button_frame, font=self.font_bold, text="Block Shapes", command=self.set_allowed_blocks)
        set_allowed_button.grid(row=5, column=0, sticky="nsew")

        # Create export schematic button
        export_button = tk.Button(button_frame, font=self.font_bold, text="Export Schematic", command=self.run_export_schematic, bg="light goldenrod", fg="red")
        export_button.grid(row=5, column=1, sticky="nsew")

        ### TEXT OUTPUT ###########################################################################################################

        self.output_box = scrolledtext.ScrolledText(root, font=self.font_monospaced, fg="white", bg="#382c1d")
        self.output_box.place(relx=0.5,rely=0.5,relheight=0.5,relwidth=0.5)
        sys.stdout = RedirectText(self.output_box)

        self.output_box.configure(state ='disabled')

        self.facing_permutation = (0,1,2,3,4,5)
        self.face_permutation   = (0,1,2,3,4,5)

        self.voxelmap = None
        self.mesh = None
        self.inverted = False
        self.export_thread = None
        self.yscale_float = 1/32 # has to be the same value as the default value of self.yscale !
        self.mesh_shape = None
        self.dx_dy_dz = None

        self.matrix = None

        self.block_shapes = {"_slab"                :["Slab"             , {'none':"disable",'all':"eighths",'246':"quarters",'4':"halfs"}],
                             "_vertical_slab"       :["Vertical Slab"    , {'none':"disable",'all':"eighths",'234':"quarters",'3':"halfs"}],
                             "_stairs"              :["Stairs"           , {'none':"disable",'all':"enable"}             ],
                             "_vertical_corner"     :["Vertical Corner"  , {'none':"disable",'all':"eighths",'234':"quarters",'3':"halfs"}],
                             "_quarter_slab"        :["Quarter"          , {'none':"disable",'all':               "quarters",'2':"halfs"}          ],
                             "_vertical_quarter"    :["Vertical Quarter" , {'none':"disable",'all':"eighths",'234':"quarters",'3':"halfs"}],
                             "_corner_slab"         :["Corner Slab"      , {'none':"disable",'all':"enable"}             ],
                             "_vertical_corner_slab":["Vert. Corner Slab", {'none':"disable",'all':"enable"}             ],
                             "_eighth_slab"         :["Eighth Slab"      , {'none':"disable",'all':"enable"}             ] }

        self.block_shape_variables = {name:tk.StringVar(value='all') for name in self.block_shapes}

    def update_font_size_button(self,value):
        self.font_bold.configure(size=int(value))
        self.font_italic.configure(size=int(value))
        self.font_underlined.configure(size=int(value))
        self.font_strikethrough.configure(size=int(value))
        self.font_normal.configure(size=int(value))

    def update_font_size_text(self,value):
        self.font_normal_text.configure(size=int(value))
        self.font_monospaced.configure(size=int(value))

    def display_image(self):
        self.ax.clear()
        if type(self.heightmap_image)!=type(None):
            self.ax.imshow(self.heightmap_image, cmap='gray')
        else:
            self.ax.text(0.5, 0.5, "No Image Loaded", ha='center', va='center', fontsize=12)
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        plt.axis('off')
        self.canvas.draw()
        self.update_xyz()

    def import_image(self):
        file_path = filedialog.askopenfilename(title="Select Heightmap Image", filetypes=[("Image Files", ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tiff")), ("All Files", "*.*")])
        if file_path:
            image = Image.open(file_path)
            self.heightmap_image = np.array(image).astype(float)
            if len(self.heightmap_image.shape)>2:
                if self.heightmap_image.shape[2]==3 or self.heightmap_image.shape[2]==4:
                    self.heightmap_image = (self.heightmap_image[:,:,0] + self.heightmap_image[:,:,1] + self.heightmap_image[:,:,2])/3
                else:
                    print("\nWARNING: unexpected file format! (needs to be RGB or Grayscale)\n")
                    return
            self.heightmap_image_original=self.heightmap_image.copy()
            self.facing_permutation = facing_perms[self.facing.get()]
            self.face_permutation = (0,1,2,3,4,5)
            filename = file_path[file_path.rfind('/')+1:file_path.rfind('.')]
            print(f"\nheightmap image '{file_path[file_path.rfind('/')+1:]}' loaded")
            self.out_filename.set("schem_"+filename)
            self.voxelmap = None
            self.inverted = False
            self.mesh = None
            self.yscale.set("1/32")

            #!!!

            self.param_label_scale.configure(text="Pixels per block", bg=self.default_bg_color)

            menu = self.param_menu_scale["menu"]
            menu.delete(0, "end")
            for option in ["8","4","2","1"]:
                menu.add_command(label=option, command=lambda opt=option:self.set_parameter_scale(opt))

            self.scale.set("8")

            self.display_image()
            self.param_label_depth.configure(text="Depth Multiplier")


    def import_obj(self):
        file_path = filedialog.askopenfilename(title="Select .OBJ File", filetypes=[("obj File", ("*.obj",)), ("All Files", "*.*")])
        if file_path:
            #### get mesh from file: ####
            self.mesh = trimesh.load(file_path, force='mesh')

            #### calculate a heightmap preview image: ####
            base_resolution=96
            # Project to XZ-plane (assuming height is Y)
            coords = self.mesh.vertices[:, [0, 2]]  # X and Z
            min_xy = coords.min(axis=0)
            max_xy = coords.max(axis=0)
            size = max_xy - min_xy
            aspect_ratio = size / size.max()
            base_resolution = base_resolution/(aspect_ratio[0]*aspect_ratio[1])**.5
            # Determine resolution while keeping aspect ratio
            res_x = max(1,int(base_resolution * aspect_ratio[0]))
            res_z = max(1,int(base_resolution * aspect_ratio[1]))
            # Create 2D grid in XZ
            x = np.linspace(min_xy[0], max_xy[0], res_x)
            z = np.linspace(min_xy[1], max_xy[1], res_z)
            grid_x, grid_z = np.meshgrid(x, z)
            grid_points = np.c_[grid_x.ravel(), grid_z.ravel()]
            # Append y ray origin for downward raycasting
            ray_origins = np.c_[grid_points[:, 0], np.full(len(grid_points), max(self.mesh.vertices[:, 1]) + 1), grid_points[:, 1]]
            ray_directions = np.tile([0, -1, 0], (len(ray_origins), 1))  # shoot rays downward
            # Ray-mesh intersection
            locations, index_ray, _ = self.mesh.ray.intersects_location(ray_origins=ray_origins, ray_directions=ray_directions, multiple_hits=False)
            # Initialize heightmap with 0
            heightmap = np.full((res_z * res_x),0.0)
            # Set heights at hit points
            heightmap[index_ray] = locations[:, 1]-min(self.mesh.vertices[:, 1])  # use Y-coordinate as height

            #### (re)set heightmap + permutations and display: ####
            self.heightmap_image = heightmap.reshape((res_z, res_x))
            self.heightmap_image_original=self.heightmap_image.copy()

            self.facing_permutation = facing_perms[self.facing.get()]
            self.face_permutation = (0,1,2,3,4,5)
            filename = file_path[file_path.rfind('/')+1:file_path.rfind('.')]
            print(f"\nobj file '{file_path[file_path.rfind('/')+1:]}' loaded")
            self.out_filename.set("schem_"+filename)
            self.voxelmap = None
            self.inverted = False

            self.mesh_shape = self.mesh.vertices.max(axis=0) - self.mesh.vertices.min(axis=0)
            self.param_label_depth.configure(text="Resolution (blocks)")
            self.yscale.set("64")

            #!!!

            self.param_label_scale.configure(text="Ray direction", bg=self.default_bg_color)

            menu = self.param_menu_scale["menu"]
            menu.delete(0, "end")
            for option in ["auto","x","y","z","fastest","all"]:
                menu.add_command(label=option, command=lambda opt=option:self.set_parameter_ray_dir(opt))

            self.scale.set("auto")

            # also sets up self.dx_dy_dz:
            self.display_image()



    def export_schematic(self):
        if type(self.heightmap_image)!=type(None):
            filename = self.out_filename.get()
            material = self.material.get()
            permutation = concat_perms(self.face_permutation,self.facing_permutation)

            print()

            if self.voxelmap==None:
                if type(self.mesh)!=type(None): # use 3D obj
                    self.voxelmap = voxelmap_from_mesh(self.mesh, self.dx_dy_dz, self.inverted, self.scale.get())
                else: # use heightmap image
                    self.voxelmap = voxelmap_from_array(np.rot90(self.heightmap_image_original,-1), self.yscale_float, int(self.scale.get()))

            tmp_voxelmap = self.voxelmap

            if self.inverted and type(self.mesh)==type(None):
                tmp_voxelmap = inverted_vm(tmp_voxelmap)
                permutation = concat_perms((0, 4, 2, 3, 1, 5), permutation)

            #print(f"permutation: {permutation}")

            if type(self.matrix)!=type(None):
                tmp_voxelmap = transform_vm(tmp_voxelmap, self.matrix)
            else:
                flips = find_flips[permutation]

                if len(flips)==2:
                    print("rotating voxelmap (by flipping 2 times) ...")
                elif len(flips)==3:
                    print("rotating and flipping voxelmap (by flipping 3 times) ...")
                for direction in flips:
                    tmp_voxelmap = flip(tmp_voxelmap,direction)

            allowed_block_shapes = []
            for block_model in block_models:
                id = block_model[1]
                properties = block_model[2]
                if id not in self.block_shape_variables:
                    allowed_block_shapes.append(block_model)
                    continue
                allowed = self.block_shape_variables[id].get()
                if allowed=='none':
                    continue
                if allowed=='all':
                    allowed_block_shapes.append(block_model)
                    continue

                ind = properties.find("layer=")
                if ind==-1:
                    ind = properties.find("layers=")
                    if ind==-1:
                        print("ERROR with getting block shapes!")
                        continue
                    layer_number = properties[ind+7]
                else:
                    layer_number = properties[ind+6]

                if layer_number in allowed:
                    allowed_block_shapes.append(block_model)
            print(f"{len(allowed_block_shapes)} out of {len(block_models)} block shapes are used.")

            schematic_from_voxelmap(tmp_voxelmap, filename, material, allowed_block_shapes)

            print("DONE!\n")
        else:
            print("ERROR: no input file was loaded!")
        self.export_process_running = False

    def run_export_schematic(self):
        if not self.export_process_running:
            self.export_process_running = True
            self.export_thread = threading.Thread(target=self.export_schematic)
            self.export_thread.start()

    def flipLR(self):
        if type(self.heightmap_image)!=type(None) and type(self.matrix)==type(None):
            self.heightmap_image = np.fliplr(self.heightmap_image)
            self.display_image()
            self.face_permutation = concat_perms(self.face_permutation,(3,1,2,0,4,5))

    def flipUD(self):
        if type(self.heightmap_image)!=type(None) and type(self.matrix)==type(None):
            self.heightmap_image = np.flipud(self.heightmap_image)
            self.display_image()
            self.face_permutation = concat_perms(self.face_permutation,(0,1,5,3,4,2))

    def rotate_pos90(self):
        if type(self.heightmap_image)!=type(None) and type(self.matrix)==type(None):
            self.heightmap_image = np.rot90(self.heightmap_image)
            self.display_image()
            self.face_permutation = concat_perms(self.face_permutation,(5,1,0,2,4,3))

    def rotate_neg90(self):
        if type(self.heightmap_image)!=type(None) and type(self.matrix)==type(None):
            self.heightmap_image = np.rot90(self.heightmap_image,-1)
            self.display_image()
            self.face_permutation = concat_perms(self.face_permutation,(2,1,3,5,4,0))

    def invert(self):
        if type(self.heightmap_image)!=type(None):
            self.heightmap_image = self.heightmap_image.max()-self.heightmap_image
            self.display_image()
            self.inverted = not self.inverted
            if type(self.mesh)!=type(None):
                self.voxelmap = None

    def set_trafo_mat(self):
        #def open_matrix_popup():
        if type(self.heightmap_image)!=type(None):
            popup = tk.Toplevel()
            popup.title("Enter Transformation Matrix")

            entries = []

            default_matrix = self.matrix
            if type(self.matrix)==type(None):
                default_matrix = np.array([[1,0,0],[0,1,0],[0,0,1]])

            for i in range(3):
                row = []
                for j in range(3):
                    e = tk.Entry(popup, width=5)
                    e.grid(row=i, column=j, padx=5, pady=5)
                    if int(default_matrix[i][j])==default_matrix[i][j]:
                        e.insert(0, str(int(default_matrix[i][j])))  # Set default value
                    else:
                        e.insert(0, str(default_matrix[i][j]))  # Set default value
                    row.append(e)
                entries.append(row)

            def on_ok():
                try:
                    matrix = np.array([[float(e.get()) for e in row] for row in entries])
                    if abs(np.linalg.det(matrix))<1.0e-15:
                        print("Determinant is zero, please enter proper transformation matrix!")
                        return
                    print("Transformation matrix was set to:")
                    print(matrix)
                    print("This matrix will be used INSTEAD of the orientation set with the buttons on the left!")
                    print("(Discard matrix by pressing the 'Matrix' button again.)")

                    self.matrix = matrix
                    # reset image
                    self.heightmap_image = self.heightmap_image_original.copy()
                    if self.inverted:
                        self.heightmap_image = self.heightmap_image.max()-self.heightmap_image
                    self.facing_permutation = (0,1,2,3,4,5)
                    self.face_permutation   = (0,1,2,3,4,5)
                    self.facing.set("Down")
                    self.set_parameter_facing("Down")
                    self.display_image()

                    # gray out buttons:
                    for thing in self.orientation_setting:
                        thing.configure(state="disabled", bg=self.default_bg_color)

                    popup.destroy()
                except ValueError:
                    print("Invalid input")

            def on_cancel():
                print("Discarded matrix, you can use the buttons on the left to set the orientation!")
                self.matrix = None
                for thing in self.orientation_setting:
                    thing.configure(state="normal", bg=self.default_bg_color)
                popup.destroy()

            tk.Button(popup, text="Set Matrix", command=on_ok).grid(row=3, column=0, columnspan=3, pady=10)
            tk.Button(popup, text="Discard Matrix", command=on_cancel).grid(row=4, column=0, columnspan=3, pady=10)

    def set_allowed_blocks(self):
        popup = tk.Toplevel()
        popup.title("Set Allowed Block Shapes")

        for i in range(1,6):
            popup.columnconfigure(i, weight = 1, uniform="Silent_Creme")

        tk.Label(popup, text="Disable or set resolution for the different block shapes:").grid(row=0, column=0, columnspan=6, pady=5, padx=5, sticky="w")
        nrow=1
        for id in self.block_shapes:
            name=self.block_shapes[id][0]
            tk.Label(popup, text=name).grid(row=nrow, column=0, pady=5, padx=5, sticky="e")

            ncol=1
            name_val_pairs = self.block_shapes[id][1]
            for val in name_val_pairs:
                str_val = name_val_pairs[val]
                rb = tk.Radiobutton(popup, text=str_val, variable=self.block_shape_variables[id], value=val)
                if name=="Quarter" and ncol==2:
                    ncol=3
                if str_val=='enable':
                    rb.grid(row=nrow, column=ncol, columnspan=3, padx=5, pady=5, sticky="w")
                else:
                    rb.grid(row=nrow, column=ncol, padx=5, pady=5, sticky="w")
                ncol+=1
            nrow+=1

        def on_ok():
            popup.destroy()

        tk.Button(popup, text="Ok", command=on_ok).grid(row=nrow, column=0, columnspan=6, pady=10, sticky="we")

    def set_parameter_scale(self,scale):
        self.scale.set(scale)
        print(f"one block = {scale} pixels in the image")
        self.update_xyz()
        self.voxelmap=None

    def set_parameter_ray_dir(self,dir):
        self.scale.set(dir)
        if dir in ["x","y","z"]:
            print(f"Ray-tracing will be done in {dir}-direction.")
        elif dir=="auto":
            print(f"Ray-tracing direction(s) will be chosen automatically.")
        elif dir=="fastest":
            print(f"Ray-tracing will be done in the direction that requires the fewest rays.")
        elif dir=="all":
            print(f"Ray-tracing will be done in all 3 directions for error correction. (useful if model has 'holes')")
        else:
            raise IOError("invalid direction parameter!")
        self.voxelmap=None


    def set_parameter_yscale(self):
        self.update_xyz()
        self.voxelmap=None

    def update_xyz(self):
        if type(self.heightmap_image)!=type(None):

            yscale, success = safe_eval(0.2, self.yscale.get())

            if success==True:
                self.yscale_float = float(yscale)

                if type(self.mesh)==type(None):
                    dx,dz = self.heightmap_image.shape
                    dx //= int(self.scale.get())
                    dz //= int(self.scale.get())
                    dy = int(max(((self.heightmap_image.max()-self.heightmap_image.min())*self.yscale_float)//int(self.scale.get()),1))
                else:
                    scale_factor = self.yscale_float/float(max(self.mesh_shape))

                    dx = max(1,int(0.5 + self.mesh_shape[0] * scale_factor))
                    dy = max(1,int(0.5 + self.mesh_shape[1] * scale_factor))
                    dz = max(1,int(0.5 + self.mesh_shape[2] * scale_factor))

                self.dx_dy_dz   = (dx,dy,dz)

                self.dim_label.config(text=f"Size: {dx}x{dz}x{dy} ({dx*dy*dz} blocks)")

            elif success==False:
                self.yscale.set("STOP IT!")
            else: # success==None: Syntax Error, do nothing
                pass

    def set_parameter_facing(self,facing):
        match facing:
            case "Up":
                self.labelU.config(text="S")
                self.labelR.config(text="E")
                self.labelD.config(text="N")
                self.labelL.config(text="W")
            case "Down":
                self.labelU.config(text="N")
                self.labelR.config(text="E")
                self.labelD.config(text="S")
                self.labelL.config(text="W")
            case "North":
                self.labelU.config(text="U")
                self.labelR.config(text="E")
                self.labelD.config(text="D")
                self.labelL.config(text="W")
            case "East":
                self.labelU.config(text="U")
                self.labelR.config(text="S")
                self.labelD.config(text="D")
                self.labelL.config(text="N")
            case "South":
                self.labelU.config(text="U")
                self.labelR.config(text="W")
                self.labelD.config(text="D")
                self.labelL.config(text="E")
            case "West":
                self.labelU.config(text="U")
                self.labelR.config(text="N")
                self.labelD.config(text="D")
                self.labelL.config(text="S")

        self.facing_permutation = facing_perms[self.facing.get()]




if __name__ == "__main__":
    root = tk.Tk()
    #app = ImageToSchematic(root,16,24)
    app = ImageToSchematic(root,10,16)
    root.mainloop()
