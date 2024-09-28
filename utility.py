from vvrpywork.constants import Key, Mouse, Color
from vvrpywork.scene import Scene3D, get_rotation_matrix, world_space
from vvrpywork.shapes import (
    Point3D, Line3D, Arrow3D, Sphere3D, Cuboid3D, Cuboid3DGeneralized,
    PointSet3D, LineSet3D, Mesh3D
)

from matplotlib import colormaps as cm
from colour import Color as clr
import numpy as np
import scipy
from scipy import sparse
import time
import open3d as o3d

import struct

def floor_gauss(gauss):
    # The result of some truly unprofessional trial and error.
    max = np.max(gauss)
    avg = np.average(gauss)
    floor_factor = avg**2/max
    floored = np.copy(gauss)
    
    for i in range(0, len(gauss)):
        if floored[i] < floor_factor:
            floored[i] = floor_factor
            
    return floored

def verticise_triangles(mesh:Mesh3D):
    # Transforms the mesh.triangles elements from lists of indices to lists of vertex positions.
    # mesh: a Mesh3D object, as defined in vrpywork.
    
    triangles = mesh.triangles
    vertices = mesh.vertices
    
    v_triangles = np.array(vertices[triangles])
    
    return v_triangles

def triangle_angles(mesh:Mesh3D):
    # Calculates the angles of the component triangles.
    # mesh: a Mesh3D object, as defined in vrpywork.
    
    triangles = verticise_triangles(mesh)
    
    # Initialise vectors corresponding to the triangle edges
    v1 = triangles[:,0] - triangles[:,1]
    v2 = triangles[:,1] - triangles[:,2]
    v3 = triangles[:,2] - triangles[:,0]
    
    # Extract unit vectors
    v1_hat = v1/(v1**2).sum()**0.5
    v2_hat = v2/(v2**2).sum()**0.5
    v3_hat = v3/(v3**2).sum()**0.5
    
    # Initialise angle array.
    angles = np.zeros((len(triangles), 3))
    magv1v2 = np.array([np.linalg.norm(v1[i]) * np.linalg.norm(v2[i]) for i in range(0, len(v1))])
    magv2v3 = np.array([np.linalg.norm(v2[i]) * np.linalg.norm(v3[i]) for i in range(0, len(v2))])
    magv3v1 = np.array([np.linalg.norm(v3[i]) * np.linalg.norm(v1[i]) for i in range(0, len(v3))])
    
    #Calculate angles with arccos
    #####angles[:,0] = np.arccos(np.clip(np.dot(v1_hat, v2_hat), -1.0, 1.0))
    ##angles[:,0] = np.arccos(np.clip(np.einsum('ij,ij->i', v1_hat, v2_hat)/, -1.0, 1.0))
    #####angles[:,1] = np.arccos(np.clip(np.dot(v2_hat, v3_hat), -1.0, 1.0))
    ##angles[:,1] = np.arccos(np.clip(np.einsum('ij,ij->i', v2_hat, v3_hat), -1.0, 1.0))
    #####angles[:,2] = np.arccos(np.clip(np.dot(v3_hat, v1_hat), -1.0, 1.0))
    ##angles[:,2] = np.arccos(np.clip(np.einsum('ij,ij->i', v3_hat, v1_hat), -1.0, 1.0))
    
    ###angles[:,0] = np.arccos(np.clip(np.dot(v1_hat, v2_hat), -1.0, 1.0))
    angles[:,0] = np.arccos(np.clip(np.einsum('ij,ij->i', v1, v2)/magv1v2, -1.0, 1.0))
    ###angles[:,1] = np.arccos(np.clip(np.dot(v2_hat, v3_hat), -1.0, 1.0))
    angles[:,1] = np.arccos(np.clip(np.einsum('ij,ij->i', v2, v3)/magv2v3, -1.0, 1.0))
    ###angles[:,2] = np.arccos(np.clip(np.dot(v3_hat, v1_hat), -1.0, 1.0))
    angles[:,2] = np.arccos(np.clip(np.einsum('ij,ij->i', v3, v1)/magv3v1, -1.0, 1.0))
    
    return angles

def triangle_areas(mesh:Mesh3D):
    # Calculates the area of each component triangle.
    # mesh: a Mesh3D object, as defined in vrpywork.
    
    triangles = verticise_triangles(mesh)
    
    # Prepare formula variables for code readability
    x0 = triangles[:,0,0]
    x1 = triangles[:,1,0]
    x2 = triangles[:,2,0]
    
    y0 = triangles[:,0,1]
    y1 = triangles[:,1,1]
    y2 = triangles[:,2,1]
    
    z0 = triangles[:,0,2]
    z1 = triangles[:,1,2]
    z2 = triangles[:,2,2]
    
    x01 = x1 - x0
    x02 = x2 - x0
    y01 = y1 - y0
    y02 = y2 - y0
    z01 = z1 - z0
    z02 = z2 - z0
    
    # Initialise area array.
    areas = (np.sqrt((y01*z02 - z01*y02)**2 + (z01*x02 - x01*z02)**2 + (x01*y02 - y01*x02)**2))/2
    
    return areas

def Gauss_curvature(mesh:Mesh3D):
    # Calculates the discrete gaussian curvature for each vertex of the mesh.
    # mesh: a Mesh3D object, as defined in vrpywork.
    
    vertices = mesh.vertices
    triangles = mesh.triangles 
    angles = triangle_angles(mesh)
    print("================================================\nAngles maximum {} and minimum {}\n=====================".format(np.max(angles),np.min(angles)))
    areas = triangle_areas(mesh)
    print("================================================\nAreas maximum {} and minimum {}\n=====================".format(np.max(areas),np.min(areas)))
    ###is_in_angle = [[0,2],[0,1],[1,2]]
    
    # Calculate the sum of all angles formed on each vertex and the areas of all triangles it is a component of. A traditional "for" is the only way this was made possible.
    area_totals = np.zeros(len(vertices))
    angle_totals = np.zeros(len(vertices))
    
    # Preparation for readability. The array contains the increase in each triangle's vertices' total angles due to participation in the triangle formation.
    angles_aligned = angles[:, [2,0,1]]
    
    for i in range(0, len(triangles)):
        area_totals[triangles[i]] += areas[i]
        angle_totals[triangles[i]] += angles_aligned[i]
    
    # Initialise curvature per vertex array. The numberer is divided by the barycentric area of each vertex, as defined by Nira Dyn, Kai Hormann, Sun-Jeong Kim, and David Levin
    # in "Optimizing 3D Triangulations Using Discrete Curvature Analysis"
    ###gauss_curvature = (np.ones(len(vertices)) * (2*np.pi) - angle_totals) / ((area_totals/3)+10**(-9)) #DIVISION BY ZERO ENCOUNTERED
    gauss_curvature = (np.ones(len(vertices)) * (2*np.pi) - angle_totals) / (area_totals/3)
    
    return np.abs(np.array(gauss_curvature))
    

def normal_angles(normals, smoothed_normals):
    
    return(angles)