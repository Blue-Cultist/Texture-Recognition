from vvrpywork.constants import Key, Mouse, Color
from vvrpywork.scene import Scene3D, get_rotation_matrix, world_space
from vvrpywork.shapes import (
    Point3D, Line3D, Arrow3D, Sphere3D, Cuboid3D, Cuboid3DGeneralized,
    PointSet3D, LineSet3D, Mesh3D
)

from matplotlib import colormaps as cm
import numpy as np
import scipy
from scipy import sparse
import time
import open3d as o3d

import struct
import utility as util


WIDTH = 1000
HEIGHT = 800
MESH_NAME = "1.ply"
CURRENT_MESH = 1

class Texture_Detection(Scene3D):
    def __init__(self):
        super().__init__(WIDTH, HEIGHT, "Final project", output=True, n_sliders=2) # :3
        self.reset_mesh()
        self.reset_sliders()
        self.printHelp()

    def reset_mesh(self):
        # Choose mesh
        # self.mesh = Mesh3D.create_bunny(color=Color.GRAY)
        self.mesh = Mesh3D("models/{}".format(MESH_NAME), color=Color.GRAY)
        # self.mesh = Mesh3D("resources/bun_zipper_res2.ply", color=Color.GRAY)
        # self.mesh = Mesh3D("resources/dragon_low_low.obj", color=Color.GRAY)

        self.mesh.remove_degenerate_triangles()
        self.mesh.remove_duplicated_vertices()
        self.mesh.remove_degenerate_triangles()
        self.mesh.remove_unreferenced_vertices()
        vertices = self.mesh.vertices
        vertices -= np.mean(vertices, axis=0)
        distanceSq = (vertices ** 2).sum(axis=-1)
        max_dist = np.sqrt(np.max(distanceSq))
        self.mesh.vertices = vertices / max_dist
        self.removeShape("mesh")
        self.addShape(self.mesh, "mesh")

        self.wireframe = LineSet3D.create_from_mesh(self.mesh)
        self.removeShape("wireframe")
        self.addShape(self.wireframe, "wireframe")
        self.show_wireframe = True

        self.eigenvectors = None
        self.eigenvector_idx = 0

        # more ....
        self.vertex_adjacency_lists = find_all_adjacent_vertices(self.mesh)

    def reset_sliders(self):
        self.set_slider_value(0, 0)
        self.set_slider_value(1, 0.5)

        
    @world_space
    def on_mouse_press(self, x, y, z, button, modifiers):
        if button == Mouse.MOUSELEFT and modifiers & Key.MOD_SHIFT:
            if np.isinf(z):
                return
            
            self.selected_vertex = find_closest_vertex(self.mesh, (x, y, z))

            vc = self.mesh.vertex_colors
            vc[self.selected_vertex] = (1, 0, 0)
            self.mesh.vertex_colors = vc
            self.updateShape("mesh", True)

    def on_key_press(self, symbol, modifiers):

        if symbol == Key.R:
            self.reset_mesh()

        if symbol == Key.W:
            if self.show_wireframe:
                self.removeShape("wireframe")
                self.show_wireframe = False
            else:
                self.addShape(self.wireframe, "wireframe")
                self.show_wireframe = True
                
        if symbol == Key.A and hasattr(self, "selected_vertex"):

            if hasattr(self, "vertex_adjacency_lists"):
                adj =  self.vertex_adjacency_lists[self.selected_vertex]
            else:
                adj = find_adjacent_vertices(self.mesh, self.selected_vertex)

            colors = self.mesh.vertex_colors
            for idx in adj:
                colors[idx] = (0, 0, 1)
            self.mesh.vertex_colors = colors
            self.updateShape("mesh")

        if symbol == Key.D and not modifiers & Key.MOD_CTRL and hasattr(self, "selected_vertex"):
            d = delta_coordinates_single(self.mesh, self.selected_vertex)
            # self.print(d)
            self.print(np.sqrt(np.sum(d**2)))

        if symbol == Key.D and modifiers & Key.MOD_CTRL:
            start = time.time()

            # ...

            self.print(f"Took {(time.time() - start):.3f} seconds.")

            # self.display_delta_coords(...)

        if symbol == Key.L:
            start = time.time()
            d_coords = delta_coordinates(self.mesh)
            self.print(f"Took {(time.time() - start):.3f} seconds.")
            self.display_delta_coords(d_coords)

        if symbol == Key.S:
            start = time.time()
            d_coords = delta_coordinates_sparse(self.mesh)
            self.print(f"Took {(time.time() - start):.3f} seconds.")
            ###print(np.shape(d_coords))
            self.display_delta_coords(d_coords)

        if symbol == Key.E:
            _, vecs = eigendecomposition_full(self.mesh)
            self.eigenvectors = vecs
            self.display_eigenvector(vecs[:, self.eigenvector_idx])

        if symbol == Key.B:
            self.mesh = reconstruct(self.mesh, self.percent)
            self.updateShape("mesh")

            self.wireframe.points = self.mesh.vertices
            self.updateShape("wireframe")

        if symbol == Key.C:
            self.mesh = reconstruct2(self.mesh, self.percent, 5)
            self.updateShape("mesh")

            self.wireframe.points = self.mesh.vertices
            self.updateShape("wireframe")
        
        if symbol == Key.M:
            Taubin_recursive(self.mesh, 50, 0.2, -0.105)
            self.updateShape("mesh")
            
            self.wireframe.points = self.mesh.vertices
            self.updateShape("wireframe")
        
        if symbol == Key.Q:
            Laplace_recursive(self.mesh, 50, 0.2)
            self.updateShape("mesh")
            
            self.wireframe.points = self.mesh.vertices
            self.updateShape("wireframe")
        
        if symbol == Key.G:
            gauss = util.Gauss_curvature(self.mesh)
            print("======\nGauss maximum {}, average {} and minimum {}\n".format(np.max(gauss), np.average(gauss), np.min(gauss)))
            print("======\nGauss elements lesser than avg**2/max condition: {}\n".format(np.count_nonzero(gauss<np.average(gauss)**2/np.max(gauss))))
            print("======\nGauss elements lesser than avg+max/2 condition: {}\n".format(np.count_nonzero(gauss<(np.max(gauss)-np.average(gauss))/2)))
            print("======\navg**2/max = {}\n======".format(np.average(gauss)**2/np.max(gauss)))
            self.display_gauss_curv(gauss)

            
        if symbol == Key.J:
            self.reset_colours()
            
        ##if symbol == Key.U:
        ##    spectral_filter(self.mesh, 10, 0.2)
        ##    self.updateShape("mesh")
        ##    ###print("To be implemented")
        ##    
        ##    self.wireframe.points = self.mesh.vertices
        ##    self.updateShape("wireframe")
        
        if symbol == Key.N:
            global MESH_NAME
            global CURRENT_MESH
            
            MESH_NAME = input("Please input the file containing the mesh to visualise (extension included).\n")
            CURRENT_MESH = MESH_NAME[0]
            self.reset_mesh()

        if symbol == Key.RIGHT:            
            CURRENT_MESH += 1
            MESH_NAME = str(CURRENT_MESH) + ".ply"
            self.reset_mesh()

        if symbol == Key.LEFT:
            CURRENT_MESH -= 1
            MESH_NAME = str(CURRENT_MESH) + ".ply"
            self.reset_mesh()
            
        if symbol == Key.SLASH:
            self.printHelp()

    def on_slider_change(self, slider_id, value):
        # if slider_id == 0:
        #     self.eigenvector_idx = int(value * (len(self.mesh.vertices) - 1))
        #     if self.eigenvectors is not None:
        #         self.display_eigenvector(self.eigenvectors[:, self.eigenvector_idx])

        if slider_id == 1:
            self.percent = 0.2 * value

    def printHelp(self):
        self.print("\
        SHIFT+M1: Select vertex\n\
        R: Reset mesh\n\
        W: Toggle wireframe\n\
        A: Adjacent vertices\n\
        D: Delta coordinates single\n\
        CTRL+D: Delta coordinates loop\n\
        L: Delta coordinates laplacian\n\
        S: Delta coordinates sparse\n\
        E: Eigendecomposition\n\
        M: Apply 50 iterations of Taubin smoothing\n\
        Q: Apply 50 iterations of Laplace smoothing\n\
        U: Apply 10 iterations of spectral filter smoothing (non-functional)\n\
        J: Reset mesh colour (non-functional)\n\
        G: Gaussian curvature\n\
        N: Terminal prompt to change which file is visualised\n\
        <- ->: Show previous/next mesh\n\
        ?: Show this list\n\n")

    def display_delta_coords(self, delta: np.ndarray):
        norm = np.sqrt((delta * delta).sum(-1))

        # linear interpolation
        norm = (norm - norm.min()) / (norm.max() - norm.min()) if norm.max() - norm.min() != 0 else np.zeros_like(norm)

        # sigmoid pass 
        norm -= np.quantile(norm,.85)
        norm = 1 / (1 + np.exp(-200*norm))
        
        colormap = cm.get_cmap("plasma")
        colors = colormap(norm)
        ###print(colors)
        self.mesh.vertex_colors = colors[:,:3]
        ###print(np.shape(colors[:,:3]))
        self.updateShape("mesh")
    
    def display_gauss_curv(self, gauss):
        colourmap = cm.get_cmap("plasma")
        ###gauss_modified = util.floor_gauss(gauss)
        ###colours = colourmap(gauss_modified)
        colours = colourmap(gauss)
        self.mesh.vertex_colors = colours[:,:3]

        self.updateShape("mesh")

    def display_eigenvector(self, vec: np.ndarray):
        # linear interpolation
        vec = (vec - vec.min()) / (vec.max() - vec.min()) if vec.max() - vec.min() != 0 else np.zeros_like(vec)
        
        vec -= .5
        vec = 1 / (1 + np.exp(-20*vec))
        
        colormap = cm.get_cmap("plasma")
        colors = colormap(vec)
        self.mesh.vertex_colors = colors[:,:3]
        self.updateShape("mesh")
    
    def reset_colours(self):
        ###self.mesh.vertex_colors = Color.GRAY
        self.color = Color.GRAY
        self.updateShape("mesh")


def find_closest_vertex(mesh: Mesh3D, query: tuple) -> int:

    dist_sq = np.sum((mesh.vertices - query) ** 2, -1)

    return np.argmin(dist_sq)

def find_adjacent_vertices(mesh: Mesh3D, selected_vertex: int) -> np.ndarray:
    """
    Find the vertices adjacent to a given vertex in a 3D mesh.

    Parameters:
    mesh (Mesh3D): The 3D mesh containing vertices and triangles.
    selected_vertex (int): The index of the vertex for which to find the adjacent vertices.

    Returns:
    list: A list of indices of adjacent vertices.
    """

    list_of_adjacent_vertices = []
    # degree = 0

    # Iterate through each triangle in the mesh
    for triangle in mesh.triangles:

        # Check if the the current triangle contains the selected vertex
        if selected_vertex in triangle: 

            # Examine the vertices opposite the selected vertex
            for vertex in triangle:

                if vertex != selected_vertex:

                    # Add vertex to the adjacency list if not already present
                    if vertex not in list_of_adjacent_vertices:

                        list_of_adjacent_vertices.append(vertex)
                        # degree += 1 
    return list_of_adjacent_vertices

def delta_coordinates_single(mesh: Mesh3D, idx: int) -> np.ndarray:
    """
    Calculate the delta coordinates for a single vertex in a 3D mesh.

    Parameters:
    mesh (Mesh3D): The 3D mesh containing the vertices and triangles.
    idx (int): The index of the vertex for which to calculate delta coordinates.

    Returns:
    np.ndarray: The delta coordinates of the vertex.
    """
    
    # Find the indices of the vertices adjacent to the vertex 'idx'
    adj = find_adjacent_vertices(mesh, idx)
    deg = len(adj)  # Determine the degree of the vertex (number of adjacent vertices)

    # Create a patch consisting of the coordinates of the vertex at 'idx' and its adjacent vertices
    patch = mesh.vertices[[idx] + adj, :]

    # Create the filter coefficients array
    coeffs = np.concatenate(([1], -(1/deg) * np.ones(deg)))[:, np.newaxis]

    # Apply the filter to the patch to calculate the delta coordinates
    delta_coords = coeffs.T @ patch

    return delta_coords

def find_all_adjacent_vertices(mesh: Mesh3D):
    """
    Find all adjacent vertices for each vertex in a 3D mesh.

    Parameters:
    mesh (Mesh3D): The 3D mesh containing vertices and triangles.

    Returns:
    list of lists: A list where each element is a list of adjacent vertex indices for each vertex.
    """
    # Initialize a list of empty lists to store adjacent vertices for each vertex
    lists_of_adjacent_vertices = [[] for _ in range(len(mesh.vertices))]

    # Iterate through each triangle in the mesh
    for triangle in mesh.triangles:
        # Iterate through each vertex index in the triangle
        for j in range(3):
            # Get the current vertex index and the next vertex index in the triangle
            k, l = triangle[j], triangle[(j+1)%3]

            # Add each vertex to the other's adjacency list if not already present
            if l not in lists_of_adjacent_vertices[k]:
                lists_of_adjacent_vertices[k].append(l)
            if k not in lists_of_adjacent_vertices[l]:
                lists_of_adjacent_vertices[l].append(k)

    # Return the list of adjacency lists
    return lists_of_adjacent_vertices

def delta_coordinates0(mesh: Mesh3D) -> np.ndarray:
    """
    Calculate the delta coordinates for all vertices in a 3D mesh.

    Parameters:
    mesh (Mesh3D): The 3D mesh containing vertices and triangles.

    Returns:
    np.ndarray: An array of the same shape as mesh.vertices containing the delta coordinates for each vertex.
    """
    
    vertices = mesh.vertices  # Get the vertices from the mesh
    filtered_vertices = np.zeros_like(vertices)  # Initialize an array to store the filtered vertices

    # Get the adjacency lists for all vertices in the mesh - Preprocessing
    vertex_adjacency_lists = find_all_adjacent_vertices(mesh)

    # Iterate over each vertex in the mesh
    for i in range(len(vertices)):
        adj = vertex_adjacency_lists[i]  # Get the adjacent vertices for vertex i
        deg = len(adj)  # Get the degree of vertex i

        # Create a patch that includes vertex i and its adjacent vertices
        patch = vertices[[i] + adj, :]

        # Create the filter coefficients array
        filter_coeff = np.concatenate(([1], -(1/deg) * np.ones(deg)))[:, np.newaxis]
    
        # Apply the filter to the patch to calculate the filtered (delta) coordinates for vertex i
        filtered_vertex = filter_coeff.T @ patch

        # Store the filtered coordinates in the corresponding row of the output array
        filtered_vertices[i, :] = filtered_vertex

    return filtered_vertices

def adjacency(mesh: Mesh3D) -> np.ndarray:
    """
    Build the adjacency matrix for a 3D mesh.

    Parameters:
    mesh (Mesh3D): The 3D mesh containing vertices and triangles.

    Returns:
    np.ndarray: The adjacency matrix of the mesh.
    """
    num_vertices = len(mesh.vertices)
    adjacency_matrix = np.zeros((num_vertices, num_vertices), dtype=int)

    # Iterate through each triangle in the mesh
    for triangle in mesh.triangles:
        # Iterate through each vertex index in the triangle
        for j in range(3):
            # Get the current vertex index and the next vertex index in the triangle
            k, l = triangle[j], triangle[(j+1)%3]

            # Set the adjacency matrix entries to 1
            adjacency_matrix[k, l] = 1
            adjacency_matrix[l, k] = 1

    return adjacency_matrix

def degree(A: np.ndarray) -> np.ndarray:
    """
    Compute the degree matrix from the adjacency matrix.

    Parameters:
    A (np.ndarray): The adjacency matrix of the graph.

    Returns:
    np.ndarray: The degree matrix of the graph.
    """
    # Sum the non-zero elements in each row to get the degree of each vertex
    degrees = np.sum(A, axis=1)
    
    # Create a diagonal matrix with the degrees
    D = np.diag(degrees)
    
    return D

def diagonal_inverse(mat: np.ndarray) -> np.ndarray:
    """
    Compute the inverse of a diagonal matrix.

    Parameters:
    mat (np.ndarray): The diagonal matrix.

    Returns:
    np.ndarray: The inverse of the diagonal matrix.
    """
    # Ensure the diagonal elements are nonzero
    if np.any(np.diag(mat) == 0):
        raise ValueError("Diagonal elements must be nonzero for inversion")

    # Extract the diagonal elements of the matrix
    diagonal_elements = np.diag(mat)
    
    # Compute the reciprocals of the diagonal elements
    reciprocals = 1 / diagonal_elements
    
    # Create a diagonal matrix with the reciprocals as diagonal elements
    inverse = np.diag(reciprocals)
    
    return inverse

def random_walk_laplacian(mesh: Mesh3D) -> np.ndarray:
    """
    Compute the random walk Laplacian matrix for a 3D mesh.

    Parameters:
    mesh (Mesh3D): The 3D mesh containing vertices and triangles.

    Returns:
    np.ndarray: The random walk Laplacian matrix of the mesh.
    """
    N = len(mesh.vertices)

    # Step 1: Compute the adjacency matrix
    A = adjacency(mesh)
    
    # Step 2: Compute the degree matrix
    D = degree(A)
    
    # Step 3: Compute the inverse of the degree matrix
    D_inv = diagonal_inverse(D)

    # Step 4: Build the identity matrix
    I = np.identity(N)

    # Step 4: Compute the random walk Laplacian matrix
    L = I - D_inv @ A
    
    return L

def delta_coordinates(mesh: Mesh3D) -> np.ndarray:
    """
    Compute the delta coordinates using the random walk Laplacian matrix.

    Parameters:
    mesh (Mesh3D): The 3D mesh containing vertices and triangles.

    Returns:
    np.ndarray: The delta coordinates of the mesh.
    """
    # Step 1: Compute the random walk Laplacian matrix
    L_rw = random_walk_laplacian(mesh)
    
    # Step 2: Compute the delta coordinates by multiplying L_rw with the vertex coordinates matrix
    delta_coords = L_rw @ mesh.vertices
    
    return delta_coords

def graph_laplacian(mesh: Mesh3D) -> np.ndarray:
    """
    Compute the graph Laplacian matrix for a 3D mesh.

    Parameters:
    mesh (Mesh3D): The 3D mesh containing vertices and triangles.

    Returns:
    np.ndarray: The graph Laplacian matrix of the mesh.
    """
    N = len(mesh.vertices)

    # Step 1: Compute the adjacency matrix
    A = adjacency(mesh)
    
    # Step 2: Compute the degree matrix
    D = degree(A)

    # Step 3: Compute the graph Laplacian matrix
    L = D - A
    
    return L

def adjacency_sparse(mesh: Mesh3D) -> sparse.lil_array:
    """
    Build the adjacency matrix for a 3D mesh using a sparse representation.

    Parameters:
    mesh (Mesh3D): The 3D mesh containing vertices and triangles.

    Returns:
    sparse.lil_array: The adjacency matrix of the mesh in a sparse format.
    """
    # Get the number of vertices in the mesh
    N = mesh.vertices.shape[0]

    # Find adjacent vertices for each vertex
    vertex_adjacency_lists = find_all_adjacent_vertices(mesh)

    # Initialize a sparse matrix in LIL (List of Lists) format
    A = sparse.lil_matrix((N, N))

    # Set the rows of the sparse matrix
    A.rows = np.array(vertex_adjacency_lists, dtype='object')

    # Set the data of the sparse matrix
    A.data = np.array([[1.0] * len(adj) for adj in vertex_adjacency_lists], dtype='object')

    return A
    
def degree_sparse(A: sparse.lil_array) -> sparse.lil_array:
    """
    Compute the degree matrix from the adjacency matrix in a sparse format.

    Parameters:
    A (sparse.lil_array): The adjacency matrix of the graph in a sparse format.

    Returns:
    sparse.lil_array: The degree matrix of the graph in a sparse format.
    """
    # Get the adjacency lists for each vertex
    vertex_adjacency_lists = A.rows

    # Compute the degrees of each vertex
    degrees = np.array([len(adj) for adj in vertex_adjacency_lists])

    # Create a sparse diagonal matrix representing the degrees
    D = sparse.dia_matrix((degrees, 0), shape=A.shape)

    return D

def diagonal_inverse_sparse(D: sparse.dia_array) -> sparse.dia_array:
    """
    Compute the inverse of a diagonal matrix in a sparse format.

    Parameters:
    D (sparse.dia_array): The diagonal matrix in a sparse format.

    Returns:
    sparse.dia_array: The inverse of the diagonal matrix in a sparse format.
    """
    # Compute the reciprocals of the diagonal elements
    data_inv = np.array([(1 / element) for element in D.data])

    # Create a sparse diagonal matrix with the reciprocals as diagonal elements
    D_inv = sparse.dia_matrix((data_inv, 0), shape=D.shape)

    return D_inv

def random_walk_laplacian_sparse(mesh: Mesh3D) -> sparse.lil_array:
    """
    Compute the sparse random walk Laplacian matrix for a 3D mesh.

    Parameters:
    mesh (Mesh3D): The 3D mesh containing vertices and triangles.

    Returns:
    sparse.lil_array: The sparse random walk Laplacian matrix of the mesh.
    """
    # Step 1: Compute the sparse adjacency matrix
    A = adjacency_sparse(mesh)
    
    # Step 2: Compute the sparse degree matrix
    D = degree_sparse(A)
    
    # Step 3: Compute the inverse of the sparse degree matrix
    D_inv = diagonal_inverse_sparse(D)
    
    # Step 4: Convert adjacency matrix to CSR format
    A = A.tocsr()
    
    # Step 5: Create sparse identity matrix
    I = sparse.eye(A.shape[0], format='csr')
    
    # Step 6: Compute the sparse random walk Laplacian matrix
    L = I - D_inv @ A
    
    return L

def delta_coordinates_sparse(mesh: Mesh3D) -> np.ndarray:
    """
    Compute the delta coordinates using the sparse random walk Laplacian matrix.

    Parameters:
    mesh (Mesh3D): The 3D mesh containing vertices and triangles.

    Returns:
    np.ndarray: The delta coordinates of the mesh.
    """
    # Step 1: Compute the sparse random walk Laplacian matrix
    L_rw_sparse = random_walk_laplacian_sparse(mesh)
    
    # Step 2: Compute the delta coordinates by multiplying L_rw_sparse with the vertex coordinates matrix
    delta_coords = L_rw_sparse @ mesh.vertices
    
    return delta_coords
   
def graph_laplacian_sparse(mesh: Mesh3D) -> sparse.csr_array:
    """
    Compute the sparse graph Laplacian matrix for a 3D mesh.

    Parameters:
    mesh (Mesh3D): The 3D mesh containing vertices and triangles.

    Returns:
    sparse.lil_array: The sparse graph Laplacian matrix of the mesh.
    """
    # Step 1: Compute the sparse adjacency matrix
    A = adjacency_sparse(mesh)
    
    # Step 2: Compute the sparse degree matrix
    D = degree_sparse(A)
    
    # Step 3: Convert adjacency matrix to CSR format
    A = A.tocsr()
    
    # Step 4: Convert degree matrix to CSR format
    D = D.tocsr()
    
    # Step 5: Compute the sparse graph Laplacian matrix
    L = D - A
    
    return L

def eigendecomposition_full(mesh: Mesh3D) -> tuple[np.ndarray, np.ndarray]:
    """
    Perform the full eigendecomposition on the graph Laplacian of a 3D mesh.

    Parameters:
    mesh (Mesh3D): The 3D mesh containing vertices and triangles.

    Returns:
    tuple[np.ndarray, np.ndarray]: The eigenvalues and eigenvectors of the graph Laplacian.
    """
    # Step 1: Compute the graph Laplacian matrix
    L = graph_laplacian(mesh)
    ##L = graph_laplacian_sparse(mesh)
    
    # Step 2: Perform the eigendecomposition
    eigenvalues, eigenvectors = scipy.linalg.eigh(L)
    ##eigenvalues, eigenvectors = scipy.sparse.linalg.eigsh(L, k=len(mesh.vertices))
    
    return eigenvalues, eigenvectors

def eigendecomposition_partial(mesh: Mesh3D, percent: float, which: str = "SM") -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the eigenvalues and eigenvectors for a percentage of the smallest magnitude of the graph Laplacian.

    Parameters:
    mesh (Mesh3D): The 3D mesh containing vertices and triangles.
    percent (float): The percentage of eigenvalues and eigenvectors to compute.
    which (str): The type of eigenvalues to compute ("SM" for smallest magnitude by default).

    Returns:
    tuple[np.ndarray, np.ndarray]: The computed eigenvalues and eigenvectors.
    """
    L = graph_laplacian_sparse(mesh)
    k = int(mesh.vertices.shape[0] * percent)
    if k <= 0 or k > mesh.vertices.shape[0]:
        raise ValueError("Percent must result in at least one and at most the number of vertices")
    
    # Compute the k smallest magnitude eigenvalues and corresponding eigenvectors
    eigenvalues, eigenvectors = scipy.sparse.linalg.eigsh(L, k=k, which=which)
    
    return eigenvalues, eigenvectors

def reconstruct(mesh: Mesh3D, percent: float) -> Mesh3D:
    """
    Reconstruct the vertices of a mesh using a subset of eigenvectors.

    Parameters:
    mesh (Mesh3D): The 3D mesh containing vertices and triangles.
    percent (float): The percentage of eigenvectors to use for reconstruction.

    Returns:
    Mesh3D: The mesh with reconstructed vertices.
    """
    # Step 1: Perform eigendecomposition to get the eigenvectors
    _, vecs = eigendecomposition_partial(mesh, percent, "SM")

    # Step 2: Extract the original vertices from the mesh
    original_vertices = mesh.vertices

    # Step 3: Reconstruct the vertices using a subset of the eigenvectors
    new_vertices = vecs @ vecs.T @ original_vertices

    # Step 4: Create a new mesh with the reconstructed vertices
    mesh.vertices = new_vertices
    
    return mesh

def reconstruct2(mesh: Mesh3D, percent: float, idx: int) -> Mesh3D:
    """
    Reconstruct the vertices of a mesh using a subset of eigenvectors.

    Parameters:
    mesh (Mesh3D): The 3D mesh containing vertices and triangles.
    percent (float): The percentage of eigenvectors to use for reconstruction.

    Returns:
    Mesh3D: The mesh with reconstructed vertices.
    """
    # Step 1: Perform eigendecomposition to get the eigenvectors
    _, vecs = eigendecomposition_partial(mesh, percent, "SM")
    
    vecs = np.hstack((vecs[:, :idx], vecs[:, idx + 1:]))

    # Step 2: Extract the original vertices from the mesh
    vertices = mesh.vertices

    # Step 3: Reconstruct the vertices using the eigenvectors
    new_vertices = vecs @ vecs.T @ vertices

    # Step 4: Create a new mesh with the reconstructed vertices
    mesh.vertices = new_vertices
    
    return mesh

def Laplace_recursive(mesh:Mesh3D, iterations, l):
    functional = 0<l
    if not functional:
        print("The parameter l must satisfy the 0<l condition for this algorithm to work.")
    
    Dp = -delta_coordinates_sparse(mesh)
    vertices = mesh.vertices
    
    if iterations == 0:
        return
        
    vertices_new = vertices + l*Dp 
    
    mesh.vertices = vertices_new
    
    Laplace_recursive(mesh, iterations-1, l)
    
    return

def Taubin_recursive(mesh:Mesh3D, iterations, l, m):
    functional = 0<l and m<0
    if not functional:
        print("The parameters l and m must satisfy the m<0<l condition for this algorithm to work.")
    
    Dp = -delta_coordinates_sparse(mesh)
    vertices = mesh.vertices
    
    if iterations == 0:
        return
        
    vertices_new = vertices + l*Dp
    ###vertices_new = vertices_new + m*Dp
    mesh.vertices = vertices_new    
    
    Dp = -delta_coordinates_sparse(mesh)
    vertices = mesh.vertices
    
    vertices_new = vertices + m*Dp
    mesh.vertices = vertices_new
    
    Taubin_recursive(mesh, iterations-1, l, m)

#Implement this    
def spectral_filter(mesh:Mesh3D, iterations, a):
    functional = a<=1 and a>=0
    if not functional:
        print("The parameter a must satisfy the 0<=a<=1 condition for this algorithm to work.")
    mesh_decomp = eigendecomposition_full(mesh)
    kernel = np.exp(-mesh_decomp[1] * a)**iterations
    
    filtered_eigenvectors = mesh_decomp[1] @ np.diag(kernel) @ mesh_decomp[1].T
    mesh.vertices = filtered_eigenvectors @ self.vertices

if __name__ == "__main__":
    app = Texture_Detection()
    app.mainLoop()
