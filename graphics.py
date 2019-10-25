import numpy as np
import tensorflow as tf
import trimesh

import tensorflow_graphics.geometry.transformation as tfg_transformation
from tensorflow_graphics.notebooks import threejs_visualization

#tf.enable_eager_execution()

# Download the mesh.
#!wget https://storage.googleapis.com/tensorflow-graphics/notebooks/index/cow.obj
# Load the mesh.
mesh = trimesh.load("./cow.obj")
mesh = {"vertices": mesh.vertices, "faces": mesh.faces}
# Visualize the original mesh.
threejs_visualization.triangular_mesh_renderer(mesh, width=400, height=400)
# Set the axis and angle parameters.
axis = np.array((0., 1., 0.))  # y axis.
angle = np.array((np.pi / 4.,))  # 45 degree angle.
# Rotate the mesh.
mesh['vertices'] = tfg_transformation.axis_angle.rotate(mesh["vertices"], axis,
                                                        angle).numpy()
# Visualize the rotated mesh.
threejs_visualization.triangular_mesh_renderer(mesh, width=400, height=400)