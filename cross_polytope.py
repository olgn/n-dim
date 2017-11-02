import os
from collections import deque
import math
import numpy as np
import utils
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

cos = math.cos
sin = math.sin


# This is the code for visualizing cross-polytopes. Other names
# for these shapes include orthoplexes, hyperoctahedrons, and cocubes. 
# These shapes are defined as follows: all vertices are permutations 
# of (+- 1, 0, 0, ..., 0). Aka all values are 0 except one, the value of 
# which is either positive or negative 1.

def create_cross_polytope_vertices(dimensions):
	# Creates a list of octahedron vertices in an n-dimensional space,
	# defined by the input variable 'dimensions'.
	#
	# Inputs:
	# dimensions (integer): the number of dimensions in which the cross-polytope exists. 
	# Must be at least 3.
	#
	# Outputs:
	# octahedron_vertices (matrix, m x n where n is the number of dimensions): the vertices of the cross-polytope 

	# The n-dimensional cross-polytope is given by all permutations of (+- 1, 0, ..., 0).
	vertices = []
	zeros = [0] * (dimensions - 1)
	pos = deque([1]  + zeros)
	neg = deque([-1] + zeros)
	for i in range(0, dimensions):
		vertices.append(np.array(list(pos)))
		vertices.append(np.array(list(neg)))
		pos.rotate()
		neg.rotate()

	# print('vertices:', vertices)
	return vertices



def create_cross_polytope(dimensions):
	vertices = create_cross_polytope_vertices(dimensions)
	vertex_graph = utils.create_associated_vertex_graph(vertices, 'orthoplex')
	return vertices, vertex_graph

def octahedron(animate=False):
	vertices, vertex_graph = create_cross_polytope(3)
	# print('vertices:', vertices, 'vertex_graph:', vertex_graph)

def octahedron_4(animate=False):

	# Set up the camera and screen parameters
	p4 = {
		'origin': np.array([0,0,0,0]),
		'orientations': [[-1,-1,0,0], [0,0,1,0]],
		'location': np.array([1,3,.2,0]),
		'theta': math.pi / 4,
		'd_theta': .005,
		'extent': np.array([5,5,5]),
		'center': np.array([0,0,0])
	}

	# Set up the 4d rotation matrix:
	R4 = [	[cos(p4['d_theta']), -sin(p4['d_theta']), 0, 0],
			[sin(p4['d_theta']), cos(p4['d_theta']), 0, 0],
			[0, 0, 1, 0],
			[0, 0, 0, 1]]

		# Set up the 4d rotation matrix:
	R42= [	[1, 0, 0, 0],
			[0, cos(p4['d_theta']), -sin(p4['d_theta']), 0],
			[0, sin(p4['d_theta']), cos(p4['d_theta']), 0],
			[0, 0, 0, 1]]

	# Create a 4d octahedron
	vertices_4, graph = create_cross_polytope(4)

	# Configure the plot
	plt.ion()
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	axis_limits = 2

	# Get the 4d camera bases
	p4['bases'] = utils.get_camera_bases(p4['origin'], p4['location'], p4['orientations'])

	if animate:
		# Exports a gif of the plots as time evolves
		num_steps = np.round(2*math.pi / p4['d_theta']).astype(int)

		def update(i, vertices_4, p4, num_steps):
				# Set up the 4d rotation matrix:
				R4 = [	[cos(p4['d_theta'] * i), -sin(p4['d_theta'] * i), 0, 0],
						[sin(p4['d_theta'] * i), cos(p4['d_theta'] * i), 0, 0],
						[0, 0, 1, 0],
						[0, 0, 0, 1]]

					# Set up the 4d rotation matrix:
				R42= [	[1, 0, 0, 0],
						[0, cos(p4['d_theta'] * i), -sin(p4['d_theta'] * i), 0],
						[0, sin(p4['d_theta'] * i), cos(p4['d_theta'] * i), 0],
						[0, 0, 0, 1]]

				# rotate the vertices
				vertices_4 = np.dot(vertices_4, R4)
				vertices_4 = np.dot(vertices_4, R42)

				# Using the camera's basis, calculate the new vertex coordinates
				# on the screen of the camera:
				vertices_3 = []
				for vertex in vertices_4:

					# Get the point in the camera's coordinate system:
					new_vertex = np.dot((vertex - p4['location']), p4['bases'])

					# Calculate the position of the new vertex on the camera's 'screen':
					screen_position = utils.get_screen_coordinates(new_vertex, p4['extent'], p4['center'], p4['theta'])

					# Put the new position into an array of 3d vertices:
					vertices_3.append(screen_position)

				# Chart the 3d points:
				ax.clear()
				points = np.array(vertices_3)

				for pair in graph:
					end_points = np.array(points[pair])
					p1 = end_points[0]
					p2 = end_points[1]
					ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]])

				ax.grid(False)
				ax.axis('off')
				ax.set_xlim(-axis_limits,axis_limits)
				ax.set_ylim(-axis_limits,axis_limits)
				ax.set_zlim(-axis_limits,axis_limits)
				ax.view_init(azim=-23, elev=-71)
				print('frame ', i, ' of ', num_steps)
				plt.show()
				return ax

		anim = FuncAnimation(fig, update, np.arange(0,num_steps), fargs=(vertices_4, p4, num_steps), interval=5)
		anim.save('C:/Users/Teal/Documents/Python/Projects/Dimensions/squirt.mp4', writer='ffmpeg', dpi=500)

	else:
		# Visualizes the hyperoctahedron in matplotlib
		while True:
			# Using the camera's basis, calculate the new vertex coordinates
			# on the screen of the camera:
			vertices_3 = []
			for vertex in vertices_4:

				# Get the point in the camera's coordinate system:
				new_vertex = np.dot((vertex - p4['location']), p4['bases'])

				# Calculate the position of the new vertex on the camera's 'screen':
				screen_position = utils.get_screen_coordinates(new_vertex, p4['extent'], p4['center'], p4['theta'])

				# Put the new position into an array of 3d vertices:
				vertices_3.append(screen_position)


			# Chart the 3d points:
			ax.clear()
			points = np.array(vertices_3)

			for pair in graph:
				end_points = np.array(points[pair])
				p1 = end_points[0]
				p2 = end_points[1]
				ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]])

			ax.grid(False)
			ax.axis('off')
			ax.set_xlim(-axis_limits,axis_limits)
			ax.set_ylim(-axis_limits,axis_limits)
			ax.set_zlim(-axis_limits,axis_limits)
			ax.view_init(azim=-23, elev=-71)
			plt.show()
			plt.pause(0.01)
			vertices_4 = np.dot(vertices_4, R4)
			vertices_4 = np.dot(vertices_4, R42)


octahedron_4(True)