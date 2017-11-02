import os
import itertools as it
import math
import numpy as np
import utils
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

cos = math.cos
sin = math.sin

def create_cube_vertices(dimensions):
	# Creates a cube in (dimensions)-space, with 
	# side length 1 and one vertex at the origin.
	# Returns the vertices as an array.
	vertices = np.array(list(it.product([0,1], repeat=dimensions)))
	return vertices


def create_cube(dimensions):
	# Creates the vertices and vertex-graph for a cube in (dimensions)-space
	vertices = create_cube_vertices(dimensions)
	vertex_graph = utils.create_associated_vertex_graph(vertices, 'cube')
	return vertices, vertex_graph

def project_cube():
	origin = np.array([0,0,0])
	camera_location = np.array([5,6,7])
	camera_orientation_array = [[1,0,0]]
	theta = math.pi / 4
	extent_2d = [100,100]
	center_2d = [50,50]

	vertices, graph = create_cube(4)

	points = []

	camera_bases = utils.get_camera_bases(origin, camera_location, camera_orientation_array)

	for vertex in vertices:
		new_point = np.dot((vertex - camera_location), camera_bases)
		screen_coords = utils.get_screen_coordinates(new_point, extent_2d, center_2d, theta)
		points.append(np.round(screen_coords).astype(int))

	print('points:', points)

	points = np.array(points)

	for pair in graph:
		print(pair)
		end_points = np.array(points[pair])
		p1 = end_points[0]
		p2 = end_points[1]
		plt.plot([p1[0], p2[0]], [p1[1], p2[1]])

	plt.show()

def project_tesseract():
	origin4 = np.array([0,0,0,0]) # 4d camera points to this point
	camera_range = np.arange(-7,7,0.3) # use this to generate list of camera angles
	smooth_range = np.concatenate((camera_range, np.fliplr([camera_range])[0])) # make camera range go there -> back
	C = np.array([[-i,i,i*.3,0] for i in smooth_range]) # create the matrix of different camera angles
	co4 = [[-.71,.71,0,0], [0,0,1,.02]]
	theta4 = math.pi / 4
	extent_3d = np.array([2,2,2])
	center_3d = extent_3d * .5

	w = 5 # the max / min of the x,y,and z axes

	vertices, graph = create_cube(4)

	vertices = (vertices * 2) - 1

	plt.ion()
	ax = plt.axes(projection='3d')
	ax.autoscale(enable=False)
	plt.show()

	while True:
		points3 = []
		C = np.roll(C, -1, 0)
		c4 = C[0]
		# print('c4:',c4)
		cb4 = utils.get_camera_bases(origin4, c4, co4)

		for vertex in vertices:
			new_point = np.dot((vertex - c4), cb4)
			# print('new_point:', new_point)
			screen_coords = utils.get_screen_coordinates(new_point, extent_3d, center_3d, theta4)
			points3.append(screen_coords)

		# for i in range(0, len(vertices)):
		# 	print('point ',i,': ', vertices[i], '->', points3[i])

		# print('graph:')
		# print(graph)

		ax.clear()		
		points = np.array(points3)

		for pair in graph:
			# print(pair)
			end_points = np.array(points[pair])
			# print('end_points:', end_points)
			p1 = end_points[0]
			p2 = end_points[1]
			ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]])
		
		ax.grid(False)
		ax.axis('off')
		ax.set_xlim(-w,w)
		ax.set_ylim(-w,w)
		ax.set_zlim(-w,w)
		# plt.canvas.draw()
		plt.pause(0.001)

def moving_camera_tesseract_animation():
	origin4 = np.array([0,0,0,0]) # 4d camera points to this point
	camera_range = np.arange(-7,7,0.3) # use this to generate list of camera angles
	smooth_range = np.concatenate((camera_range, np.fliplr([camera_range])[0])) # make camera range go there -> back
	C = np.array([[-i,0,i*.3,0] for i in smooth_range]) # create the matrix of different camera angles
	co4 = [[-.71,.71,0,0], [0,0,1,.02]]
	theta4 = math.pi / 4
	extent = np.array([2,2,2])
	center = extent * .5

	w = 7 # the max / min of the x,y,and z axes

	vertices, graph = create_cube(4)

	vertices = (vertices * 2) - 1

	plt.ion()
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	# plt.show()

	def update(i, C, origin, co, extent, center, theta):
		points3 = []
		C = np.roll(C, -i - 1, 0)
		c4 = C[0]
		# print('c4:',c4)
		cb4 = utils.get_camera_bases(origin, c4, co)

		for vertex in vertices:
			new_point = np.dot((vertex - c4), cb4)
			# print('new_point:', new_point)
			screen_coords = utils.get_screen_coordinates(new_point, extent, center, theta)
			points3.append(screen_coords)

		# for i in range(0, len(vertices)):
		# 	print('point ',i,': ', vertices[i], '->', points3[i])

		# print('graph:')
		# print(graph)

		ax.clear()		
		points = np.array(points3)

		for pair in graph:
			# print(pair)
			end_points = np.array(points[pair])
			# print('end_points:', end_points)
			p1 = end_points[0]
			p2 = end_points[1]
			ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]])

		
		ax.grid(False)
		ax.axis('off')
		ax.set_xlim(-w,w)
		ax.set_ylim(-w,w)
		ax.set_zlim(-w,w)
		return ax
		
	anim = FuncAnimation(fig, update, np.arange(0,len(smooth_range)), fargs=(C, origin4, co4, extent, center, theta4), interval=100)
	anim.save('C:/Users/Teal/Documents/Python/Projects/Dimensions/squirt.mp4', writer='ffmpeg')

	os.system("ffmpeg -i C:\\Users\\Teal\\Documents\\Python\\Projects\\Dimensions\\squirt.mp4 -vf scale=500:-1 -t 10 -r 10 C:\\Users\\Teal\\Documents\\Python\\Projects\\Dimensions\\squirt.gif")
	
def rotating_vertices_tesseract_animation():
	origin4 = np.array([0,0,0,0]) # 4d camera points to this point
	delta_theta = 0.05
	num_steps = np.round(math.pi * 2 / delta_theta)
	c4 = [2, 0, 0, 0]
	co4 = [[-12,.71,0,0], [0,1,0,-.01]]
	theta4 = math.pi / 4
	extent = np.array([2,2,2])
	center = extent * .5

	w = 4 # the max / min of the x,y,and z axes

	vertices, graph = create_cube(4)

	vertices = (vertices * 2) - 1

	plt.ion()
	fig = plt.figure(facecolor='black', edgecolor='white')
	ax = fig.add_subplot(111, projection='3d', axisbg='black')
	# plt.show()

	def update(i, vertices, delta_theta, origin, c, co, extent, center, theta, num_steps):
		t = delta_theta * i
		# R1 = [[1, 0, 0, 0], [0, math.cos(t), math.sin(t), 0], [0, -math.sin(t), math.cos(t), 0], [0,0,0,1]] #xz rotation
		R2 = [[math.cos(t), -math.sin(t), 0,0], [math.sin(t), math.cos(t), 0,0], [0,0,1,0], [0,0,0,1]] # xy rotation
		R3 = Rzw = [[math.cos(delta_theta), 0, 0, math.sin(delta_theta)], [0,1,0,0], [0,0,1,0], [-math.sin(delta_theta), 0, 0, math.cos(delta_theta)]]
		vertices = np.dot(vertices, R2)
		# vertices = np.dot(vertices, R3)
		points3 = []
		# print('c4:',c4)
		cb4 = utils.get_camera_bases(origin, c4, co)

		for vertex in vertices:
			new_point = np.dot((vertex - c4), cb4)
			# print('new_point:', new_point)
			screen_coords = utils.get_screen_coordinates(new_point, extent, center, theta)
			points3.append(screen_coords)

		# for i in range(0, len(vertices)):
		# 	print('point ',i,': ', vertices[i], '->', points3[i])

		# print('graph:')
		# print(graph)

		ax.clear()		
		points = np.array(points3)

		for pair in graph:
			# print(pair)
			end_points = np.array(points[pair])
			# print('end_points:', end_points)
			p1 = end_points[0]
			p2 = end_points[1]
			ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], 'white', linewidth=2.0)

		ax.grid(False)
		ax.axis('off')
		ax.set_xlim(-w,w)
		ax.set_ylim(-w,w)
		ax.set_zlim(-w,w)
		print('on frame ', i,'of ',num_steps)
		ax.view_init(azim=38, elev=16)
		return ax
		
	anim = FuncAnimation(fig, update, np.arange(0,num_steps), fargs=(vertices, delta_theta, origin4, c4, co4, extent, center, theta4, num_steps), interval=50)
	anim.save('C:/Users/Teal/Documents/Python/Projects/Dimensions/squirt.mp4', writer='ffmpeg')

	os.system("ffmpeg -i C:\\Users\\Teal\\Documents\\Python\\Projects\\Dimensions\\squirt.mp4 -vf scale=500:-1 -t 10 -r 10 C:\\Users\\Teal\\Documents\\Python\\Projects\\Dimensions\\squirt.gif")

def rotating_vertices_tesseract_vis():
	origin4 = np.array([0,0,0,0]) # 4d camera points to this point
	delta_theta = 0.05
	num_steps = np.round(math.pi * 2 / delta_theta)

	Rxy = [[math.cos(delta_theta), -math.sin(delta_theta), 0,0], [math.sin(delta_theta), math.cos(delta_theta), 0,0], [0,0,1,0], [0,0,0,1]]
	Ryz = [[1, 0, 0, 0], [0, math.cos(delta_theta), math.sin(delta_theta), 0], [0, -math.sin(delta_theta), math.cos(delta_theta), 0], [0,0,0,1]]
	Rzx = [[math.cos(delta_theta), 0, -math.sin(delta_theta), 0], [0,1,0,0], [math.sin(delta_theta), 0, math.cos(delta_theta), 0], [0,0,0,1]]
	Rzw = [[math.cos(delta_theta), 0, 0, math.sin(delta_theta)], [0,1,0,0], [0,0,1,0], [-math.sin(delta_theta), 0, 0, math.cos(delta_theta)]]
	c4 = [2, 0, 0, 0]
	co4 = [[-12,.71,0,0], [0,1,0,-.01]]
	theta4 = math.pi / 4
	extent = np.array([2,2,2])
	center = extent * .5

	w = 4 # the max / min of the x,y,and z axes

	vertices, graph = create_cube(4)

	vertices = (vertices * 2) - 1

	plt.ion()
	fig = plt.figure(facecolor='black', edgecolor='white')
	ax = plt.axes(projection='3d', axisbg='black')
	ax.autoscale(enable=False)
	plt.show()

	while True:

		vertices = np.dot(vertices, Rzx)
		# vertices = np.dot(vertices, Rzw)
		points3 = []
		# print('c4:',c4)
		cb4 = utils.get_camera_bases(origin4, c4, co4)

		for vertex in vertices:
			new_point = np.dot((vertex - c4), cb4)
			# print('new_point:', new_point)
			screen_coords = utils.get_screen_coordinates(new_point, extent, center, theta4)
			points3.append(screen_coords)

		ax.clear()		
		points = np.array(points3)

		for pair in graph:
			# print(pair)
			end_points = np.array(points[pair])
			# print('end_points:', end_points)
			p1 = end_points[0]
			p2 = end_points[1]
			ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], 'white', linewidth=2.5)

		ax.grid(False)
		ax.axis('off')
		ax.set_xlim(-w,w)
		ax.set_ylim(-w,w)
		ax.set_zlim(-w,w)
		ax.view_init(azim=38, elev=16)
		# plt.canvas.draw()
		plt.pause(0.001)

def lazy_cube(animate):

	# 5 dimensional parameters:
	p5 = {
		'origin': np.array([0,0,0,0,0]),
		'orientations': [[0,1,.4,0,0], [0,0,1,0,0], [0,0,0,1,0]],
		'location': np.array([7,0,3,0,0]),
		'theta': math.pi / 4,
		'd_theta': .05,
		'extent': np.array([2,2,2,2]),
		'center': np.array([1,1,1,1])
	}

	# 4 dimensional parameters:
	p4 = {
		'origin': np.array([0,0,0,0]),
		'orientations': [[0,0,.21,0], [.3,0,0,0]],
		'location': np.array([0,.2,3,0]),
		'theta': math.pi / 4,
		'd_theta': .01,
		'extent': np.array([2,2,2]),
		'center': np.array([1,1,1])
	}

	# 5d rotation matrix
	R5 = [	[cos(p5['d_theta']), -sin(p5['d_theta']), 0, 0, 0],
			[sin(p5['d_theta']), cos(p5['d_theta']), 0, 0, 0],
			[0, 0, 1, 0, 0],
			[0, 0, 0, 1, 0],
			[0, 0, 0, 0, 1]]


 	# set up the visualization
	plt.ion()
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	# ax.autoscale(enable=False)
	axis_limits = 2

	# create a 5d cube
	vertices_5, graph = create_cube(5)

	vertices_5 = ( vertices_5 * 2 ) - 1

	# get the 5d camera bases:
	p5['bases'] = utils.get_camera_bases(p5['origin'], p5['location'], p5['orientations'])

	# get the 4d camera bases:
	p4['bases'] = utils.get_camera_bases(p4['origin'], p4['location'], p4['orientations'])

	if animate:
		num_steps = np.round(2*math.pi / p4['d_theta']).astype(int)

		def update(i, vertices_5, p5, p4, num_steps):
			# 5d rotation matrix
			R5 = [	[cos(p5['d_theta'] * i), -sin(p5['d_theta'] * i), 0, 0, 0],
					[sin(p5['d_theta']* i), cos(p5['d_theta'] * i), 0, 0, 0],
					[0, 0, 1, 0, 0],
					[0, 0, 0, 1, 0],
					[0, 0, 0, 0, 1]]

			vertices_5 = np.dot(vertices_5, R5)

			# project 5d points into 4d:
			vertices_4 = []
			for vertex in vertices_5: 
				new_point = np.dot((vertex - p5['location']), p5['bases'])
				vertices_4.append(utils.get_screen_coordinates(new_point, p5['extent'], p5['center'], p5['theta']))

			# print('vertices_4:', vertices_4)
			R4 = [	[cos(p4['d_theta'] * i), -sin(p4['d_theta'] * i), 0, 0],
			[sin(p4['d_theta'] * i), cos(p4['d_theta'] * i), 0, 0],
			[0, 0, 1, 0],
			[0, 0, 0, 1]]
			# vertices_4 = np.dot(vertices_4, R4)

			# project 4d points into 3d:
			vertices_3 = []
			for vertex in vertices_4:
				new_point = np.dot((vertex - p4['location']), p4['bases'])
				vertices_3.append(utils.get_screen_coordinates(new_point, p4['extent'], p4['center'], p4['theta']))

			# print('vertices_3:', vertices_3)

			# chart the points
			ax.clear()
			points = np.array(vertices_3)

			for pair in graph:
				end_points = np.array(points[pair])
				# print('end_points', end_points)
				p1 = end_points[0]
				p2 = end_points[1]
				ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]])

			ax.grid(False)
			ax.axis('off')
			ax.set_xlim(-axis_limits,axis_limits)
			ax.set_ylim(-axis_limits,axis_limits)
			ax.set_zlim(-axis_limits,axis_limits)
			ax.view_init(azim=-154, elev=119)
			print('frame ', i, ' of ', num_steps)
			return ax

		anim = FuncAnimation(fig, update, np.arange(0,num_steps), fargs=(vertices_5,p5, p4, num_steps), interval=5)
		anim.save('C:/Users/Teal/Documents/Python/Projects/Dimensions/squirt.mp4', writer='ffmpeg', dpi=500)

		# os.system("ffmpeg -i C:\\Users\\Teal\\Documents\\Python\\Projects\\Dimensions\\squirt.mp4 -vf scale=200:-1 -r 30 C:\\Users\\Teal\\Documents\\Python\\Projects\\Dimensions\\squirt.gif")

	else:
		frame = 0
		while True:
			vertices_5 = np.dot(vertices_5, R5)

			# project 5d points into 4d:
			vertices_4 = []
			for vertex in vertices_5: 
				new_point = np.dot((vertex - p5['location']), p5['bases'])
				vertices_4.append(utils.get_screen_coordinates(new_point, p5['extent'], p5['center'], p5['theta']))

			# print('vertices_4:', vertices_4)
			R4 = [	[cos(p4['d_theta'] * frame), -sin(p4['d_theta'] * frame), 0, 0],
			[sin(p4['d_theta'] * frame), cos(p4['d_theta'] * frame), 0, 0],
			[0, 0, 1, 0],
			[0, 0, 0, 1]]
			# vertices_4 = np.dot(vertices_4, R4)

			# project 4d points into 3d:
			vertices_3 = []
			for vertex in vertices_4:
				new_point = np.dot((vertex - p4['location']), p4['bases'])
				vertices_3.append(utils.get_screen_coordinates(new_point, p4['extent'], p4['center'], p4['theta']))

			# print('vertices_3:', vertices_3)

			# chart the points
			ax.clear()
			points = np.array(vertices_3)

			for pair in graph:
				end_points = np.array(points[pair])
				# print('end_points', end_points)
				p1 = end_points[0]
				p2 = end_points[1]
				ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]])

			ax.grid(False)
			ax.axis('off')
			ax.set_xlim(-axis_limits,axis_limits)
			ax.set_ylim(-axis_limits,axis_limits)
			ax.set_zlim(-axis_limits,axis_limits)
			ax.view_init(azim=-154, elev=119)
			plt.show()
			plt.pause(0.001)
			frame += 1

def cube_6d(animate):

	delta_theta = .007
	# 6 dimensional parameters:
	p6 = {
		'origin': np.array([0,0,0,0,0,0]),
		'orientations': [[0,3,.4,0,0,0], [0,0,1,0,0,0], [0,0,0,1,0,0], [0,0,0,0,1,0]],
		'location': np.array([7,3,3,0,0,0]),
		'theta': math.pi / 4,
		'd_theta': delta_theta,
		'extent': np.array([2,2,2,2,2]),
		'center': np.array([1,1,1,1,1])
	}
	# 5 dimensional parameters:
	p5 = {
		'origin': np.array([0,0,0,0,0]),
		'orientations': [[0,1,.4,0,0], [0,0,1,0,0], [0,0,0,1,0]],
		'location': np.array([7,0,3,0,0]),
		'theta': math.pi / 4,
		'd_theta': delta_theta,
		'extent': np.array([2,2,2,2]),
		'center': np.array([1,1,1,1])
	}

	# 4 dimensional parameters:
	p4 = {
		'origin': np.array([0,0,0,0]),
		'orientations': [[0,0,.21,0], [.3,0,0,0]],
		'location': np.array([0,.2,3,0]),
		'theta': math.pi / 4,
		'd_theta': delta_theta,
		'extent': np.array([2,2,2]),
		'center': np.array([1,1,1])
	}

	# 6d rotation matrix
	R6 = [	[1, 0, 0, 0, 0, 0],
			[0, cos(p6['d_theta']), -sin(p6['d_theta']), 0, 0, 0],
			[0, sin(p6['d_theta']), cos(p6['d_theta']), 0, 0, 0],
			[0, 0, 0, 1, 0, 0],
			[0, 0, 0, 0, 1, 0], 
			[0, 0, 0, 0, 0, 1]]

	# 5d rotation matrix
	R5 = [	[cos(p5['d_theta']), -sin(p5['d_theta']), 0, 0, 0],
			[sin(p5['d_theta']), cos(p5['d_theta']), 0, 0, 0],
			[0, 0, 1, 0, 0],
			[0, 0, 0, 1, 0],
			[0, 0, 0, 0, 1]]


 	# set up the visualization
	plt.ion()
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	# ax.autoscale(enable=False)
	axis_limits = 3

	# create a 5d cube
	vertices_6, graph = create_cube(6)

	vertices_6 = ( vertices_6 * 2 ) - 1

	# get the 6d camera bases:
	p6['bases'] = utils.get_camera_bases(p6['origin'], p6['location'], p6['orientations'])

	# get the 5d camera bases:
	p5['bases'] = utils.get_camera_bases(p5['origin'], p5['location'], p5['orientations'])

	# get the 4d camera bases:
	p4['bases'] = utils.get_camera_bases(p4['origin'], p4['location'], p4['orientations'])

	if animate:
		num_steps = np.round(2*math.pi / p4['d_theta']).astype(int)

		def update(i, vertices_6, p6, p5, p4, num_steps):
			# 6d rotation matrix
			R6 = [	[1, 0, 0, 0, 0, 0],
					[0, cos(p6['d_theta'] * i), -sin(p6['d_theta'] * i), 0, 0, 0],
					[0, sin(p6['d_theta'] * i), cos(p6['d_theta'] * i), 0, 0, 0],
					[0, 0, 0, 1, 0, 0],
					[0, 0, 0, 0, 1, 0], 
					[0, 0, 0, 0, 0, 1]]

			vertices_6 = np.dot(vertices_6, R6)

			# project 6d points into 5d:
			vertices_5 = []
			for vertex in vertices_6:
				new_point = np.dot((vertex - p6['location']), p6['bases'])
				vertices_5.append(utils.get_screen_coordinates(new_point, p6['extent'], p6['center'], p6['theta']))

			# 5d rotation matrix
			R5 = [	[cos(p5['d_theta'] * i), -sin(p5['d_theta'] * i), 0, 0, 0],
					[sin(p5['d_theta'] * i), cos(p5['d_theta'] * i), 0, 0, 0],
					[0, 0, 1, 0, 0],
					[0, 0, 0, 1, 0],
					[0, 0, 0, 0, 1]]

			vertices_5 = np.dot(vertices_5, R5)

			# project 5d points into 4d:
			vertices_4 = []
			for vertex in vertices_5: 
				new_point = np.dot((vertex - p5['location']), p5['bases'])
				vertices_4.append(utils.get_screen_coordinates(new_point, p5['extent'], p5['center'], p5['theta']))

			# print('vertices_4:', vertices_4)
			R4 = [	[cos(p4['d_theta'] * i), -sin(p4['d_theta'] * i), 0, 0],
			[sin(p4['d_theta'] * i), cos(p4['d_theta'] * i), 0, 0],
			[0, 0, 1, 0],
			[0, 0, 0, 1]]
			# vertices_4 = np.dot(vertices_4, R4)

			# project 4d points into 3d:
			vertices_3 = []
			for vertex in vertices_4:
				new_point = np.dot((vertex - p4['location']), p4['bases'])
				vertices_3.append(utils.get_screen_coordinates(new_point, p4['extent'], p4['center'], p4['theta']))

			# print('vertices_3:', vertices_3)

			# chart the points
			ax.clear()
			points = np.array(vertices_3)

			for pair in graph:
				end_points = np.array(points[pair])
				# print('end_points', end_points)
				p1 = end_points[0]
				p2 = end_points[1]
				ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]])

			ax.grid(False)
			ax.axis('off')
			ax.set_xlim(-axis_limits,axis_limits)
			ax.set_ylim(-axis_limits,axis_limits)
			ax.set_zlim(-axis_limits,axis_limits)
			ax.view_init(azim=41, elev=5)
			print('frame ', i, ' of ', num_steps)
			return ax

		anim = FuncAnimation(fig, update, np.arange(0,num_steps), fargs=(vertices_6, p6, p5, p4, num_steps), interval=5)
		anim.save('C:/Users/Teal/Documents/Python/Projects/Dimensions/squirt.mp4', writer='ffmpeg', dpi=500)

		# os.system("ffmpeg -i C:\\Users\\Teal\\Documents\\Python\\Projects\\Dimensions\\squirt.mp4 -vf scale=200:-1 -r 30 C:\\Users\\Teal\\Documents\\Python\\Projects\\Dimensions\\squirt.gif")

	else:
		frame = 0
		while True:
			vertices_6 = np.dot(vertices_6, R6)

			# project 6d points into 5d:
			vertices_5 = []
			for vertex in vertices_6:
				new_point = np.dot((vertex - p6['location']), p6['bases'])
				vertices_5.append(utils.get_screen_coordinates(new_point, p6['extent'], p6['center'], p6['theta']))

			vertices_5 = np.dot(vertices_5, R5)

			# project 5d points into 4d:
			vertices_4 = []
			for vertex in vertices_5: 
				new_point = np.dot((vertex - p5['location']), p5['bases'])
				vertices_4.append(utils.get_screen_coordinates(new_point, p5['extent'], p5['center'], p5['theta']))

			# print('vertices_4:', vertices_4)
			R4 = [	[cos(p4['d_theta'] * frame), -sin(p4['d_theta'] * frame), 0, 0],
			[sin(p4['d_theta'] * frame), cos(p4['d_theta'] * frame), 0, 0],
			[0, 0, 1, 0],
			[0, 0, 0, 1]]
			# vertices_4 = np.dot(vertices_4, R4)

			# project 4d points into 3d:
			vertices_3 = []
			for vertex in vertices_4:
				new_point = np.dot((vertex - p4['location']), p4['bases'])
				vertices_3.append(utils.get_screen_coordinates(new_point, p4['extent'], p4['center'], p4['theta']))

			# print('vertices_3:', vertices_3)

			# chart the points
			ax.clear()
			points = np.array(vertices_3)

			for pair in graph:
				end_points = np.array(points[pair])
				# print('end_points', end_points)
				p1 = end_points[0]
				p2 = end_points[1]
				ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]])

			ax.grid(False)
			ax.axis('off')
			ax.set_xlim(-axis_limits,axis_limits)
			ax.set_ylim(-axis_limits,axis_limits)
			ax.set_zlim(-axis_limits,axis_limits)
			ax.view_init(azim=41, elev=0)
			plt.show()
			plt.pause(0.001)
			frame += 1

def cube_7d(animate):
	delta_theta = .007
	# 7 dimensional parameters:
	p7 = {
		'origin': np.array([0,0,0,0,0,0,0]),
		'orientations': [[0,3,.4,0,0,0,0], [0,0,1,0,0,0,0], [0,0,0,1,0,0,0], [0,0,0,0,1,0,0], [0,0,0,0,0,1,0]],
		'location': np.array([7,3,3,3,0,0,0]),
		'theta': math.pi / 4,
		'd_theta': delta_theta,
		'extent': np.array([2,2,2,2,2,2]),
		'center': np.array([1,1,1,1,1,1])
	}

	# 6 dimensional parameters:
	p6 = {
		'origin': np.array([0,0,0,0,0,0]),
		'orientations': [[1,3,.4,0,0,0], [0,0,1,0,0,0], [0,0,0,1,0,0], [0,0,0,0,1,0]],
		'location': np.array([7,3,3,0,0,0]),
		'theta': math.pi / 4,
		'd_theta': delta_theta,
		'extent': np.array([2,2,2,2,2]),
		'center': np.array([1,1,1,1,1])
	}
	# 5 dimensional parameters:
	p5 = {
		'origin': np.array([0,0,0,0,0]),
		'orientations': [[1,1,.4,0,0], [0,4,1,0,0], [0,0,0,1,0]],
		'location': np.array([7,0,3,0,0]),
		'theta': math.pi / 4,
		'd_theta': delta_theta,
		'extent': np.array([2,2,2,2]),
		'center': np.array([1,1,1,1])
	}

	# 4 dimensional parameters:
	p4 = {
		'origin': np.array([0,0,0,0]),
		'orientations': [[0,1,.0,0], [-.3,0,2,0]],
		'location': np.array([-1,.2,1,0]),
		'theta': math.pi / 4,
		'd_theta': delta_theta,
		'extent': np.array([2,2,2]),
		'center': np.array([1,1,1])
	}

	# 7d rotation matrix
	R7 = [	[1, 0, 0, 0, 0, 0, 0],
			[0, cos(p6['d_theta']), -sin(p6['d_theta']), 0, 0, 0, 0],
			[0, sin(p6['d_theta']), cos(p6['d_theta']), 0, 0, 0, 0],
			[0, 0, 0, 1, 0, 0, 0],
			[0, 0, 0, 0, 1, 0, 0], 
			[0, 0, 0, 0, 0, 1, 0],
			[0, 0, 0, 0, 0, 0, 1]]

	# 6d rotation matrix
	R6 = [	[1, 0, 0, 0, 0, 0],
			[0, cos(p6['d_theta']), -sin(p6['d_theta']), 0, 0, 0],
			[0, sin(p6['d_theta']), cos(p6['d_theta']), 0, 0, 0],
			[0, 0, 0, 1, 0, 0],
			[0, 0, 0, 0, 1, 0], 
			[0, 0, 0, 0, 0, 1]]

	# 5d rotation matrix
	R5 = [	[cos(p5['d_theta']), -sin(p5['d_theta']), 0, 0, 0],
			[sin(p5['d_theta']), cos(p5['d_theta']), 0, 0, 0],
			[0, 0, 1, 0, 0],
			[0, 0, 0, 1, 0],
			[0, 0, 0, 0, 1]]


 	# set up the visualization
	plt.ion()
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	ax.view_init(azim=29, elev=-45)
	# ax.autoscale(enable=False)
	axis_limits = 4

	# create a 7d cube
	vertices_7, graph = create_cube(7)

	vertices_7 = ( vertices_7 * 2 ) - 1

	# get the 7d camera bases:
	p7['bases'] = utils.get_camera_bases(p7['origin'], p7['location'], p7['orientations'])

	# get the 6d camera bases:
	p6['bases'] = utils.get_camera_bases(p6['origin'], p6['location'], p6['orientations'])

	# get the 5d camera bases:
	p5['bases'] = utils.get_camera_bases(p5['origin'], p5['location'], p5['orientations'])

	# get the 4d camera bases:
	p4['bases'] = utils.get_camera_bases(p4['origin'], p4['location'], p4['orientations'])

	if animate:
		num_steps = np.round(2*math.pi / p4['d_theta']).astype(int)
		# num_steps = 10

		def update(i, vertices_7, p7, p6, p5, p4, num_steps):

			# 7d rotation matrix
			R7 = [	[1, 0, 0, 0, 0, 0, 0],
					[0, cos(p6['d_theta'] * i), -sin(p6['d_theta'] * i), 0, 0, 0, 0],
					[0, sin(p6['d_theta'] * i), cos(p6['d_theta'] * i), 0, 0, 0, 0],
					[0, 0, 0, 1, 0, 0, 0],
					[0, 0, 0, 0, 1, 0, 0], 
					[0, 0, 0, 0, 0, 1, 0],
					[0, 0, 0, 0, 0, 0, 1]]

			vertices_7 = np.dot(vertices_7, R7)

			# project 7d points into 6d:
			vertices_6 = []
			for vertex in vertices_7:
				new_point = np.dot((vertex - p7['location']), p7['bases'])
				vertices_6.append(utils.get_screen_coordinates(new_point, p7['extent'], p7['center'], p7['theta']))


			# 6d rotation matrix
			R6 = [	[1, 0, 0, 0, 0, 0],
					[0, cos(p6['d_theta'] * i), -sin(p6['d_theta'] * i), 0, 0, 0],
					[0, sin(p6['d_theta'] * i), cos(p6['d_theta'] * i), 0, 0, 0],
					[0, 0, 0, 1, 0, 0],
					[0, 0, 0, 0, 1, 0], 
					[0, 0, 0, 0, 0, 1]]

			vertices_6 = np.dot(vertices_6, R6)

			# project 6d points into 5d:
			vertices_5 = []
			for vertex in vertices_6:
				new_point = np.dot((vertex - p6['location']), p6['bases'])
				vertices_5.append(utils.get_screen_coordinates(new_point, p6['extent'], p6['center'], p6['theta']))

			# 5d rotation matrix
			R5 = [	[cos(p5['d_theta'] * i), -sin(p5['d_theta'] * i), 0, 0, 0],
					[sin(p5['d_theta'] * i), cos(p5['d_theta'] * i), 0, 0, 0],
					[0, 0, 1, 0, 0],
					[0, 0, 0, 1, 0],
					[0, 0, 0, 0, 1]]

			# vertices_5 = np.dot(vertices_5, R5)

			# project 5d points into 4d:
			vertices_4 = []
			for vertex in vertices_5: 
				new_point = np.dot((vertex - p5['location']), p5['bases'])
				vertices_4.append(utils.get_screen_coordinates(new_point, p5['extent'], p5['center'], p5['theta']))

			# print('vertices_4:', vertices_4)
			R4 = [	[cos(p4['d_theta'] * i), -sin(p4['d_theta'] * i), 0, 0],
			[sin(p4['d_theta'] * i), cos(p4['d_theta'] * i), 0, 0],
			[0, 0, 1, 0],
			[0, 0, 0, 1]]
			# vertices_4 = np.dot(vertices_4, R4)

			# project 4d points into 3d:
			vertices_3 = []
			for vertex in vertices_4:
				new_point = np.dot((vertex - p4['location']), p4['bases'])
				vertices_3.append(utils.get_screen_coordinates(new_point, p4['extent'], p4['center'], p4['theta']))

			# print('vertices_3:', vertices_3)

			# chart the points
			ax.clear()
			points = np.array(vertices_3)

			for pair in graph:
				end_points = np.array(points[pair])
				# print('end_points', end_points)
				p1 = end_points[0]
				p2 = end_points[1]
				ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]])

			ax.grid(False)
			ax.axis('off')
			ax.set_xlim(-axis_limits,axis_limits)
			ax.set_ylim(-axis_limits,axis_limits)
			ax.set_zlim(-axis_limits,axis_limits)
			# ax.view_init(azim=41, elev=5)
			print('frame ', i, ' of ', num_steps)
			return ax

		anim = FuncAnimation(fig, update, np.arange(0,num_steps), fargs=(vertices_7, p7, p6, p5, p4, num_steps), interval=5)
		anim.save('C:/Users/Teal/Documents/Python/Projects/Dimensions/squirt.mp4', writer='ffmpeg', dpi=500)

		# os.system("ffmpeg -i C:\\Users\\Teal\\Documents\\Python\\Projects\\Dimensions\\squirt.mp4 -vf scale=200:-1 -r 30 C:\\Users\\Teal\\Documents\\Python\\Projects\\Dimensions\\squirt.gif")

	else:
		frame = 0
		while True:
			vertices_7 = np.dot(vertices_7, R7)

			vertices_6 = []
			for vertex in vertices_7:
				new_point = np.dot((vertex - p7['location']), p7['bases'])
				vertices_6.append(utils.get_screen_coordinates(new_point, p7['extent'], p7['center'], p7['theta']))

			vertices_6 = np.dot(vertices_6, R6)

			# project 6d points into 5d:
			vertices_5 = []
			for vertex in vertices_6:
				new_point = np.dot((vertex - p6['location']), p6['bases'])
				vertices_5.append(utils.get_screen_coordinates(new_point, p6['extent'], p6['center'], p6['theta']))

			# vertices_5 = np.dot(vertices_5, R5)

			# project 5d points into 4d:
			vertices_4 = []
			for vertex in vertices_5: 
				new_point = np.dot((vertex - p5['location']), p5['bases'])
				vertices_4.append(utils.get_screen_coordinates(new_point, p5['extent'], p5['center'], p5['theta']))

			# print('vertices_4:', vertices_4)
			R4 = [	[cos(p4['d_theta'] * frame), -sin(p4['d_theta'] * frame), 0, 0],
			[sin(p4['d_theta'] * frame), cos(p4['d_theta'] * frame), 0, 0],
			[0, 0, 1, 0],
			[0, 0, 0, 1]]
			# vertices_4 = np.dot(vertices_4, R4)

			# project 4d points into 3d:
			vertices_3 = []
			for vertex in vertices_4:
				new_point = np.dot((vertex - p4['location']), p4['bases'])
				vertices_3.append(utils.get_screen_coordinates(new_point, p4['extent'], p4['center'], p4['theta']))

			# print('vertices_3:', vertices_3)

			# chart the points
			ax.clear()
			points = np.array(vertices_3)

			for pair in graph:
				end_points = np.array(points[pair])
				# print('end_points', end_points)
				p1 = end_points[0]
				p2 = end_points[1]
				ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]])

			ax.grid(False)
			ax.axis('off')
			ax.set_xlim(-axis_limits,axis_limits)
			ax.set_ylim(-axis_limits,axis_limits)
			ax.set_zlim(-axis_limits,axis_limits)
			# ax.view_init(azim=18, elev=-40)
			plt.show()
			plt.pause(0.001)
			frame += 1


# project_tesseract()
# moving_camera_tesseract_animation()

# rotating_vertices_tesseract_vis()
# rotating_vertices_tesseract_animation()
# lazy_cube(animate=True)
# cube_6d(animate=False)
# cube_7d(animate=True)