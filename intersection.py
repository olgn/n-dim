import numpy as np
import math
import utils
import itertools as it
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

cos = math.cos
sin = math.sin

def intersect_3d(p0, p1, p_co, p_no, epsilon=1e-6):
	u = p1 - p0
	print(u)
	dot = np.dot(p_no, u)
	if abs(dot) > epsilon:
		w = p0 - p_co
		fac = -(np.dot(p_no, w)) / dot
		if (fac >= 0 and fac <= 1):
			return p0 + u * fac
	return None

def intersect_3d_tetrahedron():
	theta = 0.05
	R3 = np.array([	[cos(theta), -sin(theta), 0],
					[sin(theta), cos(theta), 0],
					[0, 0, 1]])

	
	points = np.array([	[-1, 0, 1/math.sqrt(2)],
						[1, 0, 1/math.sqrt(2)],
						[0, 1, -1/math.sqrt(2)],
						[0, -1, -1/math.sqrt(2)]])
	plt.ion()
	ax = plt.axes(projection='3d')
	plt.show()

	while True:
		ax.clear();
		points = np.dot(points, R3)
		p_no = np.array([.7,.5,1]);
		p_co = np.array([0,0,0]);

		# plot the plane:
		d = -p_co.dot(p_no)
		xx, yy = np.meshgrid(range(-2,2), range(-2,2))

		# calculate corresponding z
		z = (-p_no[0] * xx - p_no[1] * yy - d) * 1. /p_no[2]
		print('z', z)
		ax.plot_wireframe(xx, yy, z)

		edges = [comb for comb in it.combinations(points, 2)]
		print('edges:', len(edges),' - ', edges);
		print('points', points);

		intersections = []
		for edge in edges:
			print(edge)
			p0 = edge[0]
			p1 = edge[1]
			intersection = utils.intersect(p0, p1, p_co, p_no)
			if (intersection is not None):
				intersections.append(intersection)
				ax.scatter([intersection[0]], [intersection[1]], [intersection[2]], 'red', s=12)
			ax.plot([p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]])

		print('intersections', intersections)
		plt.pause(0.01)
	
def intersect_4d_tetrahedron(p_co, p_no, R3, ax, color):
	# 4 dimensional parameters:
	p4 = {
		'origin': p_co,
		'orientations': [[0,1,.21,0], [.3,0,0,0]],
		'location': p_no,
		'theta': math.pi / 4,
		'd_theta': .01,
		'extent': np.array([2,2,2]),
		'center': np.array([1,1,1])
	}

	# get the 4d camera bases:
	p4['bases'] = utils.get_camera_bases(p4['origin'], p4['location'], p4['orientations'])

	# points = np.array([	[2, 0, 0, 0],
	# 					[0, 2, 0, 0],
	# 					[0, 0, 2, 0],
	# 					[0, 0, 0, 2]])

	points = np.array([	[1/math.sqrt(10), 1/math.sqrt(6), 1/math.sqrt(3), 1],
						[1/math.sqrt(10), 1/math.sqrt(6), 1/math.sqrt(3), -1],
						[1/math.sqrt(10), 1/math.sqrt(6), -2/math.sqrt(3), 0],
						[1/math.sqrt(10), -math.sqrt(3/2), 0, 0],
						[-2*math.sqrt(2/5), 0, 0, 0]])


	while True:
		ax.clear();
		points = np.dot(points, R3)

		edges = [comb for comb in it.combinations(points, 2)]
		print('edges:', len(edges),' - ', edges)
		print('points', points)

		intersections = []
		for edge in edges:
			print(edge)
			p0 = edge[0]
			p1 = edge[1]
			intersection = utils.intersect(p0, p1, p_co, p_no)
			if (intersection is not None):
				intersections.append(intersection)
			

		print('intersections', intersections)

		# project 4d points into 3d:
		vertices_3 = []
		for vertex in intersections:
			new_point = np.dot((vertex - p4['location']), p4['bases'])
			projection = utils.get_screen_coordinates(new_point, p4['extent'], p4['center'], p4['theta'])
			ax.scatter([projection[0]], [projection[1]], [projection[2]], 'red', c=color, s=15)
			vertices_3.append(projection)

		x = [vertex[0] for vertex in vertices_3]
		y = [vertex[1] for vertex in vertices_3]
		z = [vertex[2] for vertex in vertices_3]
		ax.plot_trisurf(x, y, z, color=color, alpha=.5)
		print('vertices_3:', vertices_3)


		plt.pause(0.01)

def intersect_4d_cube(p_co, p_no, R3, color):
	# 4 dimensional parameters:
	p4 = {
		'origin': p_co,
		'orientations': [[0,1, 2 ,0], [.3,0,0,-.1]],
		'location': [1,12,0,2],
		'theta': math.pi / 4,
		'd_theta': .01,
		'extent': np.array([2,2,2]),
		'center': np.array([1,1,1])
	}

	# get the 4d camera bases:
	p4['bases'] = utils.get_camera_bases(p4['origin'], p4['location'], p4['orientations'])

	# points = np.array([	[2, 0, 0, 0],
	# 					[0, 2, 0, 0],
	# 					[0, 0, 2, 0],
	# 					[0, 0, 0, 2]])

	
	points = np.array([	[-1, 0, 0, 0],
						[0, -1, 0, 0],
						[0, 0, -1, 0],
						[0, 0, 0, -1],
						[1, 0, 0, 0],
						[0, 1, 0, 0],
						[0, 0, 1, 0],
						[0, 0, 0, 1]])

	points_2 = np.array([	[1/math.sqrt(10), 1/math.sqrt(6), 1/math.sqrt(3), 1],
					[1/math.sqrt(10), 1/math.sqrt(6), 1/math.sqrt(3), -1],
					[1/math.sqrt(10), 1/math.sqrt(6), -2/math.sqrt(3), 0],
					[1/math.sqrt(10), -math.sqrt(3/2), 0, 0],
					[-2*math.sqrt(2/5), 0, 0, 0]])
	shapes = [points, points_2]
	fig = plt.figure(1)
	subplot_list = [len(shapes)*100 + 11 + i for i in range(len(shapes))]
	axes = []
	for val in subplot_list:
		axes.append(fig.add_subplot(val, projection='3d'))

	while True:
		fig.clear()

		for idx in range(0, len(shapes)):
			axes[idx] = fig.add_subplot((len(shapes)*100) + 11 + idx, projection='3d')
			axes[idx].grid(False)
			points = shapes[idx]

			shapes[idx] = np.dot(shapes[idx], R3)

			edges = [comb for comb in it.combinations(points, 2)]

			intersections = []
			for edge in edges:
				p0 = edge[0]
				p1 = edge[1]
				intersection = utils.intersect(p0, p1, p_co, p_no)
				if (intersection is not None):
					intersections.append(intersection)

			# project 4d points into 3d:
			vertices_3 = []
			for vertex in intersections:
				new_point = np.dot((vertex - p4['location']), p4['bases'])
				projection = utils.get_screen_coordinates(new_point, p4['extent'], p4['center'], p4['theta'])
				axes[idx].scatter([projection[0]], [projection[1]], [projection[2]], s=15)
				vertices_3.append(projection)

			x = [vertex[0] for vertex in vertices_3]
			y = [vertex[1] for vertex in vertices_3]
			z = [vertex[2] for vertex in vertices_3]
			axes[idx].plot_trisurf(x, y, z, color=color)
			# print('vertices_3:', vertices_3)


		plt.pause(0.02)
plt.ion()
	
# intersect_3d_tetrahedron()
p_no = np.array([.7,1,1, -1]);
p_co = np.array([0,0,0,0]);

theta = 0.05
R3 = np.array([	[cos(theta), -sin(theta), 0, 0],
				[sin(theta), cos(theta), 0, 0],
				[0, 0, 1, 0],
				[0, 0, 0, 1]])

p_no = np.array([-.7,1,0, 1]);
p_co = np.array([0,0,.01,0]);
intersect_4d_cube(p_co, p_no, R3, (0,0,1,.5))
