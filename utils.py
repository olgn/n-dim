# Contains the utilities for
# the n-dimensional projections

import itertools as it
import numpy as np
import math

cos = math.cos
sin = math.sin



def get_vertex_index(vertex, all_vertices):
	# Determines the index of a specific vertex (vertex) in an array
	# of vertices (all_vertices)
	#
	# Inputs: 
	# vertex (array, length n): the vertex whose index we'd like to know
	# all_vertices (matrix, with m distinct length n arrays): the list of vertexes we want to search
	#
	# Outputs: 
	# index (integer): the index of the vertex in the vertex list

	index = -1
	for i in range(0, len(all_vertices)):
		if not (np.any(vertex - all_vertices[i])):
			index = i

	return index


def create_associated_vertex_graph(all_vertices, shape):
	# Given an array of vertices (all_vertices), list all the
	# points that are connected to create the n-dimensional shape (shape).

	if (shape == 'cube'):
		# calculates the norm between every combination of vertices, and associates any pair
		# that have norm 1.

		associated_vertices = []

		for i, j in it.combinations(all_vertices, 2):
			if (np.linalg.norm(i - j) == 1.0):
				idx_1 = get_vertex_index(i, all_vertices)
				idx_2 = get_vertex_index(j, all_vertices)
				# print(idx_1, idx_2)
				associated_vertices.append([idx_1, idx_2])

		return associated_vertices

	return


def get_camera_bases(point, camera_location, camera_orientation_array):
	# Given a point in space to focus on, as well as a location for a camera in (n)-space and
	# a camera orientation array of (n-1) elements each with dimension n, transform the
	# previous basis vectors into the camera's perspective.
	#
	# Inputs:
	# point (array, length n): the point the camera will be facing
	# camera_location (array, length n): the origin of the camera's basis coordinates
	# camera_orientation_array (matrix, with n-1 arrays of length n): a matrix consisting of n-1 vectors defining the orientation of the camera
	# with respect to the original origin
	#
	# Outputs:
	# camera_bases (matrix, n arrays of length n): the normalized camera basis vectors

	# Some sizes of arrays that will ensure all variables are of the proper shape:
	number_of_orientation_vectors = len(camera_orientation_array) # how many camera orientation vectors did we define?
	number_of_dimensions = len(point) # how many dimensions is each point?
	number_of_camera_dimensions = len(camera_location) # used to ensure camera dimensions are the same as point dimensions

	# Check to make sure the camera location has the same dimensionality as the point we're shifting:
	if not (number_of_dimensions == number_of_camera_dimensions):
		print('point must have the same number of dimensions as the camera location.')
		return None

	# Check to make sure that there are two fewer orientation vectors than there are dimensions
	if not (number_of_dimensions - number_of_orientation_vectors == 2):
		print('n dimensional points require n - 2 camera perspective arrays. we got ', number_of_orientation_vectors, ' bases for a point in ', number_of_dimensions, 'dimensional space.')
		if (number_of_orientation_vectors >= 1):

			# check to make sure that the every orientation vector is of the right dimension
			for i in range(0, number_of_orientation_vectors):
				perspective_array = camera_orientation_array[i]
				if not (len(perspective_array) == number_of_dimensions):
					print('at least one camera_orientation_array doesnt have the right number of dimensions.')
					return None
		return None

	# Calculate the new basis vectors from the camera's perspective:

	# The n-1 basis vector (n_n_1) is given by 
	# (point - camera_location) / || point - camera_location ||
	distance_vector = point - camera_location
	distance_norm = np.linalg.norm(distance_vector)
	n_n_1 = distance_vector / distance_norm


	# Create the basis vectors from n_(n-2) to n_1 (n_1_to_n_n_2):
	n_stack = [n_n_1]
	o_stack = camera_orientation_array

	n_1_to_n_n_2 = []

	for i in range(0, number_of_dimensions - 2):
		vectors = np.vstack((o_stack, n_stack))
		cross = cross_product(vectors)

		new_basis = cross / np.linalg.norm(cross)

		n_1_to_n_n_2.append(new_basis)

		o_stack = np.delete(o_stack, len(o_stack) - 1, 0)
		n_stack = np.vstack((n_stack, new_basis))


	# print('first_batch:', n_1_to_n_n_2)

	# Create the last basis vector:
	n_n = cross_product(n_stack)

	# Stack the bases in the right order:
	camera_bases = np.vstack((np.vstack((np.flipud(n_1_to_n_n_2), n_n_1)),n_n))

	# print('camera_bases:')
	# print(camera_bases)

	return camera_bases


def get_screen_coordinates(point, extent, center, theta):
	# Transform a point from the camera's perspective in n-space to an
	# n-l 'screen'.
	#
	# Inputs:
	# point (array, length n): the point to be transformed
	# extent (array, length n-1): the dimensions of the screen
	# center (array, length n-1): the center of the screen
	# theta (float): the angle of the field of view, in radians. should be less than pi / 2
	#
	# Outpus:
	# screen_coordinates (array, length n - 1): the new coordiantes of the point on the screen

	denominator = 2 * point[-1] * math.tan(theta / 2)
	point = point [:-1]
	if denominator == 0:
		second_term = np.array([0] * len(center))
	else:
		second_term = (extent * point) / denominator

	screen_coordinates = center + second_term

	return screen_coordinates


def cross_product(vectors):
	# Calculates the cross product of a matrix of vectors.
	# Because we will have a np.vstack of the vectors, 
	# we'll add the top row of +1, -1 that is required for determinants
	# and add it before calculation. hese represent the i,j,k, ... basis vectors
	# found in a normal cross product calculation.
	# 
	# Inputs:
	# vectors (matrix, size n-1 x n): the vectors of which we determine the cross product
	#
	# Outputs:
	# cross_product (array, length n): the cross product of the input vectors

	# print('calculating cross product of the matrix of vectors:')
	# print(vectors)

	if not (len(vectors) == len(vectors[0] ) - 1):
		print('there must be exactly one less vector than there are dimensions.')
		return

	cross_product = []
	number_of_dimensions = len(vectors[0])
	top_row = np.array([math.pow(-1, i) for i in range(0, number_of_dimensions)])
	determinant_matrix = np.vstack((top_row, vectors))

	# print('the determinant matrix is:')
	# print(determinant_matrix)

	for i in range(0, number_of_dimensions):
		sub_matrix = np.delete(np.delete(determinant_matrix, 0, 0), i, 1)
		# print('sub_matrix', sub_matrix)
		determinant = np.linalg.det(sub_matrix)
		# print('determinant', determinant)
		value = top_row[i] * determinant
		cross_product.append(value)

	# print('the calculated cross_product is:', cross_product)
	return cross_product