#granularity=0.02 / thenumberofplane=49 (=1/granularity-1=N-1에 해당, W는 0 이상 N-1 이하, W는 여기 반영 안 함) / sampling_size=200

import pyvista as pv
#import open3d as o3d

import numpy as np
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import math

#file_path = "C:/Users/user/Desktop/bunny/reconstruction/bun_zipper.ply"

#mesh = pv.read(file_path)
mesh = pv.read('bun_zipper.ply')
#mesh = o3d.io.read_point_cloud('bun_zipper.ply')

plotter = pv.Plotter()
plotter.add_mesh(mesh)

mesh_array=mesh.points

min_x, min_y, min_z = np.min(mesh_array, axis=0)
max_x, max_y, max_z = np.max(mesh_array, axis=0)

normalized_array = (mesh_array - [min_x, min_y, min_z]) / [max_x - min_x, max_y - min_y, max_z - min_z]
array_size=normalized_array.size

sampling_size=200
#sampling_size=np.floor((array_size/3)/200).astype(int)

def fps(points, n_samples):
    points = np.array(points)
    points_left = np.arange(len(points))
    sample_inds = np.zeros(n_samples, dtype='int')
    dists = np.ones_like(points_left) * float('inf')
    selected = 0
    sample_inds[0] = points_left[selected]
    points_left = np.delete(points_left, selected)
    
    for i in range(1, n_samples):
        last_added = sample_inds[i-1]
        dist_to_last_added_point = ((points[last_added] - points[points_left])**2).sum(-1)
        dists[points_left] = np.minimum(dist_to_last_added_point, dists[points_left])
        selected = np.argmax(dists[points_left])
        sample_inds[i] = points_left[selected]
        points_left = np.delete(points_left, selected)
    
    return points[sample_inds]

def hamming_fps(points, n_samples):
    points = np.array(points)
    points_left = np.arange(len(points))
    sample_inds = np.zeros(n_samples, dtype='int')
    dists = np.ones_like(points_left) * float('inf')
    selected = 0
    sample_inds[0] = points_left[selected]
    points_left = np.delete(points_left, selected)
    
    for i in range(1, n_samples):
        last_added = sample_inds[i-1]
        hamming_dist_to_last_added_point = np.sum(points[last_added] != points[points_left], axis=1)
        dists[points_left] = np.minimum(hamming_dist_to_last_added_point, dists[points_left])
        selected = np.argmax(dists[points_left])
        sample_inds[i] = points_left[selected]
        points_left = np.delete(points_left, selected)
    
    return points[sample_inds]

def hashing_function(x, y, z, granularity=0.02):
    numberofplanes=math.ceil(1/granularity) - 1
    results = np.zeros(2 * 3 * numberofplanes)
    
    for plane_x in range(numberofplanes):
        x_result = 1 if x >= (plane_x+1) * granularity else 0
        x_result_bar=1 - x_result
        results[plane_x] = x_result_bar
        results[numberofplanes + plane_x] = x_result
    
    for plane_y in range(numberofplanes):
        y_result = 1 if y >= (plane_y+1) * granularity else 0
        y_result_bar=1 - y_result
        results[2 * numberofplanes + plane_y] = y_result_bar
        results[3 * numberofplanes + plane_y] = y_result
    
    for plane_z in range(numberofplanes):
        z_result = 1 if z >= (plane_z+1) * granularity else 0
        z_result_bar=1 - z_result
        results[4 * numberofplanes + plane_z] = z_result_bar
        results[5 * numberofplanes + plane_z] = z_result
          
    return results


def inverse_hashing_function(hashed_points, granularity=0.02):
    numberofplanes = math.ceil(1/granularity) - 1
    num_points, num_planes = hashed_points.shape
    
    original_coordinates = np.zeros((num_points, 3))
    
    for i in range(num_points):
        x = 0.0
        y = 0.0
        z = 0.0
        
        for plane_x in range(numberofplanes):
            if hashed_points[i, numberofplanes+plane_x] == 1:
                x = (plane_x + 1) * granularity
    
        for plane_y in range(numberofplanes):
            if hashed_points[i, 3 * numberofplanes + plane_y] == 1:
                y = (plane_y + 1) * granularity
    
        for plane_z in range(numberofplanes):
            if hashed_points[i, 5 * numberofplanes + plane_z] == 1:
                z = (plane_z + 1) * granularity
        
        original_coordinates[i, :] = [x, y, z]
    
    return original_coordinates


random_array = normalized_array
sampled_array = fps(random_array, sampling_size)

thenumberofplane=49
hashed_array = np.zeros((int(array_size/3), thenumberofplane * 6))

for i in range(int(array_size/3)):
    x0 = random_array[i, 0]
    y0 = random_array[i, 1]
    z0 = random_array[i, 2]
    hashed_point = hashing_function(x0, y0, z0, granularity=0.02)
    hashed_array[i, :] = hashed_point

hashedandsampled_array = hamming_fps(hashed_array, sampling_size)

inverse_hashed_original_hashed_array = inverse_hashing_function(hashed_array, granularity=0.02)
inverse_hashed_sampled_array = inverse_hashing_function(hashedandsampled_array, granularity=0.02)

print("Original Array:")
print(random_array)

print("Sampled Array:")
print(sampled_array)

print("Hashed Array:")
print(hashed_array)

print("\nHashed and Sampled Array:")
print(hashedandsampled_array)


x1=random_array[:,0]
y1=random_array[:,1]
z1=random_array[:,2]
x2=sampled_array[:,0]
y2=sampled_array[:,1]
z2=sampled_array[:,2]
x3=inverse_hashed_original_hashed_array[:,0]
y3=inverse_hashed_original_hashed_array[:,1]
z3=inverse_hashed_original_hashed_array[:,2]
x4=inverse_hashed_sampled_array[:,0]
y4=inverse_hashed_sampled_array[:,1]
z4=inverse_hashed_sampled_array[:,2]


fig=plt.figure()
ax=fig.add_subplot(111,projection='3d')

ax.scatter(x1,y1,z1,c='b',label='Original Array')

plt.savefig("filename.jpg")


fig=plt.figure()
ax=fig.add_subplot(111,projection='3d')

ax.scatter(x2,y2,z2,c='r',label='Sampled Array')

plt.savefig("filename2.jpg")


fig=plt.figure()
ax=fig.add_subplot(111,projection='3d')

ax.scatter(x3,y3,z3,c='c',marker='^',label='Hashed Array')

plt.savefig("filename3.jpg")


fig=plt.figure()
ax=fig.add_subplot(111,projection='3d')

ax.scatter(x4,y4,z4,c='m',marker='^',label='Hashed and Sampled Array')

plt.savefig("filename4.jpg")


#poly_data = pv.PolyData(random_array)

#plotter = pv.Plotter()
#plotter.add_mesh(poly_data, color='m', style='wireframe')

#plotter.show("filename5.jpg")


#poly_data = pv.PolyData(sampled_array)

#plotter = pv.Plotter()
#plotter.add_mesh(poly_data, color='m', style='wireframe')

#plotter.show("filename6.jpg")


#poly_data = pv.PolyData(inverse_hashed_original_hashed_array)

#plotter = pv.Plotter()
#plotter.add_mesh(poly_data, color='m', style='wireframe')

#plotter.show("filename7.jpg")


#poly_data = pv.PolyData(inverse_hashed_sampled_array)

#plotter = pv.Plotter()
#plotter.add_mesh(poly_data, color='m', style='wireframe')

#plotter.show()

#end