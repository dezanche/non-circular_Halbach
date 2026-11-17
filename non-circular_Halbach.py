"""
This program illustrates how conformal mapping can be used to design Halbach magnet arrays on various shapes.

    Copyright (C) 2026  Nicola De Zanche

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import numpy as np
import magpylib as magpy
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches
from scipy import ndimage
from shapely.geometry import Polygon
from conformal_transformations import *
import tkinter as tk
from tkinter import ttk


N = 20              # number of magnets in Halbach array (NB for circle, Cassinian white path fails with N>20)
cube_side = 0.01    # lateral dimension of magnet cuboids [m]
cube_length = 0.2   # length of magnet cuboids [m] (along z)
radius = 0.11       # field sampling radius [m] for display
dens = 8            # sampling density parameter

# parameters for calculating field homogeneity; may need to be adjusted depending on N
shrink_scale = 0.8   # factor to shrink original shape for naive calculation
shrink_scale2 = 0.9  # factor to shrink more representative shape determined from field minima 
min_angle_sep = 0.1  # minimum angular separation (needed for star and rectangle)

# use_plotly = 1      # use plotly to display array from magpylib (looks better)
plt.ion()

# pick one shape from the options:
# shape =  "star" # circle, ellipse, rectangle, Cassinian, triangle, cardioid, rounded rectangle, star, tabletop

# %% popup
def on_select():
    selected_shape = shape_var.get()
    filled = filled_var.get()
    
    print(f"Selected shape: {selected_shape}")
    print(f"Use plotly: {filled}")
    
    root.destroy()

# Create main window
root = tk.Tk()
root.title("Select Shape")
root.geometry("300x180")
root.resizable(False, False)
root.attributes("-topmost", True)

# Label
tk.Label(root, text="Choose a shape:").pack(pady=10)

# Shape options
shapes = [
    "circle",
    "ellipse",
    "cardioid",
    "Cassinian",
    "rectangle",
    "rounded rectangle",
    "triangle",
    "tabletop",
    "star"
]

# Dropdown
shape_var = tk.StringVar(value=shapes[0])
combo = ttk.Combobox(
    root,
    textvariable=shape_var,
    values=shapes,
    state="readonly"
)
combo.pack(pady=5)

# Checkbox
filled_var = tk.BooleanVar(value=False)
tk.Checkbutton(
    root,
    text="Use plotly",
    variable=filled_var
).pack(pady=5)

# OK button
tk.Button(root, text="OK", command=on_select).pack(pady=10)

root.mainloop()

shape = shape_var.get()
use_plotly = filled_var.get()

# %%
# print("Shape selected = " + shape + ". Calculating...")

# angles needed for calculations
theta = np.linspace(0, 2*np.pi, N, endpoint=False)
alpha = theta + np.pi / 2  # orientation of each magnet (gives vertically-oriented field; remove pi / 2 for horizontal)

# %% shape-specific code
match shape:
    case "circle":
        d = 0.1  # radius
        positions = circle(theta, d)

    case "rectangle":
        # exact rectangle
        scale = 0.15
        positions, rect_X, rect_Y = rectangle(theta,0.6)
        positions = scale * positions

    case "ellipse":
        positions = Zhukovskii(theta, 0.07, 0.1)

    case "Cassinian":
        # oval of Cassini from MRM 53(1): 201-211 (2005)
        a=0.3/8         # minor radius [m]
        b=0.6/8         # major radius [m]
        c=np.sqrt((b**2-a**2)/2)      # 1/2 distance between foci
        d=np.sqrt((b**2+a**2)/2)		# radius of circle in model plane
        positions = Cassini(theta, c, d)

    case "cardioid":
        # cardioid
        scale = 1j*0.03  # scaling parameter (imaginary to have LR symmetry)
        d = 1.05        # radius in model plane (slightly >1 to avoid the cusp)
        positions = scale * cardioid(theta, d)

    case "triangle":
        # approximate triangle
        scale = 0.03 # global scaling parameter
        b = 2.5
        b_prime = scale * b
        c1 = 0.5
        c2 = -2 * 1j
        c3 = 0
        c4 = 2 * 1j
        coeffs = np.array([b_prime, c1/b_prime * scale**2 , c2/b_prime**2 * scale**3, c3/b_prime**3 * scale**4])
        positions = Laurent(theta, coeffs)

    case "rounded rectangle":
        # rounded rectangle using general Laurent series with appropriate coefficients
        scale = 0.03 # global scaling parameter
        b = 2.5
        b_prime = scale * b
        c1 = 1.5
        c2 = 0
        c3 = -3
        c4 = 0
        coeffs = np.array([b_prime, c1/b_prime * scale**2 , c2/b_prime**2 * scale**3, c3/b_prime**3 * scale**4])
        positions = Laurent(theta, coeffs)

    case "star":
        # approximate 5-pointed star shape (best if N is multiple of 5)
        scale = 0.03 # global scaling parameter
        b = 2.5
        b_prime = scale * b
        c1 = 0
        c2 = 0
        c3 = 0
        c4 = 15. * 1j
        coeffs = np.array([b_prime, c1/b_prime * scale**2 , c2/b_prime**2 * scale**3, c3/b_prime**3 * scale**4, c4/b_prime**4 * scale**5])
        positions = Laurent(theta, coeffs)

    case "tabletop":
        # flattened oval from MRM 60(2): 431-438 (2008)
        scale = 0.03 # global scaling parameter
        b = 2.5
        b_prime = scale * b
        c1 = 1
        c2 = -0.8 * 1j
        c3 = 0
        c4 = 0.5 * 1j
        coeffs = np.array([b_prime, c1/b_prime * scale**2 , c2/b_prime**2 * scale**3, c3/b_prime**3 * scale**4, c4/b_prime**4 * scale**5])
        positions = Laurent(theta, coeffs)
        
    case _:
        # throw error if unknown shape
        raise ValueError("ERROR: Unknown shape " + shape + "!")

# center shape needed in some cases for homogeneity calculation because of asymmetry
centroid = np.mean(positions, axis=0)
positions = positions - centroid

# %% create the Halbach array using magpylib
halbach = magpy.Collection()

# style for displaying with matplotlib
mstyle = {
    "mode": "color+arrow",
    "color": {
        "north": "crimson",
        "middle": "whitesmoke",
        "south": "dodgerblue"
    },
    "arrow": {"size": 0.1, "width": 2, "color": "black"}
}

for n in range(N):
    cube = magpy.magnet.Cuboid(
        # dimension = cube_side * np.ones([3, ]),
        dimension = (cube_side, cube_side, cube_length),
        polarization=(1, 0, 0),
        position = (np.real(positions[n]) , np.imag(positions[n]), 0),
        style_magnetization=mstyle
    )
    # cube.rotate_from_angax(theta[n], 'z', anchor=0, degrees = False)
    cube.rotate_from_angax(alpha[n] + theta[n], 'z', degrees = False)
    halbach.add(cube)

# display the Halbach array
if use_plotly:
    import plotly.io as pio
    pio.renderers.default = "browser"
    halbach.show(backend='plotly')  # doesn't work directly on its own in Spyder
else:
    magpy.defaults.display.style.magnet.magnetization.color.mode = "bicolor"
    halbach.show(backend='matplotlib')
# plt.show()

fig, ax = plt.subplots()

# Compute and plot field on x-y grid
grid = np.mgrid[-1.1*radius:1.1*radius:100j, -1.1*radius:1.1*radius:100j, 0:0:1j].T[0]
X, Y, _ = np.moveaxis(grid, 2, 0)

B = halbach.getB(grid)
Bx, By, _ = np.moveaxis(B, 2, 0)
Bamp = np.linalg.norm(B, axis=2)

pc = ax.contourf(X, Y, Bamp, levels=50, cmap="coolwarm")
ax.streamplot(X, Y, Bx, By, color="k", density=1.5, linewidth=1)

# Add colorbar
fig.colorbar(pc, ax=ax, label="|B|")

# Figure styling
ax.set(
    title = shape + " shape: B field lines and |B| in central plane",
    xlabel="X [m]",
    ylabel="Y [m]",
    aspect=1,
)

# plt.show()


# %% analytical results using ideal magnetic dipoles

m0 = 1           # same dipole moment for all Halbach elements
matsize = 2**dens - 1                     # size of sampling matrices
center = 2**(dens - 1)                    # location of origin
max_dev = center - 1                      # max deviation from center
delta = radius / max_dev                       # sampling step

# Create sampling grid (complex plane)
x = np.arange(-max_dev, max_dev + 1)
y = np.arange(-max_dev, max_dev + 1)
X, Y = np.meshgrid(x, y)

sampling_points = delta * (X + 1j * Y)

# Slight rotation to eliminate artifact (needed in some cases)
sampling_points *= np.exp(1j * 0.001)


# find subset of sampling points that are inside scaled down version of the shape
# NB: this shape in general is not the scaled-down region with optimal homogeneity because of non-uniform spacing of field sources

shrunk_polygon_path = Path([(z.real, z.imag) for z in shrink_scale * positions] + [(shrink_scale * positions[0].real, shrink_scale * positions[0].imag)], closed=True)
# mask = shrunk_polygon_path.contains_points([(z.real, z.imag) for z in 0.9 * sampling_points.flatten()])
stacked_points = np.column_stack((
    (1. * sampling_points).real.flatten(),
    (1. * sampling_points).imag.flatten()
))
mask = shrunk_polygon_path.contains_points(stacked_points)


# Initialize complex potential (W) and field (H)
W = np.zeros((matsize, matsize), dtype=complex)
H = np.zeros((matsize, matsize), dtype=complex)

# Loop over elements
for n in range(N):
    m = m0 * np.exp(1j * (alpha[n] + theta[n]))     # dipole moment
    pot_elm_n = -1j * m / (sampling_points - positions[n])
    W += pot_elm_n
    field_elm_n = 1j * m / (sampling_points - positions[n])**2
    H += field_elm_n

masked_H = H * mask.reshape(matsize,matsize)

# naive homogeneity calculation using scaled-down shape
non_zeros = H[np.abs(masked_H) != 0]
CoV = np.std(np.abs(non_zeros)) / np.mean(np.abs(non_zeros))  # homogeneity

print("|B| CoV = " + str(CoV * 100) + "% within " + str(shrink_scale * 100) + "% scaled shape (grey outline)")

# %% smarter homogeneity calculation finding shape of region where field is uniform

# find local field minima
local_min_filter = ndimage.minimum_filter(np.abs(H), size = (10,10))
is_min = (np.abs(H) == local_min_filter)
row_indices, col_indices = np.where(is_min)
minima_points = np.column_stack((
    delta * X[row_indices, col_indices],
    delta * Y[row_indices, col_indices]
))

# exclude minima outside the given shape
shape_path = Path([(z.real, z.imag) for z in positions] + [(positions[0].real, positions[0].imag)], closed=True)
inside_points = minima_points[shape_path.contains_points(minima_points)]

# exclude minima near the center (needed for the circle)
dist = np.sqrt((inside_points[:, 0])**2 + (inside_points[:, 1])**2)
inside_points = inside_points[dist >= 0.2*radius]

# sort points to remove intersecting lines
# centroid = np.mean(inside_points, axis=0)
inside_points_angles = np.arctan2(inside_points[:, 1] - np.imag(centroid), inside_points[:, 0] - np.real(centroid))
inside_points = inside_points[np.argsort(inside_points_angles)]
# sorted_inside_points_angles = np.arctan2(inside_points[:, 1] - np.imag(centroid), inside_points[:, 0] - np.real(centroid))
sorted_inside_points_angles = inside_points_angles[np.argsort(inside_points_angles)]
sorted_dist = dist[np.argsort(inside_points_angles)]

# keep first point and those separated at least by the minimum angle (needed for star)
keep = [0]
for i in range(1, len(sorted_inside_points_angles)):
    # if sorted_inside_points_angles[i] - sorted_inside_points_angles[i-1] >= min_angle_sep:
    #     keep.append(i)
    if sorted_inside_points_angles[i] - sorted_inside_points_angles[i-1] < min_angle_sep:
        if sorted_dist[i] < sorted_dist[i-1]:
            keep[-1] = i  # of the 2 keep the one with the smaller radius
    else: keep.append(i)

# angles = angles[keep]
inside_points = inside_points[keep]

shrunk_inside_points_path = Path(shrink_scale2 * inside_points, closed=True)
shrunk_inside_points_patch = patches.Polygon(shrink_scale2 * inside_points, closed=True,  facecolor='none', edgecolor='white', lw=1)

mask2 = shrunk_inside_points_path.contains_points(stacked_points)
masked_H = H * mask2.reshape(matsize,matsize)
non_zeros = H[np.abs(masked_H) != 0]

CoV2 = np.std(np.abs(non_zeros)) / np.mean(np.abs(non_zeros))  # homogeneity v2

print("|B| CoV = " + str(CoV2 * 100) + "% within " + str(shrink_scale2 * 100) + "% scaled polygon identified by field minima (white outline)")

# %%---- Plot shape, dipole positions and orientations, field lines ----
no_contours = 25
max_W = W.real.flatten()[mask].max()    # correct range without excess lines
contour_levels = max_W * np.arange(-no_contours, no_contours + 1) / no_contours

fig, ax = plt.subplots(1, 2, figsize=(10, 5))

# Plot dipole positions
polygon_path = Path([(z.real, z.imag) for z in positions] + [(positions[0].real, positions[0].imag)], closed=True)
polygon_patch = patches.PathPatch(polygon_path, facecolor='none', edgecolor='green', lw=2)

# draw contour and magnet positions
ax[0].plot(np.real(positions), np.imag(positions), 'ko')
ax[0].add_patch(polygon_patch)

# draw orientation of each magnet
dx = 0.025* np.cos(alpha + theta)
dy = 0.025* np.sin(alpha + theta)
ax[0].quiver(
    positions.real - dx/2,
    positions.imag - dy/2,
    dx , dy,
    angles='xy',
    scale_units='xy',
    scale=1
)

ax[0].set_xlim(-radius , radius)
ax[0].set_ylim(-radius , radius)
ax[0].set_aspect('equal', adjustable='box')
ax[0].set_title("magnet locations and orientations")
ax[0].set_xlabel("X [m]")
ax[0].set_ylabel("Y [m]")

# Plot dipole positions
ax[1].plot(np.real(positions), np.imag(positions), 'ko')
ax[1].plot(np.real(positions), np.imag(positions), 'k-')
# ax[1].plot(shrunk_polygon_path.vertices[:,0], shrunk_polygon_path.vertices[:,1], color='grey')

# Contour plot of real(W) gives B field lines
ax[1].contour(np.real(sampling_points),
            np.imag(sampling_points),
            np.real(W),  # z component of the vector potential
            contour_levels)

ax[1].set_xlim(-radius , radius)
ax[1].set_ylim(-radius , radius)
ax[1].set_aspect('equal', adjustable='box')
ax[1].set_title("B field lines (2D calculation, ideal dipoles)")
ax[1].set_xlabel("X [m]")
# ax[1].set_ylabel("Y [m]")

fig.suptitle(shape + " shape")
# plt.show()


# %% plot field magnitude

skip = 10 # points to skip to avoid arrow overcrowding
# polygon inside which version 2 of the homogeneity is calculated
# shrunk_inside_points_patch = patches.PathPatch(shrunk_inside_points_path, facecolor='none', edgecolor='white', lw=1)

# separate figure: doesn't display properly (all blue) if inside the subplots
# also with commented out statements below
plt.figure()
im = plt.imshow(np.abs(H) , vmin=0, vmax=10000, aspect='equal', extent = [-radius, radius, -radius, radius], origin='lower')
plt.colorbar(im)
# plt.xlim(-radius , radius)
# plt.ylim(-radius , radius)
# plt.set_aspect('equal', adjustable='box')
plt.title("B field magnitude and direction (2D calculation, ideal dipoles)")
plt.xlabel("X [m]")
plt.ylabel("Y [m]")

plt.quiver(
    delta * X[::skip, ::skip],
    delta * Y[::skip, ::skip],
    np.imag(masked_H)[::skip, ::skip],
    np.real(masked_H)[::skip, ::skip],
    color='white'
)

plt.plot(np.real(positions), np.imag(positions), 'ko')
# plt.plot(np.real(positions), np.imag(positions), 'k-')
plt.plot(shrunk_polygon_path.vertices[:,0], shrunk_polygon_path.vertices[:,1], color='grey')
plt.plot(inside_points[:,0],inside_points[:,1],'w.')
#plt.plot(shrunk_inside_points_path.vertices[:,0], shrunk_inside_points_path.vertices[:,1], color='white')
plt.gca().add_patch(shrunk_inside_points_patch)


# %% calculate areas

polygon = Polygon(shape_path.to_polygons()[0])
shape_area = polygon.area
print('Area enclosed within original shape =', shape_area)

polygon = Polygon(shrunk_polygon_path.to_polygons()[0])
scaled_shape_area = polygon.area   # should = shape_area * shrink_scale^2
print('Area enclosed within original shape scaled by', shrink_scale*100 , '% =', scaled_shape_area)

polygon = Polygon(shrunk_inside_points_path.to_polygons()[0])
scaled_minima_area = polygon.area
print('Area enclosed within shape determined by field minima =', scaled_minima_area)

print('[to exit back to command line close figure windows]')

plt.ioff()
plt.show()
