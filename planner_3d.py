###############################################################################
# Filename: planner_3d.py
# Author: Daniel Ajeleye
# Nov 2023
# Implementations extensions of cases to 3d space
###############################################################################


#-----------------------------------------------------------------------------#
# Import needed libraries
#-----------------------------------------------------------------------------#
import numpy as np
import random
import math as m
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.patches as patches

#---------------------------------------------------------------------------------#
# Function: getFaceFromPoint
# Purpose:  Function obtains the faces of a cube centered at a given point 
#
# Inputs:
#   - face_centre:       x, y, z -coordinate of given centre point of a cube
#   - h:           the uniform size of the discretization casted on the 3D workspace
#
# Outputs:
#   - returns the vertices of each labelled faces of the cube centred at that the given point
#---------------------------------------------------------------------------------#
def getFaceFromPoint(face_centre, h):
    # get the center of the cube
    cx, cy, cz = face_centre

    # Vertices of the front face
    front_vertices = [
        (cx - h / 2, cy - h / 2, cz + h / 2),
        (cx + h / 2, cy - h / 2, cz + h / 2),
        (cx + h / 2, cy + h / 2, cz + h / 2),
        (cx - h / 2, cy + h / 2, cz + h / 2)
    ]

    # Vertices of the right face
    right_vertices = [
        (cx + h / 2, cy - h / 2, cz + h / 2),
        (cx + h / 2, cy - h / 2, cz - h / 2),
        (cx + h / 2, cy + h / 2, cz - h / 2),
        (cx + h / 2, cy + h / 2, cz + h / 2)
    ]

    # Vertices of the back face
    back_vertices = [
        (cx + h / 2, cy - h / 2, cz - h / 2),
        (cx - h / 2, cy - h / 2, cz - h / 2),
        (cx - h / 2, cy + h / 2, cz - h / 2),
        (cx + h / 2, cy + h / 2, cz - h / 2)
    ]

    # Vertices of the left face
    left_vertices = [
        (cx - h / 2, cy - h / 2, cz - h / 2),
        (cx - h / 2, cy - h / 2, cz + h / 2),
        (cx - h / 2, cy + h / 2, cz + h / 2),
        (cx - h / 2, cy + h / 2, cz - h / 2)
    ]

    # Vertices of the top face
    top_vertices = [
        (cx - h / 2, cy + h / 2, cz + h / 2),
        (cx + h / 2, cy + h / 2, cz + h / 2),
        (cx + h / 2, cy + h / 2, cz - h / 2),
        (cx - h / 2, cy + h / 2, cz - h / 2)
    ]

    # Vertices of the bottom face
    bottom_vertices = [
        (cx - h / 2, cy - h / 2, cz - h / 2),
        (cx + h / 2, cy - h / 2, cz - h / 2),
        (cx + h / 2, cy - h / 2, cz + h / 2),
        (cx - h / 2, cy - h / 2, cz + h / 2)
    ]

    return {
        'front': front_vertices,
        'right': right_vertices,
        'back': back_vertices,
        'left': left_vertices,
        'top': top_vertices,
        'down': bottom_vertices
    }

#---------------------------------------------------------------------------------#
# Function: IsCollision
# Purpose:  Function checks for collision or overlap with any obstacle within workspace
#
# Inputs:
#   - centre:   the centre of a cube
#   - Obs:       the initial obstacles on the workspace before motion planning
#   - eps:    the uniform size of the discretization done in the 3D workspace
#
# Outputs:
#   - returns a boolean to show whether a cube intersect an obstacle or not
#---------------------------------------------------------------------------------#
def IsCollision(centre, Obs, eps):
    # obtain the cube faces
    faces = getFaceFromPoint(centre, eps)
    cff, cfr, cfb, cfl, cft, cfd = faces['front'], faces['right'], faces['back'], faces['left'], faces['top'], faces['down']
    
    bfront, bright, bback, bleft, btop, bdown, inside = False, False, False, False, False, False, False

    for obs in Obs:
        ff, fr, fb, fl, ft, fd = obs['front'], obs['right'], obs['back'], obs['left'], obs['top'], obs['down']
        
        if ff[1][2]-h > cff[0][2] and ff[0][2]-h > cff[1][2] and ff[2][2]-h > cff[3][2] and ff[3][2]-h > cff[2][2]:
            bfront = True
        if fr[1][0] < cfr[0][0]-h and fr[0][0] < cfr[1][0]-h and fr[2][0] < cfr[3][0]-h and fr[3][0] < cfr[2][0]-h:
            bright = True
        if fb[1][2] < cfb[0][2]-h and fb[0][2] < cfb[1][2]-h and fb[2][2] < cfb[3][2]-h and fb[3][2] < cfb[2][2]-h:
            bback = True
        if fl[1][0]-h > cfl[0][0] and fl[0][0]-h > cfl[1][0] and fl[2][0]-h > cfl[3][0] and fl[3][0]-h > cfl[2][0]:
            bleft = True
        if ft[1][1] < cft[0][1]-h and ft[0][1] < cft[1][1]-h and ft[2][1] < cft[3][1]-h and ft[3][1] < cft[2][1]-h:
            btop = True
        if fd[1][1]-h > cfd[0][1] and fd[0][1]-h > cfd[1][1] and fd[2][1]-h > cfd[3][1] and fd[3][1]-h > cfd[2][1]:
            bdown = True
        if not (bfront or bback or bdown or bright or btop or bleft):
            inside = True
        if inside:
            return True # The cube intersect with the obstacle
        bfront, bright, bback, bleft, btop, bdown = False, False, False, False, False, False
    return False; # No intersection detected

#---------------------------------------------------------------------------------#
# Function: getCubeFromPoint
# Purpose:  Function obtains the index of the cube which a given point belongs to
#
# Inputs:
#   - x:       x-coordinate of given point
#   - y:       y-coordinate of given point
#   - z:       z-coordinate of given point
#   - l:       the length of the 3D workspace
#   - b:       the breadth of the 3D workspace
#   - w:       the width of the 3D workspace
#   - eps:    the uniform size of the discretization done on the 3D workspace
#
# Outputs:
#   - returns the index of the grid cell that the given point belongs to
#---------------------------------------------------------------------------------#
def getCubeFromPoint(x, y, z, l, b, w, eps):
    x_min, x_max = 0, l
    y_min, y_max = 0, b
    z_min, z_max = 0, w

    # Check if the point is out of bounds
    assert x >= x_min and x < x_max and y >= y_min and y < y_max and z >= z_min and z < z_max

    x_cells = int((x_max - x_min) / eps)
    y_cells = int((y_max - y_min) / eps)
    z_cells = int((z_max - z_min) / eps)
    
    for i in range(x_cells):
        for j in range(y_cells):
            for k in range(z_cells):
                cell_x_min = x_min + i * eps
                cell_x_max = x_min + (i + 1) * eps
                cell_y_min = y_min + j * eps
                cell_y_max = y_min + (j + 1) * eps
                cell_z_min = z_min + k * eps
                cell_z_max = z_min + (k + 1) * eps
                
                if (
                    x >= cell_x_min
                    and x <= cell_x_max
                    and y >= cell_y_min
                    and y <= cell_y_max
                    and z >= cell_z_min
                    and z <= cell_z_max
                ):
                    return (i,j,k)
            
    return (-1,-1,-1)

#---------------------------------------------------------------------------------#
# Function: Grid
# Purpose:  Function performs a discretization of the workspace
#
# Inputs:
#   - Obs:    the initial obstacles on the workspace before motion planning
#   - x_g: the target point the point robot aim to reach in the workspace
#   - l:       the length of the 3D workspace
#   - b:       the breadth of the 3D workspace
#   - w:       the width of the 3D workspace
#   - h:    the uniform size of the discretization done on the 3D workspace
#
# Outputs:
#   - returns a python dictionary whose keys are respective indices of the cube 
#       with value as a list contaning the label on the cube and the centre point of that cube
#---------------------------------------------------------------------------------#
def Grid(Obs, x_g, l, b, w, h):
    D = {}

    # Iterate over the grid and populate the dictionary
    for i, x in enumerate(np.arange(0, l, h)):
        for j, y in enumerate(np.arange(0, b, h)):
            for k, z in enumerate(np.arange(0, w, h)):    
                center = (x + h/2, y + h/2, z + h/2)  # Calculate the center of the cell

                # Check for collision using IsCollision function
                if IsCollision(center, Obs, h):
                    adv_val = 1 # Assign 1 to obstacle in initial setting
                else:
                    adv_val = -1 
                # Populate the dictionary
                D[(i, j, k)] = [adv_val, center[0], center[1], center[2], 0,0,0]

    # Set goal cell
    goalCell_i, goalCell_j, goalCell_k = getCubeFromPoint(x_g[0], x_g[1], x_g[2], l, b, w, h)
    D[(goalCell_i, goalCell_j, goalCell_k)][0] = 2

    return D

#---------------------------------------------------------------------------------#
# Function: CellAdjuster
# Purpose:  Function labels each cube in the workspace as in the traditional wavefront algorithm
#
# Inputs:
#   - D:    python dictionary whose keys are respective indices of the cubes
#           with value as a list contaning the label on the cube and the centre point of that cube
#
# Outputs:
#   - returns a modified dictionary whose wavevalues has been updated
#---------------------------------------------------------------------------------#
def CellAdjuster(D):
    # Propagate the wave
    wavefrontChanged = True
    waveValue = 3

    while wavefrontChanged:
        wavefrontChanged = False
        for key in D:
            i, j, k = key
            if D[key][0] == waveValue - 1:
                for di in range(-1, 2):
                    for dj in range(-1, 2):
                        for dk in range(-1, 2):
                            if di == 0 and dj == 0 and dk == 0:
                                continue
                            ni = i + di
                            nj = j + dj
                            nk = k + dk
                            new_key = (ni, nj, nk)
                            if new_key in D and D[new_key][0] == -1:
                                D[new_key][0] = waveValue
                                wavefrontChanged = True
                # end of nested loops for di and dj and dk
            # end of if D[key][0] == waveValue - 1
        # end of loop for key
        waveValue += 1
    # end of while wavefrontChanged

    return D


#---------------------------------------------------------------------------------#
# Function: plan_move
# Purpose:  Function plans the movement of the robot based on the given dictionary
#
# Inputs:
#   - D:    dictionary whose keys are respective indices of the cubes 
#           with value as a list contaning the label on the cube and the centre point of that cube
#   - x_0: the initial point where the point robot starts its motion from, in the workspace
#   - n:    the number of steps the point robot takes, whereby a screenshot of the motion so far is taken
#   - l:       the length of the 3D workspace
#   - b:       the breadth of the 3D workspace
#   - w:       the width of the 3D workspace
#   - h:    the uniform size of the discretization done on the 3D workspace
#
# Outputs:
#   - returns a python dictionary whose values captures the path planned and the key denoting the 
#       point reached after n steps
#---------------------------------------------------------------------------------#
def plan_move(D, x_0, n, l, b, w, h):
    i, P, P_res = 0, {}, {} # initialize path
    start_cell = getCubeFromPoint(x_0[0], x_0[1], x_0[2], l, b, w, h)
    key, key1 = start_cell, (0,0,0)
    P[key] = D[key]
    while D[key][0] > 2:
        for di in range(-1, 2):
            for dj in range(-1, 2):
                for dk in range(-1, 2):
                    if di == 0 and dj == 0 and dk == 0:
                        continue
                    ni, nj, nk = key[0] + di, key[1] + dj, key[2] + dk
                    if 0 <= ni < int(l/h) and 0 <= nj < int(b/h) and 0 <= nk < int(w/h):
                        if D[(ni, nj, nk)][0] == D[key][0] - 1 and D[(ni, nj, nk)][0] != 1:
                            key = (ni, nj, nk)
                            if i < n:
                                P[key] = D[key]
                                key1 = key
                            else:
                                P_res[key] = D[key]
                            break
        i += 1

    return P, P_res, key1


#---------------------------------------------------------------------------------#
# Function: droneDynamics
# Purpose:  Function depicts the model of a drone
#
# Inputs:
#   - t:       time of system(drone)'s dynamics evolution
#   - state:       captures the configuration state of the drone at every time instance
#   - si:       captures the control input to the drone
#
# Outputs:
#   - gives the ode to generate the next state of the drone in subsequent time step
#---------------------------------------------------------------------------------#
def droneDynamics(t, state, si):
    x, y, z, psi, theta, v = state
    omega, alpha, a = si

    xdot = v * np.cos(psi) * np.cos(theta)
    ydot = v * np.sin(psi) * np.cos(theta)
    zdot = v * np.sin(theta)
    psidot = omega
    thetadot = alpha
    vdot = a

    return [xdot, ydot, zdot, psidot, thetadot, vdot]


#---------------------------------------------------------------------------------#
# Function: sortListByBound
# Purpose:  Function obtains a point in a given list within a given bound
#
# Inputs:
#   - List:       list of points, solutions to the dynamics
#   - bound:       provided bound within the workspace
#
# Outputs:
#   - gives a point in List within a given bound
#---------------------------------------------------------------------------------#
def sortListByBound(List, bound):
    lis = [pt for pt in List if bound[0] <= pt <= bound[1]]
    
    if lis:
        return random.choice(lis)
    else:
        return bound[0]+bound[2]
    

#---------------------------------------------------------------------------------#
# Function: next_state
# Purpose:  Function obtains a point in a given list within a given bound
#
# Inputs:
#   - cpt:       centre of the cube where the drone is currently in
#   - ppt:       previous point where it evolves from
#   - ss:        state information of its dynamics
#   - si:        control input information 
#   - h:         the uniform size of the discretization done on the 3D workspace
#
# Outputs:
#   - solves the dynamics to provide the point of the drone in next time step
#---------------------------------------------------------------------------------#
def next_state(cpt, ppt, ss, si, h):
    xl, yl, zl = [pt - h/2 for pt in cpt]
    xu, yu, zu = [pt + h/2 for pt in cpt]
    # Time span for simulation
    t_span = (0, 0.3)

    # Solve ODE
    solution = solve_ivp(
        fun=lambda t, state: droneDynamics(t, state, si),
        t_span=t_span,
        y0=ppt,
        method="RK45",  
        dense_output=True,
    )
    states = solution.y
    bx, by, bz = [xl, xu, h], [yl, yu, h], [zl, zu, h]
    bpsi, btheta, bv = [-np.pi/6, np.pi/6, 0.1], [-np.pi/6, np.pi/6, 0.1], [-1/2, 1/2, 0.1]
    xpt, ypt, zpt = sortListByBound(states[0],bx), sortListByBound(states[1],by), sortListByBound(states[2],bz)
    psipt, thetapt, vpt = sortListByBound(states[3],bpsi), sortListByBound(states[4],btheta), sortListByBound(states[5],bv)
    time_values = np.linspace(t_span[0], t_span[1], len(states[0]))
    return [xpt, ypt, zpt], [psipt, thetapt, vpt], time_values, states[3], states[4], states[5]


#---------------------------------------------------------------------------------#
# Function: plan_move_kinodyn
# Purpose:  Function plans the movement of the robot based on the given python dictionary and 
#           incorporating the kinodynamic constraints of the drone
#
# Inputs:
#   - D:    dictionary whose keys are respective indices of the cubes 
#           with value as a list contaning the label on the cube and the centre point of that cube
#   - x_0: the initial point where the point robot starts its motion from, in the workspace
#   - n:    the number of steps the point robot takes, whereby a screenshot of the motion so far is taken
#   - l:       the length of the 3D workspace
#   - b:       the breadth of the 3D workspace
#   - w:       the width of the 3D workspace
#   - h:    the uniform size of the discretization done on the 3D workspace
#
# Outputs:
#   - returns a python dictionary whose values captures the path planned and the key denoting the 
#       point reached after n steps
#---------------------------------------------------------------------------------#
def plan_move_kinodyn(D, x_0, n, l, b, w, h):
    i, P, P_res = 0, {}, {} # initialize path
    start_cell = getCubeFromPoint(x_0[0], x_0[1], x_0[2], l, b, w, h)
    key1, key, ppt, ss, si = (0,0,0), start_cell, x_0, [], []
    si_start1, si_start2 = random.uniform(-np.pi/6, np.pi/6), random.uniform(-1/2, 1/2)
    ss_start1, ss_start2 = random.uniform(-np.pi/2, np.pi/2), random.uniform(-1,1)
    si.append(si_start1)
    si.append(si_start1)
    si.append(si_start2)
    ss.append(ss_start1)
    ss.append(ss_start1)
    ss.append(ss_start2)
    D[key][4:] = si
    ppt = [x_0[0], x_0[1], x_0[2], ss[0], ss[1], ss[2]]
    P[key] = D[key]
    P[key][1:4], ss, t, psi, theta, v = next_state(D[key][1:4], ppt, ss, si, h)
    ppt = P[key][1:4]
    ppt.extend(ss)
    while D[key][0] > 2:
        for di in range(-1, 2):
            for dj in range(-1, 2):
                for dk in range(-1, 2):
                    if di == 0 and dj == 0 and dk == 0:
                        continue
                    ni, nj, nk = key[0] + di, key[1] + dj, key[2] + dk
                    if 0 <= ni < int(l/h) and 0 <= nj < int(b/h) and 0 <= nk < int(w/h):
                        if D[(ni, nj, nk)][0] == D[key][0] - 1 and D[(ni, nj, nk)][0] != 1:
                            key = (ni, nj, nk)
                            if i < n:
                                P[key] = D[key]
                                key1 = key
                                P[key][1:4], ss, t, psi, theta, v = next_state(D[key][1:4], ppt, ss, si, h)
                                ppt = P[key][1:4]
                                ppt.extend(ss)
                            else:
                                P_res[key] = D[key]
                            break
        i += 1

    return P, P_res, key1, ppt, t, psi, theta, v


#---------------------------------------------------------------------------------#
# Function: plot_cube
# Purpose:  Function plots a given cube in workspace
#
# Inputs:
#   - ax:       matplotlib object for plotting figure
#   - vertices:       the vertices of the cube to be plotted in workspace
#   - faces:    the faces of the cube for proper orientation
#
# Outputs:
#   - gives the figure of the required cube with dimension l x b x w
#---------------------------------------------------------------------------------#
def plot_cube(ax, vertices, faces, color='red', alpha=0.4):
    vertices = np.array(vertices)
    faces = np.array(faces)
    cube = Poly3DCollection(vertices[faces], color=color, alpha=alpha)
    ax.add_collection3d(cube)


#---------------------------------------------------------------------------------#
# Function: screenShot
# Purpose:  Function gives the control input engaged in each n steps 
#           using the matplotlib
#
# Inputs:
#   - t:     time span of considering the drone's evolution 
#   - psi:       yaw angle of drone
#   - theta:       pitch angle of drone
#   - v:       drone's linear velocity
#
# Outputs:
#   - None - gives the control input engaged in each n steps of motion
#---------------------------------------------------------------------------------#
def controlScreenShot(t, psi, theta, v):
    
    # Plot Yaw Angle (psi), Pitch Angle (theta), and Linear Velocity (v) on the same plot
    plt.plot(t, psi, label=r'$\psi$ (Yaw Angle)')
    plt.plot(t, theta, label=r'$\theta$ (Pitch Angle)')
    plt.plot(t, v, label=r'$v$ (Linear Velocity)')

    # plt.yscale('log')
    plt.xlabel('Time (s)')
    plt.ylabel('Angle Values')
    plt.legend()

    # Show the plot
    plt.show()


#---------------------------------------------------------------------------------#
# Function: screenShot
# Purpose:  Function takes screenshots after each n steps 
#           using the matplotlib for the workspace figure
#
# Inputs:
#   - D:    dictionary whose keys are respective indices of the cube 
#           with value as a list contaning the label on the cube and the centre point of that cube
#   - P:     dictionary that captures the path planned upto current state
#   - P_res:     dictionary that captures the path planned further from current state
#   - l:       the length of the 3D workspace
#   - b:       the breadth of the 3D workspace
#   - w:       the width of the 3D workspace
#   - Obs:       the initial obstacles on the workspace before motion planning
#   - x_g: the target point the point robot aim to reach in the workspace
#   - x_0: the initial point where the point robot starts its motion from, in the workspace
#   - n:    the number of steps the point robot takes, whereby a screenshot of the motion so far is taken
#
# Outputs:
#   - None - provides a snapshot of the path planned so far with the entire workspace after n steps of motion 
#---------------------------------------------------------------------------------#
def screenShot(D, P, P_res, l, b, w, Obs, x_g, x_0, n):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the obstacles
    for obs in Obs:
        ff, fr, fb, fl, ft, fd = obs['front'], obs['right'], obs['back'], obs['left'], obs['top'], obs['down']
        # Define the vertices of the cube
        cube_vertices = [ ff[0], ff[1], ff[2], ff[3], fr[1], fr[2], fb[1], fb[2] ]

        # Define the faces of the cube using vertex indices
        cube_faces = [
            [6, 4, 1, 0],  # bottom
            [3, 2, 5, 7],  # top
            [6, 0, 3, 7],  # left
            [1, 4, 5, 2],  # right
            [0, 1, 2, 3],  # front
            [4, 6, 7, 5]   # back
        ]
        plot_cube(ax, cube_vertices, cube_faces)

    # Plot the path
    path_x = [value[1] for key, value in P.items()]
    path_y = [value[2] for key, value in P.items()]
    path_z = [value[3] for key, value in P.items()]
    ax.plot(path_x, path_y, path_z, linestyle='-', color='blue', marker='*', zorder=5)

    path_x_res = [value[1] for key, value in P_res.items()]
    path_y_res = [value[2] for key, value in P_res.items()]
    path_z_res = [value[3] for key, value in P_res.items()]
    ax.plot(path_x_res, path_y_res, path_z_res, linestyle='-', color='blue', marker='*', alpha=0.3, zorder=5)

    key_start = getCubeFromPoint(x_0[0], x_0[1], x_0[2], l, b, w, h)
    key_goal = getCubeFromPoint(x_g[0], x_g[1], x_g[2], l, b, w, h)
    ax.plot(D[key_start][1], D[key_start][2], D[key_start][3], marker='o', color='purple', zorder=7)
    ax.text(D[key_start][1], D[key_start][2], D[key_start][3], 'x_start', fontsize=8, ha='right', va='top')
    ax.plot(D[key_goal][1], D[key_goal][2], D[key_goal][3], marker='o', color='green', zorder=7)
    ax.text(D[key_goal][1], D[key_goal][2], D[key_goal][3], 'x_goal', fontsize=8, ha='left', va='top')

    ax.set_box_aspect([1, 1, 1])
    # Set limits for the x, y, and z axes
    ax.set_xlim(0, l)  # Set x-axis limits
    ax.set_ylim(0, b)  # Set y-axis limits
    ax.set_zlim(0, w)  # Set z-axis limits
    # Add labels to the axes
    ax.set_xlabel('x-axis')
    ax.set_ylabel('y-axis')
    ax.set_zlabel('z-axis')
    plt.title(f'Planner Screenshot for first {n} steps')  # title format
    plt.show()


#---------------------------------------------------------------------------------#
# Function: plan_case1
# Purpose:  Function is a traditional planner with workspace having static obstacle
#
# Inputs:
#   - Obs:       the initial obstacles on the workspace before motion planning
#   - x_g: the target point the point robot aim to reach in the workspace
#   - x_0: the initial point where the point robot starts its motion from, in the workspace
#   - l:       the length of the 3D workspace
#   - b:       the breadth of the 3D workspace
#   - w:       the width of the 3D workspace
#   - h:    the uniform size of the discretization done on the 3D workspace
#   - n:    the number of steps the point robot takes, whereby a screenshot of the motion so far is taken
#   - stat: indicator whether the planner puts into consideration the kinodynamic constraints or not
#
# Outputs:
#   - None- just motion plan for case 1 and produce screenshots after each n steps 
#           using the matplotlib for the workspace figure
#---------------------------------------------------------------------------------#
def plan_case1(Obs, x_g, x_0, l, b, w, h, n, stat):
    D = Grid(Obs, x_g, l, b, w, h)
    P, D = {}, CellAdjuster(D)
    i, next_init, key = 1, x_0, (0,0,0)
    while D[key][0] != 2:
        n1, P_res = i*n, {}
        if stat == 'nonkino':
            P1, P_res, key = plan_move(D, next_init, n, l, b, w, h)
            next_init = D[key][1:]
        if stat == 'kino':
            P1, P_res, key, ppt, t1, psi1, theta1, v1 = plan_move_kinodyn(D, next_init, n, l, b, w, h)
            t1 = [value * i for value in t1]
            controlScreenShot(t1, psi1, theta1, v1)
            next_init = ppt[0:3]
        P.update(P1)
        screenShot(D, P, P_res, l, b, w, Obs, x_g, x_0, n1)   
        i += 1



#---------------------------------------------------------------------------------#
# Function: dynamics_3d
# Purpose:  Function depicts the dynamics followed by the obstacles for case 4 
#
# Inputs:
#   - x:       x-coordinate of given point
#   - y:       y-coordinate of given point
#   - z:       z-coordinate of given point
#   - h:    the uniform size of the discretization done on the 3D workspace
#
# Outputs:
#   - gives the the dynamics followed by the obstacles for case 4 
#---------------------------------------------------------------------------------#
def dynamics_3d(x, y, z, h):
    dx = random.uniform(-h, h)
    dy = random.uniform(-h, h)
    dz = random.uniform(-h, h)
    new_x = x + dx
    new_y = y + dy
    new_z = z + dz
    return new_x, new_z, new_y


#---------------------------------------------------------------------------------#
# Function: advObsInit_c4
# Purpose:  Function initializes the adversarial obstacles for case 4 
#
# Inputs:
#   - D:    dictionary whose keys are respective indices of the cube 
#           with value as a list contaning the label on the cube and the centre point of that cube
#   - P:     dictionary that captures the path planned 
#   - P_adv:     dictionary that captures the path of the adversarial obstacles so far
#   - l:       the length of the 3D workspace
#   - b:       the breadth of the 3D workspace
#   - w:       the width of the 3D workspace
#   - h:    the uniform size of the grid casted on the 3D workspace
#   - st:   indicates whether the motion is just starting or has already began
#
# Outputs:
#   - gives the the dynamics followed by the obstacles for case 4 
#---------------------------------------------------------------------------------#
def advObsInit_c4(D, P, P_adv, l, b, w, h, st):
    max_wave = max(D, key=lambda k: D[k][0])

    if st == 'start':
        # create a list of random obs 
        valid_keys = [key for key, value in D.items() if value[0] not in {max_wave, 2, -1}]

        # randomly select keys from the valid keys
        selected_keys = random.sample(valid_keys, 1)

        # ensure none of the selected keys coincide with any key in P
        selected_keys = [key for key in selected_keys if key not in P and 0<=key[0]<=l/h and 0<=key[1]<=b/h and 0<=key[2]<=w/h]

        return selected_keys
    
    if st == 'subsequent':
        # create a list of obs not init, goal 
        valid_keys = [key for key, value in D.items() if value[0] not in {max_wave, 2, -1}]

        # ensure none of the valid keys coincide with any key in P
        valid_keys = [key for key in valid_keys if key not in P and 0<=key[0]<=l/h and 0<=key[1]<=b/h and 0<=key[2]<=w/h]
       
        # ensure the next key for P_adv chosen from selected keys coincide with a key in P_adv
        selected_keys = [key for key in valid_keys if key in P_adv]

        # randomly select keys from the selected keys
        if selected_keys:
            selected_keys = random.sample(selected_keys, 1)
        else:
            selected_keys = random.sample(valid_keys, 1)

        return selected_keys


#---------------------------------------------------------------------------------#
# Function: obsAdjuster_case4
# Purpose:  Function modifies the workspace after n steps has been planned for case 4
#
# Inputs:
#   - D:    dictionary whose keys are respective indices of the grid cell 
#           with value as a list contaning the label on the cell and the centre point of that cell
#   - P:     dicctionary that captures the path planned
#   - P_adv:     dictionary that captures the path of the adversarial obstacles so far
#   - l:       the length of the 3D workspace
#   - b:       the breadth of the 3D workspace
#   - w:       the width of the 3D workspace
#   - h:    the uniform size of the grid casted on the 3D workspace
#   - st:   indicates whether the motion is just starting or has already began
#   - n:    the number of steps the point robot takes, whereby a screenshot of the motion so far is taken
# 
# Outputs:
#   - returns a modified dictionary whose wavevalues has been updated
#---------------------------------------------------------------------------------#
def obsAdjuster_case4(D, P, P_adv, l, b, w, h, st, n=4):
    modified_D, rand_obs = {}, []
    modified_D.update(D)
    
    selected_keys = advObsInit_c4(D, P, P_adv, l, b, w, h, st)

    for key in selected_keys:
        modified_D[key][0] = -1 # bring up random obs

        # Use dynamics_3d to generate the next n positions
        for i in range(n):
            new_x, new_y, new_z = dynamics_3d(modified_D[key][1], modified_D[key][2], modified_D[key][3], h)
            key1 = getCubeFromPoint(new_x, new_y, new_z, l, b, w, h)

            # Check if the key1 is valid and not in P
            if 0 <= key1[0] < int(l/h) and 0 <= key1[1] < int(b/h) and 0 <= key1[2] < int(w/h) and key1 not in P:
                modified_D[key1][0] = -1
                modified_D[key][1] = new_x
                modified_D[key][2] = new_y
                modified_D[key][3] = new_z
                P_adv[key1] = [-1, new_x, new_y, new_z]

                # Calculate the cuboid vertices
                face_centre = (new_x, new_y, new_z)
                rand_obs.append(getFaceFromPoint(face_centre, h))
            else:
                break  # Break if the new key is not valid

    return modified_D, rand_obs, P_adv


#---------------------------------------------------------------------------------#
# Function: plan_case4
# Purpose:  Function to motion plan for case where an adversarial obstacle erupts from  
#           a random point in workspace and then keep dispersing based on a dynamics.
#
# Inputs:
#   - Obs:       the initial obstacles on the workspace before motion planning
#   - x_g: the target point the point robot aim to reach in the workspace
#   - x_0: the initial point where the point robot starts its motion from, in the workspace
#   - l:       the length of the 3D workspace
#   - b:       the breadth of the 3D workspace
#   - w:       the width of the 3D workspace
#   - h:    the uniform size of the grid casted on the 2D workspace
#   - n:    the number of steps the point robot takes, whereby a screenshot of the motion so far is taken
#   - stat: indicator whether the planner puts into consideration the kinodynamic constraints or not
#
# Outputs:
#   - None- just motion plan for case 4 and produce screenshots after each n steps 
#           using the matplotlib for the workspace figure
#---------------------------------------------------------------------------------#
def plan_case4(Obs, x_g, x_0, l, b, w, h, n, stat):
    D = Grid(Obs, x_g, l, w, b, h)
    P, P_adv, D = {}, {}, CellAdjuster(D)
    i, next_init, key = 1, x_0, (0,0,0)

    while D[key][0] != 2:
        if i == 1:
            Obs_adj1 = obsAdjuster_case4(D, P, P_adv, l, b, w, h, 'start')
        else:
            Obs_adj1 = obsAdjuster_case4(D, P, P_adv, l, b, w, h, 'subsequent')
        
        Obs.extend(Obs_adj1[1])
        D1 = Obs_adj1[0]
        D1 = Grid(Obs, x_g, l, b, w, h)
        D1 = CellAdjuster(D1)

        if stat == 'nonkino':
            P1, P_res, key = plan_move(D1, next_init, n, l, b, w, h)
            D.update(D1)
            next_init = D1[key][1:]
        n1 = i*n
        if stat == 'kino':
            P1, P_res, key, ppt, t1, psi1, theta1, v1 = plan_move_kinodyn(D, next_init, n, l, b, w, h)
            t1 = [value * i for value in t1]
            controlScreenShot(t1, psi1, theta1, v1)
            next_init = ppt[0:3]
        P.update(P1)
        P_adv.update(Obs_adj1[2])
        screenShot(D1, P, P_res, l, b, w, Obs, x_g, x_0, n1)   
        i += 1


#-----------------------------------------------------------------------------#
# Run the planner for the scenarios
#-----------------------------------------------------------------------------#

# Set up parameters for the workspace
l = 10  # Length of the workspace
b = 10   # Breadth of the workspace
w = 10   # Width of the workspace
h = .4   # Discretization size

# 3D obstacles
Obs = [{'front':[(1,6.5,0),(1.5,6.5,0),(1.5,6.5,6),(1,6.5,6)],
        'right':[(1.5,6.5,0),(1.5,7,0),(1.5,7,6),(1.5,6.5,6)],
        'back':[(1.5,7,0),(1,7,0),(1,7,6),(1.5,7,6)],
        'left':[(1,7,0),(1,6.5,0),(1,6.5,6),(1,7,6)],
        'top':[(1,6.5,6),(1.5,6.5,6),(1.5,7,6),(1,7,6)],
        'down':[(1,7,0),(1.5,7,0),(1.5,6.5,0),(1,6.5,0)]},
        {'front':[(4,0,0),(5,0,0),(5,0,6),(4,0,6)],
        'right':[(5,0,0),(5,1,0),(5,1,6),(5,0,6)],
        'back':[(5,1,0),(4,1,0),(4,1,6),(5,1,6)],
        'left':[(4,1,0),(4,0,0),(4,0,6),(4,1,6)],
        'top':[(4,0,6),(5,0,6),(5,1,6),(4,1,6)],
        'down':[(4,1,0),(5,1,0),(5,0,0),(4,0,0)]},
        {'front':[(4,9,0),(5,9,0),(5,9,6),(4,9,6)],
        'right':[(5,9,0),(5,10,0),(5,10,6),(5,9,6)],
        'back':[(5,10,0),(4,10,0),(4,10,6),(5,10,6)],
        'left':[(4,10,0),(4,9,0),(4,9,6),(4,10,6)],
        'top':[(4,9,6),(5,9,6),(5,10,6),(4,10,6)],
        'down':[(4,10,0),(5,10,0),(5,9,0),(4,9,0)]},
        {'front':[(4,0,5),(5,0,5),(5,0,6),(4,0,6)],
        'right':[(5,0,5),(5,10,5),(5,10,6),(5,0,6)],
        'back':[(5,10,5),(4,10,5),(4,10,6),(5,10,6)],
        'left':[(4,10,5),(4,0,5),(4,0,6),(4,10,6)],
        'top':[(4,0,6),(5,0,6),(5,10,6),(4,10,6)],
        'down':[(4,10,5),(5,10,5),(5,0,5),(4,0,5)]},
        {'front':[(6,0,0),(7,0,0),(7,0,2),(6,0,2)],
        'right':[(7,0,0),(7,10,0),(7,10,2),(7,0,2)],
        'back':[(7,10,0),(6,10,0),(6,10,2),(7,10,2)],
        'left':[(6,10,0),(6,0,0),(6,0,2),(6,10,2)],
        'top':[(6,0,2),(7,0,2),(7,10,2),(6,10,2)],
        'down':[(6,10,0),(7,10,0),(7,0,0),(6,0,0)]}
        ]


# starting point and goal
x_0, x_g = (.5, 0, .5), (9.5,7,0)
n = 10 # number of steps for snapshot

# Creating scenerios

# static obs case implementation
# plan_case1(Obs, x_g, x_0, l, b, w, h, n, 'kino')

x_g = (9.5,7,9)
# case4 implementation
plan_case4(Obs, x_g, x_0, l, b, w, h, n, 'kino')

