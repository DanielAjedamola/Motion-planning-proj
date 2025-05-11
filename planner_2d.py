###############################################################################
# Filename: planner_2d.py
# Author: Daniel Ajeleye
# Nov 2023
# Implementations of cases for 2d space
###############################################################################


#-----------------------------------------------------------------------------#
# Import needed libraries
#-----------------------------------------------------------------------------#
import numpy as np
import random
import math as m
import matplotlib.pyplot as plt
import matplotlib.patches as patches

#---------------------------------------------------------------------------------#
# Function: calculate_distance
# Purpose:  Function computes euclidean distance between two given points
#
# Inputs:
#   - x1:       x-coordinate of first point
#   - y1:       y-coordinate of first point
#   - x2:       x-coordinate of second point
#   - y2:       y-coordinate of second point
#
# Outputs:
#   - euclidean distance between points (x1, y1) and (x2, y2)
#---------------------------------------------------------------------------------#
def calculate_distance(x1, y1, x2, y2):
    return m.sqrt((x2 - x1)**2 + (y2 - y1)**2)


#---------------------------------------------------------------------------------#
# Function: IsCollision
# Purpose:  Function checks for collision or overlap with any obstacle within workspace
#
# Inputs:
#   - centre:   the centre of a grid cell
#   - Obs:       the initial obstacles on the workspace before motion planning
#   - eps:    the uniform size of the grid casted on the 2D workspace
#
# Outputs:
#   - returns a boolean to show whether a cell overlap an obstacle or not
#---------------------------------------------------------------------------------#
def IsCollision(centre, Obs, eps):
    # get the center of the cell
    cx, cy = centre
    px1, py1 = cx-eps/2, cy-eps/2
    px2, py2 = cx+eps/2, cy-eps/2
    px3, py3 = cx+eps/2, cy+eps/2
    px4, py4 = cx-eps/2, cy+eps/2
    left, down, right, top, inside = False, False, False, False, False

    for obs in Obs:
        ox1, oy1 = obs[0][0], obs[0][1]
        ox2, oy2 = obs[1][0], obs[1][1]
        ox3, oy3 = obs[2][0], obs[2][1]
        ox4, oy4 = obs[3][0], obs[3][1]
        
        if px2 < ox1 and px2 < ox4 and px3 < ox1 and px3 < ox4:
            left = True
        if py4<oy1 and py4<oy2 and py3<oy1 and py3<oy2: 
            down = True
        if px1>ox2 and px1>ox3 and px4>ox2 and px4>ox3:
            right = True
        if py1>oy3 and py1>oy4 and py2>oy3 and py2>oy4:
            top = True
        if not (down or right or top or left):
            inside = True
        if inside:
            return True # The cell overlaps with the obstacle
        left, down, right, top = False, False, False, False
    return False; # No overlap detected


#---------------------------------------------------------------------------------#
# Function: getCellFromPoint
# Purpose:  Function obtains the grid cell which a given point belongs to
#
# Inputs:
#   - x:       x-coordinate of given point
#   - y:       y-coordinate of given point
#   - l:       the length of the 2D workspace
#   - b:       the breadth of the 2D workspace
#   - eps:    the uniform size of the grid casted on the 2D workspace
#
# Outputs:
#   - returns the index of the grid cell that the given point belongs to
#---------------------------------------------------------------------------------#
def getCellFromPoint(x, y, l, b, eps):
    x_min, x_max = 0, l
    y_min, y_max = 0, b

    # Check if the point is out of bounds
    assert x >= x_min and x < x_max and y >= y_min and y < y_max

    x_cells = int((x_max - x_min) / eps)
    y_cells = int((y_max - y_min) / eps)
    
    for i in range(x_cells):
        for j in range(y_cells):
            cell_x_min = x_min + i * eps
            cell_x_max = x_min + (i + 1) * eps
            cell_y_min = y_min + j * eps
            cell_y_max = y_min + (j + 1) * eps
            
            if (
                x >= cell_x_min
                and x <= cell_x_max
                and y >= cell_y_min
                and y <= cell_y_max
            ):
                return (i,j)
            
    return (-1,-1)

#---------------------------------------------------------------------------------#
# Function: Grid
# Purpose:  Function cast a grid on the workspace
#
# Inputs:
#   - Obs:    the initial obstacles on the workspace before motion planning
#   - x_g: the target point the point robot aim to reach in the workspace
#   - l:       the length of the 2D workspace
#   - b:       the breadth of the 2D workspace
#   - h:    the uniform size of the grid casted on the 2D workspace
#
# Outputs:
#   - returns a python dictionary whose keys are respective indices of the grid cell 
#       with value as a list contaning the label on the cell and the centre point of that cell
#---------------------------------------------------------------------------------#
def Grid(Obs, x_g, l, b, h):
    D = {}

    # Iterate over the grid and populate the dictionary
    for i, x in enumerate(np.arange(0, l, h)):
        for j, y in enumerate(np.arange(0, b, h)):
            center = (x + h/2, y + h/2)  # Calculate the center of the cell

            # Check for collision using IsCollision function
            if IsCollision(center, Obs, h):
                adv_val = 1 # Assign 1 to obstacle in initial setting
            else:
                adv_val = -1 
            # Populate the dictionary
            D[(i, j)] = [adv_val, center[0], center[1]]

    # Set goal cell
    goalCell_i, goalCell_j = getCellFromPoint(x_g[0], x_g[1], l, b, h)
    D[(goalCell_i, goalCell_j)][0] = 2

    return D

#---------------------------------------------------------------------------------#
# Function: CellAdjuster
# Purpose:  Function labels each grid cell in the workspace as in the traditional wavefront algorithm
#
# Inputs:
#   - D:    python dictionary whose keys are respective indices of the grid cell 
#           with value as a list contaning the label on the cell and the centre point of that cell
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
            i, j = key
            if D[key][0] == waveValue - 1:
                for di in range(-1, 2):
                    for dj in range(-1, 2):
                        if di == 0 and dj == 0:
                            continue
                        ni = i + di
                        nj = j + dj
                        new_key = (ni, nj)
                        if new_key in D and D[new_key][0] == -1:
                            D[new_key][0] = waveValue
                            wavefrontChanged = True
                # end of nested loops for di and dj
            # end of if D[key][0] == waveValue - 1
        # end of loop for key
        waveValue += 1
    # end of while wavefrontChanged

    # Now, dictionary D contains the result of the wavefront propagation

    return D


#---------------------------------------------------------------------------------#
# Function: plan_move
# Purpose:  Function plans the movement of the robot based on the given dictionary
#
# Inputs:
#   - D:    dictionary whose keys are respective indices of the grid cell 
#           with value as a list contaning the label on the cell and the centre point of that cell
#   - x_0: the initial point where the point robot starts its motion from, in the workspace
#   - n:    the number of steps the point robot takes, whereby a screenshot of the motion so far is taken
#   - l:       the length of the 2D workspace
#   - b:       the breadth of the 2D workspace
#   - h:    the uniform size of the grid casted on the 2D workspace
#
# Outputs:
#   - returns a dictionary whose values captures the path planned and the key denoting the 
#       point reached after n steps
#---------------------------------------------------------------------------------#
def plan_move(D, x_0, n, l, b, h):
    i, P, P_res = 0, {}, {} # initialize path
    start_cell = getCellFromPoint(x_0[0], x_0[1], l, b, h)
    key, key1 = start_cell, (0,0)
    P[key] = D[key]
    while D[key][0] > 2:
        for di in range(-1, 2):
            for dj in range(-1, 2):
                if di == 0 and dj == 0:
                    continue
                ni, nj = key[0] + di, key[1] + dj
                if 0 <= ni < int(l/h) and 0 <= nj < int(b/h):
                    if D[(ni, nj)][0] == D[key][0] - 1 and D[(ni, nj)][0] != 1:
                        key = (ni, nj)
                        if i < n:
                            P[key] = D[key]
                            key1 = key
                        else:
                            P_res[key] = D[key]
                        break
        i += 1

    return P, P_res, key1


#---------------------------------------------------------------------------------#
# Function: screenShot
# Purpose:  Function takes screenshots after each n steps 
#           using the matplotlib for the workspace figure
#
# Inputs:
#   - D:    dictionary whose keys are respective indices of the grid cell 
#           with value as a list contaning the label on the cell and the centre point of that cell
#   - P:     dictionary that captures the path planned upto current state
#   - P_res:     dictionary that captures the path planned further from current state
#   - l:       the length of the 2D workspace
#   - b:       the breadth of the 2D workspace  
#   - Obs:       the initial obstacles on the workspace before motion planning
#   - x_g: the target point the point robot aim to reach in the workspace
#   - x_0: the initial point where the point robot starts its motion from, in the workspace
#   - n:    the number of steps the point robot takes, whereby a screenshot of the motion so far is taken
#
# Outputs:
#   - None - provides a snapshot of the path planned so far with the entire workspace after n steps of motion 
#---------------------------------------------------------------------------------#
def screenShot(D, P, P_res, l, b, Obs, x_g, x_0, n):
    fig, ax = plt.subplots()

    # Plot the rectangle l x b
    ax.add_patch(patches.Rectangle((0, 0), l, b, linewidth=1, edgecolor='black', facecolor='none'))

    # Plot the obstacles
    for obs in Obs:
        ox1, oy1 = obs[0][0], obs[0][1]
        ox2, oy2 = obs[1][0], obs[1][1]
        ox3, oy3 = obs[2][0], obs[2][1]
        ox4, oy4 = obs[3][0], obs[3][1]
        bo, lo = calculate_distance(ox1, oy1, ox4, oy4), calculate_distance(ox1, oy1, ox2, oy2) 
        ax.add_patch(patches.Rectangle((ox1, oy1), lo, bo, linewidth=1, edgecolor='red', facecolor='red'))

    # Plot the path
    path_x = [value[1] for key, value in P.items()]
    path_y = [value[2] for key, value in P.items()]
    ax.plot(path_x, path_y, linestyle='-', color='blue', marker='*') 

    path_x_res = [value[1] for key, value in P_res.items()]
    path_y_res = [value[2] for key, value in P_res.items()]
    ax.plot(path_x_res, path_y_res, linestyle='-', color='blue', marker='*', alpha=0.1)

    
    key_start = getCellFromPoint(x_0[0], x_0[1], l, b, h)
    key_goal = getCellFromPoint(x_g[0], x_g[1], l, b, h)
    ax.plot(D[key_start][1], D[key_start][2], marker='o', color='purple')
    ax.text(D[key_start][1], D[key_start][2], 'x_start', fontsize=8, ha='left', va='top')
    ax.plot(D[key_goal][1], D[key_goal][2], marker='o', color='green')
    ax.text(D[key_goal][1], D[key_goal][2], 'x_goal', fontsize=8, ha='left', va='bottom')

    ax.set_aspect('equal', 'box')
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
#   - l:       the length of the 2D workspace
#   - b:       the breadth of the 2D workspace
#   - h:    the uniform size of the grid casted on the 2D workspace
#   - n:    the number of steps the point robot takes, whereby a screenshot of the motion so far is taken
#
# Outputs:
#   - None- just motion plan for case 1 and produce screenshots after each n steps 
#           using the matplotlib for the workspace figure
#---------------------------------------------------------------------------------#
def plan_case1(Obs, x_g, x_0, l, b, h, n):
    D = Grid(Obs, x_g, l, b, h)
    P, D = {}, CellAdjuster(D)
    i, next_init, key = 1, x_0, (0,0)
    while D[key][0] != 2:
        P1, P_res, key = plan_move(D, next_init, n, l, b, h)
        next_init = D[key][1:]
        n1 = i*n
        P.update(P1)
        screenShot(D, P, P_res, l, b, Obs, x_g, x_0, n1)   
        i += 1


#---------------------------------------------------------------------------------#
# Function: plan_case1_1
# Purpose:  Function to motion plan for case where random obstacles erupts  
#           from a compact space within the workspace while motion is on going, at every n steps.
#           The planner takes the worst case route for collision avoidance since random obstacles are bounded
#
# Inputs:
#   - Obs:       the initial obstacles on the workspace before motion planning
#   - x_g: the target point the point robot aim to reach in the workspace
#   - x_0: the initial point where the point robot starts its motion from, in the workspace
#   - l:       the length of the 2D workspace
#   - b:       the breadth of the 2D workspace
#   - h:    the uniform size of the grid casted on the 2D workspace
#   - n:    the number of steps the point robot takes, whereby a screenshot of the motion so far is taken
#
# Outputs:
#   - None- just motion plan for case 1_1 and produce screenshots after each n steps 
#           using the matplotlib for the workspace figure
#---------------------------------------------------------------------------------#
def plan_case1_1(Obs, x_g, x_0, l, b, h, n):
    D = Grid(Obs, x_g, l, b, h)
    P, D = {}, CellAdjuster(D)
    i, next_init, key = 1, x_0, (0,0)
    Obs_adj = []

    while D[key][0] != 2:
        Obs_adj1 = obsAdjuster_case2(Obs)
        Obs_adj.extend(Obs_adj1)
        D1 = Grid(Obs_adj1, x_g, l, b, h)
        D1 = CellAdjuster(D)

        P1, P_res1, key = plan_move(D1, next_init, n, l, b, h)
        D.update(D1)
        next_init = D1[key][1:]
        n1 = i*n
        P.update(P1)
        screenShot(D1, P, P_res1, l, b, Obs_adj, x_g, x_0, n1)   
        i += 1


#---------------------------------------------------------------------------------#
# Function: obsAdjuster_case2
# Purpose:  Function modifies the workspace after n steps has been planned for case 2
#
# Inputs:
#   - Obs:    The current obstacles present on the workspace
#
# Outputs:
#   - returns a modified dictionary whose wavevalues has been updated
#---------------------------------------------------------------------------------#
def obsAdjuster_case2(Obs):
    obs_size = random.randint(0, len(Obs))
    modified_obs = []

    for i in range(obs_size):
        min_x = min(Obs[i][0][0], Obs[i][1][0], Obs[i][2][0], Obs[i][3][0])
        max_x = max(Obs[i][0][0], Obs[i][1][0], Obs[i][2][0], Obs[i][3][0])
        min_y = min(Obs[i][0][1], Obs[i][1][1], Obs[i][2][1], Obs[i][3][1])
        max_y = max(Obs[i][0][1], Obs[i][1][1], Obs[i][2][1], Obs[i][3][1])

        # Generate random rectangle within the given region
        new_rect = []
        x1, y1 = random.uniform(min_x, min_x+(max_x-min_x)/2), random.uniform(min_y, min_y+(max_y-min_y)/2)
        x2, y3 = random.uniform(x1+.001, max_x), random.uniform(y1+.001, max_y)

        new_rect.append((x1, y1))
        new_rect.append((x2, y1))
        new_rect.append((x2, y3))
        new_rect.append((x1, y3))

        modified_obs.append(new_rect)

    return modified_obs


#---------------------------------------------------------------------------------#
# Function: plan_case2_1
# Purpose:  Function to motion plan for case where random obstacles erupts  
#           from a compact space within the workspace while motion is on going, at every n steps,
#           and the appearing obstacles at nth steps disappears at 2nth steps in motion
#
# Inputs:
#   - Obs:       the initial obstacles on the workspace before motion planning
#   - x_g: the target point the point robot aim to reach in the workspace
#   - x_0: the initial point where the point robot starts its motion from, in the workspace
#   - l:       the length of the 2D workspace
#   - b:       the breadth of the 2D workspace
#   - h:    the uniform size of the grid casted on the 2D workspace
#   - n:    the number of steps the point robot takes, whereby a screenshot of the motion so far is taken
#
# Outputs:
#   - None- just motion plan for case 2_1 and produce screenshots after each n steps 
#           using the matplotlib for the workspace figure
#---------------------------------------------------------------------------------#
def plan_case2_1(Obs, x_g, x_0, l, b, h, n):
    D = Grid(Obs, x_g, l, b, h)
    P, D = {}, CellAdjuster(D)
    i, next_init, key = 1, x_0, (0,0)

    while D[key][0] != 2:
        Obs_adj1 = obsAdjuster_case2(Obs)
        D1 = Grid(Obs_adj1, x_g, l, b, h)
        D1 = CellAdjuster(D1)

        P1, P_res1, key = plan_move(D1, next_init, n, l, b, h)
        D.update(D1)
        next_init = D1[key][1:]
        n1 = i*n
        P.update(P1)
        screenShot(D1, P, P_res1, l, b, Obs_adj1, x_g, x_0, n1)   
        i += 1

#---------------------------------------------------------------------------------#
# Function: plan_case2_2
# Purpose:  Function to motion plan for case where random obstacles erupts  
#           from a compact space within the workspace while motion is on going, at every n steps,
#           and the appearing obstacles at n steps are sustained at subsequent steps in motion
#
# Inputs:
#   - Obs:       the initial obstacles on the workspace before motion planning
#   - x_g: the target point the point robot aim to reach in the workspace
#   - x_0: the initial point where the point robot starts its motion from, in the workspace
#   - l:       the length of the 2D workspace
#   - b:       the breadth of the 2D workspace
#   - h:    the uniform size of the grid casted on the 2D workspace
#   - n:    the number of steps the point robot takes, whereby a screenshot of the motion so far is taken
#
# Outputs:
#   - None- just motion plan for case 2_2 and produce screenshots after each n steps 
#           using the matplotlib for the workspace figure
#---------------------------------------------------------------------------------#
def plan_case2_2(Obs, x_g, x_0, l, b, h, n):
    D = Grid(Obs, x_g, l, b, h)
    P, D = {}, CellAdjuster(D)
    i, next_init, key = 1, x_0, (0,0)
    Obs_adj = []

    while D[key][0] != 2:
        Obs_adj1 = obsAdjuster_case2(Obs)
        Obs_adj.extend(Obs_adj1)
        D1 = Grid(Obs_adj1, x_g, l, b, h)
        D1 = CellAdjuster(D1)

        P1, P_res1, key = plan_move(D1, next_init, n, l, b, h)
        D.update(D1)
        next_init = D1[key][1:]
        n1 = i*n
        P.update(P1)
        screenShot(D1, P, P_res1, l, b, Obs_adj, x_g, x_0, n1)   
        i += 1


#---------------------------------------------------------------------------------#
# Function: obsAdjuster_case3
# Purpose:  Function modifies the workspace after n steps has been planned for case 3
#
# Inputs:
#   - D:    dictionary whose keys are respective indices of the grid cell 
#           with value as a list contaning the label on the cell and the centre point of that cell
#   - P:     dicctionary that captures the path planned 
#   - l:       the length of the 2D workspace
#   - b:       the breadth of the 2D workspace
#   - h:    the uniform size of the grid casted on the 2D workspace
# 
# Outputs:
#   - returns a modified dictionary whose wavevalues has been updated
#---------------------------------------------------------------------------------#
def obsAdjuster_case3(D, P, l, b, h):
    max_wave = max(D, key=lambda k: D[k][0])
    modified_D, rand_obs = {}, []
    modified_D.update(D)
    
    # create a list of random obs 
    valid_keys = [key for key, value in D.items() if value[0] not in {max_wave, 2, -1}]
    obs_size = random.randint(0, 8)
    # obs_size = random.randint(0, len(valid_keys))

    # randomly select keys from the valid keys
    selected_keys = random.sample(valid_keys, obs_size)

    # ensure none of the selected keys coincide with any key in P
    selected_keys = [key for key in selected_keys if key not in P and 0<=key[0]<=l/h and 0<=key[1]<=b/h]

    for key in selected_keys:
        modified_D[key][0] = -1 # bring up random obs

        # Calculate the rectangle vertices
        x1, y1 = modified_D[key][1]-h/2, modified_D[key][2]-h/2
        x2, y2 = modified_D[key][1]+h/2, modified_D[key][2]-h/2
        x3, y3 = modified_D[key][1]+h/2, modified_D[key][2]+h/2
        x4, y4 = modified_D[key][1]-h/2, modified_D[key][2]+h/2
        rand_obs.append([(x1, y1), (x2, y2), (x3, y3), (x4, y4)])

    return modified_D, rand_obs


#---------------------------------------------------------------------------------#
# Function: plan_case3
# Purpose:  Function to motion plan for case where an adversarial obstacle erupts  
#           randomly across the workspace while motion is on going.
#
# Inputs:
#   - Obs:       the initial obstacles on the workspace before motion planning
#   - x_g: the target point the point robot aim to reach in the workspace
#   - x_0: the initial point where the point robot starts its motion from, in the workspace
#   - l:       the length of the 2D workspace
#   - b:       the breadth of the 2D workspace
#   - h:    the uniform size of the grid casted on the 2D workspace
#   - n:    the number of steps the point robot takes, whereby a screenshot of the motion so far is taken
#
# Outputs:
#   - None- just motion plan for case 3 and produce screenshots after each n steps 
#           using the matplotlib for the workspace figure
#---------------------------------------------------------------------------------#
def plan_case3(Obs, x_g, x_0, l, b, h, n):
    D = Grid(Obs, x_g, l, b, h)
    P, D = {}, CellAdjuster(D)
    i, next_init, key, n2 = 1, x_0, (0,0), n*n

    while D[key][0] != 2:
        Obs_adj1 = obsAdjuster_case3(D, P, l, b, h)
        Obs.extend(Obs_adj1[1])
        D1 = Obs_adj1[0]
        D1 = Grid(Obs, x_g, l, b, h)
        D1 = CellAdjuster(D1)

        P1, P_res1, key = plan_move(D1, next_init, n, l, b, h)
        D.update(D1)
        next_init = D1[key][1:]
        n1 = i*n
        if n1 >= n2:
            print("Road blocked completely")
            break
        P.update(P1)
        screenShot(D1, P, P_res1, l, b, Obs, x_g, x_0, n1)   
        i += 1


#---------------------------------------------------------------------------------#
# Function: dynamics_2d
# Purpose:  Function depicts the dynamics followed by the obstacles for case 4 
#
# Inputs:
#   - x:       x-coordinate of given point
#   - y:       y-coordinate of given point
#   - h:    the uniform size of the discretization done on the 2D workspace
#
# Outputs:
#   - gives the the dynamics followed by the obstacles for case 4 
#---------------------------------------------------------------------------------#
def dynamics_2d(x, y, h):
    dx = random.uniform(-h, h)
    dy = random.uniform(-h, h)
    new_x = x + 2*dx
    new_y = y + dy
    return new_x, new_y


#---------------------------------------------------------------------------------#
# Function: advObsInit_c4
# Purpose:  Function initializes the adversarial obstacles for case 4 
#
# Inputs:
#   - D:    dictionary whose keys are respective indices of the cube 
#           with value as a list contaning the label on the cube and the centre point of that cube
#   - P:     dictionary that captures the path planned 
#   - P_adv:     dictionary that captures the path of the adversarial obstacles so far
#   - l:       the length of the 2D workspace
#   - b:       the breadth of the 2D workspace
#   - h:    the uniform size of the grid casted on the 3D workspace
#   - st:   indicates whether the motion is just starting or has already began
#
# Outputs:
#   - gives the the dynamics followed by the obstacles for case 4 
#---------------------------------------------------------------------------------#
def advObsInit_c4(D, P, P_adv, l, b, h, st):
    max_wave = max(D, key=lambda k: D[k][0])

    if st == 'start':
        # create a list of random obs 
        valid_keys = [key for key, value in D.items() if value[0] not in {max_wave, 2, -1}]

        # randomly select keys from the valid keys
        selected_keys = random.sample(valid_keys, 1)

        # ensure none of the selected keys coincide with any key in P
        selected_keys = [key for key in selected_keys if key not in P and 0<=key[0]<=l/h and 0<=key[1]<=b/h]

        return selected_keys
    
    if st == 'subsequent':
        # create a list of obs not init, goal 
        valid_keys = [key for key, value in D.items() if value[0] not in {max_wave, 2, -1}]

        # ensure none of the valid keys coincide with any key in P
        valid_keys = [key for key in valid_keys if key not in P and 0<=key[0]<=l/h and 0<=key[1]<=b/h]
       
        # ensure the next key for P_adv chosen from selected keys coincide with a key in P_adv
        selected_keys = [key for key in valid_keys if key in P_adv]

        # randomly select keys from the selected keys
        selected_keys = random.sample(selected_keys, 1)

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
#   - l:       the length of the 2D workspace
#   - b:       the breadth of the 2D workspace
#   - h:    the uniform size of the grid casted on the 2D workspace
#   - st:   indicates whether the motion is just starting or has already began
#   - n:    the number of steps the point robot takes, whereby a screenshot of the motion so far is taken
# 
# Outputs:
#   - returns a modified dictionary whose wavevalues has been updated
#---------------------------------------------------------------------------------#
def obsAdjuster_case4(D, P, P_adv, l, b, h, st, n=4):
    modified_D, rand_obs = {}, []
    modified_D.update(D)
    
    selected_keys = advObsInit_c4(D, P, P_adv, l, b, h, st)

    for key in selected_keys:
        modified_D[key][0] = -1 # bring up random obs

        # Use dynamics_2d to generate the next n positions
        for i in range(n):
            new_x, new_y = dynamics_2d(modified_D[key][1], modified_D[key][2], h)
            key1 = getCellFromPoint(new_x, new_y, l, b, h)

            # Check if the key1 is valid and not in P
            if 0 <= key1[0] < int(l/h) and 0 <= key1[1] < int(b/h) and key1 not in P:
                modified_D[key1][0] = -1
                modified_D[key][1] = new_x
                modified_D[key][2] = new_y
                P_adv[key1] = [-1, new_x, new_y]

                # Calculate the rectangle vertices
                x1, y1 = new_x - h/2, new_y - h/2
                x2, y2 = new_x + h/2, new_y - h/2
                x3, y3 = new_x + h/2, new_y + h/2
                x4, y4 = new_x - h/2, new_y + h/2
                rand_obs.append([(x1, y1), (x2, y2), (x3, y3), (x4, y4)])
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
#   - l:       the length of the 2D workspace
#   - b:       the breadth of the 2D workspace
#   - h:    the uniform size of the grid casted on the 2D workspace
#   - n:    the number of steps the point robot takes, whereby a screenshot of the motion so far is taken
#
# Outputs:
#   - None- just motion plan for case 4 and produce screenshots after each n steps 
#           using the matplotlib for the workspace figure
#---------------------------------------------------------------------------------#
def plan_case4(Obs, x_g, x_0, l, b, h, n):
    D = Grid(Obs, x_g, l, b, h)
    P, P_adv, D = {}, {}, CellAdjuster(D)
    i, next_init, key, n2 = 1, x_0, (0,0), n*n

    while D[key][0] != 2:
        if i == 1:
            Obs_adj1 = obsAdjuster_case4(D, P, P_adv, l, b, h, 'start')
        else:
            Obs_adj1 = obsAdjuster_case4(D, P, P_adv, l, b, h, 'subsequent')
        
        Obs.extend(Obs_adj1[1])
        D1 = Obs_adj1[0]
        D1 = Grid(Obs, x_g, l, b, h)
        D1 = CellAdjuster(D1)

        P1, P_res1, key = plan_move(D1, next_init, n, l, b, h)
        D.update(D1)
        next_init = D1[key][1:]
        n1 = i*n
        if n1 >= n2:
            print("Road blocked completely")
            break
        P.update(P1)
        P_adv.update(Obs_adj1[2])
        screenShot(D1, P, P_res1, l, b, Obs, x_g, x_0, n1)   
        i += 1

#-----------------------------------------------------------------------------#
# Run the planner for the scenarios
#-----------------------------------------------------------------------------#

# Set up parameters for the workspace
l = 10  # Length of the workspace
b = 8   # Breadth of the workspace
h = .2   # Grid size

# Rectangle obstacles
Obs = [[(0, 7), (1, 7), (1, 8), (0, 8)], [(4, 0), (5, 0), (5, 6), (4, 6)], [(5, 3), (8, 3), (8, 4), (5, 4)], 
       [(7.5, 0.5), (8, .5), (8, 3), (7.5, 3)],
       [(6.5, 5), (10, 5), (10, 5.5), (6.5, 5.5)], [(9, 7), (10, 7), (10, 8), (9, 8)]]  # obstacle coordinates

# starting point and goal
x_0, x_g = (.5, .5), (6, 2)
n = 10 # number of steps for snapshot

# Creating scenerios

# static obs case implementation
# plan_case1(Obs, x_g, x_0, l, b, h, n)

# case1_1 implementation
# plan_case1_1(Obs, x_g, x_0, l, b, h, n)

# case2_1 implementation
# plan_case2_1(Obs, x_g, x_0, l, b, h, n)

# case2_2 implementation
# plan_case2_2(Obs, x_g, x_0, l, b, h, n)

# case3 implementation
# plan_case3(Obs, x_g, x_0, l, b, h, n)

# case4 implementation
plan_case4(Obs, x_g, x_0, l, b, h, n)

