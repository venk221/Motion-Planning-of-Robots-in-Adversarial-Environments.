"""This is an implementation of a dynamic obstacle avoidance motion planning algorithm of RRT and APF   """

import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
from matplotlib import collections
from scipy.ndimage.morphology import distance_transform_edt as bwdist
from math import *
import random
from impedance_modeles import *
import time

from progress.bar import FillingCirclesBar
from tasks import *
from threading import Thread
from multiprocessing import Process
import os

trips_completed=0
"""Please note that sometimes the algorithm gets stuck in the local minima, kindly rerun the code if that were to happen"""
#To escape local minima
def minimaChecker(V_x,V_y):
    print("checking")
    chk=2
    vx=V_x[-1]
    vy=V_y[-1]
    if(np.mean(V_x[-chk:])==V_x[0]):
        vx=vx+0.05
    if(np.mean(V_y[-chk:])==V_y[0]):
        vy=vy+0.05
    return vx,vy

def meters2grid(pose_m, nrows=500, ncols=500):
    # [0, 0](m) -> [250, 250]
    # [1, 0](m) -> [250+100, 250]
    # [0,-1](m) -> [250, 250-100]
    pose_on_grid = np.array(pose_m)*100 + np.array([ncols/2, nrows/2])
    return np.array( pose_on_grid, dtype=int)
def grid2meters(pose_grid, nrows=500, ncols=500):
    # [250, 250] -> [0, 0](m)
    # [250+100, 250] -> [1, 0](m)
    # [250, 250-100] -> [0,-1](m)
    pose_meters = ( np.array(pose_grid) - np.array([ncols/2, nrows/2]) ) / 100.0
    return pose_meters

def gradient_planner(f, current_point, v_listx,v_listy, ncols=500, nrows=500,movement_rate=0.6):
    """
    GradientBasedPlanner : This function computes the next_point
    given current location, goal location and potential map, f.
    It also returns mean velocity, V, of the gradient map in current point.
    """
    [gy, gx] = np.gradient(-f);
    iy, ix = np.array( meters2grid(current_point), dtype=int )
    print(ix.shape, iy.shape)
    w = 50 # smoothing window size for gradient-velocity
    vx = np.mean(gx[ix-int(w/2) : ix+int(w/2), iy-int(w/2) : iy+int(w/2)])
    vy = np.mean(gy[ix-int(w/2) : ix+int(w/2), iy-int(w/2) : iy+int(w/2)])
    v_listx=np.append(v_listx,vx)
    v_listy=np.append(v_listy,vy)
    vx,vy=minimaChecker(v_listx[-10:],v_listy[-10:])
    V = np.array([vx, vy])
    dt = 0.06 / norm(V);
    next_point = current_point + dt*V;
    print(current_point)
    #print("Current force is: ",f[current_point[0],current_point[1]])
    print("Current velocity is : ",V)
    return next_point, V, v_listx[-10:],v_listy[-10:]

def combined_potential(obstacles_poses, goal, nrows=500, ncols=500):
    """ Repulsive potential """
    obstacles_map = map(obstacles_poses,goal)
    goal = meters2grid(goal)
    d = bwdist(obstacles_map==0);
    d2 = (d/100.) + 1; # Rescale and transform distances
    d0 = 2;
    nu = 300;
    repulsive = nu*((1./d2 - 1./d0)**2);
    repulsive [d2 > d0] = 0;
    """ Attractive potential """
    [x, y] = np.meshgrid(np.arange(ncols), np.arange(nrows))
    xi = 1/700.;
    attractive = xi * ( (x - goal[0])**2 + (y - goal[1])**2 );
    """ Combine terms """
    f = attractive + repulsive;
    #f[goal] = -1

    return f
    
def map(obstacles_poses, goal, nrows=500, ncols=500):
    """ Obstacles map """
    obstacles_map = np.zeros((nrows, ncols));
    [x, y] = np.meshgrid(np.arange(ncols), np.arange(nrows))
    for pose in obstacles_poses:
        pose = meters2grid(pose)
        x0 = pose[0]; y0 = pose[1]
        # cylindrical obstacles
        t = ((x - x0)**2 + (y - y0)**2) < (100*R_obstacles)**2
        obstacles_map[t] = 1
    # rectangular obstacles
    obstacles_map[360:380,150:345] = 1; #hori
    obstacles_map[185:225, 100:130] = 1; #small
    obstacles_map[120:260, 320:350] = 1; #main
    obstacles_map[460:490, 250:350] = 1; #up
    obstacles_map[170:190, 170:270] = 1; #new
    obstacles_map[15:17, :] = 1
    obstacles_map[500:, :] = 1
    obstacles_map[:, 20:25] = 1
    obstacles_map[:, 495:500] = 1
    goal = 0
 
    return obstacles_map

def move_obstacles(obstacles_poses, obstacles_goal_poses):
    """ All of the obstacles tend to go to the origin, (0,0) - point """
    # for pose in obstacles_poses:
    #   dx = random.uniform(0, 0.03);        dy = random.uniform(0,0.03);
    #   pose[0] -= np.sign(pose[0])*dx;      pose[1] -= np.sign(pose[1])*dy;

    """ Each obstacles tends to go to its selected goal point with random speed """
    for p in range(len(obstacles_poses)):
        pose = obstacles_poses[p]; goal = obstacles_goal_poses[p]
        dx, dy = (goal - pose) / norm(goal-pose) * random.uniform(0,0.05)
        pose[0] += dx;      pose[1] += dy;

    return obstacles_poses


def formation(num_robots, leader_des, v, R_swarm):
    if num_robots<=1: return [] 
    u = np.array([-v[1], v[0]])
    des4 = leader_des - v*R_swarm*sqrt(3)                 # follower
    if num_robots==2: return [des4]
    des2 = leader_des - v*R_swarm*sqrt(3)/2 + u*R_swarm/2 # follower
    des3 = leader_des - v*R_swarm*sqrt(3)/2 - u*R_swarm/2 # follower
    if num_robots==3: return [des2, des3]
    
    return [des2, des3, des4]

""" initialization """
animate              = 1   # show 1-each frame or 0-just final configuration
random_obstacles     = 1   # randomly distributed obstacles on the map
num_random_obstacles = 10   # number of random circular obstacles on the map
num_robots           = 4   # <=4, number of drones in formation
moving_obstacles     = 1   # 0-static or 1-dynamic obstacles
impedance            = 1   # impedance links between the leader and followers (leader's velocity)
formation_gradient   = 1   # followers are attracting to their formation position and repelling from obstacles
draw_gradients       = 1  # 1-gradients plot, 0-grid
postprocessing       = 1   # show processed data figures after the flight
""" human guided swarm params """
interactive          = 0      # 1-human guided swarm (requires MoCap system), 0-potential fields as a planner to goal pose
human_name           = 'palm' # vicon mocap object
pos_coef             = 3.0    # scale of the leader's movement relatively to the human operator
initialized          = False  # is always inits with False: for relative position control
max_its              = 600 if interactive else 280 # max number of allowed iters for formation to reach the goal
# movie writer
progress_bar = FillingCirclesBar('Number of Iterations', max=max_its)
should_write_movie = 0; movie_file_name = os.getcwd()+'/videos/output.avi'
movie_writer = get_movie_writer(should_write_movie, 'Simulation Potential Fields', movie_fps=10., plot_pause_len=0.01)

R_obstacles = 0.05 # [m]
R_swarm     = 0.03 # [m]
start = np.array([-1.6, 1.6]); goal = np.array([1.7, -0.5])
V0 = (goal - start) / norm(goal-start)    # initial movement direction, |V0| = 1
U0 = np.array([-V0[1], V0[0]]) / norm(V0) # perpendicular to initial movement direction, |U0|=1
imp_pose_prev = np.array([0, 0])
imp_vel_prev  = np.array([0, 0])
imp_time_prev = time.time()

if random_obstacles:
    obstacles_poses      = np.random.uniform(low=-2.5, high=2.5, size=(num_random_obstacles,2)) # randomly located obstacles
    obstacles_goal_poses = np.random.uniform(low=-3.5, high=3.5, size=(num_random_obstacles,2)) # randomly located obstacles goal poses
else:
    obstacles_poses      = np.array([[-2, 1], [1.5, 0.5], [-1.0, 1.5], [0.1, 0.1], [1, -2], [-1.8, -1.8]]) # 2D - coordinates [m]
    obstacles_goal_poses = np.array([[-0, 0], [0.0, 0.0], [ 0.0, 0.0], [0.0, 0.0], [0,  0], [ 0.0,  0.0]])

""" Main loop """

# drones polygonal formation
route1 = start # leader
current_point1 = start
robots_poses = [start] + formation(num_robots, start, V0, R_swarm)
routes = [route1] + robots_poses[1:]
centroid_route = [ sum([p[0] for p in robots_poses])/len(robots_poses), sum([p[1] for p in robots_poses])/len(robots_poses) ]
des_poses = robots_poses
#print("Robot_Poses:", robots_poses)
vels = [];
VX=np.zeros((1,10))
VY=np.zeros((1,10))
for r in range(num_robots): vels.append([])
norm_vels = [];
for r in range(num_robots): norm_vels.append([])

# variables for postprocessing and performance estimation
area_array = []
start_time = time.time()

fig = plt.figure(figsize=(10, 10))
with movie_writer.saving(fig, movie_file_name, max_its) if should_write_movie else get_dummy_context_mgr():
    for i in range(max_its):
        if moving_obstacles: obstacles_poses = move_obstacles(obstacles_poses, obstacles_goal_poses)

        """ Leader's pose update """
        if interactive:
            # human palm pose and velocity using Vicon motion capture
            if not initialized:
                human_pose_init = human_pose[:2]
                drone1_pose_init = start
                initialized = True
            dx, dy = human_pose[:2] - human_pose_init
            des_poses[0] = np.array([  drone1_pose_init[0] + pos_coef*dx, drone1_pose_init[1] + pos_coef*dy ])
            vels[0] = pos_coef*hum_vel(human_pose)
            f1 = combined_potential(obstacles_poses, des_poses[0])
            des_poses[0], _ = gradient_planner(f1, des_poses[0])
            direction = np.array([cos(human_yaw), sin(human_yaw)]) # rotation of the swarm relatively to human orientation
        else:
            f1 = combined_potential(obstacles_poses, goal)
            des_poses[0], vels[0],VX,VY = gradient_planner(f1, current_point1,VX,VY)
            #print("jerhbfiurhgehg",des_poses)
            direction = ( goal - des_poses[0] ) / norm(goal - des_poses[0])
        norm_vels[0].append(norm(vels[0]))

        # drones polygonal formation
        # direction = ( goal - des_poses[0] ) / norm(goal - des_poses[0])
        des_poses[1:] = formation(num_robots, des_poses[0], direction, R_swarm)
        v = direction; u = np.array([-v[1], v[0]])

        if formation_gradient:
            # following drones are attracting to desired points - vertices of the polygonal formation
            for p in range(1, num_robots):
                """ including another robots in formation in obstacles array: """
                robots_obstacles = [x for i,x in enumerate(robots_poses) if i!=p]
                # obstacles_poses1 = np.array(robots_obstacles + obstacles_poses.tolist())
                # f = combined_potential(obstacles_poses1, des_poses[p])
                f = combined_potential(obstacles_poses, des_poses[p])
                des_poses[p], vels[p],VX,VY = gradient_planner(f, des_poses[p],VX,VY)
                norm_vels[p].append(norm(vels[p]))

        for r in range(num_robots):
            routes[r] = np.vstack([routes[r], des_poses[r]])

        current_point1 = des_poses[0] # update current point of the leader

        pp = des_poses
        centroid = [ sum([p[0] for p in pp])/len(pp), sum([p[1] for p in pp])/len(pp) ]
        centroid_route = np.vstack([centroid_route, centroid])
        dist_to_goal = norm(centroid - goal)
        if dist_to_goal < 2.5*R_swarm:
            print('\nReached the goal')
            trips_completed+=1
            if(trips_completed<2):
                goal=start
            else:
                break

        progress_bar.next()
        plt.cla()

        draw_map(start, goal, obstacles_poses, R_obstacles, f, draw_gradients=draw_gradients)
        draw_robots(current_point1, routes, num_robots, robots_poses, centroid, vels[0])
        if animate:
            plt.draw()
            plt.pause(0.001)

        if should_write_movie:
            movie_writer.grab_frame()
        # print('Current simulation time: ', time.time()-start_time)
    print('\nDone')
    progress_bar.finish()
    end_time = time.time()
    print('Simulation execution time: ', round(end_time-start_time,2))
    plt.show()

""" Flight data postprocessing """
if postprocessing:
    plt.figure()
    plt.title("Robot's trajectory")
    plt.plot(centroid_route[:,0]) #, centroid_route[:,1]
    for route in routes:
        plt.plot(route[:,0], route[:,1], '--')
    plt.grid()

    plt.figure()
    plt.title("Robot average velocity, <V>= %s m/s" %round(np.mean(np.array(norm_vels[0])), 2))
    # for r in range(num_robots):
    plt.plot(norm_vels[0])
    plt.xlabel('time')
    plt.ylabel('velocity, [m/s]')
    plt.grid()
    
    plt.grid()
    plt.show()

# TODO:
# 1. local minimum problem (FM2 - algorithm: https://pythonhosted.org/scikit-fmm/)
# 2. impedance controlled shape of the formation: area(velocity)
# 3. postprocessing: trajectories smoothness, etc. compare imp modeles:
#     - oscillation, underdamped, critically damped, overdamped
#     - velocity plot for all drones, acc, jerk ?
# 4. another drones are obstacles for each individual drone (done, however attractive and repelling forces should be adjusted)
# 5. import swarmlib (OOP) and test flight
# 6. add borders: see image processing (mirrow or black)
