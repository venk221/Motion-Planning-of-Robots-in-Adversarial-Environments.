import contextlib
from matplotlib import animation as anim
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from math import *
import time


def draw_map(start, goal, obstacles_poses, R_obstacles, f=None, draw_gradients=True, nrows=500, ncols=500):
    if draw_gradients and f is not None:
        skip = 4
        [x_m, y_m] = np.meshgrid(np.linspace(-2.5, 2.5, ncols), np.linspace(-2.5, 2.5, nrows))
        [gy, gx] = np.gradient(-f);
        Q = plt.quiver(x_m[::skip, ::skip], y_m[::skip, ::skip], gx[::skip, ::skip], gy[::skip, ::skip])
    else:
        plt.grid()
    plt.plot(start[0], start[1], 'ro', color='purple', markersize=35);
    plt.plot(goal[0], goal[1], 'ro', color='cyan', markersize=35);
    # plt.xlabel('X')
    # plt.ylabel('Y')
    plt.xlim(-3,3)
    plt.ylim(-3,3)
    ax = plt.gca()
    for pose in obstacles_poses:
        circle = plt.Circle(pose, R_obstacles, color='red')
        ax.add_artist(circle)

    rect1 = patches.Rectangle((-1.01,1.1),1.85,0.18,linewidth=1,color='orange',fill='True')     #horizontal
    rect2 = patches.Rectangle((0.74, -1.35), 0.28,1.47,linewidth=1,color='orange',fill='True')    #main
    rect3 = patches.Rectangle(( -1.5, -0.68), 0.28,0.45,linewidth=1,color='orange',fill='True')  #small
    rect4 = patches.Rectangle(( 0, 2), 1,0.4,linewidth=1,color='orange',fill='True')             #up
    rect5 = patches.Rectangle(( -0.8, -0.8), 1,0.2,linewidth=1,color='orange',fill='True')    #new
    ax.add_patch(rect1)
    ax.add_patch(rect2)
    ax.add_patch(rect3)
    ax.add_patch(rect4)
    ax.add_patch(rect5)

def draw_robots(current_point1, routes=None, num_robots=None, robots_poses=None, centroid=None, vel1=None):
    pp = robots_poses
    pp.sort(key=lambda p: atan2(p[1]-centroid[1],p[0]-centroid[0]))
    # formation = patches.Polygon(pp, color='grey', fill=True, linewidth=1);
    # plt.gca().add_patch(formation)
    plt.plot(centroid[0], centroid[1], 'bo', color='blue')

def get_movie_writer(should_write_movie, title, movie_fps, plot_pause_len):
    """
    :param should_write_movie: Indicates whether the animation of SLAM should be written to a movie file.
    :param title: The title of the movie with which the movie writer will be initialized.
    :param movie_fps: The frame rate of the movie to write.
    :param plot_pause_len: The pause durations between the frames when showing the plots.
    :return: A movie writer that enables writing MP4 movie with the animation from SLAM.
    """

    get_ff_mpeg_writer = anim.writers['ffmpeg']
    metadata = dict(title=title, artist='matplotlib', comment='Potential Fields Formation Navigation')
    movie_fps = min(movie_fps, float(1. / plot_pause_len))

    return get_ff_mpeg_writer(fps=movie_fps, metadata=metadata)

@contextlib.contextmanager
def get_dummy_context_mgr():
    """
    :return: A dummy context manager for conditionally writing to a movie file.
    """
    yield None


# HUMAN VELOCITY CALCULATION
hum_time_array = np.ones(10)
hum_pose_array = np.array([ np.ones(10), np.ones(10), np.ones(10) ])
def hum_vel(human_pose):

    for i in range(len(hum_time_array)-1):
        hum_time_array[i] = hum_time_array[i+1]
    hum_time_array[-1] = time.time()

    for i in range(len(hum_pose_array[0])-1):
        hum_pose_array[0][i] = hum_pose_array[0][i+1]
        hum_pose_array[1][i] = hum_pose_array[1][i+1]
        hum_pose_array[2][i] = hum_pose_array[2][i+1]
    hum_pose_array[0][-1] = human_pose[0]
    hum_pose_array[1][-1] = human_pose[1]
    hum_pose_array[2][-1] = human_pose[2]

    vel_x = (hum_pose_array[0][-1]-hum_pose_array[0][0])/(hum_time_array[-1]-hum_time_array[0])
    vel_y = (hum_pose_array[1][-1]-hum_pose_array[1][0])/(hum_time_array[-1]-hum_time_array[0])
    vel_z = (hum_pose_array[2][-1]-hum_pose_array[2][0])/(hum_time_array[-1]-hum_time_array[0])

    hum_vel = np.array( [vel_x, vel_y, vel_z] )

    return hum_vel


def euler_from_quaternion(q):
    """
    Intrinsic Tait-Bryan rotation of xyz-order.
    """
    q = q / np.linalg.norm(q)
    qx, qy, qz, qw = q
    roll = atan2(2.0*(qy*qz + qw*qx), qw*qw - qx*qx - qy*qy + qz*qz)
    pitch = asin(-2.0*(qx*qz - qw*qy))
    yaw = atan2(2.0*(qx*qy + qw*qz), qw*qw + qx*qx - qy*qy - qz*qz)
    return roll, pitch, yaw