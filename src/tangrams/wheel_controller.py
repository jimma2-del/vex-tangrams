## Alan Connor's Code (adapted slightly) ##
## Two Functions to find power and direction for each motor ##

import numpy as np

import sys

WHEEL_COUNT = 3

#vector projection function
def proj_of_u_on_v(u,v):
  v_norm = np.sqrt(sum(v**2))
  proj = (np.dot(u, v)/v_norm**2)*v
  return(proj)


#Function to calculate the power for each motor
def calcPower(Desired_Direction):

  wheels_orientation = np.array([[1, 0],
                                 [1/2, -np.sqrt(3)/2],
                                 [1/2, np.sqrt(3)/2]])


  wheel_power_vectors = np.zeros_like(wheels_orientation)

  for i in range(WHEEL_COUNT):
    v = np.array([wheels_orientation[i][0],wheels_orientation[i][1]])
    proj = proj_of_u_on_v(Desired_Direction,v)
    wheel_power_vectors[i][0] = proj[0]/3
    wheel_power_vectors[i][1] = proj[1]/3


  wheel_power = np.zeros((WHEEL_COUNT,1))

  for i in range(WHEEL_COUNT):
    if wheels_orientation[i][0] != 0:
      wheel_power[i] = wheel_power_vectors[i][0]/wheels_orientation[i][0]
    else:
      wheel_power[i] = wheel_power_vectors[i][1]/wheels_orientation[i][1]


  return(2*wheel_power)

WHEEL_DIRS =  np.array([
    [1, 0],
    [1/2, -np.sqrt(3)/2],
    [1/2, np.sqrt(3)/2]
]) # shape (n_wheels=3, n_dims=2)

WHEEL_DISTS = np.array((4, -7, 7))
  # dist from center; positive if facing CCW, negative if facing CW

WHEEL_SPEEDS_MATRIX = np.hstack((WHEEL_DIRS, WHEEL_DISTS[:, None]))
print(WHEEL_SPEEDS_MATRIX)

def get_wheel_speeds(dir, rot):
    x = np.append(dir, rot)
    speeds = WHEEL_SPEEDS_MATRIX @ x
    print(x, speeds)
    return speeds