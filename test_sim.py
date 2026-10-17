import jax.numpy as jnp
import jax
import numpy as np

from tangrams.sim import Sim, Settings, State, Action

import pygame, sys
pygame.init()

print("CONTROLS:")
print("ESDF to Move")
print("JK to Rotate")
print("1-7 to Change Controlled Shape")

env = Sim(settings=Settings(ctrl_dt=0.1, sim_dt=0.025))
reset = jax.jit(env.reset)
step = jax.jit(env.step)
render = jax.jit(env.render)

FPS = round(1 / env.settings.sim_dt)
clock = pygame.time.Clock()

pygame.display.set_caption("Tangrams Sim")
screen = pygame.display.set_mode(env.render_size)

state = reset()
collided = False

shape_i = 0
NUM_KEYS = { pygame.K_1 + i: i + 1 for i in range(7) }  # K_1..K_7 -> 1..7

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            sys.exit()
        if event.type == pygame.KEYDOWN:
            if event.key in NUM_KEYS:
                shape_i = NUM_KEYS[event.key] - 1

    keys = pygame.key.get_pressed()

    move = np.zeros(2)

    if keys[pygame.K_s]: move[0] -= 1
    if keys[pygame.K_f]: move[0] += 1

    if keys[pygame.K_d]: move[1] -= 1
    if keys[pygame.K_e]: move[1] += 1

    rotate = 0
    if keys[pygame.K_j]: rotate += 1
    if keys[pygame.K_k]: rotate -= 1

    target_vel = np.zeros((7, 2))
    vel_norm = np.linalg.norm(move)
    if vel_norm != 0:
        target_vel[shape_i] = move / vel_norm * env.settings.max_vel

    target_ang_vel = np.zeros(7)
    target_ang_vel[shape_i] = rotate * env.settings.max_ang_accel

    action = Action(target_vel=target_vel, target_ang_vel=target_ang_vel)

    if not collided:
        state, collided = step(state, action)
        if collided: print("COLLISION")

    #print(state.)

    img_arr = np.asarray(render(state, action))
    img_arr = img_arr[:, ::-1, :] # flip y-axis since pygame expects up -> -y

    pygame_surface = pygame.surfarray.make_surface(img_arr)
    screen.blit(pygame_surface, (0,0))
    pygame.display.flip()

    clock.tick(FPS)