# Developer : Hamdy Abou El Anein
# https://github.com/hamdyaea/Daylight-Pong-python3/blob/master/DaylightPong.py

import random
import pygame
import sys
from pygame import *
from easygui import *
import csv
import time

pygame.init()
fps = pygame.time.Clock()


WHITE = (255, 255, 255)
ORANGE = (255, 140, 0)
GREEN = (0, 255, 0)
BLACK = (0, 0, 0)


WIDTH = 600
HEIGHT = 400
BALL_RADIUS = 20
PAD_WIDTH = 8
PAD_HEIGHT = 80
HALF_PAD_WIDTH = PAD_WIDTH // 2
HALF_PAD_HEIGHT = PAD_HEIGHT // 2
ball_pos = [0, 0]
ball_vel = [0, 0]
paddle1_vel = 0
paddle2_vel = 0
l_score = 0
r_score = 0


window = pygame.display.set_mode((WIDTH, HEIGHT), 0, 32)
pygame.display.set_caption("Daylight Pong")


def ball_init(right):
    global ball_pos, ball_vel
    ball_pos = [WIDTH // 2, HEIGHT // 2]
    horz = 10
    vert = random.randrange(-3, 3)

    if right == False:
        horz = -horz

    ball_vel = [horz, -vert]


def init():
    global paddle1_pos, paddle2_pos, paddle1_vel, paddle2_vel, l_score, r_score  # these are floats
    global score1, score2  # these are ints
    paddle1_pos = [HALF_PAD_WIDTH - 1, HEIGHT // 2]
    paddle2_pos = [WIDTH + 1 - HALF_PAD_WIDTH, HEIGHT // 2]
    l_score = 0
    r_score = 0
    if random.randrange(0, 2) == 0:
        ball_init(True)
    else:
        ball_init(False)


def draw(canvas):
    global paddle1_pos, paddle2_pos, ball_pos, ball_vel, l_score, r_score

    canvas.fill(BLACK)
    pygame.draw.line(canvas, WHITE, [WIDTH // 2, 0], [WIDTH // 2, HEIGHT], 1)
    pygame.draw.line(canvas, WHITE, [PAD_WIDTH, 0], [PAD_WIDTH, HEIGHT], 1)
    pygame.draw.line(
        canvas, WHITE, [WIDTH - PAD_WIDTH, 0], [WIDTH - PAD_WIDTH, HEIGHT], 1
    )
    pygame.draw.circle(canvas, WHITE, [WIDTH // 2, HEIGHT // 2], 70, 1)

    if paddle1_pos[1] > HALF_PAD_HEIGHT and paddle1_pos[1] < HEIGHT - HALF_PAD_HEIGHT:
        paddle1_pos[1] += paddle1_vel
    elif paddle1_pos[1] == HALF_PAD_HEIGHT and paddle1_vel > 0:
        paddle1_pos[1] += paddle1_vel
    elif paddle1_pos[1] == HEIGHT - HALF_PAD_HEIGHT and paddle1_vel < 0:
        paddle1_pos[1] += paddle1_vel

    if paddle2_pos[1] > HALF_PAD_HEIGHT and paddle2_pos[1] < HEIGHT - HALF_PAD_HEIGHT:
        paddle2_pos[1] += paddle2_vel
    elif paddle2_pos[1] == HALF_PAD_HEIGHT and paddle2_vel > 0:
        paddle2_pos[1] += paddle2_vel
    elif paddle2_pos[1] == HEIGHT - HALF_PAD_HEIGHT and paddle2_vel < 0:
        paddle2_pos[1] += paddle2_vel

    ball_pos[0] += int(ball_vel[0])
    ball_pos[1] += int(ball_vel[1])

    pygame.draw.circle(canvas, ORANGE, ball_pos, 20, 0)
    pygame.draw.polygon(
        canvas,
        GREEN,
        [
            [paddle1_pos[0] - HALF_PAD_WIDTH, paddle1_pos[1] - HALF_PAD_HEIGHT],
            [paddle1_pos[0] - HALF_PAD_WIDTH, paddle1_pos[1] + HALF_PAD_HEIGHT],
            [paddle1_pos[0] + HALF_PAD_WIDTH, paddle1_pos[1] + HALF_PAD_HEIGHT],
            [paddle1_pos[0] + HALF_PAD_WIDTH, paddle1_pos[1] - HALF_PAD_HEIGHT],
        ],
        0,
    )
    pygame.draw.polygon(
        canvas,
        GREEN,
        [
            [paddle2_pos[0] - HALF_PAD_WIDTH, paddle2_pos[1] - HALF_PAD_HEIGHT],
            [paddle2_pos[0] - HALF_PAD_WIDTH, paddle2_pos[1] + HALF_PAD_HEIGHT],
            [paddle2_pos[0] + HALF_PAD_WIDTH, paddle2_pos[1] + HALF_PAD_HEIGHT],
            [paddle2_pos[0] + HALF_PAD_WIDTH, paddle2_pos[1] - HALF_PAD_HEIGHT],
        ],
        0,
    )

    if int(ball_pos[1]) <= BALL_RADIUS:
        ball_vel[1] = -ball_vel[1]
    if int(ball_pos[1]) >= HEIGHT + 1 - BALL_RADIUS:
        ball_vel[1] = -ball_vel[1]

    if int(ball_pos[0]) <= BALL_RADIUS + PAD_WIDTH and int(ball_pos[1]) in range(
        paddle1_pos[1] - HALF_PAD_HEIGHT, paddle1_pos[1] + HALF_PAD_HEIGHT, 1
    ):
        ball_vel[0] = -ball_vel[0]
        ball_vel[0] *= 1.1
        ball_vel[1] *= 1.1
    elif int(ball_pos[0]) <= BALL_RADIUS + PAD_WIDTH:
        r_score += 1
        ball_init(True)
        paddle1_pos = [HALF_PAD_WIDTH - 1, HEIGHT // 2]
        paddle2_pos = [WIDTH + 1 - HALF_PAD_WIDTH, HEIGHT // 2]
        time.sleep(0.100)

    if int(ball_pos[0]) >= WIDTH + 1 - BALL_RADIUS - PAD_WIDTH and int(
        ball_pos[1]
    ) in range(paddle2_pos[1] - HALF_PAD_HEIGHT, paddle2_pos[1] + HALF_PAD_HEIGHT, 1):
        ball_vel[0] = -ball_vel[0]
        ball_vel[0] *= 1.1
        ball_vel[1] *= 1.1
    elif int(ball_pos[0]) >= WIDTH + 1 - BALL_RADIUS - PAD_WIDTH:
        l_score += 1
        ball_init(False)
        paddle1_pos = [HALF_PAD_WIDTH - 1, HEIGHT // 2]
        paddle2_pos = [WIDTH + 1 - HALF_PAD_WIDTH, HEIGHT // 2]
        time.sleep(0.100)

    myfont1 = pygame.font.SysFont("Comic Sans MS", 20)
    label1 = myfont1.render("Score " + str(l_score), 1, (255, 255, 0))
    canvas.blit(label1, (50, 20))

    myfont2 = pygame.font.SysFont("Comic Sans MS", 20)
    label2 = myfont2.render("Score " + str(r_score), 1, (255, 255, 0))
    canvas.blit(label2, (470, 20))

    # just save game status and score
    with open('game_report.csv', mode='w') as csv_file:
        fieldnames = ['ball_pos_c', 'ball_pos_l', 'r_score', 'l_score']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

        writer.writeheader()
        writer.writerow({'ball_pos_c': ball_pos[0], 'ball_pos_l': ball_pos[1], 'r_score': r_score, 'l_score': l_score})


def keydown(event):
    global paddle1_vel, paddle2_vel

    if event.key == K_UP:
        paddle2_vel = -8
    elif event.key == K_DOWN:
        paddle2_vel = 8
    elif event.key == K_z:
        paddle1_vel = -8
    elif event.key == K_s:
        paddle1_vel = 8


def keyup(event):
    global paddle1_vel, paddle2_vel

    if event.key in (K_z, K_s):
        paddle1_vel = 0
    elif event.key in (K_UP, K_DOWN):
        paddle2_vel = 0



def track_event():


    while True:

        draw(window)

        for event in pygame.event.get():

            if event.type == KEYDOWN:
                keydown(event)
            elif event.type == KEYUP:
                keyup(event)
            elif event.type == QUIT:
                pygame.display.quit()
                pygame.quit()
                sys.exit()

        pygame.display.update()
        fps.tick(60)