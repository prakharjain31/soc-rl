import pygame 
from pygame.locals import *
import random

class Square(pygame.sprite.Sprite):
    def __init__(self, x, y):
        super(Square, self).__init__()
        self.surf = pygame.Surface((20, 20))
        self.surf.fill((0, 200, 255))
        # self.rect = self.surf.get_rect()
        self.pos = [x, y]


pygame.init()
# Set up the drawing window
window_x = 800
window_y = 600
screen = pygame.display.set_mode((window_x, window_y))
 
square = Square(40, 40)

# Defining the beginning position and size of the snake
snake_front_position = [200,150]
snake_body = [ [200,150], 
               [190,150],
               [180,150] ]

direction = 'R'
change_dir_to = direction

is_fruit_spawned = True
fruit = pygame.image.load('red.png')
fruit = pygame.transform.scale(fruit, (20, 20))
fruit_position = [random.randrange(15, (window_x//10)) * 10 - 21, random.randrange(15, (window_y//10)) * 10 - 21]



score = 0

# Update the display using flip
pygame.display.flip()


gameOn = True
# Our game loop
while gameOn:
    # for loop through the event queue
    pygame.time.Clock().tick(60)
    for event in pygame.event.get():
        if event.type == QUIT:
            gameOn = False
    keys = pygame.key.get_pressed()

    # Removing old snake
    square.surf.fill((0, 0, 0))
    for pos in snake_body:
        square.pos = pos
        screen.blit(square.surf, tuple(square.pos)) 
    square.surf.fill((0, 200, 255))

    # Changing the direction of the snake
    if keys[K_w] or keys[K_UP]:
        change_dir_to = 'U'
    if keys[K_a] or keys[K_LEFT]:
        change_dir_to = 'L'
    if keys[K_s] or keys[K_DOWN]:
        change_dir_to = 'D'
    if keys[K_d] or keys[K_RIGHT]:
        change_dir_to = 'R'

    # Making sure the snake cannot move in the opposite direction instantaneously
    if change_dir_to == 'U' and direction != 'D':
        direction = 'U'
    if change_dir_to == 'D' and direction != 'U':
        direction = 'D'
    if change_dir_to == 'L' and direction != 'R':
        direction = 'L'
    if change_dir_to == 'R' and direction != 'L':
        direction = 'R'
    
    # Moving the snake
    if direction == 'U':
        snake_front_position[1] -= 5
    if direction == 'D':
        snake_front_position[1] += 5
    if direction == 'L':
        snake_front_position[0] -= 5
    if direction == 'R':
        snake_front_position[0] += 5
    
    # Checking if the snake has eaten the fruit
    if (abs(snake_front_position[0] - fruit_position[0]) < 10  and abs(snake_front_position[1] - fruit_position[1]) < 10):
        score += 1
        is_fruit_spawned = False
    else:
        snake_body.pop()

    # Spawn fruit if it is not already spawned
    if(not is_fruit_spawned):
        fruit_position = [random.randrange(1, (window_x//10)) * 10 - 21, random.randrange(1, (window_y//10)) * 10 - 21]
        is_fruit_spawned = True

    snake_body.insert(0, list(snake_front_position))

    # Putting the snake on the screen
    screen.fill((0,0,0))
    for pos in snake_body:
        square.pos = pos
        screen.blit(square.surf, tuple(square.pos))
    
    # Putting the fruit on the screen
    square.pos = fruit_position
    screen.blit(fruit, tuple(square.pos))
    
    # Checking if the snake is out of bounds
    if snake_front_position[0] < 0 or snake_front_position[0] > window_x-20:
        gameOn = False
    if snake_front_position[1] < 0 or snake_front_position[1] > window_y-20:
        gameOn = False


    # Checking if the snake has hit itself
    for block in snake_body[1:]:
        if snake_front_position[0] == block[0] and snake_front_position[1] == block[1]:
            gameOn = False

    # Update the display using flip
    pygame.display.flip()
print("Your score is: ", score)
pygame.quit()