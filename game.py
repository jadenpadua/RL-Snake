import pygame
import random
import config
from enum import Enum
from collections import namedtuple
import numpy as np

pygame.init()
Point = namedtuple('Point', 'x, y')
font = pygame.font.Font('arial.ttf', 25)


"""
@Agent-Reward: 3
    - eat food: +10
    - game over: -10
    - else: 0

@Agent-States: 11

    - danger_straight, danger_right, danger_left,

    - direction_left, direction_right, direction_up, direction_down,

    - food_left, food_right, food_up, food_down

@Agent-Actions: 3

    - [1, 0, 0] --> straight
    - [0, 1, 0] --> turn right
    - [0, 0, 1] --> turn left

"""

"""
@description: State holding current direction of agent
"""


class Direction(Enum):
    UP = 1
    DOWN = 2
    LEFT = 3
    RIGHT = 4


"""
@description: Basic methods for snake game functionality
"""


class SnakeGameAI:
    """
    @description: constructor of SnakeGame classs
    """

    def __init__(self, height=config.HEIGHT, width=config.WIDTH):
        self.height, self.width = height, width
        self.display = pygame.display.set_mode((self.width, self.height))

        self.clock = pygame.time.Clock()
        pygame.display.set_caption('RL Snake')

        self.reset()
    """
    @description: After each game agent should be able to call reset and start with a new game
    """

    def reset(self):
        self.direction = Direction.RIGHT

        self.head = Point(self.width/2, self.height/2)
        self.snake = [self.head,
                      Point(self.head.x-config.BLOCK_SIZE, self.head.y),
                      Point(self.head.x-(2*config.BLOCK_SIZE), self.head.y)]

        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0

    """
    @description: places food at random point in grid
    """

    def _place_food(self):
        x = random.randint(0, (self.width-config.BLOCK_SIZE) //
                           config.BLOCK_SIZE)*config.BLOCK_SIZE
        y = random.randint(0, (self.height-config.BLOCK_SIZE) //
                           config.BLOCK_SIZE)*config.BLOCK_SIZE
        self.food = Point(x, y)
        if self.food in self.snake:
            self._place_food()

    """
    @description: Takes in an agent action and computes a direction to move, and based on this action  we will either:
        - Give a positive reward for eating food (+10) and place another food
        - Give a negative reward for a collision (-10) and end game
        - Do nothing, direction just updates and frame_iteration increments
    """

    def play_step(self, action):

        self.frame_iteration += 1

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        self._move(action)
        self.snake.insert(0, self.head)

        reward = 0
        game_over = False
        if self.is_collision() or self.frame_iteration > 100*len(self.snake):
            game_over = True
            reward -= 10
            return reward, game_over, self.score

        if self.head == self.food:
            self.score += 1
            reward += 10
            self._place_food()
        else:
            self.snake.pop()

        self._update_ui()
        self.clock.tick(config.SPEED)
        return reward, game_over, self.score

    """
    @description: Collision conditions if snake hits border boundaries
    """

    def is_collision(self, pt=None):
        if pt is None:
            pt = self.head
        if pt.x > self.width - config.BLOCK_SIZE or pt.x < 0 or pt.y > self.height - config.BLOCK_SIZE or pt.y < 0 or pt in self.snake[1:]:
            return True

        return False

    """
    @description: Updates our UI grid to reflect changes in our game states
    """

    def _update_ui(self):
        self.display.fill(config.GREEN)

        for pt in self.snake:
            pygame.draw.rect(self.display, config.BLUE1, pygame.Rect(
                pt.x, pt.y, config.BLOCK_SIZE, config.BLOCK_SIZE))
            pygame.draw.rect(self.display, config.BLUE2,
                             pygame.Rect(pt.x+4, pt.y+4, 12, 12))

        pygame.draw.rect(self.display, config.RED, pygame.Rect(
            self.food.x, self.food.y, config.BLOCK_SIZE, config.BLOCK_SIZE))

        text = font.render("Score: " + str(self.score), True, config.BLACK)
        self.display.blit(text, [0, 0])
        pygame.display.flip()

    """
    @description: Moves in the calculated next_dir based on the action.
    Since your resultant direction is relative based on your current direction when you apply an action,
    we need to make a clock_wise list to account for this relationship
    """

    def _move(self, action):
        clock_wise = [Direction.RIGHT, Direction.DOWN,
                      Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)

        new_dir = 0

        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx]
        elif np.array_equal(action, [0, 1, 0]):
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx]
        else:
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx]

        self.direction = new_dir

        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT:
            x += config.BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= config.BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += config.BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= config.BLOCK_SIZE

        self.head = Point(x, y)
