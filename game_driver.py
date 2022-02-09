import pygame
import random
import config
from enum import Enum
from collections import namedtuple


WHITE = (255, 255, 255)
RED = (200, 0, 0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0, 0, 0)

SPEED = 20

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


class SnakeGame:
    """
    @description: constructor of SnakeGame classs
    """

    def __init__(self, height=config.HEIGHT, width=config.WIDTH):
        self.height, self.width = height, width
        self.display = pygame.display.set_mode((self.width, self.height))
        self.direction = Direction.RIGHT

        self.head = Point(self.width/2, self.height/2)
        self.snake = [self.head,
                      Point(self.head.x-config.BLOCK_SIZE, self.head.y),
                      Point(self.head.x-(2*config.BLOCK_SIZE), self.head.y)]

        self.score = 0
        self.food = None
        self._place_food()

        self.clock = pygame.time.Clock()
        pygame.display.set_caption('RL Snake')

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
    @description: Event loop that checks for user input
    """

    def play_step(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP and self.direction != Direction.DOWN:
                    self.direction = Direction.UP
                elif event.key == pygame.K_DOWN and self.direction != Direction.UP:
                    self.direction = Direction.DOWN
                elif event.key == pygame.K_LEFT and self.direction != Direction.RIGHT:
                    self.direction = Direction.LEFT
                elif event.key == pygame.K_RIGHT and self.direction != Direction.LEFT:
                    self.direction = Direction.RIGHT

        self._move(self.direction)
        self.snake.insert(0, self.head)

        game_over = False
        if self._is_collision():
            game_over = True
            return game_over, self.score

        if self.head == self.food:
            self.score += 1
            self._place_food()
        else:
            self.snake.pop()

        self._update_ui()
        self.clock.tick(SPEED)
        return game_over, self.score

    """
    @description: Collision conditions if snake hits border boundaries
    """

    def _is_collision(self):
        if self.head.x > self.width - config.BLOCK_SIZE or self.head.x < 0 or self.head.y > self.height - config.BLOCK_SIZE or self.head.y < 0 or self.head in self.snake[1:]:
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
    @description: Move by block size in particular direction
    Head of snake gets updated to the new point x,y
    """

    def _move(self, direction):
        x = self.head.x
        y = self.head.y
        if direction == Direction.RIGHT:
            x += config.BLOCK_SIZE
        elif direction == Direction.LEFT:
            x -= config.BLOCK_SIZE
        elif direction == Direction.DOWN:
            y += config.BLOCK_SIZE
        elif direction == Direction.UP:
            y -= config.BLOCK_SIZE
        self.head = Point(x, y)


"""
@description: Main method of game execution, init game and establish game loop
"""
if __name__ == '__main__':
    pygame.init()
    Point = namedtuple('Point', 'x, y')
    font = pygame.font.Font('arial.ttf', 25)

    game = SnakeGame()
    while True:
        game_over, score = game.play_step()

        if game_over == True:
            break

    print('Final Score:', score)

    pygame.quit()
