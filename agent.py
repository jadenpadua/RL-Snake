import torch
import random
import numpy as np
from collections import deque
from game import SnakeGameAI, Direction, Point

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

"""
@description: AI Agent used to play our snake game with our model
"""


class Agent:
    """
    @description: init agent with certain properties
        - epsilon = randomness
        - gamma = discount rate
    """

    def __init__(self):
        self.games = 0
        self.epsilon = 0
        self.gamma = 0
        self.memory = deque(maxlen=MAX_MEMORY)

        self.model = None
        self.trainer = None

        # TODO, model, trainer
    """
    @description: Calculates current 11 state list at our head position
        - generate 4 points adjacent to the head
        - generate 4 0-1 values for current game direction
        - For each danger dir, we calc based on u,d,l,r if we get in to that danger dir, if 1 of them is true that danger dir is true
        - then 4 states used to hold what move direction we are
        - then 4 states ised to hold where food is located
        - returns np array with list of boolean values as required
    """

    def get_state(self, game):
        head = game.snake[0]
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)

        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            # Danger straight
            (dir_r and game.is_collision(point_r)) or
            (dir_l and game.is_collision(point_l)) or
            (dir_u and game.is_collision(point_u)) or
            (dir_d and game.is_collision(point_d)),
            # Danger right
            (dir_u and game.is_collision(point_r)) or
            (dir_d and game.is_collision(point_l)) or
            (dir_l and game.is_collision(point_u)) or
            (dir_r and game.is_collision(point_d)),
            # Danger left
            (dir_d and game.is_collision(point_r)) or
            (dir_u and game.is_collision(point_l)) or
            (dir_r and game.is_collision(point_u)) or
            (dir_l and game.is_collision(point_d)),
            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            # Food location
            game.food.x < game.head.x,  # food left
            game.food.x > game.head.x,  # food right
            game.food.y < game.head.y,  # food up
            game.food.y > game.head.y  # food down
        ]

        return np.array(state, dtype=int)
    """
    @description: Remembers tuple of variables and stores in our deque
        - NOTE: when entries exceed 100,000, popleft() is called
    """

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    """
    @description: 
        - Takes random batch sample if memory is above 10,000
        - If below 10,000 we let our sample be the entire memory
        - Then zips the sample into multiple tuples with the respective vars inside
        - Sends these tuples of data into long term training step
    """

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            sample = random.sample(self.memory, BATCH_SIZE)
        else:
            sample = self.memory

        states, actions, rewards, next_states, dones = zip(*sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    """
    @description: Takes state vars for (1 step) and sends it to our model to train
    """

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)
        pass

    """
    @description: Calculates what action we should take
        - Tradeoff between exploration vs. explotation
        - In the beggining we elect for randomness and explore
        - As model gets more defined we elect to exploit that instead
        - more games --> smaller epsilon --> more exploitation
        - if we exploit we pass our state list into tensor
        - obtain prediction --> get index with max argument (max probability)
        - set final move of that index to 1
    """

    def get_action(self, state):
        self.epsilon = 80 - self.games
        final_move = [0, 0, 0]
        # Exploration
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        # Exploitation
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model.predict(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move


"""
@description: Init method which kicks off the training and instantiates game + agent
    - defines variables to plot later
    - kicks off training loop to train model until we exit out
    - get old state --> predict final move --> apply final move --> get new state
    - saves our Model if we get a high score
"""


def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGameAI()
    # Training loop
    while True:
        state_old = agent.get_state(game)
        final_move = agent.get_action(state_old)
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        agent.train_short_memory(
            state_old, final_move. reward, state_new, done)

        agent.remember(state_old, final_move. reward, state_new, done)

        if done:
            game.reset()
            games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save()

            print('Game:', agent.games, 'Score:', score, 'Record:', record)


if __name__ == '__main__':
    train()
