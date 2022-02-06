"""
2-input XOR example -- this is most likely the simplest possible example.
"""

from __future__ import print_function

import json
import os
import uuid

import cupy as np
import neat
from tqdm import tqdm

from displaySimulation import GameOfRealSimulation

# import visualize

# size of the input grid
size_x = 50
size_y = 50

# size of the whole grid
size_X = 1000
size_Y = 1000

prefix = uuid.uuid4()

sim_steps = 4

content: dict = None
with open('./configuations/scores.json', 'r') as file:
    content = json.load(file)

global epoch
epoch = 0


def saveConfig(field: np.ndarray, eval: np.ndarray, name: str):
    # save the field
    with open(f"./configuations/{name}", 'wb') as file:
        np.save(file, field)

    # save the saved file in a json
    if (content.get(str(float(eval))) == None):
        content[str(float(eval))] = f"{name}"
        with open('./configuations/scores.json', 'w') as file:
            json.dump(content, file, indent=4)


def eval_genomes(genomes, config):
    global epoch
    epoch += 1
    for genome_id, genome in tqdm(genomes):
        genome.fitness = 4.0
        net = neat.nn.FeedForwardNetwork.create(genome, config)

        array = np.zeros((size_x, size_y))
        s = 0  # sum of the board
        n = 0

        for x in range(size_x):
            for y in range(size_y):
                out = net.activate((x, y, s, n))[0]
                array[x, y] = out
                s += out
                if out > 0.1:
                    n += 1

        sim = GameOfRealSimulation(size_X, size_Y, array, sim_steps)
        sim.run()
        sim.evalField()
        saveConfig(array, sim.score, f"{prefix}__{epoch}__{sim.score}")
        genome.fitness += float(sim.score)


def run(config_file):
    # Load configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(5, filename_prefix='./run/'))

    # Run for up to 300 generations.
    winner = p.run(eval_genomes, 300)

    # Display the winning genome.
    print('\nBest genome:\n{!s}'.format(winner))

    # Show output of the most fit genome against training data.
    print('\nOutput:')
    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)


if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward')
    run(config_path)
