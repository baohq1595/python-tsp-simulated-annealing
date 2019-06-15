from nodes_generator import NodeGenerator
from simulated_annealing import SimulatedAnnealing
from tsplib_processor import TSPProcessor
import numpy as np
import argparse
import os

def main():
    parser = argparse.ArgumentParser(description='Input TSP file and hyperparameters')
    parser.add_argument('--tsp_file', '-f', help='TSP file path')

    args = parser.parse_args()
    tsp_file = args.tsp_file
    processor = TSPProcessor(tsp_file)
    nodes = processor.euc2d_process()


    '''set the simulated annealing algorithm params'''
    temp = 10000
    stopping_temp = 0.00000001
    alpha = 0.999995
    stopping_iter = 10000000

    '''set the dimensions of the grid'''
    size_width = 200
    size_height = 200

    '''set the number of nodes'''
    population_size = 70

    '''generate random list of nodes'''
    # nodes = NodeGenerator(size_width, size_height, population_size).generate() 

    '''run simulated annealing algorithm with 2-opt'''
    sa = SimulatedAnnealing(nodes, temp, alpha, stopping_temp, stopping_iter)
    sa.anneal()

    '''animate'''
    save_path = os.path.join(os.path.join(os.path.dirname(tsp_file), 'result_gif'),
                                os.path.basename(tsp_file) + '.gif')
    sa.animateSolutions(save_path)

    '''show the improvement over time'''
    sa.plotLearning()


if __name__ == "__main__":
    main()