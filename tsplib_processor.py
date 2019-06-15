import numpy as np
import os

class TSPProcessor():
    def __init__(self, tsp_file):
        self.tsp_file = tsp_file

    
    def euc2d_process(self):
        '''
        Process for 2D coordinates with euclidian distance
        Generate numpy array like to represent cites coordinates
        '''
        with open(self.tsp_file) as f:
            content = f.readlines()
        
        # Remove unnecessary character
        content = [line.strip() for line in content]
        
        # Check metric type
        if 'EUC_2D' not in content[4]:
            raise Exception('Metric should be EUC_2D')
        
        # Remove tsp file specs
        content = content[6:]
        
        # Map coordinates ~ (x, y)
        xs = []
        ys = []
        
        for coord_sec in content[:-1]:
            temp = coord_sec.split()
            xs.append(int(temp[1]))
            ys.append(int(temp[2]))

        return np.column_stack((xs, ys))

        



