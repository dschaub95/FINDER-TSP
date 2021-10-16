import numpy as np
from sklearn.decomposition import PCA
import networkx as nx
from itertools import combinations
import math
import random

class TSP_EucTransformer:
    
    def __init__(self, sampling_steps=64) -> None:
        self.problems = None
        self.transformed_problems = None
        
        self.problem_size = None
        self.dimension = None
        self.num_problems = None
        self.sampling_steps = sampling_steps

    def load_TSPs_from_nx(self, g_list):
        # extract node features (node position)
        self.problems = np.array([[g.nodes[k]['coord'] for k in g.nodes()] for g in g_list])
        self.transformed_problems = np.tile(np.expand_dims(self.problems, axis=1), (1,self.sampling_steps,1,1))
        # self.transformed_problems = np.copy(self.problems)
        self.dimension = self.problems.shape[-1]
        self.problem_size = self.problems.shape[-2]
        self.num_problems = self.problems.shape[0]

    def load_TSP_from_coords(self, problems):
        self.problems = np.array(problems)
        self.transformed_problems = np.tile(np.expand_dims(self.problems, axis=1), (1,self.sampling_steps,1,1))
        # self.transformed_problems = np.copy(self.problems)
        self.dimension = self.problems.shape[-1]
        self.problem_size = self.problems.shape[-2]
        self.num_problems = self.problems.shape[0]

    def save_TSP_as_nx(self, problems=None):
        if problems is None:
            problems = self.problems
        g_list = []
        for problem in problems:
            edges = [(s[0],t[0],np.linalg.norm(s[1]-t[1])) for s,t in combinations(enumerate(problem),2)]
            g = nx.Graph()
            g.add_weighted_edges_from(edges)
            feature_dict = {k: {'coord': problem[k]} for k in range(self.problem_size)} 
            nx.set_node_attributes(g, feature_dict)
            g_list.append(g)
        return g_list
    
    def reset_TSPs_to_default(self):
        self.transformed_problems = np.tile(self.problems, (1,self.sampling_steps,1,1))
    
    def scale_TSP(self, scaler=1.0, problems=None):
        if problems is None:
            problems = self.problems
        self.transformed_problems = problems * scaler

    def flip_TSP_coordinates(self, problems=None):
        # if self.dimension == 2:
        #     matrix = np.array([[0,1],[1,0]])
        #     self.coordinates = self.coordinates @ matrix
        # else:
        #     temp = self.coordinates.transpose()
        #     np.random.shuffle(temp)
        #     self.coordinates = temp.transpose()
        if problems is None:
            problems = self.problems
        self.problems = self.problems[:,:,::-1]

    def pomo_TSP(self, variant=0):
        assert self.dimension == 2
        x = self.problems[:,:,0:1]
        y = self.problems[:,:,1::]
        if variant == 0:
            self.problems = np.concatenate((x, y), -1)
        elif variant == 1:
            self.problems = np.concatenate((1 - x, y), 2)
        elif variant == 2:
            self.problems = np.concatenate((x, 1 - y), -1)
        elif variant == 3:
            self.problems = np.concatenate((1 - x, 1 - y), -1)
        elif variant == 4:
            self.problems = np.concatenate((y, x), -1)
        elif variant == 5:
            self.problems = np.concatenate((1 - y, x), -1)
        elif variant == 6:
            self.problems = np.concatenate((y, 1 - x), -1)
        elif variant == 7:
            self.problems = np.concatenate((1 - y, 1 - x), -1)
    
    def flip_TSP_simple(self, refit=False):
        # can be realized as a (N+1)D rotation along various axis
        assert self.dimension == 2
        self.center_TSP(refit)
        temp = np.concatenate((self.problems, np.zeros((self.num_problems, self.problem_size, 1))), 2)
        temp = rotate_3D(temp, degree=180)
        self.problems = temp[:,:,0:self.dimension]
        if refit:
            self.fit_TSP_into_square()
    
    def flip_TSP(self, flip_axis=0, refit=False):
        if flip_axis == 0:
            pass
        elif flip_axis == 1:
            self.flip_TSP_simple()
        elif flip_axis == 2:
            self.rotate_TSP(degree=90, refit=refit)
            self.flip_TSP_simple()
            self.rotate_TSP(degree=-90, refit=refit)
        elif flip_axis == 3:
            self.rotate_TSP(degree=45, refit=refit)
            self.flip_TSP_simple()
            self.rotate_TSP(degree=-45, refit=refit)
        elif flip_axis == 4:
            self.rotate_TSP(degree=-45, refit=refit)
            self.flip_TSP_simple()
            self.rotate_TSP(degree=45, refit=refit)
    
    def apply_PCA_to_TSP(self, variant=1):
        if variant == 0:
            return 0
        pca = PCA(n_components=self.dimension) # center & rotate coordinates
        self.problems = np.array([pca.fit_transform(problem) for problem in self.problems])
        self.fit_TSP_into_square()

    def rotate_TSP(self, degree, refit=True):
        assert self.dimension == 2
        self.center_TSP(refit)
        self.problems = rotate_2D(self.problems, degree)
        if refit:
            self.fit_TSP_into_square()
        else:
            self.translate_TSP(shift=(0.5,0.5))

    def translate_TSP(self, shift=(0.5,0.5)):
        assert self.dimension == 2
        x = self.problems[:,:,0:1]
        y = self.problems[:,:,1::]
        self.problems = np.concatenate((x - shift[0], y - shift[1]), -1)


    def fit_TSP_into_square(self):
        for j, problem in enumerate(self.problems):
            maxima = []
            minima = []
            for k in range(self.dimension):
                maxima.append(np.max(problem[:,k]))
                minima.append(np.min(problem[:,k]))

            differences = [maxima[k] - minima[k] for k in range(self.dimension)]

            scaler = 1 / np.max(differences)

            for k in range(self.dimension):
                self.problems[j,:,k] = scaler * (problem[:,k] - minima[k])

    def center_TSP(self, refit=False):
        if refit:
            self.fit_TSP_into_square()
        self.problems = self.problems - 0.5

    def apply_random_transfo(self):
        flip_axis = [0, 1, 2, 3, 4]
        variants = [0, 1, 2, 3, 4, 5, 6, 7, 8]
        degrees = np.arange(0, 360, 45)
        pcas = [-1, 0]
        
        # make random choices
        axis = random.choice(flip_axis)
        degree = random.choice(degrees)
        variant = random.choice(variants)
        pca = random.choice(pcas)
        
        self.pomo_TSP(variant=variant)
        if pca == -1:
            self.apply_PCA_to_TSP()
        self.flip_TSP(axis)
        self.rotate_TSP(degree, refit=True)

    

def rotate_3D(vectors, degree):
    assert vectors.shape[-1] == 3
    radians = (degree / 360) * 2 * math.pi
    rotmatrix = np.array([[1, 0, 0],
                          [0, math.cos(radians), -math.sin(radians)],
                          [0, math.sin(radians), math.cos(radians)]])

    return vectors @ rotmatrix

def rotate_2D(vectors, degree):
    assert vectors.shape[-1] == 2
    radians = (degree / 360) * 2 * math.pi
    rotmatrix = np.array([[math.cos(radians),-math.sin(radians)],
                          [math.sin(radians),math.cos(radians)]])
    return vectors @ rotmatrix
    