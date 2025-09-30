"""
Evolve the same representation as in image_learn.py, but using evolutionary strategies.
Uses the python package 'deap' for the evolutionary algorithm.
"""
import os
import logging
import matplotlib.pyplot as plt
import numpy as np
import multiprocessing
from deap import base, creator, tools
import pickle
import random
from util import make_input_grid
from copy import deepcopy
import cv2


class ScaleFreeImage(object):
    """
    Vectorization of an image, decompose it into regions of constant color.
    Evaluate color at any (x,y) coordinate.  Extrapolate, interpolate, etc.

    The plane is partitioned by circles and lines (the "dividers"), each partition a constant color.
    Partition colors are computed to minimize MSE to the target image.

    """
    _DEFAULT_MUTATION_PARAMS = {
        'single-mutations':{
            'line': {'p_mutate': 0.1, 'angle_std': 0.1, 'center_std': 0.05},
            'circle': {'p_mutate': 0.1, 'radius_std': 0.05, 'center_std': 0.05},},
        'mating': {}  # Not implemented yet
    }
    
    def __init__(self, target_image, n_div, dividers=None, mutation_params=None):
        """
        
        
        target_image: train to reproduce this image
        n_div: dict with number of units for each type, e.g. {'linear': 10, 'circular': 5} in layer 1
        dividers: optional dict of divider parameters, if None then random initialization, provided, target image can be None
        
        aspect ratio of image (width/height), inputs scaled so larger dimension is 
            in [-1,1], smaller in [-a,a] where a = min(aspect, 1/aspect)
            NOTE:  This is NOT a trainable parameter, since training is always on one image.
            This is to preserve giving equal weight to pixels x and y extent.
        

        
        mutation_params: dict of mutation parameters, see _DEFAULT_MUTATION_PARAMS
        """
        self.mutation_params = ScaleFreeImage._DEFAULT_MUTATION_PARAMS.copy()
        if mutation_params is not None:
            self.mutation_params.update(mutation_params)
            
        self.n_div = n_div
        if dividers is None:
            self.dividers = self._init_random_dividers(target_image)
        else:
            if n_div['circular']!=len(dividers['circles']['centers']):
                raise ValueError("n_div['circular'] does not match provided dividers")
            if n_div['linear']!=len(dividers['lines']['angles']):
                raise ValueError("n_div['linear'] does not match provided dividers")
            self.dividers = dividers
            
        self.n_dividers = self.n_div['circular'] + self.n_div['linear']
        
          
        
    def _init_random_dividers(self, target_image):

        n_circles = self.n_div['circular']
        n_lines = self.n_div['linear']
        
        
        aspect = target_image.shape[1] / target_image.shape[0]
        if aspect>0:
            xlim=np.array([-1.0, 1.0])
            
            ylim=np.array([-1.0 / aspect, 1.0 / aspect])
        else:
            xlim=np.array([-aspect, aspect])
            ylim=np.array([-1.0, 1.0])
            
        theta_lim = np.array([-np.pi, np.pi])
        log_r_lim = np.log(np.array([0.1, 2.0]))

        
        circles = {'centers': np.array([np.random.uniform(xlim[0], xlim[1], n_circles),
                                                             np.random.uniform(ylim[0], ylim[1], n_circles)]).T,
                                        'log-radii':np.random.uniform(log_r_lim[0], log_r_lim[1], n_circles)} if n_circles>0 else None
        lines = {'angles': np.random.uniform(theta_lim[0], theta_lim[1], n_lines),
                                      'centers': np.array([np.random.uniform(xlim[0], xlim[1], n_lines),
                                                           np.random.uniform(ylim[0], ylim[1], n_lines)]).T} if n_lines>0 else None
        color_lut = self._compute_color_lut(circles, lines, target_image)
        
        dividers = {'circles': circles, 
                    'lines': lines,
                    'color_lut': color_lut, 
                    'aspect': aspect, 
                    'xlim': xlim, 'ylim': ylim, }
        return dividers
    

        
        
    def __deepcopy__(self, memo):
        return type(self)(
            input_aspect=self.input_aspect,
            n_div=self.n_div,
            dividers=deepcopy(self.dividers),  # deep copy
        )

        
    
    def _encode(self, x, y):
        """
        Algorithm:
           maintain a bitmask for each code, which 
        
        
        
        """
    
        
    def _eval(self, x, y):
        """
        Evaluate the network for a given input.
        """
        shape_in = x.shape
        x = x.reshape(-1)
        y = y.reshape(-1)
        codes = self._encode(x,y)
        output = self._paint_by_code(codes)
        return output.reshape((*shape_in, 3))

    def render(self, shape_wh, keep_aspect=True):
        """
        Render the image at the specified width and height.
        """
        x,y = make_input_grid(shape_wh, resolution=1.0, keep_aspect=keep_aspect)
        
        img = self._eval(x, y)
        img = np.clip(img*255.0, 0, 255).astype(np.uint8)
        return img
    
    def evaluate_sse(self, target_image_normalized):
        """
        Evaluate the sum of squared errors (SSE) between the network output and target data.
        :param target_image_normalized: target image data, shape (n_points, 3), values in [0, 1]
        """
        x, y= make_input_grid(target_image_normalized.shape[:2], resolution=1.0, keep_aspect=True)
        output_image_normalized = self._eval(x, y)
        sse = np.sum((output_image_normalized - target_image_normalized) ** 2)
        return sse

    def mutate(self):
        """
        Mutate the network weights in place.
        """
        # Mutate circle parameters
        if self.n_div['circular']>0:
            for i in range(self.n_div['circular']):
                if np.random.rand() < 0.5:
                    # Mutate center
                    self.weights['circle_centers'][i, 0] += np.random.normal(0, self.mutation_params['single-mutations']['circle']['center_std'])
                    self.weights['circle_centers'][i, 1] += np.random.normal(0, self.mutation_params['single-mutations']['circle']['center_std'])
                    self.weights['circle_centers'][i, 0] = np.clip(self.weights['circle_centers'][i, 0], self.xlim[0], self.xlim[1])
                    self.weights['circle_centers'][i, 1] = np.clip(self.weights['circle_centers'][i, 1], self.ylim[0], self.ylim[1])
                else:
                    # Mutate radius
                    self.weights['circle_log_radii'][i] += np.random.normal(0, self.mutation_params['single-mutations']['circle']['radius_std'])
                    self.weights['circle_log_radii'][i] = np.clip(self.weights['circle_log_radii'][i], self.log_r_lim[0], self.log_r_lim[1])
        
        # Mutate line parameters
        if self.n_div['linear']>0:
            for i in range(self.n_div['linear']):
                if np.random.rand() < 0.5:
                    # Mutate center
                    self.weights['line_centers'][i, 0] += np.random.normal(0, self.mutation_params['single-mutations']['line']['center_std'])
                    self.weights['line_centers'][i, 1] += np.random.normal(0, self.mutation_params['single-mutations']['line']['center_std'])
                    self.weights['line_centers'][i, 0] = np.clip(self.weights['line_centers'][i, 0], self.xlim[0], self.xlim[1])
                    self.weights['line_centers'][i, 1] = np.clip(self.weights['line_centers'][i, 1], self.ylim[0], self.ylim[1])
                else:
                    # Mutate angle
                    self.weights['line_angles'][i] += np.random.normal(0, self.mutation_params['single-mutations']['line']['angle_std'])
                    self.weights['line_angles'][i] = np.unwrap(self.weights['line_angles'][i], self.theta_lim[0], self.theta_lim[1])
                    self.weights['line_lengths'][i] += np.random.normal(0, self.mutation_params['single-mutations']['line']['length_std'])
                    self.weights['line_lengths'][i] = np.unwrap(self.weights['line_lengths'][i], self.length_lim[0], self.length_lim[1])

    @classmethod
    def random_init(cls, input_aspect, n_div, n_struct, n_color):
        """Create a new DivNet with random weights."""
        return cls(img_aspect=input_aspect, n_div=n_div, n_struct=n_struct, n_color=n_color)


class EvolvingImage(object):
    def __init__(self, image, seed_pop, project_dir, evolution_params, resume=False):
        self.image = image
        self.evolution_params = evolution_params
        self.project_dir = project_dir
        self.seed_pop = seed_pop
        self.resume = resume
        self.aspect = image.shape[1] / image.shape[0]
        
        
        if seed_pop is not None and resume:
            raise ValueError("Cannot both resume and provide a seed population")

        self._img_shape = image.shape

        # Make sure project directory exists
        os.makedirs(self.project_dir, exist_ok=True)

        # Initialize DEAP evolution environment and population
        self._init_evolution()
        self._init_population()

    def _init_evolution(self):
        """Initialize DEAP creators, toolbox, and parallel pool."""

        # ---- Extract params ----
        self.pop_size = self.evolution_params.get("pop_size", 100)
        self.cx_prob  = self.evolution_params.get("cx_prob", 0.5)
        self.mut_prob = self.evolution_params.get("mut_prob", 0.2)
        self.ngen     = self.evolution_params.get("ngen", 100)
        self.n_div    = self.evolution_params.get("n_div", {})
        self.n_struct = self.evolution_params.get("n_struct", 50)
        self.n_color  = self.evolution_params.get("n_color", 20)

        # ---- Create Fitness/Individual types (guard against redeclaration) ----
        if not hasattr(creator, "FitnessMin"):
            creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        if not hasattr(creator, "Individual"):
            # Individual is just a DivNet with a fitness property
            creator.create("Individual", DivNet, fitness=creator.FitnessMin)

        # ---- Toolbox ----
        self.toolbox = base.Toolbox()

        # Initialization
        def init_divnet():
            return creator.Individual.random_init(
                input_aspect=self.aspect,
                n_div=self.n_div,
                n_struct=self.n_struct,
                n_color=self.n_color
            )
        self.toolbox.register("individual", init_divnet)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)

        # Evaluation
        def evaluate(individual):
            return (individual.evaluate_sse(self.image),)  # must return tuple
        self.toolbox.register("evaluate", evaluate)

        # Crossover, mutation, selection
        def mate(ind1, ind2):
            ind1.crossover(ind2)
            return ind1, ind2
        def mutate(ind):
            ind.mutate()
            return (ind,)
        self.toolbox.register("mate", mate)
        self.toolbox.register("mutate", mutate)
        self.toolbox.register("select", tools.selTournament, tournsize=3)

        # ---- Parallelization ----
        self.pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
        self.toolbox.register("map", self.pool.map)
        
    def _init_population(self):
        """Initialize population from seed_pop or fresh random individuals."""

        # Case 1: resume is True → load from disk (to implement separately)
        if getattr(self, "resume", False):
            # placeholder: you’d load a checkpointed population here
            raise NotImplementedError("Resume functionality not yet implemented")

        # Case 2: seeded population
        if self.seed_pop is not None:
            self.population = []
            for indiv in self.seed_pop:
                # If indiv is already a DivNet, wrap into Individual
                if not isinstance(indiv, creator.Individual):
                    indiv = creator.Individual(indiv)  # copy constructor
                self.population.append(indiv)

            # If fewer than pop_size, fill with random
            if len(self.population) < self.pop_size:
                n_missing = self.pop_size - len(self.population)
                self.population.extend(self.toolbox.population(n=n_missing))

        # Case 3: no seed → full random population
        else:
            self.population = self.toolbox.population(n=self.pop_size)

    def run(self):
        """Run the evolutionary optimization with manual loop and reporting."""

        # Ensure output dir exists
        os.makedirs(self.project_dir, exist_ok=True)

        # Evaluate initial population
        invalid = [ind for ind in self.population if not ind.fitness.valid]
        fitnesses = list(self.toolbox.map(self.toolbox.evaluate, invalid))
        for ind, fit in zip(invalid, fitnesses):
            ind.fitness.values = fit

        # Track stats
        log = []
        best_ind = tools.selBest(self.population, 1)[0]

        for gen in range(1, self.ngen + 1):
            # --- Selection ---
            offspring = self.toolbox.select(self.population, len(self.population))
            offspring = list(map(self.toolbox.clone, offspring))

            # --- Crossover ---
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if np.random.rand() < self.cx_prob:
                    self.toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

            # --- Mutation ---
            for mutant in offspring:
                if np.random.rand() < self.mut_prob:
                    self.toolbox.mutate(mutant)
                    del mutant.fitness.values

            # --- Evaluate new individuals ---
            invalid = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = list(self.toolbox.map(self.toolbox.evaluate, invalid))
            for ind, fit in zip(invalid, fitnesses):
                ind.fitness.values = fit

            # --- Replacement ---
            self.population[:] = offspring

            # --- Logging ---
            fits = [ind.fitness.values[0] for ind in self.population]
            mean_fit = float(np.mean(fits))
            best_fit = float(np.min(fits))
            worst_fit = float(np.max(fits))
            log.append((gen, mean_fit, best_fit, worst_fit))

            # Track best individual overall
            current_best = tools.selBest(self.population, 1)[0]
            if current_best.fitness.values[0] < best_ind.fitness.values[0]:
                best_ind = current_best

            # --- Reporting ---
            print(f"Gen {gen}: mean={mean_fit:.2f}, best={best_fit:.2f}, worst={worst_fit:.2f}")

            # Plot fitness curve
            plt.figure(figsize=(6,4))
            gens, means, bests, worsts = zip(*log)
            plt.plot(gens, means, label="mean")
            plt.plot(gens, bests, label="best")
            plt.plot(gens, worsts, label="worst")
            plt.xlabel("Generation")
            plt.ylabel("Fitness (SSE)")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(self.project_dir, f"fitness_gen{gen:04d}.png"))
            plt.close()

            # Save checkpoint
            ckpt = {
                "gen": gen,
                "population": self.population,
                "best_ind": best_ind,
                "log": log,
            }
            with open(os.path.join(self.project_dir, f"checkpoint_gen{gen:04d}.pkl"), "wb") as f:
                pickle.dump(ckpt, f)

        # Final result
        return best_ind, log
