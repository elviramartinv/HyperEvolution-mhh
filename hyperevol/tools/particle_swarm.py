''' Particle swarm optimization algorithm '''
import os
import json
import glob
import numpy as np
import chaospy as cp


class Particle():
    ''' The class representing one particle in the swarm '''
    def __init__(
            self,
            hyperparameter_info,
            hyperparameters,
            iterations,
            n_informants,
            c_max=1.62,
            w_init=0.8,
            w_fin=0.4,
            **kwargs
    ):
        self.c_max = c_max
        self.w = w_init
        self.n_informants = n_informants
        self.w_step = (w_init - w_fin) / iterations
        self.hyperparameter_info = hyperparameter_info
        self.hyperparameters = hyperparameters
        self.initialize_speeds()
        self.total_iterations = iterations
        self.iteration = 0

    def initialize_speeds(self):
        ''' The speeds are initialized to be up to 25% of the whole range of
        the hyperparameter space in a given direction'''
        self.speed = {}
        for key in self.hyperparameter_info.keys():
            v_max = (self.hyperparameter_info[key]['max'] - \
                     self.hyperparameter_info[key]['min']) / 4
            self.speed[key] = np.random.uniform() * v_max

    def set_fitness(self, fitness):
        ''' Sets the fitness of the particle. Also updates the personal and/or
        global best fitness if the current fitness is better. '''
        self.fitness = fitness
        if self.fitness < self.personal_best_fitness:
            self.set_personal_best()
        if self.fitness < self.global_best_fitness:
            self.set_global_best(self.hyperparameters, self.fitness)

    def set_personal_best(self):
        ''' Sets the personal best '''
        self.personal_best = self.hyperparameters.copy()
        self.personal_best_fitness = self.fitness

    def set_global_best(self, hyperparameters, fitness):
        ''' Sets the global best'''
        self.global_best = hyperparameters.copy()
        self.global_best_fitness = fitness

    def set_initial_bests(self, fitness):
        ''' Before any espionage step, sets the current fitness and location
        to be both the personal and global test '''
        self.fitness = fitness
        self.set_personal_best()
        self.set_global_best(self.hyperparameters, self.fitness)

    def update_speeds(self):
        ''' Upadetes the speed, taking into account the influences from all the
        different components like cognitive, social and inertial component '''
        for key in self.hyperparameter_info.keys():
            rand1 = np.random.uniform()
            rand2 = np.random.uniform()
            cognitive_component = self.c_max * rand1 * (
                self.personal_best[key] - self.hyperparameters[key])
            social_component = self.c_max * rand2 * (
                self.global_best[key] - self.hyperparameters[key])
            inertial_component = self.w * self.speed[key]
            self.speed[key] = sum([
                    cognitive_component, social_component, inertial_component])

    def update_location(self):
        ''' Updates the location of the particle. In case the next location
        is not within bounds of the hyperparameter space, the location will be
        set to be the value of the bound and the speed will be set to 0 in that
        given direction'''
        for key in self.hyperparameter_info.keys():
            self.hyperparameters[key] += self.speed[key]
            if self.hyperparameter_info[key]['exp'] == 1:
                max_value = np.exp(self.hyperparameter_info[key]['max'])
                min_value = np.exp(self.hyperparameter_info[key]['min'])
            else:
                max_value = self.hyperparameter_info[key]['max']
                min_value = self.hyperparameter_info[key]['min']
            if self.hyperparameters[key] > max_value:
                self.hyperparameters[key] = max_value
                self.speed[key] = 0
            if self.hyperparameters[key] < min_value:
                self.hyperparameters[key] = min_value
                self.speed[key] = 0
            if self.hyperparameter_info[key]['int'] == 1:
                self.hyperparameters[key] = int(np.ceil(self.hyperparameters[key]))

    def gather_intelligence(self, swarm):
        ''' Queries a subset of particles in the whole swarm for their personal
        best location they have visited. Having this information the particle
        infers the global best location it has seen so far.'''
        informants = np.random.choice(swarm, self.n_informants)
        idx = np.argmin(informant.personal_best_fitness for informant in informants)
        if informants[idx].personal_best_fitness < self.global_best_fitness:
            self.set_global_best(
                                  informants[idx].personal_best,
                                  informants[idx].personal_best_fitness)

    def next_iteration(self, swarm):
        ''' Particle undergoes the evolutionary loop, consisting our of 3 steps:
        espionage, location update and speed update '''
        self.gather_intelligence(swarm)
        self.update_location()
        self.update_speeds()
        self.w -= self.w_step


class ParticleSwarm:
    ''' Class representing the whole swarm, that consists out of multiple
    particles '''
    def __init__(
            self,
            objective_function,
            hyperparameter_info,
            settings={},
            population_size=50,
            n_informants=5,
            iterations=50,
            seed=42,
            output_dir='',
            **kwargs
    ):
        self.settings = settings
        self.population_size = settings['population_size']
        self.seed = settings['seed']
        self.n_informants = settings['nr_informants']
        self.iterations = settings['iterations']
        self.output_dir = settings['output_dir']
        self.objective_function = objective_function
        self.hyperparameter_info = hyperparameter_info
        self.global_bests = []
        self.global_best = 99e99
        self.swarm = self.create_swarm()

    def create_swarm(self):
        ''' Initializes the swarm by making a list of locations where particles
        will be spawned. This kind of solution is needed in order to fill the
        space more evenly using for example the latin hypercube distributing
        method '''
        particle_swarm = []
        locations = self.create_initial_locations()
        for location in locations:
            single_particle = Particle(
                self.hyperparameter_info,
                location,
                self.iterations,
                self.n_informants
            )
            particle_swarm.append(single_particle)
        return particle_swarm

    def create_initial_locations(self):
        ''' Creates the locations where the particles will spawn in the
        beginning of the algorithm. This kind of solution is needed in order to
        fill the space more evenly using for example the latin hypercube
        distributing method '''
        list_of_distributions = []
        for hyperparameter in self.hyperparameter_info.values():
            list_of_distributions.append(
                cp.Uniform(hyperparameter['min'], hyperparameter['max'])
            )
        distribution = cp.J(*list_of_distributions)
        samples = distribution.sample(self.population_size, rule='latin_hypercube')
        sample_points = np.transpose(samples)
        locations = []
        for sample_point in sample_points:
            location = {}
            for coord, name in zip(sample_point, self.hyperparameter_info.keys()):
                if self.hyperparameter_info[name]['int']:
                    value = np.round(coord).astype(int)
                elif self.hyperparameter_info[name]['exp']:
                    value = np.exp(coord)
                else:
                    value = coord
                location[name] = value
            locations.append(location)
        return locations

    def get_fitnesses_and_location(self, group):
        best_locations = []
        best_fitnesses = []
        for particle in group:
            best_fitnesses.append(particle.personal_best_fitness)
            best_locations.append(particle.personal_best)
        return best_fitnesses, best_locations

    def set_particle_fitnesses(self, fitnesses, initial=False):
        for particle, fitness in zip(self.swarm, fitnesses):
            if initial:
                particle.set_initial_bests(fitness)
            else:
                particle.set_fitness(fitness)

    def find_best_hyperparameters(self):
        best_fitnesses, best_locations = self.get_fitnesses_and_location(
            self.swarm)
        index = np.argmin(best_fitnesses)
        best_fitness = best_fitnesses[index]
        best_location = best_locations[index]
        return best_fitness, best_location

    def check_global_best(self):
        ''' Checks whether a new global best location for all the particles
        has been found '''
        for particle in self.swarm:
            if particle.fitness < self.global_best:
                self.global_best = particle.fitness
        self.global_bests.append(self.global_best)

    def optimize(self):
        ''' The main function to call when swarm is to be start optimizing'''
        iteration = 0
        np.random.seed(self.seed)
        all_locations = [particle.hyperparameters for particle in self.swarm]
        fitnesses = self.objective_function(all_locations, self.settings)
        self.set_particle_fitnesses(fitnesses, initial=True)
        self.check_global_best()
        for particle in self.swarm:
            particle.next_iteration(self.swarm)
        iteration = 1
        while iteration < self.iterations:
            print('%s/%s' %(iteration, self.iterations))
            iteration += 1
            all_locations = [particle.hyperparameters for particle in self.swarm]
            fitnesses = self.objective_function(all_locations, self.settings)
            self.set_particle_fitnesses(fitnesses)
            self.check_global_best()
            for particle in self.swarm:
                particle.next_iteration(self.swarm)
        best_fitness, best_location = self.find_best_hyperparameters()
        return best_location, best_fitness
