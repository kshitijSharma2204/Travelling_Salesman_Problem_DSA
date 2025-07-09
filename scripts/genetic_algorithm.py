import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
import numpy as np
import pandas as pd
import random
import math, sys
import random
from utilities import City, read_cities, write_cities_and_return_them, generate_cities


class Fitness:
    def __init__(self, route):
        self.route = route
        self.distance = 0
        self.fitness = 0.0

    def path_cost(self):
        if self.distance == 0:
            distance = 0
            for index, city in enumerate(self.route):
                distance += city.distance(self.route[(index + 1) % len(self.route)])
            self.distance = distance
        return self.distance

    def path_fitness(self):
        if self.fitness == 0:
            self.fitness = 1 / float(self.path_cost())
        return self.fitness


class GeneticAlgorithm:
    def __init__(self, iterations, population_size, cities, elites_num, mutation_rate,
                 greedy_seed=0, roulette_selection=True, plot_progress=True):
        self.plot_progress = plot_progress
        self.roulette_selection = roulette_selection
        self.progress = []
        self.mutation_rate = mutation_rate
        self.cities = cities
        self.elites_num = elites_num
        self.iterations = iterations
        self.population_size = population_size
        self.greedy_seed = greedy_seed

        self.population = self.initial_population()
        self.average_path_cost = 1
        self.ranked_population = None

    def best_chromosome(self):
        return self.ranked_population[0][0]

    def best_distance(self):
        return 1 / self.ranked_population[0][1]

    def random_route(self):
        return random.sample(self.cities, len(self.cities))

    def initial_population(self):
        p1 = [self.random_route() for _ in range(self.population_size - self.greedy_seed)]
        greedy_population = [greedy_route(start_index % len(self.cities), self.cities)
                             for start_index in range(self.greedy_seed)]
        return [*p1, *greedy_population]

    def rank_population(self):
        fitness = [(chromosome, Fitness(chromosome).path_fitness()) for chromosome in self.population]
        self.ranked_population = sorted(fitness, key=lambda f: f[1], reverse=True)

    def selection(self):
        selections = [self.ranked_population[i][0] for i in range(self.elites_num)]
        if self.roulette_selection:
            df = pd.DataFrame(self.ranked_population, columns=["route","fitness"])
            self.average_path_cost = sum(1 / df.fitness) / len(df.fitness)
            df['cum_sum'] = df.fitness.cumsum()
            df['cum_perc'] = 100 * df.cum_sum / df.fitness.sum()

            for _ in range(0, self.population_size - self.elites_num):
                pick = 100 * random.random()
                for i in range(0, len(self.ranked_population)):
                    if pick <= df.iat[i, 3]:
                        selections.append(self.ranked_population[i][0])
                        break
        else:
            for _ in range(0, self.population_size - self.elites_num):
                pick = random.randint(0, self.population_size - 1)
                selections.append(self.ranked_population[pick][0])
        self.population = selections

    @staticmethod
    def produce_child(parent1, parent2):
        gene_1 = random.randint(0, len(parent1))
        gene_2 = random.randint(0, len(parent1))
        gene_1, gene_2 = min(gene_1, gene_2), max(gene_1, gene_2)
        child = [parent1[i] for i in range(gene_1, gene_2)]
        child.extend([gene for gene in parent2 if gene not in child])
        return child

    def generate_population(self):
        length = len(self.population) - self.elites_num
        children = self.population[:self.elites_num]
        for i in range(0, length):
            child = self.produce_child(self.population[i],
                                       self.population[(i + random.randint(1, self.elites_num)) % length])
            children.append(child)
        return children

    def mutate(self, individual):
        for index, city in enumerate(individual):
            if random.random() < max(0, self.mutation_rate):
                sample_size = min(min(max(3, self.population_size // 5), 100), len(individual))
                random_sample = random.sample(range(len(individual)), sample_size)
                sorted_sample = sorted(random_sample,
                                       key=lambda c_i: individual[c_i].distance(individual[index - 1]))
                random_close_index = random.choice(sorted_sample[:max(sample_size // 3, 2)])
                individual[index], individual[random_close_index] = \
                    individual[random_close_index], individual[index]
        return individual

    def next_generation(self):
        self.rank_population()
        self.selection()
        self.population = self.generate_population()
        self.population[self.elites_num:] = [self.mutate(chromosome)
                                             for chromosome in self.population[self.elites_num:]]

    def run(self):
        if self.plot_progress:
            plt.ion()
        for ind in range(0, self.iterations):
            self.next_generation()
            self.progress.append(self.best_distance())
            if self.plot_progress and ind % 10 == 0:
                self.plot()
            elif not self.plot_progress and ind % 10 == 0:
                print(self.best_distance())

    def plot(self):
        # prepare best-distance and route coords
        best = self.best_distance()
        x_list = [c.x for c in self.best_chromosome()] + [self.best_chromosome()[0].x]
        y_list = [c.y for c in self.best_chromosome()] + [self.best_chromosome()[0].y]

        # on first call, create a single figure with 2 subplots
        if not hasattr(self, 'fig'):
            self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(12, 5))
            self.fig.suptitle('Genetic Algorithm Progress')

        # left subplot: distance over generations
        self.ax1.clear()
        self.ax1.plot(self.progress, 'g-')
        self.ax1.set_title('Distance per Generation')
        self.ax1.set_xlabel('Generation')
        self.ax1.set_ylabel('Distance')

        # right subplot: current best TSP route
        self.ax2.clear()
        self.ax2.plot(x_list, y_list, 'g-')
        self.ax2.plot(x_list, y_list, 'ro')
        self.ax2.set_title('TSP Route')

        # draw and pause for animation
        if self.plot_progress:
            self.fig.canvas.draw()
            plt.pause(0.05)


def greedy_route(start_index, cities):
    unvisited = cities[:]
    del unvisited[start_index]
    route = [cities[start_index]]
    while len(unvisited):
        index, nearest_city = min(enumerate(unvisited), key=lambda item: item[1].distance(route[-1]))
        route.append(nearest_city)
        del unvisited[index]
    return route


if __name__ == '__main__':
    # Prepare GA without live plotting
    cities = read_cities(64)
    ga = GeneticAlgorithm(
        cities=cities,
        iterations=1200,
        population_size=100,
        elites_num=20,
        mutation_rate=0.008,
        greedy_seed=1,
        roulette_selection=True,
        plot_progress=False
    )

    # Set up figure and axes
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Genetic Algorithm Progress')

    # Animation function: advance GA one generation, then redraw both subplots
    def animate(frame):
        ga.next_generation()
        ga.progress.append(ga.best_distance())

        # Left: distance vs. generation
        ax1.clear()
        ax1.plot(ga.progress, 'g-')
        ax1.set_title('Distance per Generation')
        ax1.set_xlabel('Generation')
        ax1.set_ylabel('Distance')

        # Right: best TSP route
        route = ga.best_chromosome()
        xs = [c.x for c in route] + [route[0].x]
        ys = [c.y for c in route] + [route[0].y]
        ax2.clear()
        ax2.plot(xs, ys, 'g-')
        ax2.plot(xs, ys, 'ro')
        ax2.set_title('TSP Route')

        return ax1, ax2

    # Build and save animation using FFMpeg
    anim = FuncAnimation(fig, animate, frames=ga.iterations, blit=False)
    writer = FFMpegWriter(fps=10, metadata={'artist': 'Kshitij'}, bitrate=1800)
    # Save with a simple progress indicator, then close the figure
    anim.save(
        'ga_tsp_evolution.mp4',
        writer=writer,
        progress_callback=lambda i, n: print(f"Rendering frame {i+1}/{n}", end='\r')
    )
    print("\nSaved animation to ga_tsp_evolution.mp4")
    plt.close(fig)