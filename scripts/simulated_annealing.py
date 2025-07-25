import math
import random
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from utilities import City, read_cities, write_cities_and_return_them, generate_cities, visualize_tsp, path_cost


class SimAnneal(object):
    def __init__(self, cities, temperature=-1, alpha=-1, stopping_temperature=-1, stopping_iter=-1):
        self.cities = cities
        self.num_cities = len(cities)
        self.temperature = math.sqrt(self.num_cities) if temperature == -1 else temperature
        self.T_save = self.temperature
        self.alpha = 0.999 if alpha == -1 else alpha
        self.stopping_temperature = 1e-8 if stopping_temperature == -1 else stopping_temperature
        self.stopping_iter = 100000 if stopping_iter == -1 else stopping_iter
        self.iteration = 1
        self.route = None
        self.best_fitness = float("Inf")
        self.progress = []
        self.cur_cost = None

    def greedy_solution(self):
        start_node = random.randint(0, self.num_cities - 1)  # start from a random node
        unvisited = self.cities[:]
        del unvisited[start_node]
        route = [self.cities[start_node]]
        while len(unvisited):
            index, nearest_city = min(enumerate(unvisited), key=lambda item: item[1].distance(route[-1]))
            route.append(nearest_city)
            del unvisited[index]
        current_cost = path_cost(route)
        self.progress.append(current_cost)
        return route, current_cost

    def accept_probability(self, candidate_fitness):
        return math.exp(-abs(candidate_fitness - self.cur_cost) / self.temperature)

    def accept(self, guess):
        guess_cost = path_cost(guess)
        if guess_cost < self.cur_cost:
            self.cur_cost, self.route = guess_cost, guess
            if guess_cost < self.best_fitness:
                self.best_fitness, self.route = guess_cost, guess
        else:
            if random.random() < self.accept_probability(guess_cost):
                self.cur_cost, self.route = guess_cost, guess

    def run(self):
        self.route, self.cur_cost = self.greedy_solution()
        while self.temperature >= self.stopping_temperature and self.iteration < self.stopping_iter:
            guess = list(self.route)
            left_index = random.randint(2, self.num_cities - 1)
            right_index = random.randint(0, self.num_cities - left_index)
            guess[right_index: (right_index + left_index)] = reversed(guess[right_index: (right_index + left_index)])
            self.accept(guess)
            self.temperature *= self.alpha
            self.iteration += 1
            self.progress.append(self.cur_cost)

        print("Best fitness obtained: ", self.best_fitness)

    def visualize_routes(self):
        visualize_tsp('simulated annealing TSP', self.route)

    def plot_learning(self):
        fig = plt.figure(1)
        plt.plot([i for i in range(len(self.progress))], self.progress)
        plt.ylabel("Distance")
        plt.xlabel("Iterations")
        plt.show(block=False)


if __name__ == "__main__":
    cities = read_cities(64)
    sa = SimAnneal(cities, stopping_iter=15000)
    sa.run()

    # — build and save a two-pane animation of progress + route —
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Simulated Annealing TSP Progress')

    def animate(i):
        # left: learning curve up to iteration i
        ax1.clear()
        ax1.plot(range(i+1), sa.progress[:i+1], 'g-')
        ax1.set_xlim(0, len(sa.progress))
        ax1.set_ylim(min(sa.progress), max(sa.progress))
        ax1.set_title('Distance per Iteration')
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Distance')

        # right: best route (final) as static reference
        route = sa.route
        xs = [c.x for c in route] + [route[0].x]
        ys = [c.y for c in route] + [route[0].y]
        ax2.clear()
        ax2.plot(xs, ys, 'g-')
        ax2.plot(xs, ys, 'ro')
        ax2.set_title('Best TSP Route')
        return ax1, ax2

    anim = FuncAnimation(fig, animate, frames=len(sa.progress), blit=False)
    writer = FFMpegWriter(fps=10, metadata={'artist': 'Kshitij'}, bitrate=1800)
    anim.save(
        'sa_tsp_progress.mp4',
        writer=writer,
        progress_callback=lambda i, n: print(f"Rendering frame {i+1}/{n}", end='\r')
    )
    print("\nSaved animation to sa_tsp_progress.mp4")
    plt.close(fig)

    # optionally still output a static plot or visualization
    sa.visualize_routes()