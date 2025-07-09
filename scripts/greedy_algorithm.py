import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
from utilities import City, read_cities, path_cost


class Greedy:
    def __init__(self, cities):
        # Initialize unvisited cities and starting route
        self.unvisited = cities[1:]
        self.route = [cities[0]]
        self._frames = []  # store segments for animation

    def run(self):
        # Build the route and record each segment
        while self.unvisited:
            index, nearest_city = min(
                enumerate(self.unvisited),
                key=lambda item: item[1].distance(self.route[-1])
            )
            prev_city = self.route[-1]
            self.route.append(nearest_city)
            del self.unvisited[index]
            # record this edge for animation
            self._frames.append(((prev_city.x, prev_city.y), (nearest_city.x, nearest_city.y)))

        # Close the loop back to the start
        self.route.append(self.route[0])
        last_city = self.route[-2]
        first_city = self.route[-1]
        self._frames.append(((last_city.x, last_city.y), (first_city.x, first_city.y)))

        # Return the total path cost
        return path_cost(self.route)


if __name__ == "__main__":
    # Read city data and compute greedy TSP
    cities = read_cities(64)
    greedy = Greedy(cities)
    total_cost = greedy.run()
    print(f"Total cost: {total_cost}")

    # --- Setup plotting ---
    fig, ax = plt.subplots()
    ax.set_title("Greedy TSP Animation")
    # Plot all city points
    xs = [c.x for c in cities]
    ys = [c.y for c in cities]
    ax.scatter(xs, ys, c='red')

    # Line object for building the path
    line, = ax.plot([], [], 'g-')

    def update(frame):
        (x1, y1), (x2, y2) = frame
        prev_xs, prev_ys = line.get_data()
        # Append new segment to existing path
        new_xs = list(prev_xs) + [x1, x2]
        new_ys = list(prev_ys) + [y1, y2]
        line.set_data(new_xs, new_ys)
        return line,

    # Create animation object
    anim = FuncAnimation(
        fig,
        update,
        frames=greedy._frames,
        blit=True,
        interval=200
    )

    # Save as MP4 (requires ffmpeg)
    writer_mp4 = FFMpegWriter(fps=5)
    anim.save("greedy_tsp.mp4", writer=writer_mp4)

    plt.pause(3)
    plt.show(block=False)