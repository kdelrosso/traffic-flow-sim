from __future__ import division
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')

def increment_coord(coord, coord_max):
    """Return the incremented coordinate, accounting for the wrap around
    feature of BML traffic model.

    Parameters
    ----------
    coord: int, an x or y coordinates
    coord_max: int, the maximum value along x or y axis
    """

    if coord == coord_max:
        return 0
    else:
        return coord + 1

class Car(object):
    """Represents either a red or blue car in the Biham-Middleton-Levine (BML) traffic model.

    Parameters
    ----------
    car_id: int, unique car id
    color: string, either 'red' or 'blue'
    x_coord: int, x coordinate of the car's starting locations
    y_coord: int, y coordinate of the car's starting locations
    """

    def __init__(self, car_id, color, x_coord, y_coord):
        self.car_id = car_id
        self.color = color
        self.x = x_coord
        self.y = y_coord

        # record the number of times the car moves
        self.move_count = 0

    def __repr__(self):
        return "id: {0}, color: {1}, location: {2}".format(
            self.car_id,
            self.color,
            self.get_location()
        )

    def get_location(self):
        """Return the current (x, y) coordinates of the car."""

        return (self.x, self.y)

    def move_car(self, row_max, col_max):
        """Move red cars rightwards and blue cars downwards.

        Parameters
        ----------
        row_max: int, maximum rows of the grid
        col_max: int, maximum columns of the grid
        """

        if self.color == 'red':
            self.x = increment_coord(self.x, col_max)
        else:
            self.y = increment_coord(self.y, row_max)

        self.move_count += 1

class Grid(object):
    """Represents the rectangular lattice where cars move in
    the Biham-Middleton-Levine (BML) traffic model.

    In this model blue cars move at time periods t = 1, 3, 5, ... and
    the red cars move at time periods t = 2, 4, 6, ...

    Also, red cars move horizontally rightwards while blue cars move vertically downwards.

    Parameters
    ----------
    r / c: int, the grid size r x c
    n_red / n_blue: int, number of red and blue cars
    p: float in the interval (0, 1), probability of a red car
    n_cars: int, number of cars
    density: float in the interval (0, 1), density of cars in the grid

    Notes
    -----
    input either
    n_red & n_blue or
    p and n_cars or density
    """

    def __init__(self, r, c, n_red=None, n_blue=None, p=None, n_cars=None, density=None):
        self.n_red = n_red
        self.n_blue = n_blue

        # update if p and n_cars / density are specified
        if density:
            n_cars = int(r * c * density)

        if p:
           self.n_red = np.random.binomial(n_cars, p)
           self.n_blue = n_cars - self.n_red

        self.n_cars = self.n_red + self.n_blue

        # for storing the current location of each car
        self.car_loc = {}

        # for the grid
        self.r = r - 1
        self.c = c - 1
        self.coords = self._grid_coord(r, c)

        # controlling car movement by turn
        self.step = 0
        self.color_paused = 'blue'
        self.color_move = 'red'

        # for summary
        self.jammed_dict = {'red': False, 'blue': False}
        self.movement = {'red': 0, 'blue': 0}

        self._initialize_cars()

    def __repr__(self):
        return "red: {0}, blue: {1} \nrow: {2}, col: {3}".format(
            self.n_red,
            self.n_blue,
            self.r + 1,
            self.c + 1
        )

    def _grid_coord(self, row, col):
        """Return array containing all points in traffic model lattice."""

        return np.array( [(r, c) for r in xrange(row) for c in xrange(col)] )

    def is_jammed(self):
        """Return True is system is jammed, False otherwise."""

        return sum( self.jammed_dict.itervalues() ) == 2

    def _initialize_cars(self):
        """Run once to begin simulation, randomly selects a starting location for all cars."""

        # randomly select n_car locations
        keep_coords = np.random.choice(len(self.coords), self.n_cars, replace=False)
        car_coords = self.coords[keep_coords]

        # assign n_red and n_blue cars to these locations
        car_id = 1
        col = 'red'
        for coord in car_coords:
            # add a car at this location
            self.car_loc[tuple(coord)] = Car(car_id, col, coord[0], coord[1])

            # the remaining cars will be blue
            if car_id == self.n_red:
                col = 'blue'
            car_id += 1

    def _get_new_location(self, x, y, color):
        """Return new x and y coordinates incremented appropriately."""

        if self.step % 2 == 0:
            # red moves
            if color == 'red':
                x = increment_coord(x, self.c)
        else:
            # blue moves
            if color == 'blue':
                y = increment_coord(y, self.r)

        return (x, y)

    def _update_colors(self):
        """Update the color of the moving cars. Blue moves during odd steps, red during even steps."""

        if self.step % 2 == 0:
            # red moves
            self.color_move, self.color_paused = 'red', 'blue'
        else:
            # blue moves
            self.color_move, self.color_paused = 'blue', 'red'

    def _move_cars(self, cars_to_move):
        """Move cars in cars_to_move, updates car_loc dict with current location of cars.

        Parameters
        ----------
        cars_to_move: set, coordinates of cars to move
        """

        # updated the jammed dictionary whether or not the current color can move
        self.jammed_dict[self.color_move] = len(cars_to_move) == 0
        self.movement[self.color_move] = len(cars_to_move)

        # copy the location dictionary so info isn't overwritten
        loc_copy = self.car_loc.copy()

        # keep track of all new locations
        new_locations = set()
        for loc in cars_to_move:
            c = loc_copy[loc]

            # move the car, record the new location
            c.move_car(self.r, self.c)
            new_loc = c.get_location()

            # update dictionary with new location of car
            self.car_loc[new_loc] = c
            new_locations.add(new_loc)

        # only delete old locations which are not new locations
        for loc in cars_to_move.difference(new_locations):
            del self.car_loc[loc]

    def take_step(self):
        """Take a single step in the BML traffic model. Move cars and update their new locations.

        Suppose were are at an odd time step and blue moves. We'll check whether or not each
        car can move. If the car is red then it won't move. If it's blue there are three options
        to consider:
            - the blue car can move into an empty location (we can move)
            - the blue car tries to move into a location occupied by a red car (we can't move)
            - the blue car tried to move into a location occupied by a blue car (call this the
                blocking blue car)

        In the last option, we'll continually check whether the blocking blue car can move.
        """

        # update the information for this step
        self.step += 1
        self._update_colors()

        # record the car locations
        visited = set()
        to_move = []
        possible_moves = []
        for (x, y), v in self.car_loc.iteritems():

            # if we've also seen this car or it's a paused color, continue
            if ((x, y) in visited) or (v.color == self.color_paused):
                continue

            visited.add((x, y))
            possible_moves.append((x, y))

            # get the location where the car will move
            new_loc = self._get_new_location(x, y, v.color)

            # if this new location is empty, the car can move
            if new_loc not in self.car_loc:
                to_move.extend(possible_moves)
                possible_moves = []
            else:
                # there is a car in the new location, get the color of that car and
                # if the color is the paused color, cars can't move
                while new_loc in self.car_loc:
                    visited.add(new_loc)

                    # get the color of the blocking car
                    car_color = self.car_loc[new_loc].color

                    # we have a duplicate point, i.e. the entire row / column is the same color
                    dup = len(set(possible_moves)) < len(possible_moves)

                    # exit the loop, none of these cars can move
                    if (car_color == self.color_paused) or dup:
                        possible_moves = []
                        break

                    # blocking car is of the same color, add as possible move and update location
                    possible_moves.append(new_loc)
                    new_loc = self._get_new_location(new_loc[0], new_loc[1], car_color)

                # loop ends if we reach an open point, i.e. all points found can move
                else:
                    to_move.extend(possible_moves)
                    possible_moves = []

        # move all the cars
        self._move_cars(set(to_move))

    def run_simulation(self, n=100, directory='./images', plot_freq=0.1, update_freq=0.05):
        """Run n steps of the BML traffic simulation.

        Parameters
        ----------
        n: int, number of simulation steps
        plot_frequency: float, percentage of steps to save plots
        """

        get_num = lambda n, x: max(int( n * x ), 1)

        if plot_freq > 1:
            plot_num = plot_freq
        else:
            plot_num = get_num(n, plot_freq)

        # how often to print simulation step number
        update_num = get_num(n, update_freq)
        for i in xrange(n):
            if i % update_num == 0:
                print i
            if i % plot_num == 0:
                self.plot( directory = directory )

            self.take_step()

            if self.is_jammed():
                print "Traffic jam!"
                print i
                self.plot(directory = directory)
                break

        self.summary()

    def _fix_file_num(self, n, digits):
        """Return n padded with leading '0's so total length is digits."""

        n = str(n)
        mult = digits - len(n)
        return '0' * mult + n

    def plot(self, directory='./images', file_num=None, digits=5):
        """Plot the current location of cars in the simulation.

        Parameters
        ----------
        directory: string, directory to save simulation plots
        file_num: int, suffix to append to filename 'simulation_step_{file_num}'
        digits: int, how many digits to use while padding file_num with zeros
        """

        # create DataFrame with x, y, and color columns for all cars
        coord_col = [( x, y, v.color) for (x, y), v in self.car_loc.iteritems()]
        coord_col = pd.DataFrame(coord_col, columns=['x', 'y', 'color'])

        # create scatter plot by car color
        coord_col.plot(kind='scatter', x='x', y='y', c=coord_col['color'], figsize = (12, 8))

        # remove axis and extra white space
        plt.gca().set_axis_off()
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0.01, 0.01)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.title( 'Simulation Step: {0}'.format(self.step) )

        # save the plot
        if not file_num:
            file_num = self._fix_file_num(self.step, digits)

        # make directory if it doesn't already exist
        if not os.path.exists(directory):
            os.makedirs(directory)

        filename = '{0}/simulation_step_{1}.png'.format(directory, file_num)
        plt.savefig(filename)
        plt.close()

    def summary(self):
        """Print summary statistics from the simulation."""

        density = round((self.n_red + self.n_blue) / np.prod((self.r + 1, self.c + 1)), 2)
        total_cars = self.n_red + self.n_blue
        total_move = int( sum([c.move_count for c in self.car_loc.itervalues()]) )
        possible_moves = int(self.n_red * self.step / 2 + self.n_blue * self.step / 2)
        percent_move = round(total_move * 100 / possible_moves, 1)

        print '''
        Summary
        -------
        Grid size: {r} x {c}, (rows x cols)
        # red cars: {n_red}, # blue cars: {n_blue}, density: {density}
        Total steps taken: {n_steps}
        Cars moved {percent_move} percent of the time ({total_move} of {possible_moves})
        Car movement last round: {movement} (of {total_cars})
        The system is jammed: {is_jammed}
        '''.format(
            r=self.r + 1,
            c=self.c + 1,
            n_red=self.n_red,
            n_blue=self.n_blue,
            density=density,
            n_steps=self.step,
            percent_move=percent_move,
            total_move=total_move,
            possible_moves=possible_moves,
            movement=sum(self.movement.itervalues()),
            total_cars=total_cars,
            is_jammed=self.is_jammed()
        )

if __name__ == "__main__":

    g = Grid(150, 150, p=0.3, density=0.2)
    g.run_simulation(1000, './sim_images', plot_freq=8)
