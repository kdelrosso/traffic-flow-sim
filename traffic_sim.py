from __future__ import division
import time
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('whitegrid')

def grid_coord(row, col):
    '''
    Creates an array containing all points in traffic model lattice
    '''
    return np.array( [ (r, c) for r in xrange(row) for c in xrange(col) ] )

def increment_coord( coord, coord_max ):
    '''
    INPUT:
        - coord: an x or y coordinates
        - coord_max: the maximum value along x or y axis 
    OUTPUT: the incremented coordinate
    DOC: accounts for wrap around feature of BML traffic model
    '''

    if coord == coord_max:
        return 0
    else:
        return coord + 1

class Car(object):
    '''
    This class represents either a red or blue car in the Biham-Middleton-Levine (BML) traffic model
    '''

    def __init__(self, car_id, color, x_coord, y_coord):
        '''
        Initialize car with an id, color (either red or blue) and a starting location (x, y)
        '''

        self.car_id = car_id
        self.color = color
        self.x = x_coord
        self.y = y_coord

        # record the number of times the car moves
        self.move_count = 0

    def __repr__(self):
        return "id: {0}, color: {1}, location: {2}".format( self.car_id, 
            self.color, self.get_location() )

    def get_location(self):
        return (self.x, self.y)

    def get_color(self):
        return self.color

    def get_movement(self):
        return self.move_count

    def move_car(self, r_max, c_max):
        '''
        INPUT: r_max / c_max are maximum rows and columns of the grid,
        used for wrapping cars around
        
        DOC: 
            - red cars move horizontally rightwards
            - blue cars move vertically downwards
        '''

        if self.color == 'red':
            self.x = increment_coord( self.x, c_max )
        else:
            self.y = increment_coord( self.y, r_max )

        self.move_count += 1

class Grid(object):
    '''
    This class represents the rectangular lattice where cars move in 
    the Biham-Middleton-Levine (BML) traffic model.

    In this model blue cars move at time periods t = 1, 3, 5, ... and 
    the red cars move at time periods t = 2, 4, 6, ...

    Also, red cars move horizontally rightwards while blue cars move vertically downwards.
    '''

    def __init__(self, r, c, n_red = None, n_blue = None, p = None, n_cars = None, density = None):
        '''
        INPUT:
            - r and c: the grid size r x c
            - input n_red / n_blue: number of red and blue cars
            - or input p / n_cars / density: where p is the probability of a red car, 
                n_cars is the number of cars, and density is the density of cars in the grid

        DOC: either input n_red & n_blue, or p & n_cars / density
        '''

        # for the number of each colored car
        self.n_red = n_red
        self.n_blue = n_blue

        # update if p, n_cars / density are specified
        if density:
            n_cars = int( r * c * density )
        
        if p:
           self.n_red = np.random.binomial( n_cars, p ) 
           self.n_blue = n_cars - self.n_red

        self.n_cars = self.n_red + self.n_blue

        # for storing the current location of each car
        self.car_loc = {}
        
        # for the grid
        self.r = r - 1
        self.c = c - 1
        self.coords = grid_coord(r, c)

        # controlling car movement by turn
        self.step = 0
        self.color_paused = 'blue'
        self.color_move = 'red'

        # for summary
        self.jammed_dict = {'red': False, 'blue': False}
        self.movement = {'red': 0, 'blue': 0}

    def __repr__(self):
        return "red: {0}, blue: {1} \nrow: {2}, col: {3}".format( self.n_red, 
            self.n_blue, self.r, self.c )

    def get_all_cars(self):
        return self.car_loc

    def is_jammed(self):
        '''return True is system is jammed, False otherwise'''
        return sum( self.jammed_dict.itervalues() ) == 2

    def initialize_cars(self):
        '''
        Run one to begin simulation, we randomly select starting locations for all the cars
        '''

        # first randomly select n_car locations
        keep_coords = np.random.choice( len( self.coords ), self.n_cars, replace = False )
        car_coords = self.coords[ keep_coords ]

        # then assign n_red and n_blue cars to these locations
        id_count = 0
        col = 'red'
        for coord in car_coords:
            # adding a car at this location
            self.car_loc[ tuple(coord) ] = Car( id_count, col, coord[0], coord[1] )

            # the remaining cars will be blue
            if id_count == self.n_red - 1:
                col = 'blue'
            id_count += 1

    def _get_new_location(self, new_x, new_y, color):
        '''
        INPUT: previous x, y coordinates and color of the car
        OUTPUT: new coordinates incremented appropriately
        DOC: internal method; blue moves during odd steps, red during even steps
        '''
        if self.step % 2 == 0:
            # red moves
            if color == 'red':
                new_x = increment_coord( new_x, self.c )
        else:
            # blue moves
            if color == 'blue':
                new_y = increment_coord( new_y, self.r )

        return (new_x, new_y)

    def _update_colors(self):
        '''
        DOC: 
            - called once during take_step() to update which cars are moving / paused during
                this step
            - internal method; blue moves during odd steps, red during even steps
        '''

        if self.step % 2 == 0:
            # red moves
            self.color_move = 'red'
            self.color_paused = 'blue'
        else:
            # blue moves
            self.color_move = 'blue'
            self.color_paused = 'red'

    def _move_cars(self, cars_to_move):
        '''
        INPUT: set of coordinates of cars to move
        OUTPUT: None
        DOC:
            - move cars in cars_to_move
            - update self.car_loc dict with current location of cars
            - internal method call by take_step()
        '''

        # updated the jammed dictionary whether or not the current color can move
        self.jammed_dict[ self.color_move ] = len( cars_to_move ) == 0
        self.movement[ self.color_move ] = len( cars_to_move )

        # copy the location dictionary so info isn't overwritten
        loc_copy = self.car_loc.copy()

        # keep track of all new locations
        new_locations = set()
        for loc in cars_to_move:
            c = loc_copy[ loc ] 

            # move the car, record the new location
            c.move_car( self.r, self.c )
            new_loc = c.get_location()
            # new_loc = self._get_new_location( loc[0], loc[1], self.color_move )

            # update dictionary with new location of car
            self.car_loc[ new_loc ] = c
            new_locations.add( new_loc )

        # only delete old locations which are not also new locations
        for loc in cars_to_move.difference( new_locations ):
            del self.car_loc[ loc ]

    def take_step(self):
        '''
        Take a single step in the BML traffic model. Move cars and update their new locations.

        DOC: Suppose were are at an odd time step and blue moves. We'll check whether or not each 
        car can move. If the car is the red when the car can't move. Then if the car is blue we 
        have three possibly outcomes to consider.
            - the blue car can move into an empty location (we can move)
            - the blue car tries to move into a location occupied by a red car (we can't move)
            - the blue car tried to more into a location occupied by a blue car (call this the 
                blocking blue car)

        In this last situation, we'll continually check whether the blocking blue car can move as 
        done above. We'll then have a list of possible car which can move.
        '''
        # update the information for this step
        self.step += 1
        self._update_colors()
        
        # record the car locations
        visited = set()
        to_move = []
        possible_moves = []

        for (x, y), v in self.car_loc.iteritems():
            
            # if we've also seen this car or it's a paused color, continue
            if ( (x, y) in visited ) or ( v.get_color() == self.color_paused ):
                continue
            
            visited.add( (x, y) )
            possible_moves.append( (x, y) )

            # get the location where the car will move
            new_loc = self._get_new_location( x, y, v.get_color() )

            # if this new location is empty, the car can move
            if new_loc not in self.car_loc:
                to_move.extend( possible_moves )
                possible_moves = []
            else:
                # there is a car in the new location, get the color of that car and
                # if the color is the paused color, cars can't move
                while new_loc in self.car_loc:
                    visited.add( new_loc )

                    # get the color of the blocking car
                    car_color = self.car_loc[ new_loc ].get_color()

                    # we have a duplicate point, i.e. the entire row / column is the same color
                    dup = len( set(possible_moves) ) < len(possible_moves)

                    # exit the loop, none of these cars can move
                    if (car_color == self.color_paused) or dup:
                        possible_moves = []
                        break

                    # blocking car is of the same color, add as possible move and update location
                    possible_moves.append( new_loc )
                    new_loc = self._get_new_location( new_loc[0], new_loc[1], car_color )

                # loop ends if we reach an open point, i.e. all points found can move    
                else:
                    to_move.extend( possible_moves )
                    possible_moves = []

        # move all the cars
        self._move_cars( set(to_move) )

    def run_simulation(self, n = 100, directory = './images', plot_freq = 0.1, update_freq = 0.05):
        '''
        INPUT: 
            - n: number of simulation steps
            - plot_frequency: percentage of steps to save plot

        OUTPUT: run time of the simulation in minutes
        '''

        self.initialize_cars()
        
        get_num = lambda n, x: max( int( n * x ), 1 )

        if plot_freq > 1:
            plot_num = plot_freq
        else:
            plot_num = get_num( n, plot_freq )

        update_num = get_num( n, update_freq )
        
        start = time.time()
        for i in xrange(n):

            if i % update_num == 0: 
                print i
            if i % plot_num == 0: 
                self.plot( directory = directory )

            self.take_step()

            if self.is_jammed():
                print "Traffic jam!"
                print i
                self.plot( directory = directory )
                break

        self.summary()

        return (time.time() - start) / 60

    def plot(self, directory = './images', file_num = None, show = False, digits = 5, full_zoom = True):
        '''
        INPUT: (optional arguments)
            - directory: directory to save simulation plots
            - file_num: extension to filename 'simulation_step_{file_num}'
            - show: whether or not to show plot or just save
            - digits: how many digits to use while padding file_num with zeros
            - full_zoom: allows for zooming in further

        DOC: plot the current location of cars in the simulation
        '''

        def fix_file_num( n, digits ):
            ''' pad n with '0' so total length is digits'''
            n = str(n)
            mult = digits - len(n)
            return '0' * mult + n

        # create data frame with x, y, and color columns for all cars
        coord_col = [ ( x, y, v.get_color() ) for (x, y), v in self.car_loc.iteritems() ]
        coord_col = pd.DataFrame( coord_col, columns = ['x', 'y', 'color'] )

        # create scatter plot by car color
        coord_col.plot( kind='scatter', x='x', y='y', c=coord_col['color'], figsize = (12, 8) )

        # remove axis and extra white space
        plt.gca().set_axis_off()

        if full_zoom:
            plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
            plt.margins(0.01,0.01)
        else:
            plt.margins(0,0)

        plt.gca().xaxis.set_major_locator( plt.NullLocator() )
        plt.gca().yaxis.set_major_locator( plt.NullLocator() )

        plt.title( 'Simulation Step: {0}'.format( self.step ) )

        # save the plot
        if not file_num:
            file_num = fix_file_num( self.step, digits )

        # make directory is it doesn't already exist
        if not os.path.exists(directory):
            os.makedirs(directory)

        filename = '{0}/simulation_step_{1}.png'.format( directory, file_num )
        plt.savefig( filename )

        if show: plt.show()

        plt.close()

    def summary(self):
        '''
        Print summary statistics from the simulation
        '''

        # density
        d = round( (self.n_red + self.n_blue) / np.prod( (self.r + 1, self.c + 1) ), 2 )
        total_cars = self.n_red + self.n_blue
        total_move = int( sum( [ c.get_movement() for c in self.car_loc.itervalues() ] ) )
        possible_moves = int( self.n_red * self.step / 2 + self.n_blue * self.step / 2 )
        percent_move = round( total_move * 100 / possible_moves, 1 )

        title = "\n\nSummary:\n"
        grid_size = "Grid size: {0} x {1}, (r x c)\n".format( self.r + 1, self.c + 1 )
        num_cars = "red cars: {0}, blue cars: {1}, density: {2}\n".format( self.n_red, self.n_blue, d )
        steps_taken = "Total steps taken: {0}\n".format( self.step )
        average_movement = "Cars moved {0} percent of the time ({1} of {2})\n".format( percent_move,
            total_move , possible_moves )
        car_movement = "Car movement last round: {0} (of {1})\n".format( sum( self.movement.itervalues() ),
            total_cars )
        jammed = "The system is jammed: {0}\n".format( self.is_jammed() )

        print title, grid_size, num_cars, steps_taken, average_movement, car_movement, jammed

if __name__ == "__main__":

    # np.random.seed( 293 )

    g = Grid( 150, 150, p = 0.3, density = 0.2 )
    print g.run_simulation( 1000, './sim_images', plot_freq = 3 )





    

