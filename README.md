# Overview
Python class to simulate the Biham-Middleton-Levine traffic model

## The Simulation Process

This project involves a simulation of a simple traffic flow model that exhibits a phase transition. The simulation is about cars moving on a grid. We have two types of cars, _blue_ and _red_. These move on a two-dimensional grid of size _r_ by _c_. We populate the grid by placing _ρ_ × _r_ × _c_ (with 0 < _ρ_ < 1) cars at random positions in the _r_ × _c_ cells, but with no two cars occupying the same cell. 
Now the cars can move. In our configuration, the blue cars move at odd time periods _t_ = 1, 3, 5, ... and the red cars move at even time periods _t_ = 2, 4, 6, ..., i.e. they alternate in time. The blue cars move vertically upward while the red cars move horizontally rightwards. When a car gets to the edge of the grid, it “wraps” around, i.e. when a blue car gets to the top row, the next time it moves it goes to the bottom row of the same column. Similarly a red car that gets to the right edge of the grid will move next to the first column of the grid, i.e. the extreme left. A car cannot move to a cell if that cells is already occupied by another car (of any color).
This process is called the Biham-Middleton-Levine traffic model and is of interest because it is one of the very simplest processes that exhibits a phase-transition, and also self-organizing behavior. More information is available <a href="https://en.wikipedia.org/wiki/Biham%E2%80%93Middleton%E2%80%93Levine_traffic_model">here</a>.

## Examples

To run a simulation we simply run 

```
g = Grid(r=150, c=150, n_red=2000, n_blue=2000)
g.run_simulation(2000)
```

which creates a 150 x 150 grid with 2,000 red cars and 2,000 blue cars and runs the simulation for 2,000 steps. Summary statistics will print once the simulation finishes. Alternatively, we could specify the probability of a red car (p = 0.3) and the density of the cars on the grid (d = 0.2).

```
g = Grid(r=150, c=150, p=0.3, density=0.2)
g.run_simulation(2000)
```

Finally, the following code runs the same simulation while capturing the plot of every 8th step.

```
g = Grid(r=150, c=150, p=0.3, density=0.2)
g.run_simulation(2000, plot_freq=8)
```

This allows us to create the following animations. Notice that eventually all the red cars become grouped together.

<img src="./images/traffic_sim_1.gif" alt="traffic animation 1">

The next animation shows a similar process at a reduced speed.

```
g = Grid(r=150, c=150, p=0.3, density=0.3)
g.run_simulation(1000, plot_freq=3)
```

<img src="./images/traffic_sim_2.gif" alt="traffic animation 2">

Finally, since these are random events, we can sometimes reach a traffic jam.

```
g = Grid(r=150, c=150, p=0.5, density=0.3)
g.run_simulation(1000)
```

<img src="./images/traffic_jam.gif" alt="traffic jam">