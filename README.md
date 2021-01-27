# Take it Easy

A short introduction to the game implemented in the evironment, taken from [Wikipedia](https://en.wikipedia.org/wiki/Take_It_Easy_(game)):

> Take It Easy is an abstract strategy board game created by Peter Burley. It can be characterized as a strategic bingo-like game, and has been published by several publishers since 1983. Each player gets a board with places for 19 hexagon tiles to place in a hexagon shape. Additionally, players get identical sets of tiles which have different types of colored/numbered lines crossing in three directions. One player draws a tile randomly and then tells the others which was drawn. Each player then puts their matching tile on their board in any available spot. This is repeated until the board is filled. The object is to complete colored/numbered lines across the board, for which points are scored according to the numbers on those lines. The maximum score possible is 307.

Each episode consists of 19 transitions as this is the number of tiles that can be placed on the board.
The environment has two reward modes, either returning reward for the current state of the game by calculating the score of lines that are connected and
undisturbed or by simply calculating the reward at the last timestep for the full board.
A visual representation is available and an interactive mode, where one can load (fully trained) DRL agents and select the drawn piece to evaluate the agents in real games is planned.

Further, this repository will feature simple DRL algorithms to test the environment and compare their usability for highly stochastic environments.
