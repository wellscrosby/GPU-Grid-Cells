# GPU Grid Cells

## About

A GPU accelerated simulation of grid cells. Heavily inspired by Burak and Fiete et al. (2009) [Paper](https://arxiv.org/abs/0811.1826).

This implementation demonstrates the potential for grid cells to detect velocity of input stimuli via feedback (reading out the velocity is not implemented).

Created as a final project for Projects in the Science of Intelligence (9.58) at MIT. In paper.pdf is the admitedly rough and oddly structured paper detailing this work further (Wells Crosby's version).

## Usage

Run grid_cells.ipynb to see an interactive grid cell simulation. An Nvidia GPU will allows the program to run extremely quickly!

## Futher Work

- Improving kernel generation
- Improving performance
- Use activation sum to create velocity detector
- Break global velocity into seperate cardianal directions

I will likely not work on this project further so please give those improvements a try!
