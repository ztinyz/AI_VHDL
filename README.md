# AI_VHDL 

> A work-in-progress VHDL project aiming to implement an Ai model that can identify drawn numbers on an FPGA

## Description

AI_VHDL is an experimental VHDL-based hardware design project.\
The goal is to explore hardware implementation of a neural network.\
The project is currently under development.\
This repository serves as a sandbox for learning and experimentation.

## Motivation & Goals

- Serve as a learning / educational project: to deepen understanding of Ai architecture, VHDL, Python, simulation & hardware design.  
- Offer a basis for further expansion — once design stabilizes: testing, simulation, possibly synthesis for FPGA or hardware targets.

## Current Status

Unfinished — not all components are implemented or verified.\
The Ai model is trained and working properly.\
The FPGA implements now only a canvas(VGA) where you can draw numbers using buttons.

## (Proposed) Project Structure
AI_VHDL\
├── VHDL_CODE/ # VHDL source files\
├── models/ # all the trained Ai models I experimented with\
├── Model_training/ # all the python files I needed in order to train and quantiteze a model using pytorch.\
└──other files as project evolves

## Requirements / Tools

To work with or build this project (when ready):

- A VHDL-capable toolchain (Vivado)
- Basic familiarity with VHDL, hardware description, synchronous logic.  
- For hardware targets: FPGA / board, constraints file, I/O definition, etc.
