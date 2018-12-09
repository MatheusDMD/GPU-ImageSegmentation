# Background Remover

This project has the goal of comparing performance between a sequential implementation of the **Single Source Shortest Path** algorithm and a GPGPU (General Purpose Graphic Processing Unit) implementation.

## Compiling and Running

Compile both the sequetial and the GPGPU (nvgraph) using the Make command.

To execute run with the following inputs:

```
./nvgraph <input_path>.pgm <output_path>.pgm < <input_seeds>.txt
```
and

```
./sequencial <input_path>.pgm <output_path>.pgm < <input_seeds>.txt
```
an example would be to run it for the ski example:

```
./nvgraph sample_images/ski.pgm sample_output/ski_output.pgm --show < sample_input/input_ski.txt
``` 

to see the original picture as the result of the background remotion program, simply run it with the flag **--show**.

## Input

The input file uses the following structure:

```
<n_foreground_seeds> <n_background_seeds>
<x> <y>
<x> <y>
...
```

## Limitations
- The program only works with PGM images (without comments)
- The program calls the SSSP functions twice for BG and FG
- The nvgraph program calls the GraphGen function twice for BG and FG



