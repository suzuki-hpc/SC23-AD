# How to reproduce the experiments

## (ii) Making executables

We adopt GNU Make to compile our C++ code. Thus, we following commands generates the executables: `auto.out` and `manual.out`.

Note that the `Result` directory making in the last command is necessary for storing the test results in step (iii).

```shell
cd (path to this directory)
make -f Makefile
mkdir Result
```

## (iii) Running executables

We provide all test cases in two Zsh scripts: `run.sh` and `run_manual.sh`. Only running these scripts you can obtain the numerical results in the `Result` directory.

```shell
zsh run.sh
zsh run_manual.sh
```

For each test problems, two directories is generated: `Result/(name)` and `Result/(name)_manual`. The former contains the results of the tests using `run.sh` and the latter stores the results of the tests using `run_manual.sh`.

Each text file is named according to the following rule.

```
(name)_(fp64 or fp32 or int)_(seq or multi)_(5 or 10 or 20).txt
```

`fp64`, `fp32`, and `int` identify the used solver, and `seq` and `multi` specify whether the solve is implemented sequentially or multi-threaded. `5`, `10`, and `20` indicate the restart period of the solver.

## (iv) Drawing figures

Using the Python scripts, you can produce similar tables and figures to the ones in the articles. The running instructions are written in the README file in the [Artwork](Artwork) directory.