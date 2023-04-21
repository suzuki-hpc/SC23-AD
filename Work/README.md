# Experiment workflow

## Making executables

In the CA, GNU Make compiles our C++ source code as follows:

```shell
cd <path to the CA>/Work
make -f Makefile
```

The `Makefile` assumes Intel OneAPI or GCC. If necessary, the compiler and the options should be changed. `CC` and `CFLAGs` correspond to the computer and the options, respectively. 

If all the compilations is successful, three executables are created in the `<path to the CA>/Work` directory: `seq.exe`, `multi.exe`, and `sub.exe`.

## Running executables

We provide all test cases in two Zsh scripts: `run.sh` and `run_sub.sh`. Only running these scripts, all the tests can be performed:

```shell
zsh run.sh
zsh run_sub.sh
```

After running all the test properly, two subdirectories, `Result/<test name>` and `Result/<test name>_sub`, are  created, which save the test results as text files. The former contains the results of the tests that uses `run.sh` and the latter stores the results of  `run_sub.sh`.

The text files in `Result/<test name>` are named according to the following rule.

```
<test name>_<fp64 or fp32 or int>_<seq or multi>_<5 or 10 or 20>.txt
```

Here, `fp64`, `fp32`, and `int` identify the used solver, and `seq` and `multi` specify whether the solve is implemented sequentially or multi-threaded. `5`, `10`, and `20` indicate the restart period of the solver.

The files in `Result/<test name>_sub` are name similarly as follows:

```
<test name>_int_seq_20_<bits>.txt
```

Because the tests of `run_sub.sh` evaluate the effect of the fractional bit length on the sequential integer-based solver with restart cycle of 20, the names include `int`, `seq`, and `20`. `bits` indicates the assigned fractional bit length in that test.

## Drawing tables and figures

In the [Artwork](Artwork) directory, the Python scripts reproduce similar tables and figures to the ones in the articles. The running instructions are written in the README file in the Artwork directory.