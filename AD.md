# Artifact Description

## Artifact Identification

Our article proposed integer (fixed-point) arithmetic-based implementations of the algebraic multigrid (AMG) method and the preconditioned FGMRES method with iterative refinement for solving a linear system. Combining the two integer arithmetic-based implementations, we developed an integer arithmetic-based linear solver. Then, through numerical experiments, we compared it with two floating-point arithmetic-based AMG preconditioned FGMRES solvers: one based on only FP64 arithmetic and one based on a mixed-precision algorithm of FP32 and FP64 arithmetic. The numerical results showed that our integer-based solver had a comparable convergence rate to the two floating-point-based solvers. Additionally, the results demonstrated that our solver had a similar timing performance to the mixed-precision solver and ran faster than the FP64-based solver.



Our computational artifact (CA) provides the implementations of all three solvers and the test suites with scripts. Because the CA is self-contained, it can reproduce the experiments of the article by itself, except for the effect of the compiler implementation and the execution environment. 

The CA is available at a GitHub repository (https://github.com/suzuki-hpc/SC23_AD). The CA comprises C++ source files and some supplemental Zsh and Python scripts.

The C++ source code does not depend on any external libraries, and thus it may work on any CPU-based hardwares. However, we suggest an CPU supporting a C++20 compiler for adequate reproducibility considering the use of the bit shifting operations. The Python scripts use Matplotlib to draw graph figures, which can be installed by following the installation instruction in the CA. The CA supports matrix files encoded by the Matrix Market format. All test matrices used in the article are available from the SuiteSparse Matrix Collection (). The CA provides the script for downloading them, which can be used instead of downloading from the website manually.

## Reproducibility of Experiments

The workflow of the experiment reproduction consists of four steps. First, the test matrices are downloaded from the SuiteSparse Matrix Collection as follows.

```shell
$ cd <path to the CA>/Matrix
zsh download.sh
```

Note that the matrices can be downloaded manually from the website.

(ii) Next, the C++ source code is compiled using GNU Make.

In step (ii), GNU Make generates two executable files. One performs the main experiments for comparing the three solvers, `auto.out` , and the other achieves the additional test for evaluating the effect of the fractional bit length of fixed-point formats on our integer-based solver, `manual.out`. Note that we assume the use of GCC or Intel oneAPI.

```shell
$ cd <path to the CA>/Work
$ make
```



(iii) Then, the Zsh scripts perform all test suites using the executables.

In step (iii), the Zsh scripts execute the two executables and write the results to text files in a directory with the same name as the test matrix. For example, the test results for `wang3` are stored in files in `Work/Result/wang3`. Each file contains the convergence history, the implicit relative residual norm, the execution time, the number of iterations, and the explicit relative residual norm, in that order. The pair of the execution time and the number of iterations corresponds to an entry of the tables in the article. If the executables and scripts perform correctly, the number of iterations should be several dozen, and the execution time should be a few seconds.

```shell
zsh run.sh
zsh run_manual.sh
```

 (iv) Finally, the Python scripts draw the graphs of the convergence history.

```shell
cd <path to the CA>/Work/Artwork
pip3 install -r requirements.txt
python3 tab_reqult.py
python3 fig_history.py <matrix name> <5 or 10 or 20> <seq or multi>
python3 fig_manual.py
```



The convergence histories correspond to the graphs in the article. In step (iv), similar figures can be reproduced using the Python scripts in the CA. For detailed execution instructions, such as setting runtime arguments, refer to the README file in the GitHub repository.



 Each step will take about 3, 1, 15, or 1 minute, respectively; the total execution time will be about 20 minutes.



