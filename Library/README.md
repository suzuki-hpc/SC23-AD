# Header-only library for our solvers

The foundation of the three solvers are provided in [solver](solver), and the AMG preconditioner is implemented in [preconditioner](preconditioner). The Gauss-Seidel smoother used in the AMG preconditioner is defined in [smoother](smoother).

[blas](blas) provides some functions similar to the level-1 and level-2 BLAS, and [io](io) give a functions for reading a matrix of the Matrix Market format and store it in the CSR format. The other folders and files, such as helper, present some supplemental functions related with their filenames.



The`intAMGflex` class in [preconditioner/INTAMG.hpp](preconditioner/INTAMG.hpp) is the implementation of our integer-based AMG preconditioner, and the `FGMRESflex` class in [solver/gmres.hpp](solver/gmres.hpp) is the implementation of our integer-based FGMRES solver.