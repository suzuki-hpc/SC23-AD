KEY := gnu # gnu or intel
OMP := openmp
ifeq ($(KEY), intel)
CC      := icpx
CFLAGs  := -fast -xCORE-AVX512 -std=c++20 -q$(OMP) -Wall -Wopenmp
else
CC      := g++-12
CFLAGs  := -Ofast -std=c++20 -f$(OMP) -Wall
endif
EXEC1    := auto.out
EXEC2    := manual.out
MYLIBDIR := ../Library
INCDIR  := $(MYLIBDIR)
INCS    := $(wildcard $(INCDIR)/*.hpp) $(wildcard $(INCDIR)/**/*.hpp) $(wildcard $(INCDIR)/**/**/*.hpp)

all: $(EXEC1) $(EXEC2)

$(EXEC1): main.cpp $(INCS)
	$(CC) $(CFLAGs) -I $(INCDIR) $< -o $@

$(EXEC2): main_manual.cpp $(INCS)
	$(CC) $(CFLAGs) -I $(INCDIR) $< -o $@

.PHONY: clean test
clean:
	rm -f ./$(EXEC1) ./$(EXEC2)

