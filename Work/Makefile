KEY := gnu # gnu or intel
OMP := openmp
NOOMP := no-openmp
ifeq ($(KEY), intel)
CC      := icpx
CFLAGs  := -fast -std=c++20 -q$(OMP) -Wall -Wopenmp
else
CC      := g++
CFLAGs  := -Ofast -std=c++20 -f$(OMP) -Wall
endif
EXEC1    := main_seq.out
EXEC2    := main_multi.out
EXEC3    := sub.out
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

