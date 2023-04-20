KEY := intel# gnu or intel
OMP := openmp
NOOMP := no-openmp
ifeq ($(KEY),intel)
CC      := icpx
CFLAGs  := -fast -std=c++20 -q$(OMP)
NOCFLAGs:= -fast -std=c++20 -q$(NOOMP)
else 
CC      := g++
CFLAGs  := -Ofast -std=c++20 -f$(OMP)
NOCFLAGs:= -Ofast -std=c++20 -f$(NOOMP)
endif
EXEC1    := seq.exe
EXEC2    := multi.exe
EXEC3    := sub.exe
MYLIBDIR := ../Library
INCDIR  := $(MYLIBDIR)
INCS    := $(wildcard $(INCDIR)/*.hpp) $(wildcard $(INCDIR)/**/*.hpp) $(wildcard $(INCDIR)/**/**/*.hpp)

all: $(EXEC1) $(EXEC2) $(EXEC3)

$(EXEC1): main.cpp $(INCS)
	$(CC) $(NOCFLAGs) -I $(INCDIR) $< -o $@

$(EXEC2): main.cpp $(INCS)
	$(CC) $(CFLAGs) -I $(INCDIR) $< -o $@

$(EXEC3): sub.cpp $(INCS)
	$(CC) $(NOCFLAGs) -I $(INCDIR) $< -o $@

.PHONY: clean test
clean:
	rm -f ./$(EXEC1) ./$(EXEC2) ./$(EXEC3)

