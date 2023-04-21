KEY := intel# gnu or intel
ifeq ($(KEY),intel)
CC      := icpx
CFLAGs  := -fast -std=c++20 -qopenmp
SEQCFLAGs:= -fast -std=c++20 -qno-openmp
else 
CC      := g++
CFLAGs  := -Ofast -std=c++20 -fopenmp
SEQCFLAGs:= -Ofast -std=c++20 -fno-openmp
endif
EXEC1    := seq.exe
EXEC2    := multi.exe
EXEC3    := sub.exe
MYLIBDIR := ../Library
INCDIR  := $(MYLIBDIR)
INCS    := $(wildcard $(INCDIR)/*.hpp) $(wildcard $(INCDIR)/**/*.hpp) $(wildcard $(INCDIR)/**/**/*.hpp)

all: $(EXEC1) $(EXEC2) $(EXEC3)

$(EXEC1): main.cpp $(INCS)
	$(CC) $(SEQCFLAGs) -I $(INCDIR) $< -o $@

$(EXEC2): main.cpp $(INCS)
	$(CC) $(CFLAGs) -I $(INCDIR) $< -o $@

$(EXEC3): sub.cpp $(INCS)
	$(CC) $(SEQCFLAGs) -I $(INCDIR) $< -o $@

.PHONY: clean test
clean:
	rm -f ./$(EXEC1) ./$(EXEC2) ./$(EXEC3)

