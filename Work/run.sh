BIN_SEQ=./seq.exe
BIN_MULTI=./multi.exe

matrices=(wang3 wang4 atmosmodd atmosmodm t2em Transport G3_circuit thermal1 thermal2)

periods=(5 10 20)

typeset -A solvers
solvers=(0 fp64 1 fp32 2 int)

mkdir -p ./Result
for mat in $matrices; do
    mkdir -p ./Result/$mat
    for id in ${(k)solvers}; do
        for j in $periods; do
            $BIN_SEQ $mat.mtx 0 $id $j > ./Result/$mat/${mat}_$solvers[$id]_seq_$j.txt
        done
    done
done

export OMP_NUM_THREADS=40

for mat in $matrices; do
    mkdir -p ./Result/$mat
    for id in ${(k)solvers}; do
        for j in $periods; do
            $BIN_MULTI $mat.mtx 1 $id $j > ./Result/$mat/${mat}_$solvers[$id]_multi_$j.txt
        done
    done
done

