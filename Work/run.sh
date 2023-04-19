BIN=./auto.out

matrices=(wang3 wang4 atmosmodd atmosmodm t2em Transport G3_circuit thermal1 thermal2)
periods=(5 10 20)

typeset -A smoothers
smoothers=(0 seq 1 multi)

typeset -A solvers
solvers=(0 fp64 1 fp32 2 int)

mkdir -p ./Result
for mat in $matrices; do
    mkdir -p ./Result/$mat
    for s_id in ${(k)smoothers}; do
        for id in ${(k)solvers}; do
            for j in $periods; do
                $BIN $mat.mtx $s_id $id $j > ./Result/$mat/${mat}_$solvers[$id]_$smoothers[$s_id]_$j.txt
            done
        done
    done
done


