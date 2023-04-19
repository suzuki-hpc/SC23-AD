BIN=./manual.out

matrices=(wang4 t2em Transport thermal2)

for mat in $matrices; do
    mkdir -p ./Result/${mat}_manual
    $BIN $mat.mtx 0 20 > ./result/${mat}_manual/${mat}_int_seq_20.txt
done


