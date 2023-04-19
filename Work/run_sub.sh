BIN=./sub.exe

matrices=(wang4 t2em Transport thermal2)
bits=(20 21 22 23 24 25 26 27 28 29 30)

mkdir -p ./Result
for mat in $matrices; do
    mkdir -p ./Result/${mat}_sub
    for bit in $bits; do
        $BIN $mat.mtx 0 20 $bit > ./result/${mat}_sub/${mat}_int_seq_20_${bit}.txt
    done
done


