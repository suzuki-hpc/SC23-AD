matrices=(atmosmodd atmosmodm G3_circuit thermal2 thermal1 Transport t2em wang3 wang4)
groups=(Bourchtein Bourchtein AMD Schmid Schmid Janna CEMW Wang Wang)

for (( i = 1; i < 10; i++ )); do
    curl -o $matrices[i].tar.gz http://sparse-files.engr.tamu.edu/MM/$groups[i]/$matrices[i].tar.gz
    tar -zxvf $matrices[i].tar.gz
    rm $matrices[i].tar.gz
    mv $matrices[i]/$matrices[i].mtx ./
    rm -rf $matrices[i]
done


