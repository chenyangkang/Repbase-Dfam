cd $PBS_O_WORKDIR
plink --r2 --bfile d1_6 --matrix
cut -f7- outfile.ped >geno.txt
sed -i 's/ /\//g' geno.txt
wait
