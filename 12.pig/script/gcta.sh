for i in {1..10}
do
{
gcta64 --bfile d1_7 --make-grm-part 10 ${i} --thread-num 1 --out d1_7
}
done

cat d1_7.part*.grm.id > d1_7.grm.id
cat d1_7.part*.grm.bin > d1_7.grm.bin
cat d1_7.part*.grm.N.bin > d1_7.grm.N.bin

gcta64 --grm d1_7 --make-bK-sparse 0.05 --out d1_7_sp

for i in {2..133}
do
{
python3 replace_phenotype.py T${i}
}
done

for i in {2..133}
do
{
echo '''
cd $PBS_O_WORKDIR
gcta64 \
--fastGWA-mlm \
--bfile d1_7 \
--grm-sparse d1_7_sp \
--pheno basic_pheno.T'"${i}"'.txt \
--qcovar covar_mds.txt \
--threads 1 \
--out d1_7_assoc_gcta_T'"${i}"'
#--covar fixed.txt

cat d1_7_assoc_gcta_T'"${i}"'.fastGWA|head -1 > d1_7_assoc_gcta_T'"${i}"'.fastGWA.filtered.txt
cat d1_7_assoc_gcta_T'"${i}"'.fastGWA|awk '"'"'$10<=0.01{print $0}'"'"'|sort -u -k10,10 -r >> d1_7_assoc_gcta_T'"${i}"'.fastGWA.filtered.txt

Rscript Manhattan_plot.R d1_7_assoc_gcta_T'"${i}"'.fastGWA.filtered.txt ${i}.fastGWA
Rscript QQ_plot.R d1_7_assoc_gcta_T'"${i}"'.fastGWA.filtered.txt ${i}.fastGWA
'''>gcta_T${i}.sh
chmod 755 gcta_T${i}.sh
}
done







for i in {2..133}
do
{
gcta64 \
--fastGWA-mlm \
--bfile d1_7 \
--grm-sparse d1_7_sp \
--pheno basic_pheno.T${i}.txt \
--qcovar covar_mds.txt \
--threads 1 \
--out d1_7_assoc_gcta_T${i}
#--covar fixed.txt

cat d1_7_assoc_gcta_T${i}.fastGWA|head -1 > d1_7_assoc_gcta_T${i}.fastGWA.filtered.txt
cat d1_7_assoc_gcta_T${i}.fastGWA|awk '$10<=0.01{print $0}'|sort -u -k10,10 -r >> d1_7_assoc_gcta_T${i}.fastGWA.filtered.txt

Rscript Manhattan_plot.R d1_7_assoc_gcta_T${i}.fastGWA ${i}.fastGWA
Rscript QQ_plot.R d1_7_assoc_gcta_T${i}.fastGWA ${i}.fastGWA
}
done
