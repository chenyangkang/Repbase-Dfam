for i in {2..133}
do
{
echo ${i}
gcta64 \
--fastGWA-mlm \
--bfile d1_7 \
--grm-sparse d1_7_sp \
--pheno basic_pheno.T${i}.txt \
--qcovar covar_mds.txt \
--threads 10 \
--out d1_7_assoc_gcta_T${i}  
#--covar fixed.txt

cat d1_7_assoc_gcta_T${i}.fastGWA|head -1 > d1_7_assoc_gcta_T${i}.fastGWA.filtered.txt
cat d1_7_assoc_gcta_T${i}.fastGWA|awk '$10<=0.01{print $0}'|sort -u -k10,10 -r -g >> d1_7_assoc_gcta_T${i}.fastGWA.filtered.txt

Rscript Manhattan_plot.R d1_7_assoc_gcta_T${i}.fastGWA ${i}.fastGWA
Rscript QQ_plot.R d1_7_assoc_gcta_T${i}.fastGWA ${i}.fastGWA
}
done
