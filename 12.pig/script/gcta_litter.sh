cd $PBS_O_WORKDIR
gcta64 \
--fastGWA-mlm \
--bfile d1_7 \
--grm-sparse d1_7_sp \
--pheno basic_pheno.Tlitter.txt \
--qcovar covar_mds.txt \
--threads 1 \
--out d1_7_assoc_gcta_Tlitter
#--covar fixed.txt

cat d1_7_assoc_gcta_litter.fastGWA|head -1 > d1_7_assoc_gcta_Tlitter.fastGWA.filtered.txt
cat d1_7_assoc_gcta_litter.fastGWA|awk '$10<=0.01{print $0}'|sort -u -k10,10 -r >> d1_7_assoc_gcta_Tlitter.fastGWA.filtered.txt

Rscript Manhattan_plot.R d1_7_assoc_gcta_litter.fastGWA ${i}.fastGWA
Rscript QQ_plot.R d1_7_assoc_gcta_litter.fastGWA ${i}.fastGWA

