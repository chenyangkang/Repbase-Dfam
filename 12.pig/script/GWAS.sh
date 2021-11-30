#plink --bfile d1_7 --read-genome pihat_min0.0.genome --cluster --mds-plot 10 --out d1_7_mds
plink --bfile d1_7 --cluster --mds-plot 10 --out d1_7_mds --exclude nonF2.list

awk '{print$1, $2, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13}' d1_7_mds.mds > covar_mds.txt


for i in $(<Traits.list)
do
{
plink --bfile d1_7 --covar covar_mds.txt --hide-covar --pheno all.pheno --pheno-name ${i} --allow-no-sex --assoc --adjust --out ${i}.assoc

#awk '!/'NA'/' logistic_results.assoc.logistic > logistic_results.assoc_2.logistic
#plink --bfile HapMap_3_r3_13 -assoc --adjust --out adjusted_assoc_results

Rscript Manhattan_plot.R ${i}.assoc.qassoc ${i}.assoc.qassoc
Rscript QQ_plot.R ${i}.assoc.qassoc ${i}.assoc.qassoc
}
done
