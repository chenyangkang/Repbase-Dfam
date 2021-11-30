#### step0 ##### remove non-F2 indiv ########
plink --bfile d1 --remove nonF2.list --make-bed --out d1_1



### Step1##
plink --bfile d1_1 --missing
Rscript --no-save hist_miss.R
plink --bfile d1_1 --geno 0.2 --make-bed --out d1_2
plink --bfile d1_2 --mind 0.2 --make-bed --out d1_3
plink --bfile d1_3 --geno 0.02 --make-bed --out d1_4
plink --bfile d1_4 --mind 0.02 --make-bed --out d1_5


###### step2 ########
#plink --bfile d1_5 --check-sex 
#Rscript --no-save gender_check.R
#grep "PROBLEM" plink.sexcheck| awk '{print$1,$2}'> sex_discrepancy.txt
#plink --bfile d1_5 --remove sex_discrepancy.txt --make-bed --out d1_6 

###### step3 #######
##### autosomal already selected ############
plink --bfile d1_5 --freq --out MAF_check --nonfounders
Rscript --no-save MAF_check.R
plink --bfile d1_5 --maf 0.05 --make-bed --out d1_6 --nonfounders


###### step4 ######
plink --bfile d1_6 --hardy --nonfounders
awk '{ if ($9 <0.00001) print $0 }' plink.hwe>plinkzoomhwe.hwe
Rscript --no-save hwe.R
plink --bfile d1_6 --hwe 1e-5 --make-bed --out d1_hwe_filter_step1 --nonfounders
plink --bfile d1_hwe_filter_step1 --hwe 1e-10 --hwe-all --make-bed --out d1_7 --nonfounders


####### LD ########
#plink --r2 --bfile d1_7 --allow-no-sex --ld-window 99999 --ld-window-kb 1000 --ld-window-r2 0.2 --out out_file
#plink --r2 --bfile d1_7 --matrix
#cut -f7- outfile.ped >geno.txt
#sed -i 's/ /\//g' geno.txt

###### step5 #######
#plink --bfile d1_7 --exclude inversion.txt --range --indep-pairwise 50 5 0.2 --out indepSNP
plink --bfile d1_7 --indep-pairwise 500kb 5 0.5 --out indepSNP
plink --bfile d1_7 --extract indepSNP.prune.in --make-bed --out d1_7
plink --bfile d1_7 --extract indepSNP.prune.in --het --out R_check
Rscript --no-save check_heterozygosity_rate.R
Rscript --no-save heterozygosity_outliers_list.R
sed 's/"// g' fail-het-qc.txt | awk '{print$1, $2}'> het_fail_ind.txt
plink --bfile d1_7 --remove het_fail_ind.txt --make-bed --out d1_8


###### step6 #######
plink --bfile d1_8 --extract indepSNP.prune.in --genome --min 0.2 --out pihat_min0.2
awk '{ if ($8 >0.9) print $0 }' pihat_min0.2.genome>zoom_pihat.genome
Rscript --no-save Relatedness.R
plink --bfile d1_8 --filter-founders --make-bed --out d1_8
plink --bfile d1_9 --extract indepSNP.prune.in --genome --min 0.2 --out pihat_min0.2_in_founders
plink --bfile d1_9 --missing
### edit remove file manually
plink --bfile d1_9 --remove 0.2_low_call_rate_pihat.txt --make-bed --out d1_10
