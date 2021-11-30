for i in `seq 18`
do
{
plink --bfile d1_7 --chr $i --sheep --recode tab --make-bed --out d1_7_chr$i
}
done
wait
for i in {1..18}
do
{
plink --r2 --bfile d1_7_chr${i} --matrix --out d1_7_chr${i} --make-founders
plink --r2 --bfile d1_7_chr${i} --allow-no-sex --ld-window 99999 --ld-window-kb 1000 --ld-window-r2 0.2 --out d1_7_chr${i}_ld_window --make-founders
}
done
