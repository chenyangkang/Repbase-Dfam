for i in `seq 18`
do
{
plink --bfile d1_7 --chr $i --sheep --recode tab --out d1_7_chr$i
}
done
