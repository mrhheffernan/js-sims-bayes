
cd $1
for design_pt in ./*
do
  cd $design_pt
  cat *.dat > results.dat
  cd ..
done
