
cd $1
for design_pt in ./*
do
  cd $design_pt
  rm obs.dat
  rm results.dat
  cd ..
done
