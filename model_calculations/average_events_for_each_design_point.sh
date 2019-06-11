
#loop over design point directories, running average_obs.py once for each design point
cd $1
for design_pt in ./*
do
  cd $design_pt
  echo $design_pt  
  python3 ../../calculations_average_obs.py results.dat
  cd ..
done
cd ..
