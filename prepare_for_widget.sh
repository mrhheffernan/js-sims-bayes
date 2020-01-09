#download the expt data
sh download_experimental_data.sh

#unzip the design points
cd production_designs/500pts/
unzip -q design_pts_Au_Au_200_production.zip
unzip -q design_pts_Pb_Pb_2760_production.zip
cd ../../
