
# fully specify numeric data types, including endianness and size, to
# ensure consistency across all machines
float_t = '<f8'
int_t = '<i8'
complex_t = '<c16'

#how many versions of the model are run, for instance
# 4 versions of delta-f with SMASH and a fifth model with UrQMD totals 5
number_of_models_per_run = 5

#the Collision systems
systems = [('Pb', 'Pb', 2760)]

#the number of design points
n_design_pts = 100

# Number of principal components to keep in the emulator
npca=4
f_model_calculations = 'model_calculations/Obs/main.dat'
f_model_validations = 'model_calculations/Obs/validation.dat'
design_dir = 'design_pts'

