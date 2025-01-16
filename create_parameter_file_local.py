import sys
import os
CURR_DIR = os.getcwd()
mayawaves_path = CURR_DIR[:CURR_DIR.rfind('/')]
sys.path.append(mayawaves_path)


from mayawaves.utils.preprocessingutils import *

#output_directory = "/home/hli75/mayawaves_data_analysis_projects/COM_corrections_project/simulations"
output_directory = ""

mass_ratio = 50 
initial_separation = 3
primary_dimensionless_spin = np.array([0, 0, 0])
secondary_dimensionless_spin = np.array([0, 0, 0])
eccentricity = 0
create_bbh_parameter_file(output_directory=output_directory, mass_ratio=mass_ratio,
                          initial_separation=initial_separation,
                          primary_dimensionless_spin=primary_dimensionless_spin,
                          secondary_dimensionless_spin=secondary_dimensionless_spin,
                          eccentricity=eccentricity)
