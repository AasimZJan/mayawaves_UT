#!/usr/bin/env python
#The purpose of this code is to create a metadata from MAYA NR h5 files. The h5 files need to be in MAYA format as this code gather metadata by creating a mayawaves coalescence object from the h5 files.

#Usage: python CreateMetadata.py --path /Users/aasim/Desktop/Research/Codes/steps_marginalisation/NR-manager/Sequence-MAYA-Generic/all --output-path `pwd` --output MAYAmetadata
#       python CreateMetadata.py --path /Users/aasim/Desktop/Research/Codes/steps_marginalisation/NR-manager/Sequence-MAYA-Generic/all
# TO DO:
# Add the ability to combine an old metadata with a new one. Add the ability to check for duplicates. Make a new code to combine n metadata files.
# some waveforms have None spin. Need to look into that
import pandas as pd
import argparse
import os
import sys
import numpy as np
try:
    from mayawaves.coalescence import Coalescence
except:
    print("Unable to import mayawaves, exiting")
    sys.exit()
 
parser = argparse.ArgumentParser()
parser.add_argument("--path", help = "Path to h5 files,the folder should contain all the NR h5 files you want to create the metadata for.")
parser.add_argument("--output", default = "metadata", help = "Name for the metadata file, default is 'metadata'.")
parser.add_argument("--output-path", default = False, help = " (Optional) Where you want the code to save the metadata file, by default it saves it in the running directory.")
parser.add_argument("--add-to-metadata", default = False, help = "(Optional) path to existing metadata file whose information you want to add to the new metadata file.")
opts = parser.parse_args()


#check for the h5 file path, exit if the path doesn't exist.
path = opts.path 
curr_dir = os.getcwd()
if os.path.exists(path) == False:
    print("Provided path doesn't exist. Exiting")
    sys.exit()

#load in files and initialize arrays
os.chdir(path)
files = list(os.popen("ls *h5").read().split("\n")[:-1])
print(f"---------------------Loading files from {path}--------------------")
print(f"Total number of h5 files in this folder is {len(files)}")

#Creating empty lists to store metadata
name = [] #coalescence
m1 = [] #compact_obj
m2 = [] #compact_obj
m1_irr = [] #compact_obj
m2_irr = [] #compact_obj
q = []  #coalescence
eta = []  #coalescence
a1x, a1y, a1z = [],[],[] #compact_obj 
a2x, a2y, a2z = [], [], [] #compact_obj
fmin = [] #calculated
omega = []  #coalescence
separation = []  #coalescence
merge_time = [] #coalescence
eccentricity = [] #coalescence
mpa_array = [] #coalescence

#to calculate fmin from omega, need to convert into real units
G = 6.67428e-11 
mass_sun = 1.98892e30 
c = 2.99792458e8  
factor = mass_sun* G/c**3  #this factor converts time from code units to real units


print("Eccentricity being evaluated from 0 M to 75 M.")
for i in files:
    print(f"Reading {i}")
    #gather data from the coalecence object
    c_obj = Coalescence(f"{opts.path}/{i}")
    name.append(c_obj.catalog_id)
    eta.append(c_obj.symmetric_mass_ratio)
    q.append(c_obj.mass_ratio)
    omega.append(c_obj.initial_orbital_frequency)
    separation.append(c_obj.initial_separation)
    ecc , mpa = c_obj.eccentricity_and_mean_anomaly_at_time(75,75)  #use 75,75 instead of 0,0 to avoid the effect of junk radiation
    eccentricity.append(ecc)
    mpa_array.append(mpa)
    merge_time.append(c_obj.merge_time)
    fmin.append(omega[-1]/factor/np.pi)

    #gather data from compact objects
    primary_obj = c_obj.primary_compact_object
    secondary_obj = c_obj.secondary_compact_object
    a1 = primary_obj.initial_dimensionless_spin
    a2 = secondary_obj.initial_dimensionless_spin
    m1.append(primary_obj.initial_horizon_mass)
    m2.append(secondary_obj.initial_horizon_mass)
    m1_irr.append(primary_obj.initial_irreducible_mass)
    m2_irr.append(secondary_obj.initial_irreducible_mass)
    try:
        a1x.append(a1[0])
        a1y.append(a1[1])
        a1z.append(a1[2])
        a2x.append(a2[0])
        a2y.append(a2[1])
        a2z.append(a2[2])
    except:#test why we need this
        a1x.append(None)
        a1y.append(None)
        a1z.append(None)
        a2x.append(None)
        a2y.append(None)
        a2z.append(None)


final_data = {"m1":m1, "m2":m2, "m1_irr":m1_irr, "m2_irr":m2_irr, "q":q, "eta":eta, "a1x":a1x, "a1y":a1y, "a1z":a1z, "a2x":a2x, "a2y":a2y, "a2z":a2z, "f_lower_at_1MSUN":fmin, "omega":omega, "separation":separation, "eccentricity":eccentricity, "mean_per_anomaly":mpa_array,\
    "merge_time":merge_time}

df = pd.DataFrame(final_data)


df.index = name #name the rows as waveform name
print(f"Information stored {list(df.keys())}")


#saving metadata, where depends on if output path was provided.
if opts.output_path:
    print(f"Saving to {opts.output_path}")
    df.to_pickle(f"{opts.output_path}/{opts.output}.pkl")
else:
    print(f"Saving to {curr_dir}")
    df.to_pickle(f"{curr_dir}/{opts.output}.pkl")
