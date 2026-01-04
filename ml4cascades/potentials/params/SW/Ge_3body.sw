# DATE: 2025-12-17 UNITS: metal CONTRIBUTOR: Ruoyan Jin, ruoyan.jin@aalto.fi 

# Stillinger-Weber parameters for various elements and mixtures
# multiple entries can be added to this file, LAMMPS reads the ones it needs
# these entries are in LAMMPS "metal" units:
#   epsilon = eV; sigma = Angstroms
#   other quantities are unitless

# format of a single entry (one or more lines):
#   element 1, element 2, element 3, 
#   epsilon, sigma, a, lambda, gamma, costheta0, A, B, p, q, tol
# set A = 0, to remove 2 body part

# Here are the original parameters in metal units, for Germanium from:
# CITATION: Kejian Ding and Hans C. Andersen,  Phys Rev B, 34, 6987, (1986)
# CITATION: K. Nordlund et al., Phys Rev B, 57, 7556, (1998)


Ge Ge Ge 1.58  2.181  1.80  21.0  1.20  -0.333333333333
         0  0.6022245584  4.0  0.0 0.0
