# mode_analyze
Python code analyzing gyre output (including calculating overlap integrals)

IApprox/IApprox.m -- mathematica codes for computing I that appears in tidal coupling constants

coupling_n3.csv, coupling_n32.csv -- tidal coupling constants from Lee&Ostriker '86

poly*tsv -- tabulated polytropic models (first column is radius, second column is density).

mode_analyze.py -- creates ModeAnalyzer object that organized basic info about the stellar model
and associated oscillation modes. 

poly_3, poly_32 -- contains mode data computed from GYRE version 5. 

example.ipynb -- show some examples of the the code in action (getting basic mode info, tidal coupling constants). 
