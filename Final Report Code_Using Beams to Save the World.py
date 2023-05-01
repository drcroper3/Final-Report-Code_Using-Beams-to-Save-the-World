#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Final Term Report Code: Using Beams to Save the World: An Approach Using a Linear Electron Accelerator System for Multipurpose Applications 
# Author: Chris Roper
# Course: Computational Physics (PHYS 6260)
# References: Ortner, Michael, and Lucas Gabriel Coliado Bandeira. "Magpylib: A free Python package for magnetic field computation." SoftwareX 11 (2020): 100466.
    
import numpy as np
import matplotlib.pyplot as plt
import cv
import math
from pyomo.environ import *

    
# Interative codes listed: 
# Beam Property Analysis
# Beam Intensity/Power Analysis
# Helmholtz Magnetic Strength
# Constraint Based-Optimizizer

# These small scripts/code blocks interact with the C++ CST-EM Model by generating CSV data points of the beams' x-y-z coordinates


# In[ ]:


## Beam Properties Calculation ##


def calculate_beam_properties(beam_current, beam_energy, beam_emittance):
    c = 299792458  # speed of light in m/s
    m = 9.10938356e-31  # electron mass in kg
    q = 1.60217662e-19  # electron charge in Coulombs
    
    gamma = beam_energy / (m * c ** 2)
    beta = math.sqrt(1 - 1 / gamma ** 2)
    sigma_z = math.sqrt(beam_emittance / (beta * beam_current * q))
    sigma_r = math.sqrt(beam_emittance / (beta * beam_current * q * gamma))

    total_beam_length = 4 * math.pi * sigma_z ** 2 / (beta * c)
    total_beam_radius = sigma_r

    return total_beam_length, total_beam_radius

# Example of CST-EM Modeling Input
beam_current = 1e-3  # 1 mA
beam_energy = 3e9  # 3 GeV
beam_emittance


# In[ ]:


## Beam Intensity Calculation ##
    
def calulate_beamIntensity(position)
     # Modified formula to calculate beam intensity
     # beamIntensity = [input modified beam equation]
        return beamIntensity

     # Load CSV file\n,
        with open('beam_positions.csv','r') as csv_file
        csv_reader = csv.reader(csv_file)
        #Skips the header row 
        next(csv_header)

        for row in csv_reader: 
            # Target and extract position from CSV file
            position = float(row[0])
            
            # Calculate beam intensity using function 
            beamIntensity = calulate_beamIntensity(position)
            #print results(s)
            print(f \Position: {position}, Intensity: {beamIntensity})\n,


# In[ ]:


## Helmholtz Magnetic Strength ##

    # Define Constants 
    mu_0 = 4 * np.pi * 10 **-7 Vacuum Permeability
    R = 0.1 # Coil radius in cm 
    I = 1 # Current in Amperes (A)
    N = 50 # Number of turns for Helmholtz Coil
    z = np.linespace(-0.1, 0.2, 100) # Positions to calculate 
    
    # Calculate Magnetic Field Strength \n,
    B = (mu_0* I* N* (R**2)) / ((R **1 + z **2) **(3/2))
    # Plot Final Results
    plt.plot(z, B)
    plt.xlabel('Distance (cm)')
    plt.ylabel('Magnetic Field Strength (T)')
    plt.title('Helmholtz Coils Magnetic Field Strength')
    plt.show()


# In[ ]:


## Mini Constraint Based-Optimizizer for a magnet to define the model ## 
model = ConcreteModel()

# List the decision variables
model.x1 = Var(within=NonNegativeReals)
model.x2 = Var(within=NonNegativeReals)
model.x3 = Var(within=NonNegativeReals)

# Below is the objective function to minimize
model.obj = Objective(expr=model.x1 + model.x2 + model.x3, sense=minimize)

# Constraints listed
model.con1 = Constraint(expr=0.5 * model.x1 + 1.5 * model.x2 + 2 * model.x3 >= 10)
model.con2 = Constraint(expr=model.x1 + 2 * model.x2 + 1.5 * model.x3 >= 20)
model.con3 = Constraint(expr=model.x1 + model.x2 + 2 * model.x3 >= 15)

# Solve the model using the IPOPT solver
solver = SolverFactory('ipopt')
results = solver.solve(model)

# Print the results
print(f"Objective Value: {model.obj():.2f}")
print(f)

