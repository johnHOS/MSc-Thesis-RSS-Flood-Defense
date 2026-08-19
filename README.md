# MSc-Thesis-RSS-Flood-Defense
Python Notebook files supporting the MSc thesis "Assessment of reinforced soil in flood defense systems" at TU Delft.

# Assessment of reinforced soil in flood defense systems

This repository contains the Python implementation used for the reliability assessment performed as part of the MSc thesis **“Assessment of reinforced soil in flood defense systems”**.

The repository serves as the digital appendix to the thesis. Only an overview of the computational framework is included in the written report; the complete Python scripts and calculation notebooks are provided here to improve transparency and reproducibility of the reliability assessment.

## Reliability assessment

The reliability framework is divided into separate Python files to distinguish between:

* deterministic design parameters;
* limit state functions;
* reliability methods;
* hydraulic joint probability modelling; 
* processing and visualisation of the numerical results.

The reliability assessment includes deterministic unity checks, the First Order Reliability Method (FORM), and Monte Carlo Simulation (MCS), depending on the considered failure mechanism.

The probabilistic calculations are primarily performed using the Python package **OpenTURNS**.

## Repository structure

| File                                | Description                                                                                                                                                                                                     |
| ----------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `Parameters.py`                     | Defines the deterministic input values, design variables, stochastic variables, system failure probability, failure probability budget, and target failure probabilities for the individual failure mechanisms. |
| `Probabilistic.py`                  | Contains the general reliability functions. It creates the OpenTURNS input distributions, defines the failure event (Z < 0), and contains the functions used for FORM and Monte Carlo Simulation.               |
| `Overtopping_joint_dist_updated.py` | Derives the joint probability distribution between significant wave height and still water level. The fitted copula is used in the reliability assessment of wave overtopping.                                  |
| `Lsf_overtopping.py`                | Contains the limit state and verification functions associated with overflow and wave overtopping.                                                                                                              |
| `Lsf_erosion.py`                    | Contains the limit state functions for toe protection, including top-layer stability and the required protection width.                                                                                         |
| `Lsf_external.py`                   | Contains the limit state functions for external stability of the reinforced soil structure, including vertical bearing capacity, sliding, and rotational stability.                                             |
| `Lsf_internal.py`                   | Contains the limit state functions for internal stability, including reinforcement rupture and reinforcement pull-out for the individual reinforcement layers.                                                  |
| `Main-FORM.ipynb`                   | Runs the FORM reliability analyses and returns the probability of failure, reliability index, design point, and importance factors of the stochastic variables.                                                 |
| `Output.py`                         | Processes the numerical results and generates plots and output tables used in the reliability assessment.                                                                                                       |


## Author

**Johannes Thiruchelvam**
MSc Civil Engineering (Hydraulic and Offshore structure engineering)
Delft University of Technology

## Academic purpose
This repository was created as part of my MSc thesis and is provided for academic transparency. The calculations remain subject to the assumptions, limitations, and scope described in the thesis report.
