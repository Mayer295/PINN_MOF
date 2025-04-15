# Physics-Informed Machine Learning for Fast Screening High Hydrogen Storage MOFs with Monotonicity Constraints
 This repository contains an open source implementation of the machine learning model described in the literature and the corresponding dataset.

 ## Abstract
Hydrogen represents a highly promising alternative energy vehicle that facilitates the transition toward a carbon-neutral economy. However, the current hydrogen storage technologies are both inefficient and costly, which significantly limits the overall efficiency of hydrogen energy systems. The method of storing hydrogen via physical adsorption in Metal-Organic Frameworks (MOFs) has emerged as a particularly promising solution for hydrogen storage, attributed to their extensive surface area and remarkable tunability. Machine learning techniques have demonstrated great potential in screening a theoretically limitless number of MOFs that meet specific hydrogen storage targets, but these models often lack physical consistency. To address this limitation, this study presents a Physics-Informed Neural Network (PINN) that incorporates monotonic relationships between the crystallographic characteristics and the hydrogen storage capacity into the neural network architecture to effectively screen the high hydrogen storage capacity MOFs. The performance of the identified top MOFs were validated by the Grand Canonical Monte Carlo simulations.  Compared to other machine learning methods, the top MOFs identified by the PINN exhibit meaningful crystallographic features via heatmap analysis. The proposed PINN model successfully identified two experimentally synthesized MOFs, LADQEM_CSD17 and LADQEM01_CSD17, that meet the targets for onboard hydrogen storage systems established by the U.S. Department of Energy for the year 2025.

 ## Model
 The Model folder contains the training code for seven different machine learning models: Random Forest (RF), Extremely Randomized Trees (Extra Trees/ERT), LightGBM, XGBoost(XGB), Support Vector Machine (SVM), Neural Network (NN), and Physics-Informed Neural Network (PINN).In addition to the PINN, five other machine learning models—RF, LightGBM, XGB, SVM, and NN—are considered for comparative analysis. To optimize the performance of these five ML models, Optuna is utilized to identify the optimal hyperparameters, which is a machine learning tool grounded in Bayesian optimization. 
 
The ERT model originates from this paper:https://www.sciencedirect.com/science/article/pii/S2666389921001240
 

 ## Dataset
 The dataset utilized in this study comprises 98,695 MOFs, as detailed by Ahmed et al, and was sourced from the MOF database maintained on the Hydrogen Materials Advanced Research Consortium (HyMARC) platform. 
 https://datahub.hymarc.org/dataset/computational-prediction-of-hydrogen-storage-capacities-in-mofs
 
 The ps_usable_hydrogen_storage_capacity_gcmcv2 data is used for model training.
 
 The 820k_mofs_crytal_props_ml_h2_ps_v17feb2021 data is used for fast screening

## GCMC
After Screening, GCMC simulations of the MOFs were performed using the RASPA package to compare with the predictions of the PINN model. The framework atoms of the porous materials were modeled using the Universal Force Field (UFF), with atomic charges derived from the Density Electrostatic and Chemical (DDEC) method. Interactions between the framework atoms and H₂ molecules were characterized using the Lennard-Jones (LJ) potential, incorporating Coulombic interactions, with cross LJ parameters calculated using the Lorentz-Berthelot combining rules. Lennard-Jones (LJ) potential was modified to account for the quantum effects of hydrogen molecules at cryogenic temperatures employing the Feynman-Hibbs adjustments. The H₂-H₂ interactions were modeled with the three-site Darkrim-Levesque potential. All calculations were carried out using a 12.8 Å cut-off radius with compensating long-range corrections.GCMC simulations were executed over a total of 2000 cycles, consisting of 1000 initialization cycles and 1000 production cycles focusing on computing H₂ adsorption properties. Each GCMC cycle consisted of translation, rotation, insertion, and deletion moves with equal probabilities. Detailed LJ parameters for H₂ and the MOFs are presented in Tables S1 and S2 of the Supporting Information.

The DDEC charge calculation was performed using MEPO-ML: a robust graph attention network model for rapid generation of partial atomic charges in metal-organic frameworks.
https://www.nature.com/articles/s41524-024-01413-4


## Note
Code and data for academic purpose only.
