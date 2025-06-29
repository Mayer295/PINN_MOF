# Physics-Informed Machine Learning for Fast Screening High Hydrogen Storage MOFs with Monotonicity Constraints
 This repository contains an open source implementation of the machine learning model described in the literature and the corresponding dataset.

![TOC graphic](https://github.com/user-attachments/assets/cb3ad311-73ca-4cec-b678-580fb8229d24)

 ## Abstract
The key contributions of this paper include:

1.A novel PINN framework that enforces physical constraints for fast MOF screening

2.The proposed PINN model exhibits not only superior prediction performance but also enhanced physics consistence and robust generalization capabilities.

3.Two experimentally synthesized materials identified by the PINN model have met DOE 2025 targets for onboard hydrogen storage, demonstrating significant real-world impact.


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

The DDEC charge calculation was performed using PACMAN

PACMAN: a Partial Atomic Charge Predicter for Porous Materials based on Graph Convolutional Neural Network. https://pacman-charge-mtap.streamlit.app

## Reference
if this work is helpful your reasearch please cite the following publication:
https://www.sci-hub.ru/10.1021/acs.jctc.5c00242  

## Note
Code and data for academic purpose only.
