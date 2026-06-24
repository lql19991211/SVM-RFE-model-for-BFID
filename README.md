# SVM-RFE-model-for-BFID
A multi-label classifier that can be used to infer the tissues origin of body fluids in forensic sciences.
# Usage:
(1) '1.data_simulation.py': This script is designed to generate simulated mixture data using large-scale sequencing read data derived from real body fluid samples.  
(2) '2.SVM-RFE_pipeline.py': This script includes the complete pipeline for model training, evaluation (using both independent test sets and external validation sets), and SHAP (SHapley Additive exPlanations) analysis.  
(Note: The training data, validation data, and simulated mixture datasets can be downloaded directly from the supplementary files of the published paper.)

# Notes:
The SVM-RFE classifier constructed in this repository—based on the provided dataset—is specifically validated only for the identification of the following body fluids and their mixtures:
* Peripheral Blood (PB)
* Saliva (SA)
* Menstrual Blood (MB)
* Vaginal Secretions (VS)
* Semen (SE)

Any other types of biological stains or fluids not listed above will be classified as "Unknown".

# Contact information:
2534312024@qq.com
