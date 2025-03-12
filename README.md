
# Project README

This project consists of three example： Python scripts for training, transfer learning, and inference using a Transformer-based model .

## Required Libraries
Make sure you have the following Python libraries installed:
- torch
- pandas
- scikit-learn
- numpy

## Running the Scripts
Execute the scripts in the following order:

1. **Train the initial model on MDD data**:
   ```bash
   python 1_train_model.py
   ```

2. **Perform transfer learning using MA data**:
   ```bash
   python 2_transform_learning_model.py
   ```

3. **Run inference**:
   ```bash
   python 3_inference.py
   ```

After running the inference script, the results will be saved in a file named `Predicted_SNPs.txt`.

## Note on Example Data
The example data used in these scripts is limited to SNPs from chromosome 6. This is intended for demonstration purposes, and the results do not represent the model's full performance as described in the research paper.
