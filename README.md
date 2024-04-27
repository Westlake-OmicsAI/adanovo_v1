# Installation
## Create a conda environment

First, open the Linux terminal. All of the commands that follow should be entered into this terminal. 
To create a new conda environment for Adanovo and activate this environment, run the following commands:

```
conda create --name adanovo python=3.10
conda activate adanovo
```
## Install package
Enter the working directory and install the corresponding environment dependencies for running Adanovo:

```
cd adanovo_v1
pip install -r requirements.txt
```

# Running
## Training
For training a Adanovo model from scratch,
```
python adanovo.py --mode=train --peak_path=case.mgf --peak_path_val=case.mgf --config=config.yaml --output=log_file/case3
```
Ensure that your training data (e.g. in. mgf format) is similar in format to the case.mgf provided here.
In config.yaml, you can modify the save path of the model, the number of epochs trained, the types of residues in the training data, and the specified AA with PTM types (which must be a subset of the previous residues types)

## Validation
For validating a trained model,
```
python adanovo.py --mode=eval --model=xx.ckpt --peak_path=case.mgf --config=config.yaml --output=log_file/case3
```
Replace xx.ckpt with the path of the model you have already trained.
Replace case.mgf with the dataset you are preparing to validate.

## Sequencing
For sequencing peptides from mass spectra in an MGF file
```
python adanovo.py --mode=denovo --model=xx.ckpt --peak_path=case.mgf --config=config.yaml --output=log_file/case4
```
Replace xx.ckpt with the path of the model you have already trained.
Replace case.mgf with the dataset you are preparing to sequence.
