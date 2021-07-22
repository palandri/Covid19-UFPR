# Covid19-UTFPR

This project is the code used to generate the results presented in ##PENDING PAPER##.  
Database used: https://github.com/ari-dasci/OD-covidgr

### preprocess_database.ipynb
This notebook loads and preprocess the images from the database, and save the results in the files "images/[data, labels, lb].pkl". Each .pkl file contain only the reffered dataset.

### process_[dense, eff, mobile, res, vgg].ipynb
These notebooks contains the driver code to each of the networks. They are all very similar, the difference is basically the network generated.

### src/generic/generic.py
This file contains all the functions and helpers to generate and load .pkl files, train, predict, crossvalidate, etc.

## Cite Covid19-UTFPR
Placeholder
