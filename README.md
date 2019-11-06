### Analysis of diffrent recommendation engines for movie recommendation
#### Personalization: Theory & Applications Project 1
#### IEOR 4571
#### Fall 2019
_____________________________________________________________________________________________________________________________


##### Authors: 
1. [Chandrasekaran Anirudh Bhardwaj](https://github.com/anirudhsekar96) ()

2. [Ridhi Mahajan](https://github.com/rmahajan14) (rm3601)

3. [Sheetal Reddy](https://github.com/Sheetalkreddy) (kr2793)

4. [Akshat Mittal](https://github.com/aksmit94) (amxxxx)
_____________________________________________________________________________________________________________________________

> In this project we perform an in-depth analysis of different algorithms for movie recomendation. 

The report is in [Final.ipynb](./Final.ipynb)

Code is structured as follows

    .
    ├── utils
    |     ├── data_loader.py			# Load data & Sampling functions
    │     └── yapf_format.py			# pep8 code standard
    |
    ├── model
    |     ├── baseline_model.py			# Bias based model
    |     ├── als_model.py				# Alternating Least Squares based Matrix Factorization
    |     ├── lightfm_model.py			# LightFm
    │     └── nearest_neighbor_model.py		# Nearest Neighbors model with Z-score scaling of users
    ├── data                     			# Data files
    ├── cache                    			# Data cache used to avoid re-reading the data each time model changes are made
    ├── Final.ipynb                   		# Report Markdown
    ├── LICENSE
    └── README.md



