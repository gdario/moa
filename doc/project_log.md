# MoA Challenge

## Introduction

- Gene expression and cell viability data.
- Pool of 100 different cell types
- MoA annotations for more than 5000 drugs.
- One approach for measuring MoA is to treat sample of human cells with the drug and then analyze the cellular responses searching for similarit to known patterns in large genomic databases.

## Data

For the training set there are three files, for the test set only one.

- train_features.csv :: contains an id, the type of compound (treatment or vehicle), treatment duration, dose (D1, D2), gene expression features (from g-1 to g-771) and the cell viability data (from c-1 to c-99).
- train_targets_nonscored.csv :: Additional and optional binary MoA responses for the training data. These are neither predicted nor scored.
- train_targets_scored :: binary MoA targets that have been scored.

For the test set there is only:

- test_features.csv :: features for the test data. You must predict the probability of each scored MoA for each row in the test data.

## Useful links

[https://clue.io/connectopedia/glossary](Connectopedia) is a free, web-based dictionary of terms and concepts related to the Connectivity Map (including definitions of cell viability and gene expression data in that context)

Corsello et al. “[https://doi.org/10.1038/s43018-019-0018-6](Discovering the anticancer potential of non-oncology drugs by systematic viability profiling),” Nature Cancer, 2020,

Subramanian et al. “[https://doi.org/10.1016/j.cell.2017.10.049](A Next Generation Connectivity Map: L1000 Platform and the First 1,000,000 Profiles),” Cell, 2017,

[https://www.kaggle.com/c/lish-moa/discussion/184005](This notebook) contains some potentially useful information. It says that the gene expression data are based on the L1000 assays, while the cell viability data are from the PRISM assay.

## Obervations

### Id and dataset sampling

The =sig_id= seems to be composed of hexadecimal numbers, apparently in increasing order.

### Row and column totals of the target data

Adding column-wise the entries in the data returns the number of responses in each given assay. Adding row-wise the entries, gives for each id the number of assays where a response was measured. There is no assay with zero responses, the minimum being 1. There are many ids where no response is measured. All 1866 vehicles and 7501 compounds have no response whatsoever.

We would expect compounds with similar profiles to have similar responses. Is this the case?

## TODO

- Understand if you need to define a new scoring function that ravels the data.
- [x] Understand the metric
- [x] Difference between OneVsRestClassifier and MultiOutputClassifier
What is the difference between the two?
- Use logging
- [x] Verify that vehicles always generate zero response
- Verify that training and test set are compatible.
- Verify that training and validation sets are compatible.
- Verify that the metric you are using is compatible with the leaderboard.
- Read the slides on feature engineering.
- Think of the most effective way of setting up your project.
- Submissions to this competition must be made through notebooks.
- Freely and publicly available external data is allowed, including pre-trained models.
- Put everything in a Docker container
- Take a look at the [https://www.kaggle.com/c/lish-moa/overview/useful-links](useful links.)
