# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).



## [Release 2.0] - yyyy-mm-dd

Updates centered around removing data leakage issue during feature selection in Release 1.0. After correcting data leakage, additional measure were taken to improve the score/performance

### Added 

1. Script to generate the features once rather than in each modeling script. Also added Getaway and WHIM 3D descriptos to feature set
2. Script to calculate feature correlations (binary vs binary, and numeric vs. numeric) then save a list of features to drop prior to feature selection
3. Added SVM as an additional modeling method, also include it as a candidate model in ensembling

### Changed

1. Feature Selection: Optimized grid search method to get higher resolution in less time per run. Then increase feature selection to 20 folds for more stability. Now only run feature selection a single time on training/validation combined, instead of separate selections for each of the 5 seed train splits. 
2. Logistic Regression: Transformed features with kernel PCA prior to modeling. Separated hyperparameter search into it's own function, and run it once with 20 folds on train/val combined. Then use best parameters to train models on all 5 seeds rather than have differently tuned models for each seed
3. Random Forest: Separated hyperparameter search into it's own function, and run it once with 20 folds on train/val combined. Then use best parameters to train models on all 5 seeds rather than have differently tuned models for each seed
4. DNN: Transformed features with kernel PCA prior to modeling. Created large Optuna search that functions as an optimal DNN designer, run it on 10 folds of train/val then use that same architecture and hyperparameters to train on all 5 seeds. Note that this "DNN designer" code is not included in this repo and is not run here. The optimal parameters from outside study have been set as fixed here to control run time.
5. Ensemble: Added SVM to candidate base learners. Model selection changed from recursive elimination to best subset found with forward selection
6. Changed environment (both Code Ocean and the conda_env.yml file) to include the mlxtend package

### Fixed

1. Removed use of “get_train split” TDC loader, to prevent issue where TDC included many test samples in this set. Now all scripts only use “benchmark split” method

## [Release 1.0] - 2023-03-24

Includes logistic regression, deep neural network, and random forest models independently as well as an ensemble model using the logistic and dnn models as base learners. All 4 models rank at the top of the TDC BBB-martins leaderboard as of release date
