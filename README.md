# Lantern Pharma: Blood-Brain-Barrier Permeability
**Lantern Pharma** presents this public repository for predicting a drug's ability to penetrate the blood brain barrier for Theraputic Data Commons BBB-Martins et al leaderboard

Information and data used for the project, can be found on the following TDC site:
https://tdcommons.ai/single_pred_tasks/adme/#bbb-blood-brain-barrier-martins-et-al

**Dataset Description:** As a membrane separating circulating blood and brain extracellular fluid, the blood-brain barrier (BBB) is the protection layer that blocks most foreign drugs. Thus the ability of a drug to penetrate the barrier to deliver to the site of action forms a crucial challenge in development of drugs for central nervous system From MoleculeNet.

**Task Description:** Binary classification. Given a drug SMILES string, predict the activity of BBB.
Dataset Statistics: 1,975 drugs.

**Approach:**
SMILES strings for each drug were transformed into machine learning features using RDKit tools, more information on the package resources is located here:
https://rdkit.org

# Installation

**Clone this repository as follows:**

```
git clone https://github.com/lanternpharma/tdc-bbb-martins
```

**Use conda to install all the required dependencies in a virtual environment. Use the `env.yml` file to create and activate the environment as follows.**

```
conda env create -f environment/conda_env.yml 
conda activate tdc
```

# Models included in this repository:
Each model can be run independently as the associated features are generated within each modeling python script. 

**Logistic Regression:** contained in the logistic_fe_aug.py script, this model uses fingerprints generated by RDKit as primary features along with additional engineered features and data augmentation for feature selection and model training

**Deep Neural Network:** contained in the dnn_fe_aug.py script, this model uses fingerprints, autocorrelations, and descriptors associated with known drug filters generated by RDKit as primary features along with additional engineered features and data augmentation for feature selection and model training

**Random Forest:** contained in the rf_fe_aug.py script, this model uses fingerprints, autocorrelations, and descriptors associated with known drug filters generated by RDKit as primary features along with additional engineered features and data augmentation for feature selection and model training

**Ensemble:** contained in the ensemble.py script, this model uses the logistic regression and deep neural network models above as base learners that feed into a second-level logistic regression meta-learner to ensemble predictions

# Running the models to reproduce results

With conda environment activated, change to the "code" directory then execute the shell script as follows to run all available model types:
```
sh run
```

The default run took about 8 hours on a MacBook M1 with 16 Gb RAM circa (March 2023).

If you would like to run just one model instead of all, the following run scripts are also available:

```
run_logistic
run_dnn
run_rf
run_ensemble
```

# Summary of model results

<table>
    <thead>
        <tr>
            <th>Model</th>
            <th>Evaluation</th>
            <th>Performance</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td colspan=4 style="text-align: center;"></td>
        </tr>
        <tr>
            <td>ensemble</td>
            <td>AUROC</td>
            <td>0.962 &#177; 0.003</td>
        </tr>
        <tr>
            <td>logistic_fe_aug  </td>
            <td>AUROC</td>
            <td>0.956 &#177; 0.006</td>
        </tr>
        <tr>
            <td>dnn_fe_aug</td>
            <td>AUROC</td>
            <td>0.949 &#177; 0.004</td>
        </tr>
        <tr>
            <td>rf_fe_aug</td>
            <td>AUROC</td>
            <td>0.928 &#177; 0.002</td>
        </tr>
        </tr>
        <tr>
    </tbody>
</table>

## License
Copyright (c) 2023 Lantern Pharma Inc.

This project is licensed under the GNU Affero General Public License version 3.0. See the [LICENSE](LICENSE) file for license rights and limitations (AGPLv3).