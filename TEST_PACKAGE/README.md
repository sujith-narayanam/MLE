# Median housing value prediction

The housing data can be downloaded from https://raw.githubusercontent.com/ageron/handson-ml/master/. The script has codes to download the data. We have modelled the median house value on given housing data.

The following techniques have been used:

- Linear regression
- Decision Tree
- Random Forest

## Steps performed

- We prepare and clean the data. We check and impute for missing values.
- Features are generated and the variables are checked for correlation.
- Multiple sampling techinuqies are evaluated. The data set is split into train and test.
- All the above said modelling techniques are tried and evaluated. The final metric used to evaluate is mean squared error.

## Crating environment

- Open anaconda terminal or command prompt.
- Run 'conda activate base' if not activated by default.
- Run 'conda env create -f env.yml' to create environment named mle-dev.
- Run 'conda activate mle-dev' to activate the environment.
- Refer to https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html for any issues with creating environment.

## To excute the script

- Run "python nonstandardcode.py" in terminal after activating the environment mle-dev.

## Building the package

* The files needed to build the package are all made and are ready in the folder.
* Run the following command to build the package.

  `python -m build`
* Use the files in the dist folder to install the package.

## Installing the package

* Get the `test_package_ml-0.2.tar.gz` file from the `dist` folder.
* Extract it using xtarfile package as follows
  `	tar xsf test_package_ml-0.2.tar.gz`
* Go inside the extracted folder and install the package using
  `python setup.py install`
* To check the installation of the package, run the pytests from the tests folder as

  `pytest tests/`
* The installed packages can be used in the environment as

  `import test_package`
