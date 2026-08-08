# Wine Classification and Dimensionality Reduction using PCA

A data mining project that applies Principal Component Analysis (PCA) to the Wine dataset to explore dimensionality reduction, visualize the data in a lower-dimensional space, and evaluate its effect on classification performance.

## Project Overview

The Wine dataset contains 178 samples described by 13 numerical features and divided into three different classes.

The main objectives of this project are to:

- Load and preprocess the Wine dataset.
- Standardize the numerical features before applying PCA.
- Apply Principal Component Analysis to reduce dimensionality.
- Analyze the cumulative explained variance of the principal components.
- Visualize the dataset using a two-dimensional PCA projection.
- Determine the number of components required to retain at least 95% of the variance.
- Compare Logistic Regression classification performance before and after dimensionality reduction.

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib

## Project Structure

```text
data_mining_project/
├── data/
│   └── wine.data.csv
├── data_loader.py
├── preprocessing.py
├── pca_analysis.py
├── visualisation.py
├── main.py
└── README.md
```

### File Description

- `data_loader.py` – Loads the Wine dataset and prepares the feature and target variables.
- `preprocessing.py` – Standardizes the dataset before PCA.
- `pca_analysis.py` – Performs PCA and calculates explained variance.
- `visualisation.py` – Generates the cumulative explained variance and 2D PCA plots.
- `main.py` – Runs the complete analysis and classification workflow.
- `data/wine.data.csv` – Dataset used for the analysis.

## PCA Analysis

Principal Component Analysis is used to transform the original 13 features into a smaller set of principal components while retaining as much of the original information as possible.

The project analyzes the cumulative explained variance to determine how many principal components are required to preserve at least 95% of the dataset's variance.

A two-dimensional PCA projection is also generated to visualize the distribution of the three wine classes.

## Classification

Logistic Regression is used to evaluate classification performance in two scenarios:

1. Using the original standardized features.
2. Using the PCA-transformed features.

This comparison demonstrates how dimensionality reduction affects the predictive performance of the classifier.

## Results

The analysis produced the following results:

- Dataset size: **178 samples**
- Original features: **13**
- Wine classes: **3**
- Components required for approximately 95% explained variance: **10**
- Variance explained by the first two principal components: **~55.4%**
- Logistic Regression accuracy using original features: **~97.78%**
- Logistic Regression accuracy after PCA: **~97.78%**

The results show that PCA reduced the dimensionality of the dataset while maintaining the same classification accuracy in this experiment.

## Running the Project

Clone the repository and navigate to the project directory:

```bash
git clone <repository-url>
cd data_mining_project
```

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the required dependencies:

```bash
pip install numpy pandas scikit-learn matplotlib
```

Run the project:

```bash
python main.py
```

The program will display the analysis results in the terminal and generate visualizations for the cumulative explained variance and the 2D PCA projection.

## Key Concepts

This project demonstrates several fundamental data mining and machine learning concepts, including:

- Data preprocessing
- Feature standardization
- Principal Component Analysis (PCA)
- Dimensionality reduction
- Explained variance
- Data visualization
- Logistic Regression
- Classification evaluation

## Dataset

The project uses the Wine dataset, which contains chemical analysis measurements of wines belonging to three different classes.

## Author

Noel Zani
