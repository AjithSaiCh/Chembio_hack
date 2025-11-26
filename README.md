# Bioinformatics ML Project

A bioinformatics machine learning pipeline for protein sequence classification using multiple advanced techniques to handle imbalanced data.

## Overview

This Jupyter notebook implements a complete machine learning workflow for analyzing and classifying biological sequences (proteins/DNA). The project focuses on extracting bioinformatics features from sequences and using various machine learning models to make predictions, with special attention to handling imbalanced datasets.

## Features

### Data Processing
- **Unique Value Extraction**: Parse and extract unique sequences from text files
- **GC Content Calculation**: Calculate GC content for DNA/RNA sequences
- **GRAVY Score Computation**: Calculate Grand Average of Hydropathy for protein sequences

### Feature Engineering
- **Protein Analysis Features**:
  - Molecular weight
  - Aromaticity
  - Instability index
  - Isoelectric point
  - Secondary structure fractions (helix, turn, sheet)
- **Compositional Analysis**: Amino acid composition using scikit-bio
- **Automated DataFrame Creation**: Structured feature extraction pipeline

### Exploratory Data Analysis
- Statistical visualizations
- Feature distribution analysis
- Correlation studies
- Interactive plots using Plotly

### Machine Learning Models

The project implements and compares multiple classification algorithms:

1. **Decision Tree Classifier**
2. **Random Forest Classifier**
3. **XGBoost Classifier**
4. **K-Nearest Neighbors (KNN)**
5. **LSTM Neural Network** (TensorFlow/Keras)

### Handling Imbalanced Data

The notebook addresses class imbalance using two main techniques:

#### Technique 1: Moving Threshold
- Adjusts classification threshold to balance precision and recall
- Includes hyperparameter optimization using RandomizedSearchCV
- LSTM implementation with:
  - Batch Normalization
  - Dropout layers for regularization
  - Custom architecture for sequence data

#### Technique 2: BalancedBaggingClassifier
- Implements ensemble bagging with balanced sampling
- Combines with base classifiers for improved performance

### Model Evaluation
- Accuracy metrics
- Confusion matrices
- Classification reports
- ROC-AUC scores
- Model comparison framework

## Requirements

### Core Dependencies
```
Bio>=1.6.2
biopython>=1.80
scikit-bio>=0.5.9
```

### Data Science Stack
```
numpy
pandas
matplotlib
seaborn
plotly
scikit-learn
xgboost
```

### Deep Learning
```
tensorflow
keras
```

## Installation

Install all required packages:

```bash
pip install Bio scikit-bio
pip install numpy pandas matplotlib seaborn plotly
pip install scikit-learn xgboost
pip install tensorflow
```

## Usage

1. **Data Preparation**: Load your sequence data files (FASTA format or text files)

2. **Feature Extraction**: Run the feature extraction pipeline to calculate bioinformatics features

3. **EDA**: Explore your data using the visualization cells

4. **Model Training**: Train and compare multiple models using the provided framework

5. **Evaluation**: Assess model performance using various metrics

## Project Structure

The notebook is organized into the following sections:

1. **Installing dependencies** - Setup and imports
2. **Data Preprocessing** - Loading and cleaning data
3. **Feature Extraction** - Computing bioinformatics features
4. **EDA and Visualisation** - Exploratory analysis
5. **Model Selection** - Training multiple classifiers
6. **Handling Imbalanced Data** - Specialized techniques for class imbalance
7. **Evaluation** - Model performance assessment

## Key Functions

- `unq_val(file_path)` - Extract unique values from file
- `calculate_gravy(sequence)` - Calculate GRAVY hydropathy score
- `gc_cont(sequence)` - Calculate GC content percentage
- `bio_features(sequence)` - Extract comprehensive protein features
- `create_dataframe(data)` - Generate structured feature dataframe
- `model_acc(model, X_test, y_test)` - Calculate model accuracy

## Models Implemented

| Model | Type | Use Case |
|-------|------|----------|
| Decision Tree | Tree-based | Baseline, interpretable |
| Random Forest | Ensemble | Robust, feature importance |
| XGBoost | Gradient Boosting | High performance |
| KNN | Instance-based | Pattern recognition |
| LSTM | Deep Learning | Sequential patterns |

## Hyperparameter Optimization

The project includes RandomizedSearchCV for automated hyperparameter tuning across different model types, ensuring optimal performance.

## Performance Metrics

Models are evaluated using:
- Accuracy scores
- Confusion matrices
- Precision, Recall, F1-score
- ROC-AUC curves

## Google Colab

This notebook is designed to run on Google Colab. You can open it directly using the badge at the top of the notebook or by visiting:
```
https://colab.research.google.com/github/CyScar/Chembio_hack/blob/main/ChemBio.ipynb
```

## Contributing

This project is part of the Chembio_hack repository. Contributions and improvements are welcome.

## Notes

- The notebook handles both protein and nucleotide sequences
- Special attention is given to imbalanced datasets, a common challenge in bioinformatics
- Multiple modeling approaches allow for comprehensive comparison
- Feature scaling is applied where appropriate using StandardScaler

## License

Please refer to the repository license for usage terms.

## Author

Repository: [CyScar/Chembio_hack](https://github.com/CyScar/Chembio_hack)
