# Supervised Learning Compendium

Final project for Introduction to Artificial Intelligence.

## Repository Structure

- `part1_regression/` - Regression task using the Auto MPG dataset
- `part2_classification` - Classification task using CIFAR-10
- `PyTorchNN.ipynb` - Neural network classification using PyTorch (3rd part).

## Datasets

- Part 1: **Auto MPG**: Publicly available at https://archive.ics.uci.edu/dataset/9/auto+mpg
- Part 2: **CIFAR-10**: Publicly available at https://www.cs.toronto.edu/~kriz/cifar.html  

## Report

The full report can be viewed here: [Overleaf Report](https://www.overleaf.com/read/hmmcwwgdxcfr#d47f7a)

## Results Summary
| Part | Model | Test Performance |
|------|-------|-----------------|
| 1 - Regression | KNN (k=5) | RMSE: 2.437, R²: 0.895 |
| 2 - Classification | Logistic Regression | Accuracy: 40.27% |
| 3 - Neural Network | 3-block CNN | Accuracy: 83.63% |

## Notes
- Saved model files are not included in this repository as they can be reproduced by running the scripts.
- All results reported in the Overleaf report are reproducible by running the provided code.
