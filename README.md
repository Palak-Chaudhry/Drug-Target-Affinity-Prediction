# Impact of Graph Representation of Proteins in Drug-Target Affinity Prediction

This project investigates the application of graph representation techniques to improve understanding of drug-protein interactions using deep learning methodologies. We utilize Evolutionary Scale Modeling 3 (ESM3) and advanced neural network architectures to capture complex drug-protein interactions more precisely than existing methods.

## Table of Contents
- [Introduction](#introduction)
- [Data Preparation](#data-preparation)
- [Methods](#methods)
- [Results](#results)
- [Conclusion](#conclusion)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Introduction

Drug-target binding affinity (DTA) prediction is crucial for drug discovery but remains challenging due to the complexity of protein folding and molecular interactions. This project addresses limitations in current computational approaches by:

1. Developing a more precise computational approach for predicting drug-target binding affinity
2. Utilizing advanced graph representation techniques for a comprehensive understanding of molecular interactions
3. Leveraging Evolutionary Scale Modeling 3 (ESM3) to generate three-dimensional protein graph representations

## Data Preparation

We use the DAVIS dataset, which contains:
- 68 drugs
- 379 targets
- 25,771 interactions

Our data preparation pipeline includes:

1. Drug Molecule Feature Engineering
   - Creating molecular graphs where atoms are nodes and bonds are edges
   - Incorporating 78 features for each node (atom)

2. Protein Feature Engineering
   - Utilizing ESM3 to generate 3D conformations for protein sequences
   - Converting protein structures to graphs with 33 features for each node (residue)

## Methods

We implement and compare several model architectures:

1. Graph Convolutional Networks (GCN) - Baseline
2. Graph Attention Networks (GAT)
3. Hybrid models (GAT-GCN)
4. Interaction Learning with HyperAttention

## Results

Our experiments show:

1. The GAT-GCN model achieves the lowest Mean Squared Error (MSE) and highest Concordance Index (CI)
2. Feature engineering significantly improves model performance
3. Interaction learning does not boost performance over vanilla concatenation of representations

## Conclusion

Our approach, leveraging graph-based representations and deep learning architectures, addresses limitations of sequence-based methods and achieves competitive performance on the DAVIS dataset.

## Installation

```bash
git clone https://github.com/your-username/drug-target-affinity-prediction.git
cd drug-target-affinity-prediction
pip install -r requirements.txt
```

## Usage

```python
# Example code to run the model
from model import GATGCN
from data_loader import load_davis_dataset

# Load data
drugs, proteins, affinities = load_davis_dataset()

# Initialize and train model
model = GATGCN()
model.train(drugs, proteins, affinities)

# Make predictions
predictions = model.predict(new_drugs, new_proteins)
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License.

Citations:
[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/32279206/caac9f3e-54c6-4d9e-aced-00a31bbb9c7c/Project_Milestone_Report.pdf
