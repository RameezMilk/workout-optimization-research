# GNN-Based Personalized Workout Optimization System

### Author

**Rameez E. Malik**
> Department of Computer Science

> North Carolina State University

> Email: [remalik@ncsu.edu](mailto:remalik@ncsu.edu)

---

## Overview

This repository contains the implementation, experimental analysis, and report for the project **"Graph Neural Network-Based Personalized Workout Optimization System"**.
The system applies **Graph Neural Networks (GNNs)** to model inter-exercise relationships and predict optimized workout parameters (next-session weight, repetitions, and rest) using relational graph structures.

The baseline model is a **two-layer Graph Convolutional Network (GCN)**, while the proposed model is a **GraphSAGE** architecture designed for inductive generalization to unseen exercises.

All experiments were performed in Google Colab using PyTorch Geometric, with supporting data preprocessing and visualization components written in Python.

---

## Repository Structure

```
project-root/
│
├── figures/
│   ├── fig_shap_weight.png
│   ├── fig_shap_reps.png
│   ├── fig_shap_rest.png
│   ├── graph_structure_example.png
│
├── references/
│   ├── FitRec_Research_Paper.pdf
│   └── PLOS_ONE_Research_Paper.pdf
│
├── reports/
│   ├── Deep_Learning_Project_Proposal.pdf
│   └── Deep_Learning_Final_Project_Report.pdf
│
├── slidedeck/
│   └── Workout_Optimization_Research_Slidedeck.pdf
│
├── src/
│   └── Latest_Workout_Optimization_Research.ipynb
│
└── README.md
```

---

## Code Organization and Descriptions

All implementation files are contained in the `src/` directory.
The core research and code development were done by **Rameez Malik**.
Each component was written, tested, and documented by the author in Colab, then exported to `.py` and `.ipynb` formats.

### **1. Data Preprocessing and Feature Engineering**

**Key Functions:**

* `compute_overload_targets(row)`:
  Generates next-session targets (`y_next_weight`, `y_next_reps`, `y_next_rest`) based on fatigue and RIR logic inspired by sports science literature.

* Derived metrics:

  * `training_load = volume × intensity`
  * `training_monotony = load / std(load)`
  * `strain = weekly_load × monotony`
  
These variables quantify training variability and stress, grounding model features in physiological science (from PLOS ONE, 2019).

**Purpose:**
Transform raw workout logs into normalized numerical tensors suitable for graph-based learning.

**Developed by:** Rameez Malik

---

### **2. Graph Construction**

**Key Components:**

* Uses **NetworkX** to build a heterogeneous exercise graph:

  * **Same-day edges:** Connect exercises within the same session (shared fatigue/recovery).
  * **Temporal edges:** Connect identical exercises across weeks to track progression.
* Converts the graph to a PyTorch Geometric `Data` object using `from_networkx()`.

**Purpose:** Define relational graph structure ( G = (V, E) ) for GNN message passing.

**Developed by:** Rameez Malik

---

### **3. Model Definitions**

**File:** Implemented directly within notebook (`Latest_Workout_Optimization_Research.ipynb`).

#### **Baseline GCN**

```python
class BaselineGCN(nn.Module):
    def __init__(self, in_channels, hidden=64, out_channels=3):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden)
        self.conv2 = GCNConv(hidden, hidden)
        self.fc_out = nn.Linear(hidden, out_channels)
```

**Purpose:**
Implements a 2-layer Graph Convolutional Network mirroring FitRec’s 2-layer LSTM depth for fair baseline comparison.

#### **Proposed GraphSAGE**

```python
class ProposedGraphSAGE(nn.Module):
    def __init__(self, in_channels, hidden=64, out_channels=3):
        super().__init__()
        self.sage1 = SAGEConv(in_channels, hidden, aggr='mean')
        self.sage2 = SAGEConv(hidden, hidden, aggr='mean')
        self.fc_out = nn.Linear(hidden, out_channels)
```

**Purpose:**
Implements an inductive GNN that uses neighborhood aggregation to generalize to unseen exercises.

**Developed by:** Rameez Malik

---

### **4. Training and Evaluation**

**Function:** `train_model(model, data, optimizer, criterion, epochs=400)`
Performs forward and backward propagation, prints loss every 20 epochs, and updates weights using Adam optimizer.

**Metrics Used:**

* **Mean Absolute Error (MAE)**
* **Root Mean Square Error (RMSE)**

These were computed on both normalized and real scales for fair interpretability.

**Developed by:** Rameez Malik

---

### **5. Visualization and Interpretability**

**Plots Generated:**

* **Predicted vs Actual Scatter Plots:** Compare model outputs for each target variable.
* **SHAP Feature Importance:** Highlights the most influential features (training load, monotony, fatigue).

**Purpose:**
Visualize predictive reliability and physiological interpretability of the model.

**Developed by:** Rameez Malik

---

## Running the Code

### **1. Environment Setup**

All experiments were conducted on **Google Colab** using **Python 3.10** and **PyTorch Geometric**.

Install required libraries:

```bash
!pip install torch torchvision torchaudio
!pip install torch-geometric networkx pandas matplotlib scikit-learn shap
```

### **2. Execute Notebook**

Run the Jupyter Notebook in order (`src/Latest_Workout_Optimization_Research.ipynb`).
Each section corresponds to a logical stage of the pipeline:

1. Environment Setup
2. Data Preprocessing and Feature Engineering
3. Graph Construction
4. Model Definitions (GCN and GraphSAGE)
5. Training and Evaluation
6. Metric Calculation and SHAP Visualization

### **3. Running from Python Script**

To execute from `.py` file instead:

```bash
python src/latest_workout_optimization_research.py
```

Ensure that the dataset file (Google Sheets CSV export) is accessible via the URL specified inside the script.

---

## Authored Code Summary

| Component                                  | Description                                                           | Developed by |
| ------------------------------------------ | --------------------------------------------------------------------- | ------------ |
| Data preprocessing and feature engineering | Added computation of load, monotony, strain, and overload targets     | Rameez Malik |
| Graph construction and PyTorch conversion  | Created NetworkX graph and edge logic                                 | Rameez Malik |
| Baseline and proposed models               | Implemented GCN and GraphSAGE architectures                           | Rameez Malik |
| Training, evaluation, and visualization    | Implemented custom training loop, metrics, and interpretability tools | Rameez Malik |
| Report and figures                         | Authored NeurIPS-style report and SHAP/graph figures                  | Rameez Malik |

---

## Citation References

* Ni, J., Muhlstein, L., and McAuley, J. (2019). *Modeling heart rate and activity data for personalized fitness recommendation.* Proceedings of the 2019 World Wide Web Conference (WWW '19), 1343–1353.
* Foster, G. G., Smith, R. E., and Jones, T. M. (2019). *Quantifying training load and monotony in athletes: A week-to-week analysis of performance variation.* PLOS ONE, 14(3), e0213562.

---

## Notes

* The codebase, figures, and reports were entirely developed by **Rameez Malik**.
* All figures and tables in the report are generated directly from this codebase.
* This project satisfies the deliverable criteria for the **CSC 592/Deep Learning Final Project** under the NeurIPS report format.
