# Polymerization-Aware Machine Learning (PA-ML) Framework for Self-Healing Polyurethanes


## 📄 Overview

Self-healing polyurethanes (PUs) typically suffer from an intrinsic trade-off between mechanical strength (requiring dynamic chain restriction) and healing efficiency (requiring dynamic chain mobility). 

Conventional polymer informatics relies on monomer-level SMILES or fingerprints, failing to capture polymerization-induced structural features such as chain-segment sequences, hard/soft segment distributions, and local hydrogen-bonding networks.

To address these limitations, we introduce the **Polymerization-Aware Machine Learning (PA-ML)** framework:

1. **"Polymerization Process" Descriptors:** Encodes prepolymer architectures, representative oligomeric chain fragments (`Fragment1`, `Fragment2`, `Fragment2*`), and local intermolecular environments directly into the descriptor space.
2. **Multi-Objective Virtual Screening:** Screens over 30,000 candidate formulations across monomer combinations and feed ratios using an optimized Gradient Boosting Regression (GBR) surrogate model and a composite geometric index ($f$).
3. **Experimental Breakthrough:** Guided by PA-ML, we synthesized **PUSA**, an elastomer demonstrating an extraordinary combination of **101.86 MPa** tensile strength ($T_s$) and **87.6%** healing efficiency ($\eta$) at 40 °C after 24 hours.

---

## 📂 Repository Structure

```
.
├── Data, ML model/
│   ├── dataset.csv                  # Main experimental dataset extracted via LLM & manual curation
│   ├── Ts.csv                       # Cleaned input dataset for Tensile Strength modeling
│   ├── Eb.csv                       # Cleaned input dataset for Elongation at Break modeling
│   ├── η.csv                        # Cleaned input dataset for Healing Efficiency modeling
│   ├── GBR.py                       # Gradient Boosting Regression modeling pipeline
│   ├── RF.py / SVR.py / GPR.py ...  # Alternative baseline model implementations
│   ├── Sensitivity analysis.py      # Noise injection sensitivity evaluation
│   ├── Importance verification.py   # Descriptor feature importance ablation scripts
│   ├── stability evaluation.py      # Multi-seed stability & MAE distribution checks
│   ├── feature nosie.py               # feature nosie
│   └── Gaussian-noise augmentation.py # Robustness analysis via data augmentation
├── Model interpretation/
│   ├── SHAP analysis.py             # SHapley Additive exPlanations feature analysis
│   └── Fragment analysis.py         # BRICS "Polymer Gene" enrichment and Z-score statistics
├── polymerization process descriptors/
│   ├── Fragment1.py                 # Simplest prepolymer/segment structure generator
│   ├── Fragment2.py / Fragment2'.py               # Extended oligomeric segment generator (Fragment2/Fragment2*)
│   ├── Feature Selection.py         # Variance, correlation, and feature importance filter
│   ├── molecular descriptors.py     # RDKit chemical descriptor extractor
│   ├── example.csv                  # sample SMILES for formulations
│   └── Fragment example.py          # Fragment generator
├── requirements.txt                 # Python dependencies
└── README.md                        # Project documentation
```

---

## 📥 Data Input Paths & Structure

Because the Python scripts do not hardcode input data internally, inputs are passed via structured CSV files. 

### 1. Data File Paths
* **Primary Raw Dataset:** `Data, ML model/dataset.csv`
* **Property-Specific Modeling Inputs:**
  * Tensile Strength ($T_s$): `Data, ML model/Ts.csv`
  * Elongation at Break ($E_b$): `Data, ML model/Eb.csv`
  * Healing Efficiency ($\eta$): `Data, ML model/η.csv`

## ⚙️ Feature Generation Workflow

The descriptor generation process constructs **Polymerization Process Descriptors** by dynamically transforming monomer structures into representative oligomer segment fragments before calculating 2D/3D chemical properties.

```
[Monomer SMILES & Ratios]
       │
       ▼
[Fragment Builder: Fragment1.py / Fragment2.py]
       │  (Generates prepolymer & chain-segment oligomer SMILES)
       ▼
[RDKit Feature Calculation: molecular descriptors.py]
       │  (Extracts 2D/3D physicochemical descriptors)
       ▼
[Feature Selection: Feature Selection.py]
       │  (Removes low-variance & highly correlated features)
       ▼
[Final Model Input Feature Matrix]
```

---

## 🛠️ Installation & Setup

### Prerequisites
* Python 3.8.18
* [RDKit](https://www.rdkit.org/)

### Environment Setup

1. Clone the repository:
   ```bash
   git clonehttps://github.com/chemwangl/Code-for-PAML.git
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

---

## 🚀 Usage Examples & Step-by-Step Workflow

### Example 1: Generate Descriptors from Input Monomers

1. Define or check your target monomer SMILES and ratios in `polymerization process descriptors/example.csv` or your input CSV.
2. Build representative polymer chain fragments (`Fragment2` / `Fragment2*`):
   ```bash
   python "polymerization process descriptors/Fragment2.py"
   ```
   Example:
   ```bash
   python "polymerization process descriptors/Fragment example.py"
   ```
3. Compute RDKit molecular descriptors for the generated fragment SMILES:
   ```bash
   python "polymerization process descriptors/molecular descriptors.py"
   ```
4. Perform feature selection to generate the finalized dataset:
   ```bash
   python "polymerization process descriptors/Feature Selection.py"
   ```

---

### Example 2: Train & Evaluate Surrogate Models

To train the Gradient Boosting Regression (GBR) model using `Data, ML model/Ts.csv` as the target input:

```bash
python "Data, ML model/GBR.py"
```

To run a stability evaluation over 100 random train-test splits:

```bash
python "Data, ML model/stability evaluation.py"
```

---

### Example 3: Model Interpretability & Feature Importance

Run SHAP analysis to assess descriptor contributions across dataset inputs:

```bash
python "Model interpretation/SHAP analysis.py"
```

Extract dynamic fragment "Polymer Genes" using BRICS fragmentation:

```bash
python "Model interpretation/Fragment analysis.py"
```

---
Note: The code in this repository is intended solely as sample code for research on this specific materials system.
