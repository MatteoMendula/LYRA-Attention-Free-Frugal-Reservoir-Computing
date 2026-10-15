# LYRA: Experimental Repository

This repository contains the code, experimental results, and analysis scripts accompanying the anonymous submission:

**"LYRA: An Attention-Free Reservoir Computing Architecture for Frugal Multitask Time-Series Representations"**. 【1-6b552e】

The repository includes:

- Implementation of the proposed LYRA architecture.
- Benchmark comparisons against Deep Reservoir Computing and MOMENT-based representations.
- Experiments on:
  - Server Machine Dataset (SMD)
  - Electricity Transformer Temperature (ETT)
  - ECG5000
- Representation analysis and topology-related studies.
- Plot generation scripts used to produce figures included in the manuscript.

---

## Repository Structure

```text
.
├── experiments-code/
│   ├── SMD-Lyra.py
│   ├── ETT-Lyra.py
│   ├── ECG5000-Lyra.py
│   ├── SMD-moment.py
│   ├── ETT-moment.py
│   ├── ECG5000-moment.py
│   └── ...
│
├── topological analysis/
│   ├── plot_eigenvalues_merged_ieee.py
│   ├── psi_DeepRC.py
│   ├── psi_DeepRC_IP.py
│   ├── utils.py
│   └── spectra/
│
├── representation-metrics/
│   └── *.json
│
*── csv/
│  *└── experimental results
│
├──*plots/
*   └── generated figures
│
├*─*imgs/
│  *└── manuscript figures
│*└── analysis scripts
``*

---

## Main Experiments

### S*D

Scripts related*to the Server Machine Dataset*

```bash*python experiments-code/SMD-Lyra.p*
python experiments*code/SMD-moment.py
python experime*ts*code/SMD-DeepRC-MultiTap_Standard.*y
```

Additional analyses:

```*ash*python smd-plots.py
python sm*-efficiency.py
*ython comparison-smd-reservoirpy.p*
```

---

### ETT

Scripts relate* to the Electricity Transformer Te*perature benchmarks:

```bash
*ython experiments-code/ETT-Lyra.py*python experiments-code/ETT-moment*py
```

Additional analyses:

```b*sh*python ETT_local_variability.py
py*hon ETT-sr-study.py
python*ETT**lot-variability.py
python ett-plot*.py
```

---

### ECG5000

Scripts*related to ECG5000 classification:*
```bash
python experiments-code/E*G5000-Lyra.py
python experiments-c*de/ECG5000-moment.py
```

Visualiz*tion:

```bash
python experiments-*ode/ECG5000-tsne.py
python ecg-500*.py
```

---

## Topological and D*namical Analysis

The*folder `*opological analysis/`*contains scripts used to study:

* Effective transition-matrix resca*ing.
- Spectral*properties of reservoir dynamics.
* Edge-of*chaos behaviour.
- Ly*punov-guided reservoir configurati*n. 【1-6b552e】*
Example:

```bash*python "topological analysis/plot*eigenvalues_merged_ieee.py"
```

*--

## Generated Outputs

The repo*itory*stores generated artifacts in:

| *irectory*| Contents |
|------------|-------*---|
| `csv*` | Experimental results and inter*ediate outputs |
| `*lots/` | Final*figures used for analysis |
| `*mgs*` | Manuscript graphics*|
| `*epresentation-metrics/` | Represen*ation quality metrics |

---

*# Dependencies

The code*ase relies*on commonly used scientific Python*packages, including:

```*ext
numpy
scipy
pandas**cikit-learn
matplotlib
seaborn
tor*h
mlflow
reservoir*computing related libraries
```

*epending on the*experiment, additional packages ma* be required.

---

## Reproducibi*ity

The repository contains:

- E*periment scripts.
- Analysis utili*ies.
- Stored*metrics*and generated figures.

*ataset acquisition*follows*the original dataset providers'*distribution policies and therefor**datasets are not redistributed wit*in this repository.

---

## Citat*on

If*this work*is accepted and*publicly released, a citation*entry will be provided in the fina* camera-ready version.

For the*purpose of anonymous review, pleas* cite the submitted manuscript onl*.

---

## Disclaimer

This reposi*ory is released exclusively*to support the anonymous*peer-review process.*Any information*that could reveal the identity*of the authors has been intentiona*ly removed.
# LYRA-Attention-Free-Frugal-Reservoir-Computing
# LYRA-Attention-Free-Frugal-Reservoir-Computing
# LYRA-Attention-Free-Frugal-Reservoir-Computing
