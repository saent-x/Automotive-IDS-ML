# Automotive In-Vehicle Intrusion Detection with Machine Learning

Research project (2024). A hybrid intrusion detection system for in-vehicle CAN networks that combines **signature-based detection** (supervised learning, for known attacks) with **anomaly-based detection** (unsupervised learning, for attacks the model has never seen).

## Approach

```
labelled CAN traffic ──► clean + merge attack subsets ──► features
                                                            │
                         ┌──────────────────────────────────┴───────────────┐
                         ▼                                                  ▼
        Signature tier (supervised)                         Anomaly tier (unsupervised)
        Random Forest · Gradient Boosting                             K-Means clustering
                         └──────────────────┬───────────────────────────────┘
                                            ▼
                          per-attack classification reports + confusion matrices
```

| Model | Tier | Why it was chosen |
| --- | --- | --- |
| Random Forest | Signature | Accurate and robust, resists overfitting |
| Gradient Boosting | Signature | Strong accuracy on large tabular datasets |
| K-Means | Anomaly | Simple, scales well, flags traffic that doesn't fit learned clusters |

Decision trees and SVMs were evaluated and dropped; the reasoning is in [Docs.md](Docs.md).

## Evaluation design

Models are tested on four scenarios, so the results show how well detection generalises beyond the training vehicle and attack set:

| Scenario | Attacks |
| --- | --- |
| Known vehicle, known attacks | DoS, force_neutral, rpm, standstill |
| Known vehicle, unknown attacks | double, fuzzing, interval, speed, systematic, triple |
| Unknown vehicle, known attacks | DoS, force_neutral, rpm, standstill |
| Unknown vehicle, unknown attacks | double, fuzzing, interval, speed, systematic, triple |

Datasets: [can-train-and-test](https://data.dtu.dk/articles/dataset/can-train-and-test/24805533/1) (Lampe & Meng, 2024), plus the [Car-Hacking dataset](https://ocslab.hksecurity.net/Datasets/CAN-intrusion-dataset) and a heavy-duty truck CAN dataset for comparison. Full pipeline and dataset notes: [Docs.md](Docs.md).

## Layout

| Path | What |
| --- | --- |
| `app.py` | Entry point: trains all models and runs every test scenario |
| `py_pkg/algos` | Model training and prediction |
| `py_pkg/testing` | Scenario runner and metrics reports |
| `py_pkg/core` | Dataset loading, splitting and reporting helpers |
| `go-pkg` | Experimental Go port of the training pipeline using mlpack |
| `repl.ipynb` | Exploration notebook |

## Run it

```bash
pip install pandas numpy scipy scikit-learn matplotlib
# place the cleaned dataset at ./datasets/clean-data/updated_dataset.csv
python app.py
```

## Related work

- [nn-anomaly-detection](https://github.com/saent-x/nn-anomaly-detection): PyTorch anomaly detector exported to ONNX with live SocketCAN inference
- [neural-network-module](https://github.com/saent-x/neural-network-module): neural network written from scratch in Go, applied to the same CAN data
