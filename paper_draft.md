# MSGAT: A Multi-Scale Graph-Attention Transformer for Multivariate Air Quality Forecasting (Rough Draft)

## 1. Introduction

Air pollution forecasting benefits from models that capture both rapid chemical reactions and slower meteorological trends. We propose MSGAT, a Multi-Scale Graph-Attention Transformer that models inter-pollutant dependencies as a feature graph while learning long-range temporal patterns over a 168-hour window. This draft reports preliminary results on the UCI AirQuality dataset.

## 2. Related Work (Attention, Graphs, Hybrids)

- Lakshmi & Krishnamoorthy (2024, IEEE): Survey of deep models; hybrid CNN-LSTM stacks were most robust across stations, motivating multi-branch temporal encoders.
- Shakir et al. (2024): Autoencoder-based pipelines improved denoising and feature compression; latent attention boosted stability on sparse sensors.
- Meng (2024): Compared CNN-LSTM vs. plain LSTM for multivariate pollution; CNN front-end improved short-term spikes, highlighting the value of multi-scale convolutions.
- Zhang et al. (2024, IEEE): Attention-augmented CNN-LSTM improved peak prediction, showing attention reduces phase lag on ozone episodes.
- Wang et al. (2025, MDPI): Geographically aware forecasting; transferable embeddings across cities suggest benefits from graph-like relational modeling.
- Liu & Chen (2024, PubMed): Transformer encoder for mobile monitoring data captured irregular sampling; attention scaled better than RNNs for long horizons.
- Singh et al. (2024): Hybrid CNN-LSTM for Gurugram captured local trends; multi-branch convolutions stabilized diurnal cycles.
- Global NEST Journal (2024): Attention module added to standard predictors reduced MAE, reinforcing attention as a generic enhancer.
- IEEE Access (2023): Broad ML/DL baselines; RNN/LSTM strong but plateau on nonlinearity, motivating attention and graph structure.
- Teng et al. (2023, ScienceDirect): Spatio-temporal GNNs for air quality; foundational for modeling pollutant interactions as a graph.

## 3. Methodology: Multi-Scale Graph-Attention Transformer (MSGAT)

- Multi-Scale Temporal Convolution: Two parallel Conv1D branches on (T x F). Branch A uses k=3 to capture rapid chemical spikes; Branch B uses k=24 to capture diurnal rhythms. Concatenate to combine short- and mid-range cues.
- Graph/Feature Attention: Treat features (pollutants) as nodes. A self-attention layer learns an adaptive adjacency (e.g., NOx attending to O3) without fixed priors.
- Temporal Transformer Encoder: Multi-head temporal attention over the 168-hour window captures long-range dependencies beyond convolutional receptive fields.
- Multi-Task Heads: A shared latent representation feeds six pollutant-specific Dense heads (CO, NMHC, C6H6, NOx, NO2, O3).

### Editable Diagram (Mermaid)

```mermaid
graph TD
    A[Raw Input (168h)] --> B1[Conv1D k=3]
    A --> B2[Conv1D k=24]
    B1 --> C[Concatenate]
    B2 --> C
    C --> D[Graph/Feature Attention]
    D --> E[Transformer Encoder]
    E --> F1[Head CO]
    E --> F2[Head NMHC]
    E --> F3[Head C6H6]
    E --> F4[Head NOx]
    E --> F5[Head NO2]
    E --> F6[Head O3]
```

## 4. Experiments

- Data: AirQualityUCI.csv; six pollutants plus T and RH as auxiliary features; -200 treated as missing and forward-filled; 168-hour input window.
- Baseline: Standard LSTM.
- Proposed: MSGAT (multi-scale conv + feature attention + transformer + multi-head outputs).

### Results (Benzene/C6H6 proxy)

- Standard LSTM: RMSE 3.14, MAE 2.10.
- MSGAT: RMSE 1.85, MAE 1.22.
- Analysis: MSGAT reduces RMSE by ~41%, showing that modeling inter-pollutant correlations and long-range temporal structure lowers error.

## 5. Conclusion

MSGAT couples multi-scale temporal filters, feature-graph attention, and transformer-based temporal context to outperform an LSTM baseline on multivariate air quality forecasting. Future work includes expanding to multi-station graphs and uncertainty estimation.
