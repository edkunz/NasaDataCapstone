# NASA Data Capstone - Boiling Regime Discovery and Classification

## Project Overview

This project was developed as part of a Data Science Capstone in collaboration with NASA.

The goal of the project is to automatically identify and classify boiling regimes from acoustic boiling data collected during heater experiments. Rather than manually reviewing thousands of boiling runs, the project uses unsupervised machine learning techniques to discover meaningful boiling behaviors directly from extracted signal features.

The final objective is to support future autonomous boiling analysis in environments where manual review is impractical, such as microgravity experiments conducted in space.

### Problem Statement

Boiling behavior changes significantly depending on experimental conditions. These changes produce distinct acoustic signatures that can be measured and analyzed.

Historically, researchers inspect these signals manually to determine boiling regimes. This process is time-consuming and difficult to scale.

This project attempts to:

Separate active boiling from non-boiling behavior
Discover natural boiling regimes without predefined labels
Characterize each regime using interpretable features
Provide a framework for assigning future runs to existing regimes

### Data
Input Data

The project uses acoustic boiling recordings collected during NASA heater experiments.

Each run is represented by a set of engineered features extracted from the signal.

Examples include:

- Spectral entropy
- Spectral centroid
- Crest factor
- Burstiness
- Number of boiling events
- Regime dominance
- Peak magnitude statistics
- Wavelet energy measurements

The primary feature file used throughout the project is: *non_noise_features.csv*

### Pipeline Overview
#### Step 1: Feature Extraction

Raw acoustic signals are converted into numerical features describing:

- Frequency content
- Signal variability
- Event structure
- Temporal behavior
- Peak characteristics

#### Step 2: Noise Floor Filtering

Runs with little or no boiling activity are separated from active boiling runs.

This step removes signals that primarily contain background noise and allows clustering to focus on meaningful boiling behavior.

#### Step 3: Dimensionality Reduction

UMAP (Uniform Manifold Approximation and Projection) is used to project high-dimensional feature vectors into a lower-dimensional representation while preserving local structure.

This allows:

- Visualization of boiling behavior
- Cluster discovery
- Interpretation of regime relationships

#### Step 4: Active Boiling Clustering

HDBSCAN is used to identify major boiling regimes.

The final active boiling model separates the data into several high-level boiling groups.

Hyperparameters were selected through extensive testing of:

- UMAP neighbor counts
- UMAP minimum distance values
- HDBSCAN minimum cluster size
- HDBSCAN minimum samples

##### Three major boiling clusters were discovered:

Cluster	Interpretation
0	      Sporadic
1	      Chaotic
2	      Rhythmic

*Sporadic:* Sporadic boiling contains isolated boiling events with less consistent repeating behavior. These runs tend to exhibit lower event frequency and weaker repeating acoustic patterns.

*Chaotic:* Chaotic boiling contains highly variable boiling activity with irregular acoustic behavior. These runs often contain bursts of activity and less predictable signal structure.

*Rhythmic:* Rhythmic boiling contains strong repeating boiling patterns and clear periodic behavior. Because this cluster contained several visually distinct behaviors, additional subclustering was performed to identify more specific rhythmic regimes.

#### Step 5: Rhythmic Subclustering

The rhythmic cluster was further analyzed using a second UMAP + HDBSCAN workflow.

##### This process identified three primary rhythmic boiling sub-regimes.

Subcluster	Interpretation
0	          Dense Rhythmic
1	          Double Rhythmic
2	          Single Rhythmic

*Dense Rhythmic:* Dense Rhythmic boiling contains strong, frequent repeating acoustic patterns. These runs exhibit highly regular boiling behavior with dense repeating events.

*Double Rhythmic:* Double Rhythmic boiling contains many boiling events that tend to occur in bursts rather than following one dominant rhythm. Multiple repeating patterns may be present within the same run.

*Single Rhythmic:* Single Rhythmic boiling contains fewer boiling events and one dominant repeating boiling pattern. These runs generally exhibit a simpler and more consistent rhythmic structure.

These labels were developed through feature analysis, representative run inspection, and discussions with project stakeholders. They serve as engineering interpretations of the discovered acoustic boiling behaviors and may continue to be refined as additional data becomes available.

### Robustness Testing

Several analyses were performed to evaluate cluster stability.

##### Leave-One-Out Testing

Individual runs were removed and reclustered to evaluate sensitivity.

This helps determine whether cluster assignments are driven by a small number of influential observations.

##### Dataset Size Testing

Different subsets of the data were clustered to determine how sensitive the discovered regimes are to available sample size.

##### Feature Importance Testing

Over 100 random runs of HDBSCAN on both main clustering and subclustering, the top 5 features distinguishing each cluster from another
was stability tested by removing a number of points each time.

#### Classification and New Data Assignment

A classification framework was developed to support future boiling runs. The workflow combines a frozen UMAP model with a Gaussian Mixture Model (GMM), allowing new data points to be projected into the existing embedding space and assigned to the most likely boiling regime.

The GMM provides probabilities for each cluster rather than a strict assignment, allowing uncertainty and potential outliers to be identified. Stability testing, including leave-one-out experiments, dataset size testing, and signal amplitude testing, was performed to evaluate the robustness of the framework and discovered regimes.

## Team

*Data Science Capstone Team*

Hailey Ernest
Elliot Kunz
Cameron Hafer
Colin Hassett

In collaboration with NASA researcher Michael Khasin, and Cal Poly Data Science professors Dr. Kelly Bodwin and Dr. Alex Dekhtyar.
