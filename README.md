# audio speed detection with synthetic training data

This repository contains my solution for the NEOAI 2026 audio speed detection competition. The task was to rank test clips by how likely they were to have been sped up.

## a small clarification

The organizers provided the task, dataset, evaluation setup, statistical context, and a starter baseline. I used that baseline as a starting point instead of pretending this appeared from a completely blank notebook.

I wrote and adapted the Kaggle code myself, including synthetic sample generation, acoustic feature extraction, model training, and submission.

I am currently studying statistics independently. I use the statistical terms that the problem actually needs, but I am not pretending I have already mastered the whole subject. learning in public, basically.

Public ROC AUC: around **0.90** :)

## approach

The training set contains only normal-speed speech, so I generated the missing positive class. For each training clip, I kept the original version as class 0 and created a time-stretched version with a random speed factor between 1.5 and 2.0 as class 1. No positive labels, so I made the positive class myself. hohoho.

I then extracted a compact set of acoustic features and trained a logistic regression classifier. The predicted probability for class 1 was used as the submission score.

## features

The feature set includes:

- spectral centroid
- spectral bandwidth
- spectral rolloff
- spectral flatness
- RMS energy
- zero-crossing rate
- MFCC statistics
- first-order MFCC delta statistics
- duration and waveform statistics

For frame-level features, I used the mean, standard deviation, 10th percentile, and 90th percentile.

## model

The final pipeline uses `StandardScaler` followed by `LogisticRegression`:

```python
make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=2000, C=2.0, random_state=42),
)
```

## repository contents

- `neoai-audio-speed-detection-0.90.ipynb`: complete feature extraction, training, and submission pipeline
- `README.md`: project description

## running the notebook

1. Add the NEOAI 2026 Day 2 audio competition dataset to a Kaggle notebook.
2. Run all cells in order.
3. The final cell creates `submission.csv`.

The notebook runs on CPU and uses `joblib` to parallelize feature extraction.
