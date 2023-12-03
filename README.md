Name: **Mohamed Abdelhamid**

Email: m.abdelhamid@innopolis.university

Group: BS20-AAI

# Movie-Recommender-System
## Overview
This repository hosts the implementation of a movie recommender system developed as part of a Practical Machine Learning and Deep Learning course project. The system utilizes the MovieLens 100K dataset to suggest movies.

## Dataset
The MovieLens 100K dataset includes 100,000 movie ratings from 943 users across 1682 movies. It features user demographics (age, gender, occupation, zip code) and movie details (genres, titles).

## Approach
The project employs a Matrix Factorization approach to build the recommender system. This collaborative filtering method decomposes the user-item interaction matrix into user and movie embeddings to predict user preferences.

## Key Metrics
- **Root Mean Square Error (RMSE):** Measures the average magnitude of prediction errors.
- **Precision at Top-5:** Evaluates the accuracy of the top five recommendations made to the users.

## Repository Structure
```
movie-recommender-system
├── README.md               # The top-level README
│
├── data
│   ├── external            # Data from third party sources
│   ├── interim             # Intermediate data that has been transformed.
│   └── raw                 # The original, immutable data
│
├── models                  # Trained and serialized models, final checkpoints
│
├── notebooks               #  Jupyter notebooks.
│                               
│                                        
│ 
├── references              # Data dictionaries, manuals, and all other explanatory materials.
│
├── reports
│   ├── figures             # Generated graphics and figures to be used in reporting
│   └── final_report.pdf    # Report containing data exploration, solution exploration, training process, and evaluation
│
└── benchmark
    ├── data                # dataset used for evaluation 
    └── evaluate.py         # script that performs evaluation of the given model
```


## Results
- RMSE Score: 1.2
- Precision at Top-5: 0.013

## Usage
To run the project:
1. Clone the repository.
2. Install required dependencies.
3. Run the Jupyter notebooks in the `notebooks` directory for data exploration and model training.



