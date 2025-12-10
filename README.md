# Reddit Post Virality Predictor

## Authors

- Author 1: Kundyz Serzhankyzy

- Author 2: Uyen Pham

## Description of Question and Research Topic

This project aims to predict whether a Reddit post will go viral based on its content, metadata, and engagement metrics. We want to understand which factors, such as text, time of posting, or user influence, most affect virality. By analyzing Reddit posts from different datasets, we hope to identify generalizable patterns of viral content. This research can provide insights for marketers, content creators, and social media analysts. Ultimately, we aim to build a machine learning model that predicts a post's virality with high accuracy.

## Project Outline/Plan

- Collect and preprocess Reddit datasets from Kaggle.  

- Extract relevant features: text features, metadata, engagement ratios.  

- Train machine learning models to predict post virality.  

- Evaluate models using metrics like accuracy, precision, recall, F1-score, and ROC-AUC.  

- Analyze feature importance and document findings in a final report.

## Data Collection Plan

### Author 1

- Use the Reddit Conversations dataset from Kaggle: [https://www.kaggle.com/datasets/jerryqu/reddit-conversations](https://www.kaggle.com/datasets/jerryqu/reddit-conversations)

- Extract text, conversation threads, posting time, and engagement metrics from the dataset.

- Posts are labeled as viral using a **composite virality score** that combines multiple engagement features (conversation length, thread depth, engagement metrics, etc.). The top 30% of posts by this composite score are labeled as viral (1), the bottom 30% as not viral (0), and the middle 40% are labeled based on a median threshold. This approach creates a balanced dataset (approximately 50/50 class distribution) and prevents the model from trivially memorizing simple patterns, leading to more robust predictions.

### Author 2

- Use the Reddit dataset from Kaggle: [https://www.kaggle.com/datasets/rohitrajesh/reddit-dataset](https://www.kaggle.com/datasets/rohitrajesh/reddit-dataset)

- Dataset contains 6,187 Reddit posts with the following fields:
  - `title`: Post title text
  - `body`: Post body/content text
  - `url`: Post URL (used to extract subreddit)
  - `post_score`: Upvote score of the post
  - `comment`: Top comment text
  - `comment_score`: Upvote score of the top comment

- Additional data file: `reddit_data_counts.json` contains subreddit frequency counts (e.g., "business": 311, "datascience": 1031, etc.)

- Extracted features include:
  - **Text features**: Character count, word count, title/body lengths, punctuation patterns (question marks, exclamation marks), uppercase ratio
  - **Engagement features**: Post score, comment score, comment-to-score ratio, total engagement, comment presence and length
  - **Subreddit features**: Subreddit frequency (from `reddit_data_counts.json`)

- Posts in the top 30% of the virality score distribution are labeled as viral. The virality score is computed as `v = z_post + α × z_comment`, where α (alpha) is a hyperparameter tuned through grid search.

## Model Plans

### Author 1: Reddit Conversations Dataset + LSTM Model

Author 1 (Kundyz Serzhankyzy) will work with the **Reddit Conversations dataset** and develop a **Long Short-Term Memory (LSTM) network** to model how temporal patterns and sequential engagement affect post virality. This model will process Reddit posts and conversation threads as sequences, capturing how engagement evolves over time and identifying temporal patterns in posting behavior (such as peak activity hours and optimal posting times). The LSTM architecture is well-suited for this task because it can learn long-term dependencies in sequential data, allowing it to understand how early engagement signals predict future virality. By processing text sequences and temporal metadata together, the LSTM will learn to recognize patterns that indicate a post is likely to go viral based on both its content and the timing of its release. The output will be a probability score indicating how likely a post is to become viral based on these temporal and sequential patterns.

### Author 2: Reddit Dataset + Random Forest Model

Author 2 (Uyen Pham) will work with the **Reddit dataset** and design a **Random Forest classifier** to analyze how multiple features—including content characteristics, metadata, and engagement metrics—contribute to post virality. The Random Forest model will integrate diverse feature types into a single ensemble classifier. It combines:

- **Text-based features**: Text length, word count, title/body lengths, punctuation patterns (question marks, exclamation marks), uppercase ratio
- **Engagement features**: Post score, comment score, comment-to-score ratio, total engagement, comment presence indicators
- **Subreddit features**: Subreddit frequency (loaded from `reddit_data_counts.json`)

## Data Access Statements

### Author 1 (Kundyz Serzhankyzy)
- **Dataset**: Reddit Conversations dataset from Kaggle
- **Source**: [https://www.kaggle.com/datasets/jerryqu/reddit-conversations](https://www.kaggle.com/datasets/jerryqu/reddit-conversations)
- **Access**: The dataset must be downloaded from Kaggle. You will need a Kaggle account and API credentials to access the data. Place the downloaded dataset files in the appropriate directory as specified in the data preparation notebook.

### Author 2 (Uyen Pham)
- **Dataset**: Reddit dataset from Kaggle
- **Source**: [https://www.kaggle.com/datasets/rohitrajesh/reddit-dataset](https://www.kaggle.com/datasets/rohitrajesh/reddit-dataset)
- **Access**: The dataset must be downloaded from Kaggle. You will need a Kaggle account and API credentials to access the data.
- **Preprocessed Data**: The `Uyen/data/` directory contains preprocessed feature files (`reddit_features.csv`) and saved model files (`random_forest_model.pkl`) that can be used directly without re-running the full preprocessing pipeline.

## Installation Instructions

These installation instructions apply to both the `Uyen/` (Random Forest) and `Kundyz/` (LSTM) implementations. To set up and run this project, follow these steps:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/kundyzs/trend-teller.git
   cd trend-teller
   ```

2. **Create a conda environment (recommended):**
   ```bash
   conda create -n trend-teller python=3.12
   conda activate trend-teller
   ```

3. **Install required packages:**
   ```bash
   pip install -r requirements.txt
   ```
   
   Or install using conda:
   ```bash
   conda install --file requirements.txt
   ```
   
   **Note**: Both `Uyen/` and `Kundyz/` folders may have their own `requirements.txt` files. Install the root `requirements.txt` for common dependencies, and check individual folders for additional requirements if needed.

4. **Launch Jupyter Notebook:**
   ```bash
   jupyter notebook
   ```

5. **Open the notebooks:**
   - **For Random Forest implementation** (Author 2 - Uyen Pham), navigate to the `Uyen/` folder:
     - `1_Data_Preprocessing.ipynb` - Data loading, feature engineering, and virality score computation
     - `2_Model_Construction.ipynb` - Random Forest model training and saving
     - `3_Model_Evaluation.ipynb` - Model evaluation, metrics, and visualizations
   - **For LSTM implementation** (Author 1 - Kundyz Serzhankyzy), navigate to the `Kundyz/` folder:
     - `LSTM_data_preparation.ipynb` - LSTM data preparation
     - `LSTM_model.ipynb` - LSTM model training
     - `LSTM_evaluation_analysis.ipynb` - LSTM model evaluation

The project requires Python 3.8 or higher and is compatible with Windows, macOS, and Linux operating systems. The same conda environment can be used for both implementations.

## Future Updates and Improvements

If we were to continue working on this project, we would pursue the following updates:

1. **Enhanced Feature Engineering:** Incorporate sentiment analysis, topic modeling, and temporal features
2. **Model Improvements:** Experiment with ensemble methods, deep learning architectures (BERT, GPT), and attention mechanisms
3. **Data Expansion:** Collect real-time Reddit data and incorporate multi-platform data (Twitter, Instagram)
4. **Deployment:** Create a web application or API for real-time virality predictions with a visualization dashboard
5. **Research Extensions:** Investigate causal relationships and study how external events impact post virality

## Project Timeline

### Phase 1: Data Collection and Initial Preparation (Weeks 1-2)
- [x] Set up project structure and documentation
- [x] Create data preparation notebook with automated data loading and cleaning pipeline
- [x] Download and load Reddit Conversations dataset (Author 1)
- [x] Download and load Reddit dataset (Author 2)
- [x] Complete initial data exploration and quality assessment
- [x] Implement virality labeling (top 30% threshold) 

### Phase 2: Feature Engineering and Preprocessing (Week 3) 
- [x] Extract and engineer text-based features (character count, word count, punctuation patterns)
- [x] Calculate engagement metrics and ratios (comment-to-score ratios, total engagement)
- [x] Perform advanced feature engineering:
  - Author 1: Sequence features for LSTM (conversation thread patterns, temporal sequences)
  - Author 2: Text and engagement features for Random Forest (text length, word count, engagement ratios, subreddit encoding)
- [x] Handle missing values and outliers
- [x] Compute virality scores with tunable alpha hyperparameter

### Phase 3: Model Development (Weeks 4-5)
- [x] **Author 1 - LSTM Model:**
  - [x] Prepare sequential data for LSTM input
  - [x] Design and implement LSTM architecture
  - [x] Train model with temporal and text sequence data
  - [x] Tune hyperparameters (learning rate, batch size, LSTM units, dropout)
  
- [x] **Author 2 - Random Forest Model:**
  - [x] Prepare feature matrix with engineered features
  - [x] Implement Random Forest classifier
  - [x] Train model with diverse feature set
  - [x] Tune hyperparameters (n_estimators=100, max_depth=10, min_samples_split=5, class_weight='balanced')

### Phase 4: Model Evaluation and Analysis (Week 5)
- [x] Split data into train/validation/test sets  
- [x] Evaluate Random Forest model using multiple metrics:
  - Accuracy, Precision, Recall, F1-score  
  - ROC-AUC curves  
  - Confusion matrices  
- [x] Analyze feature importance (Random Forest)
- [x] Identify key factors driving virality predictions
- [x] Perform subgroup analysis by text length quartiles
- [x] Create visualizations of model behavior and performance  

### Phase 5: Results and Documentation (Week 6) 
- [x] Compile comprehensive results summary  
- [x] Create visualizations of model performance and feature importance  
- [x] Document findings and insights about viral content patterns  
- [x] Organize project files into author-specific folders (`Uyen/` and `Kundyz/`)  
