# Post Virality Predictor

## Authors

- Author 1: Kundyz Serzhankyzy

- Author 2: Uyen Pham

## Description of Question and Research Topic

This project aims to predict whether a Reddit post will go viral based on its content, metadata, and engagement metrics. We want to understand which factors—such as text, time of posting, or user influence—most affect virality. By analyzing Reddit posts from different datasets, we hope to identify generalizable patterns of viral content. This research can provide insights for marketers, content creators, and social media analysts. Ultimately, we aim to build a machine learning model that predicts a post's virality with high accuracy.

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

- Posts in the top 10% of the score distribution will be labeled as viral to maintain consistency across both datasets.

## Model Plans

### Author 1: Reddit Conversations Dataset + LSTM Model

Author 1 (Kundyz Serzhankyzy) will work with the **Reddit Conversations dataset** and develop a **Long Short-Term Memory (LSTM) network** to model how temporal patterns and sequential engagement affect post virality. This model will process Reddit posts and conversation threads as sequences, capturing how engagement evolves over time and identifying temporal patterns in posting behavior (such as peak activity hours and optimal posting times). The LSTM architecture is well-suited for this task because it can learn long-term dependencies in sequential data, allowing it to understand how early engagement signals predict future virality. By processing text sequences and temporal metadata together, the LSTM will learn to recognize patterns that indicate a post is likely to go viral based on both its content and the timing of its release. The output will be a probability score indicating how likely a post is to become viral based on these temporal and sequential patterns.

### Author 2: Reddit Dataset + Random Forest Model

Author 2 (Uyen Pham) will work with the **Reddit dataset** and design a **Random Forest classifier** to analyze how multiple features—including content characteristics, metadata, and engagement metrics—contribute to post virality. The Random Forest model will integrate diverse feature types into a single ensemble classifier. It combines:

- **Text-based features**: Text length, word count, title/body lengths, punctuation patterns (question marks, exclamation marks), uppercase ratio
- **Engagement features**: Post score, comment score, comment-to-score ratio, total engagement, comment presence indicators
- **Subreddit features**: Subreddit frequency (loaded from `reddit_data_counts.json`)

The ensemble nature of Random Forest allows it to capture non-linear relationships between features and virality, while its ability to provide feature importance scores will help identify which factors most strongly predict viral content. By training on multiple decision trees and aggregating their predictions, Random Forest will be robust to overfitting and capable of handling the mixed data types common in social media analysis. The model will output both a binary classification (viral or not viral) and feature importance rankings to understand what drives virality.

## Project Timeline

### Phase 1: Data Collection and Initial Preparation (Weeks 1-2) ✅ In progress
- [x] Set up project structure and documentation
- [x] Create data preparation notebook with automated data loading and cleaning pipeline
- [x] Download and load Reddit Conversations dataset (Author 1)
- [x] Download and load Reddit dataset (Author 2)
- [x] Complete initial data exploration and quality assessment
- [x] Implement virality labeling (composite score approach with top 30% threshold for Author 1)

### Phase 2: Feature Engineering and Preprocessing (Week 3) 
- [ ] Extract and engineer text-based features (character count, word count, punctuation patterns)
- [ ] Extract temporal features (hour, day of week, posting patterns)
- [ ] Calculate engagement metrics and ratios (comment-to-score ratios, total engagement)
- [ ] Perform advanced feature engineering:
  - Author 1: Sequence features for LSTM (conversation thread patterns, temporal sequences)
  - Author 2: Text and engagement features for Random Forest (text length, word count, engagement ratios, subreddit encoding)
- [ ] Handle missing values and outliers
- [ ] Normalize and scale features as needed

### Phase 3: Model Development (Weeks 4-5)
- [ ] **Author 1 - LSTM Model:**
  - [ ] Prepare sequential data for LSTM input
  - [ ] Design and implement LSTM architecture
  - [ ] Train model with temporal and text sequence data
  - [ ] Tune hyperparameters (learning rate, batch size, LSTM units, dropout)
  
- [ ] **Author 2 - Random Forest Model:**
  - [ ] Prepare feature matrix with engineered features
  - [ ] Implement Random Forest classifier
  - [ ] Train model with diverse feature set
  - [ ] Tune hyperparameters (n_estimators, max_depth, min_samples_split)

## Phase 4: Model Evaluation and Analysis (Week 5)
- [ ] Split data into train/validation/test sets  
- [ ] Evaluate both models using multiple metrics:
  - Accuracy, Precision, Recall, F1-score  
  - ROC-AUC curves  
  - Confusion matrices  
- [ ] Compare model performance  
- [ ] Analyze feature importance (Random Forest) and attention patterns (LSTM)  
- [ ] Identify key factors driving virality predictions  

## Phase 5: Results and Documentation (Week 6)
- [ ] Compile comprehensive results summary  
- [ ] Create visualizations of model performance and feature importance  
- [ ] Document findings and insights about viral content patterns  
- [ ] Prepare final report with methodology, results, and conclusions  
- [ ] Review and refine documentation  
