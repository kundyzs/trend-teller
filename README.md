# Post Virality Predictor

## Authors

- Author 1: Kundyz Serzhankyzy

- Author 2: Uyen Pham

## Description of Question and Research Topic

This project aims to predict whether a social media post will go viral based on its content, metadata, and engagement metrics. We want to understand which factors—such as text, hashtags, time of posting, or user influence—most affect virality. By analyzing multiple platforms like Twitter and Reddit, we hope to identify generalizable patterns of viral content. This research can provide insights for marketers, content creators, and social media analysts. Ultimately, we aim to build a machine learning model that predicts a post's virality with high accuracy.

## Project Outline/Plan

- Collect and preprocess datasets from Twitter and Reddit (one dataset per author).  

- Extract relevant features: text features, metadata, engagement ratios.  

- Train machine learning models to predict post virality (one model per author).  

- Evaluate models using metrics like accuracy, precision, recall, F1-score, and ROC-AUC.  

- Analyze feature importance and document findings in a final report.

## Data Collection Plan

### Author 1

- Collect Twitter datasets using the Twitter API, focusing on tweets with engagement metrics (likes, retweets, replies).  

- Extract text, hashtags, posting time, and user follower count.

- A post will be labeled as viral if its engagement exceeds a certain threshold, such as being in the top 10% by retweet count.

### Author 2

- Collect Reddit posts and comments from various subreddits using the Pushshift API.  

- Include metadata such as score, number of comments, post text, and posting time.

- Posts in the top 10% of the score distribution will be labeled as viral to maintain consistency across both platforms.

## Model Plans

### Author 1: Twitter Dataset + LSTM Model

Author 1 (Kundyz Serzhankyzy) will work with the **Twitter dataset** and develop a **Long Short-Term Memory (LSTM) network** to model how temporal patterns and sequential engagement affect post virality. This model will process tweets as sequences, capturing how engagement evolves over time and identifying temporal patterns in posting behavior (such as peak activity hours and optimal posting times). The LSTM architecture is well-suited for this task because it can learn long-term dependencies in sequential data, allowing it to understand how early engagement signals predict future virality. By processing text sequences and temporal metadata together, the LSTM will learn to recognize patterns that indicate a tweet is likely to go viral based on both its content and the timing of its release. The output will be a probability score indicating how likely a tweet is to become viral based on these temporal and sequential patterns.

### Author 2: Reddit Dataset + Random Forest Model

Author 2 (Uyen Pham) will work with the **Reddit dataset** and design a **Random Forest classifier** to analyze how multiple features—including content characteristics, metadata, and engagement metrics—contribute to post virality. The Random Forest model will integrate diverse feature types into a single ensemble classifier. It will combine text-based features (such as sentiment scores, keyword counts, and topic keywords), metadata (posting time, user influence metrics), and engagement ratios (early engagement rates, comment-to-upvote ratios). The ensemble nature of Random Forest allows it to capture non-linear relationships between features and virality, while its ability to provide feature importance scores will help identify which factors most strongly predict viral content. By training on multiple decision trees and aggregating their predictions, Random Forest will be robust to overfitting and capable of handling the mixed data types common in social media analysis. The model will output both a binary classification (viral or not viral) and feature importance rankings to understand what drives virality.

## Project Timeline

- Week 1–2: Data collection and cleaning  

- Week 3: Feature engineering and preprocessing  

- Week 4–5: Model training and evaluation  

- Week 6: Feature analysis, results summary, and final report preparation

