# Post Virality Predictor

## Authors
- Author 1: Kundyz Serzhankyzy
- Author 2: Uyen Pham

## Description of Question and Research Topic
This project aims to predict whether a social media post will go viral based on its content, metadata, and engagement metrics. We want to understand which factors—such as text, hashtags, time of posting, or user influence—most affect virality. By analyzing multiple platforms like Twitter and Reddit, we hope to identify generalizable patterns of viral content. This research can provide insights for marketers, content creators, and social media analysts. Ultimately, we aim to build a machine learning model that predicts post virality with high accuracy.

## Project Outline/Plan
- Collect and preprocess datasets from Twitter, Reddit, and Instagram.  
- Extract relevant features: text features, metadata, engagement ratios.  
- Train machine learning models to predict post virality.  
- Evaluate models using metrics like accuracy, precision, recall, F1-score, and ROC-AUC.  
- Analyze feature importance and document findings in a final report.

## Data Collection Plan
### Part 1 (Author 1)
- Collect Twitter datasets using the Twitter API, focusing on tweets with engagement metrics (likes, retweets, replies).  
- Extract text, hashtags, posting time, and user follower count.  

### Part 2 (Author 2)
- Collect Reddit posts and comments from various subreddits using the Pushshift API.  
- Include metadata such as score, number of comments, post text, and posting time.

## Model Plans
### Part 1 (Author 1)
- Implement Random Forest and Logistic Regression models.  
- Focus on tabular features like hashtags count, posting time, and engagement ratios.

### Part 2 (Author 2)
- Implement NLP and CNNs.  
- Focus on NLP features such as TF-IDF vectors, sentiment analysis, and embeddings from post text.

## Project Timeline
- Week 1–2: Data collection and cleaning  
- Week 3: Feature engineering and preprocessing  
- Week 4–5: Model training and evaluation  
- Week 6: Feature analysis, results summary, and final report preparation

## .gitignore
