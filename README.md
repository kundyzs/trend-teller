# Post Virality Predictor

## Authors
- Author 1: Kundyz Serzhankyzy
- Author 2: Uyen Pham

## Description of Question and Research Topic
This project aims to predict whether a social media post will go viral based on its content, metadata, and engagement metrics. We want to understand which factors—such as text, hashtags, time of posting, or user influence—most affect virality. By analyzing multiple platforms like Twitter and Reddit, we hope to identify generalizable patterns of viral content. This research can provide insights for marketers, content creators, and social media analysts. Ultimately, we aim to build a machine learning model that predicts a post's virality with high accuracy.

## Project Outline/Plan
- Collect and preprocess datasets from Twitter, Reddit, and Instagram.  
- Extract relevant features: text features, metadata, engagement ratios.  
- Train machine learning models to predict post virality.  
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
### Part 1 
Author 1 will develop a Hybrid Temporal–Graph Attention Network (HT-GAT) to model how timing and user influence affect post virality. This model will represent users and their posts as nodes in a graph, with connections formed through interactions such as mentions, replies, and retweets. The model will incorporate temporal embeddings to capture posting time patterns, such as peak activity hours, and use Graph Attention Networks (GAT) to learn how influence spreads across the network. By combining these two components, HT-GAT aims to understand both when a post is likely to go viral and who plays the biggest role in amplifying it. The output will be a probability score indicating how likely a post is to become viral based on these dynamic and relational patterns.

### Part 2
Author 2 will design a Multi-Channel Text Fusion Transformer (MTFT) to analyze how the content and emotional tone of a post contribute to its virality. The model will integrate multiple text-based signals into a single Transformer architecture. It will combine semantic representations from BERT embeddings, emotional intensity from sentiment analysis tools, and topical context from hashtag embeddings. These different representations will be fused through multi-head attention, allowing the model to weigh various linguistic and emotional cues simultaneously. By doing so, MTFT will capture both the meaning and the emotional resonance of content—two critical elements that often determine whether a post spreads widely online.

## Project Timeline
- Week 1–2: Data collection and cleaning  
- Week 3: Feature engineering and preprocessing  
- Week 4–5: Model training and evaluation  
- Week 6: Feature analysis, results summary, and final report preparation
