# Graph-Augmented Abstractive Summarization Model  

This repository contains the implementation of a **graph-augmented sequence-to-sequence (Seq2Seq) model** designed to enhance **factual consistency** in abstractive text summarization. By integrating **knowledge graphs** with contextual embeddings, the model addresses challenges in accurately capturing entity relationships and minimizing factual inconsistencies in generated summaries.  

<img width="1119" alt="model_arc" src="https://github.com/user-attachments/assets/c1186283-dbf6-4de0-9ee1-3c622baa1669" />
(model architecture)

## Key Features  
- **Dual Cross-Attention Mechanism**: Combines contextual embeddings from a document encoder with graph-based representations from a graph encoder.  
- **Knowledge Graph Integration**: Constructs structured representations of entities and relations from source texts to enrich semantic understanding.  
- **Dataset Versatility**: Evaluated on CNN/DailyMail, XSum, and Fiction datasets, representing diverse levels of abstraction and complexity.  
- **Comprehensive Metrics**:  
  - **Structural and Semantic Evaluation**: ROUGE, BERTScore, Novel N-grams.  
  - **Factual Consistency Assessment**: FENICE Score, Relation Matching Rate.  

## Highlights  
- **Marginal Improvements in Factual Consistency**: Particularly evident in highly abstractive datasets (e.g., XSum, Fiction).  
- **Analysis of Knowledge Graph Effectiveness**: Findings indicate the potential of knowledge graphs, though constrained by current triple extraction methods.  
- **Insights for Future Research**: Emphasizes the need for improved knowledge extraction techniques to maximize the benefits of graph-based summarization.  

## Applications  
This repository is valuable for researchers and practitioners in natural language processing (NLP) working on:  
- **Abstractive Summarization**  
- **Factual Consistency in Text Generation**  
- **Graph-Enhanced NLP Models**  

