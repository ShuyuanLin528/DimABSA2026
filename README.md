Task-Adaptive Multilingual Framework for DimABSA (SemEval-2026)

This repository contains the official implementation of our unified multilingual framework for Dimensional Aspect-Based Sentiment Analysis (DimABSA), originally developed for Track-A of SemEval-2026 Task 3.  

Our framework successfully tackles the challenge of fine-grained continuous sentiment prediction (Valence-Arousal) and structured opinion extraction across 6 languages and 4 domains.  

1. Key Highlights: Instead of relying on massive, computationally expensive models, this project demonstrates how targeted architectural design and instruction tuning can out-punch heavyweights:
   
  🏆 State-of-the-Art Efficiency in Regression (DimASR): Our aspect-aware label distribution learning (LDL) model, built on the lightweight XLM-RoBERTa-Large, outperformed the massive GPT-OSS 120B in 9 out of 10 language-domain pairs.

  🔥 Superior Extraction Power (DimASTE): Utilizing an instruction-based supervised fine-tuning paradigm with Qwen3 (14B & 4B), our framework exceeded the performance of Llama3.3-70B across the vast majority of evaluated subsets.  
  
  🥇 Top-Tier Leaderboard Rankings: 
  
        Ranked 5/19 in Chinese Finance (DimASR).  
        Ranked 5/14 in Chinese Restaurant (DimASTE).  
        Ranked 6/18 in Ukrainian Restaurant (DimASR).  

2. Core Architecture
   
Our solution is decoupled into two specialized modules to handle the unique challenges of regression and structured extraction:

2.1. DimASR: Aspect-Aware Label Distribution Learning (LDL)Predicting Arousal is notoriously harder than predicting Valence. To solve this, we abandoned single-pipeline constraints:

Decoupled Dual-Model Training: We utilized two physically independent predictors for Valence and Arousal, eliminating optimization interference.

Label Distribution Learning: Continuous VA values (1.00 to 9.00) are transformed into Gaussian-based soft distributions over predefined anchors, recovering precise continuous scores via expected-value decoding.

2.2. DimASTE: Generative Extraction with LLMsTo handle complex, implicit sentiment expressions and avoid the hallucination pitfalls of standard autoregressive decoding:  

Instruction-Tuned Qwen3: We applied LoRA-based supervised fine-tuning on chat-formatted sequences to adapt the LLM specifically for information extraction (IE).

Structured Consistency Filtering (SCF): A custom post-processing pipeline that utilizes stochastic decoding and strict structural constraints (substring validation, distance-based pairing) to ensure 100% numerically valid and semantically logical triplet outputs.

Cross-lingual Multi-view Augmentation: For low-resource languages, we introduced parallel texts from other languages as unlabelled context during training to enhance semantic alignment.
