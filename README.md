# Task-Adaptive Multilingual Framework for DimABSA (SemEval-2026)

This repository contains the official implementation of our unified multilingual framework for Dimensional Aspect-Based Sentiment Analysis (DimABSA), originally developed for Track-A of SemEval-2026 Task 3.  

Our framework successfully tackles the challenge of fine-grained continuous sentiment prediction (Valence-Arousal) and structured opinion extraction across 6 languages and 4 domains.  

## 1. Key Highlights

Instead of relying on massive, computationally expensive models, this project demonstrates how targeted architectural design and instruction tuning can out-punch heavyweights:
   
  🏆 State-of-the-Art Efficiency in Regression (DimASR): Our aspect-aware label distribution learning (LDL) model, built on the lightweight XLM-RoBERTa-Large, outperformed the massive GPT-OSS 120B in 9 out of 10 language-domain pairs.

  🔥 Superior Extraction Power (DimASTE): Utilizing an instruction-based supervised fine-tuning paradigm with Qwen3 (14B & 4B), our framework exceeded the performance of Llama3.3-70B across the vast majority of evaluated subsets.  
  
  🥇 Top-Tier Leaderboard Rankings: 
  
        Ranked 5/19 in Chinese Finance (DimASR).  
        Ranked 5/14 in Chinese Restaurant (DimASTE).  
        Ranked 6/18 in Ukrainian Restaurant (DimASR).  

## 2. Core Architecture
   
Our solution is decoupled into two specialized modules to handle the unique challenges of regression and structured extraction:

### 2.1 DimASR

Aspect-Aware Label Distribution Learning (LDL)Predicting Arousal is notoriously harder than predicting Valence. To solve this, we abandoned single-pipeline constraints:

**Decoupled Dual-Model Training**: We utilized two physically independent predictors for Valence and Arousal, eliminating optimization interference.

**Label Distribution Learning**: Continuous VA values (1.00 to 9.00) are transformed into Gaussian-based soft distributions over predefined anchors, recovering precise continuous scores via expected-value decoding.

### 2.2 DimASTE

Generative Extraction with LLMsTo handle complex, implicit sentiment expressions and avoid the hallucination pitfalls of standard autoregressive decoding:  

**Instruction-Tuned Qwen3**: We applied LoRA-based supervised fine-tuning on chat-formatted sequences to adapt the LLM specifically for information extraction (IE).

**Structured Consistency Filtering (SCF)**: A custom post-processing pipeline that utilizes stochastic decoding and strict structural constraints (substring validation, distance-based pairing) to ensure 100% numerically valid and semantically logical triplet outputs.

**Cross-lingual Multi-view Augmentation**: For low-resource languages, we introduced parallel texts from other languages as unlabelled context during training to enhance semantic alignment.

## 3. Datasets & Performance

The framework was rigorously tested across multiple language-domain pairs, including English, Chinese, Japanese, Russian, Tatar, and Ukrainian.

### 📈 DimASR Task Results (Regression)
**Table 1: Comparison of our LDL framework against LLM baselines on the DimASR task.** 

| Language | Domain | PCC_V ↑ | PCC_A ↑ | **Ours (RMSE_VA) ↓** | Kimi K2 RMSE ↓ | GPT-OSS 120B RMSE ↓ |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| ENG | Laptop | 0.8469 | 0.4975 | **1.3814** | 2.1893 | 1.5269 |
| ENG | Restaurant | 0.8570 | 0.5971 | **1.3402** | 2.1461 | 1.4605 |
| JPN | Finance | 0.8089 | 0.5410 | **0.9267** | 1.6396 | 1.0188 |
| JPN | Hotel | 0.9197 | 0.7101 | **0.7055** | 1.7553 | 0.7188 |
| RUS | Restaurant | 0.8810 | 0.5877 | **1.4650** | 1.7768 | 1.4775 |
| UKR | Restaurant | 0.8808 | 0.5865 | **1.4618** | 1.7805 | 1.7153 |
| TAT | Restaurant | 0.6696 | 0.3854 | 1.9867 | 1.9380 | 1.5166 |
| ZHO | Finance | 0.8677 | 0.6790 | **0.4944** | 1.9652 | 0.6511 |
| ZHO | Laptop | 0.8940 | 0.7187 | **0.7085** | 1.6440 | 0.8032 |
| ZHO | Restaurant | 0.8621 | 0.5859 | **0.9679** | 1.8959 | 1.0349 |

> **Note:** Our lightweight XLM-RoBERTa-Large model achieves a lower RMSE_VA than the 120B parameter GPT-OSS across 9 out of 10 evaluated language-domain pairs.

<br>

### 🎯 DimASTE Task Results (Extraction)
**Table 2: Comparison of our Qwen3-14B based extraction pipeline against baselines on the DimASTE task.** 

| Language | Domain | cPre ↑ | cRec ↑ | **Ours (cF1) ↑** | Kimi K2 cF1 ↑ | Llama3.3-70B cF1 ↑ |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| ENG | Laptop | 0.5999 | 0.4081 | 0.4858 | 0.4424 | 0.5418 |
| ENG | Restaurant | 0.6803 | 0.4972 | **0.5745** | 0.4920 | 0.4664 |
| JPN | Hotel | 0.5358 | 0.4905 | **0.5121** | 0.3464 | 0.4694 |
| RUS | Restaurant | 0.5244 | 0.5484 | **0.5362*** | 0.4242 | 0.4590 |
| UKR | Restaurant | 0.5223 | 0.5454 | **0.5336** | 0.4220 | 0.4517 |
| TAT | Restaurant | 0.4489 | 0.4801 | **0.4640*** | 0.3577 | 0.4101 |
| ZHO | Laptop | 0.4938 | 0.4715 | **0.4824** | 0.2494 | 0.4344 |
| ZHO | Restaurant | 0.5486 | 0.5283 | **0.5382** | 0.3529 | 0.4789 |

> **Note:** `*` indicates results utilizing cross-lingual Multi-View augmentation. As shown, our 14B parameter solution demonstrates highly competitive performance, surpassing the Llama3.3-70B baseline on 7 out of 8 datasets.
