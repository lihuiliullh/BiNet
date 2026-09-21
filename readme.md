
# BiNet
This is the code for our KDD 2022 paper "Joint Knowledge Graph Completion and Question Answering"

# Instructions

## Data and pre-trained models

In order to run the code, first download data.zip and pretrained_model.zip from [here](https://drive.google.com/drive/folders/1pciDTdnz8gSe6Y4bphPR5sPE6akU5Qp1?usp=share_link). Unzip these files in the main directory.

## MetaQA

Following is an example command to run the QA training code

```
python3 main_xxx.py --mode train --relation_dim 200 --hidden_dim 256 \
--gpu 2 --freeze 0 --batch_size 128 --validate_every 5 --hops 2 --lr 0.0005 --entdrop 0.1 --reldrop 0.2  --scoredrop 0.2 \
--decay 1.0 --model ComplEx --patience 5 --ls 0.0 --kg_type half
```

## WebQuestionsSP and SimpleQA

Change to directory ./KGQA/webqsp_simpleqa. Following is an example command to run the QA training code
```
python3 main.py --mode train --relation_dim 200 --do_batch_norm 1 \
--gpu 2 --freeze 1 --batch_size 16 --validate_every 10 --hops webqsp_half --lr 0.00002 --entdrop 0.0 --reldrop 0.0 --scoredrop 0.0 \
--decay 1.0 --model ComplEx --patience 20 --ls 0.05 --l3_reg 0.001 --nb_epochs 200 --outfile half_fbwq
```

Also, please note that this implementation uses embeddings created through libkge (https://github.com/uma-pi1/kge). This is a very helpful library and I would suggest that you train embeddings through it since it supports sparse embeddings + shared negative sampling to speed up learning for large KGs like Freebase.

# Dataset creation

## MetaQA

### KG dataset

There are 3 datasets: MetaQA_full, MetaQA_half and MetaQA_30. Full dataset contains the original kb.txt as train.txt with duplicate triples removed. Half and 30 contains only 50% and 30% of the triples (randomly selected without replacement). 

There are some lines like 'entity NOOP entity' in the train.txt for half dataset. This is because when removing the triples, all triples for that entity were removed, hence any KG embedding implementation would not find any embedding vector for them using the train.txt file. By including such 'NOOP' triples we are not including any additional information regarding them from the KG, it is there just so that we can directly use any embedding implementation to generate some random vector for them.

### QA Dataset

There are different files for each dataset (1, 2 and 3 hop)
- qa_train_{n}hop_xxxx.txt
- qa_dev_{n}hop.txt
- qa_test_{n}hop.txt

Details about how to generate knowledge graph completion files, and how to generate the whole train data, please refer to code in ./preprocess_kg

## WebQuestionsSP and SimpleQA

### KG dataset

There are 3 datasets: fbwq_full, fbwq_half and fbwq_30 (fbsimpleQA_half and fbsimpleQA_30)

WebQuestionsSP contains both single hop questions and multiple hop (2hop) questions. It can be completed using the same strategy as metaQA 2hop which could bring performance improvement. 

### QA Dataset

Same as the original WebQuestionsSP QA dataset.

### More Experiment Results

**compared with LLM + RAG**
<img width="650" height="654" alt="Screenshot 2026-09-21 at 9 38 30 AM" src="https://github.com/user-attachments/assets/2e4c64f4-b109-452e-8849-9879d2eee684" />

| Type | Method | WebQSP Hits@1 | WebQSP F1 | CWQ Hits@1 | CWQ F1 |
|---|---|---|---|---|---|
| Embedding | KV-Mem | 46.7 | 34.5 | 18.4 | 15.7 |
| Embedding | EmbedKGQA | 66.6 | – | 45.9 | – |
| Embedding | NSM | 68.7 | 62.8 | 47.6 | 42.4 |
| Embedding | TransferNet | 71.4 | – | 48.6 | – |
| Embedding | KGT5 | 56.1 | – | 36.5 | – |
| Retrieval | GraftNet | 66.4 | 60.4 | 36.8 | 32.7 |
| Retrieval | PullNet | 68.1 | – | 45.9 | – |
| Retrieval | SR+NSM | 68.9 | 64.1 | 50.2 | 47.1 |
| Retrieval | SR+NSM+E2E | 69.5 | 64.1 | 49.3 | 46.3 |
| Semantic Parsing | SPARQA | – | – | 31.6 | – |
| Semantic Parsing | QGG | 73.0 | 73.8 | 36.9 | 37.4 |
| Semantic Parsing | ArcaneQA | – | 75.3 | – | – |
| Semantic Parsing | RnG-KBQA | – | 76.2 | – | – |
| LLMs | Flan-T5-xl | 31.0 | – | 14.7 | – |
| LLMs | Alpaca-7B | 51.8 | – | 27.4 | – |
| LLMs | LLaMA2-Chat-7B | 64.4 | – | 34.6 | – |
| LLMs | ChatGPT | 66.8 | – | 39.9 | – |
| LLMs | ChatGPT+CoT | 75.6 | – | 48.9 | – |
| LLMs+KGs | KD-CoT | 68.6 | 52.5 | 55.7 | – |
| LLMs+KGs | UniKGQA | 77.2 | 72.2 | 51.2 | 49.1 |
| LLMs+KGs | DECAF (DPR+FiD-3B) | 82.1 | 78.8 | – | – |
| LLMs+KGs | RoG | 85.7 | 70.8 | 62.6 | 56.2 |
| | **(ours)** | **69.2** | **64.1** | **48.3** | **46.2** |


### Results on complete KG
![image](https://github.com/user-attachments/assets/ef96d90c-eb2e-4756-9e3c-94a57855d7e8)

### Results on incomplete KG
![Screenshot 2025-03-09 at 7 13 50 PM](https://github.com/user-attachments/assets/86b531cb-6722-45de-a9b2-b5167a6a0377)

### Results on other datasets
![image](https://github.com/user-attachments/assets/80f6d6c4-074c-496c-a3e8-4ea0b3c40be9)

<!--- Hic-KGQA: Improving multi-hop question answering over knowledge graph via hypergraph and inference chain -->

# How to cite

If you used our work or found it helpful, please use the following citation:

```
@inproceedings{lihui2022binet,
author = {Liu, Lihui and Du, Boxin and Xu, Jiejun and Xia, Yinglong and Tong, Hanghang},
title = {Joint Knowledge Graph Completion and Question Answering},
year = {2022},
publisher = {Association for Computing Machinery},
booktitle = {Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining},
pages = {1098–1108},
series = {KDD '22}
}
```





The old version of code and all the dataset can be download from https://drive.google.com/drive/folders/1YalkzBmBhlekepTQd3wA8o8LWfsGDdhp?usp=sharing


Please consider cite the paper "Joint Knowledge Graph Completion and Question Answering" KDD 2022 if you think the paper or code is useful.

02 mc
