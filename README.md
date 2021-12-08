# rankQuestionAnswering
Implementation of the paper RankQA
Paper url - https://arxiv.org/pdf/1906.03008.pdf

## Module 1 - Information Retrieval
For a given query, the information retrieval module retrieves the top-n (here: n = 10) matching documents from the content repository and then splits these articles into paragraphs.

```
cd scripts 

#### For SQUAD dataset
python module1-squad.py

#### For Wiki dataset
python module1-wiki.py
```


## Module 2 - Machine Comprehension
The machine comprehension module extracts and scores one candidate answer for every paragraph of all top-k documents. 
Hyperparamer **k** : I used k=20 because of memory constraints. The author used k=40.


### BERT-QA pipeline

#### Step 1 Module1 - We use the information retrieval module to retrieve top-n documents from Wikipedia for each question 
#### Step 2 Module2 - Using generate_candidates.py we pair the question and all its top-n paragraphs and pass into BERT 
#### Step 3 Module2 - Get top-k candidate answers, ranked by BERT scores
#### Step 4 Module2 - We aggregate all the top-k answers and perform feature extraction for the purpose of re-ranking


### How to run

#### Step1 Download the BERT-base-uncased model and set its directory

```
BERT_BASE_DIR=/Users/rahul/Desktop/ANLP/ProjectANLP/models/uncased_L-12_H-768_A-12/  
BERT_FINE_TUNED=/Users/rahul/Desktop/ANLP/ProjectANLP/models/uncased_L-12_H-768_A-12/ 
```

#### Step2 Run generate_candidates.py

##### On local machine
```
python generate_candidates.py  --vocab_file=/content/drive/My Drive/IIIT/ADV_NLP/Project/bertqa/models/uncased/vocab.txt \
                   --bert_config_file=/content/drive/My Drive/IIIT/ADV_NLP/Project/bertqa/models/uncased/bert_config.json \
                   --output_dir=/tmp \
                   --do_predict=True \
                   --predict_file=data/datasets/SQuAD-v1.1-train.txt \
                   --retriever_model=data/wikipedia/docs-tfidf-ngram\=2-hash\=16777216-tokenizer\=simple.npz  \
                   --doc_db=data/wikipedia/docs.db \
                   --out_name=output/squad_train 
```

##### On colab 

```
!python generate_candidates.py --vocab_file=models/uncased/vocab.txt --bert_config_file=models/uncased/bert_config.json --output_dir=/tmp --do_predict=True --predict_file=data/datasets/SQuAD-v1.1-train.txt --retriever_model=data/wikipedia/docs-tfidf-ngram\=2-hash\=16777216-tokenizer\=simple.npz --doc_db=data/wikipedia/docs.db --out_name=output/squad_train
```

#### Outputs generated 
The outputs gets generated in the folder in differen batches    
    ```bertqa/out/output-feat-batch-0.txt ``` 


##### Sample output
```
[{"text": "pio or sister maria pierina de micheli these have involved visions of jesus", "span_score": 2.2091721296310425, "start_logit": 1.0981687307357788, "end_logit": 1.1110033988952637, "doc_score": 324.03023353020836, "doc_pos": 3, "in_doc_pos": 0, "paragraph_score": 0.06557377049180328, "context_len": 67, "span_len": 13, "first_occ": 1, "num_occ": 1, "avg_span_score": 2.2091721296310425, "max_span_score": 2.2091721296310425, "min_span_score": 2.2091721296310425, "avg_doc_score": 324.03023353020836, "max_doc_score": 324.03023353020836, "min_doc_score": 324.03023353020836, "avg_doc_pos": 3, "max_doc_pos": 3, "min_doc_pos": 3, "avg_start_logit": 1.0981687307357788, "max_start_logit": 1.0981687307357788, "min_start_logit": 1.0981687307357788, "avg_end_logit": 1.1110033988952637, "max_end_logit": 1.1110033988952637, "min_end_logit": 1.1110033988952637, "avg_paragraph_score": 0.06557377049180328, "max_paragraph_score": 0.06557377049180328, "min_paragraph_score": 0.06557377049180328, "avg_in_doc_pos": 0, "max_in_doc_pos": 0, "min_in_doc_pos": 0, "target": 0, "question": "To whom did the Virgin Mary allegedly appear in 1858 in Lourdes France?", "question_len": 13}
```



## Module 3 - Reranking module
