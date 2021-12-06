# rankQuestionAnswering
Implementation of the paper RankQA : https://arxiv.org/pdf/1906.03008.pdf

# Module 1 - Information Retrieval
python module1-squad.py

# Module 2 - Machine Comprehension
BERT_BASE_DIR=/Users/rahul/Desktop/ANLP/ProjectANLP/models/uncased_L-12_H-768_A-12/
BERT_FINE_TUNED=/Users/rahul/Desktop/ANLP/ProjectANLP/models/uncased_L-12_H-768_A-12/

On local machine
``` python generate_candidates.py  --vocab_file=/content/drive/My Drive/IIIT/ADV_NLP/Project/bertqa/models/uncased/vocab.txt \
                   --bert_config_file=/content/drive/My Drive/IIIT/ADV_NLP/Project/bertqa/models/uncased/bert_config.json \
                   --output_dir=/tmp \
                   --do_predict=True \
                   --predict_file=data/datasets/SQuAD-v1.1-train.txt \
                   --retriever_model=data/wikipedia/docs-tfidf-ngram\=2-hash\=16777216-tokenizer\=simple.npz  \
                   --doc_db=data/wikipedia/docs.db \
                   --out_name=output/squad_train ```



On colab
``` !python generate_candidates.py --vocab_file=models/uncased/vocab.txt --bert_config_file=models/uncased/bert_config.json --output_dir=/tmp --do_predict=True --predict_file=data/datasets/SQuAD-v1.1-train.txt --retriever_model=data/wikipedia/docs-tfidf-ngram\=2-hash\=16777216-tokenizer\=simple.npz --doc_db=data/wikipedia/docs.db --out_name=output/squad_train```


# Module 3 - Reranking module
