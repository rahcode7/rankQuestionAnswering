
import json 
from drqa import retriever
from drqa.retriever import TfidfDocRanker
from drqa.retriever import DocDB 
from utils import split_doc
import prettytable
import csv

#ranker = retriever.get_class('tfidf')#(tfidf_path='./data/wikipedia/docs-tfidf-ngram\=2-hash\=16777216-tokenizer\=simple.npz)
ranker = TfidfDocRanker()
print(ranker)
k=5

db_class = DocDB()
PROCESS_DB = db_class

# def fetch_text(doc_id):
#     global PROCESS_DB
#     return 
total=0

## 1. Read sample questions from SQUAD
question = []
answer = []

i = 0
with open('/Users/rahul/Desktop/ANLP/ProjectANLP/src/DrQA-main/scripts-2/data/datasets/SQuAD-v1.1-train.txt') as f:
    for line in f:
        i+=1
        # if i>3:
        #     break
        q = json.loads(line)['question']
        a = json.loads(line)['answer']
        question.append(q)
        answer.append(a)

#print(question[0:10])
#print(answer[0:10])
print(len(question))

## 2. Get top k documents from wiki for each question query
for j,query in enumerate(question):
    print(j)
    print(query)
    doc_names, doc_scores = ranker.closest_docs(query, k)
    #filename = '../results/query_topdocs.csv'
   
    # table = prettytable.PrettyTable(
    #     ['Rank', 'Doc Id', 'Doc Score']
    # )

    # for i in range(len(doc_names)):
    #              # writing the data rows 
    #     table.add_row([i + 1, doc_names[i], '%.5g' % doc_scores[i]])

    #print(table)
    # with open(filename, 'w') as csvfile: 
    #         # creating a csv writer object 
    #         csvwriter = csv.writer(csvfile) 
    #         for i in range(len(doc_names)):
    #             # writing the data rows 
    #             csvwriter.writerows([query,i + 1, doc_names[i], '%.5g' % doc_scores[i]])
    #             table.add_row([query,i + 1, doc_names[i], '%.5g' % doc_scores[i]])
    # #print(table)

    # docidtoidx = {did: didx for didx, did in enumerate(doc_names)} 
    # print(docidtoidx)                        
                                                                                                         
    ### Fetch text first from DB 
    candidates_para = []
    for doc_id in doc_names:
        doc_text = PROCESS_DB.get_doc_text(doc_id)
        #print(doc_text) 

        ## Split into paragraphs 
        splits = split_doc(doc_text)
        for split in splits:
            candidates_para.append(split)

    total+=len(candidates_para)
    query_para_pair = map(lambda e: (j,e), candidates_para)
    #query_ids = map((j,query))

    # Save query and paragraph pairs which will be input to the second module
    with open('../results/query_para_pair.txt', 'a') as fp:
        fp.write('\n'.join('%s %s' % x for x in query_para_pair))

    with open('../results/query_ids.txt', 'a') as f:
        f.write(str(j) + " " + str(query) + "\n")

with open('../results/stats.txt', 'a') as f:
    f.write("Num of Queries : {0}".format(j))
    f.write("Candidates found : {0}".format(total))
    




    





