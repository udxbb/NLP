## Import Necessary Modules...
import pickle
import project_part2 as project_part2
import time

## Read the data sets...
start_time = time.time()
### Read the Training Data2
train_file = './Data2/train.pickle'
train_mentions = pickle.load(open(train_file, 'rb'))

### Read the Training Labels...
train_label_file = './Data2/train_labels.pickle'
train_labels = pickle.load(open(train_label_file, 'rb'))

### Read the Dev Data2... (For Final Evaluation, we will replace it with the Test Data2)
dev_file = './Data2/dev.pickle'
dev_mentions = pickle.load(open(dev_file, 'rb'))
# for key, value in dev_mentions.items():
#     if value['doc_title'] == '68_CRICKET':
#         print(key, value)

### Read the Parsed Entity Candidate Pages...
fname = './Data2/parsed_candidate_entities.pickle'
parsed_entity_pages = pickle.load(open(fname, 'rb'))

### Read the Mention docs...
mens_docs_file = "./Data2/men_docs.pickle"
men_docs = pickle.load(open(mens_docs_file, 'rb'))

### Read the Dev Labels... (For Final Evaluation, we will replace it with the Test Data2)
dev_label_file = './Data2/dev_labels.pickle'
dev_labels = pickle.load(open(dev_label_file, 'rb'))
# for key, value in dev_mentions.items():
#     print(value['mention'], value['candidate_entities'], '\t\t\n', dev_labels[key]['label'])

## Result of the model...
result = project_part2.disambiguate_mentions(train_mentions, train_labels, dev_mentions, men_docs, parsed_entity_pages)

## Here, we print out sample result of the model for illustration...
for key in list(result)[:5]:
    print('KEY: {} \t VAL: {}'.format(key,result[key]))


## We will be using the following function to compute the accuracy...
def compute_accuracy(result, data_labels):
    assert set(list(result.keys())) - set(list(data_labels.keys())) == set()
    TP = 0.0
    for id_ in result.keys():
        if result[id_] == data_labels[id_]['label']:
            TP +=1
    assert len(result) == len(data_labels)
    return TP/len(result)



accuracy = compute_accuracy(result, dev_labels)
print("Accuracy = ", accuracy)

train_file = './Data1/train.pickle'
train_mentions = pickle.load(open(train_file, 'rb'))

### Read the Training Labels...
train_label_file = './Data1/train_labels.pickle'
train_labels = pickle.load(open(train_label_file, 'rb'))

### Read the Dev Data1... (For Final Evaluation, we will replace it with the Test Data1)
dev_file = './Data1/dev.pickle'
dev_mentions = pickle.load(open(dev_file, 'rb'))
# for key, value in dev_mentions.items():
#     if value['doc_title'] == '68_CRICKET':
#         print(key, value)

### Read the Parsed Entity Candidate Pages...
fname = './Data1/parsed_candidate_entities.pickle'
parsed_entity_pages = pickle.load(open(fname, 'rb'))

### Read the Mention docs...
mens_docs_file = "./Data1/men_docs.pickle"
men_docs = pickle.load(open(mens_docs_file, 'rb'))

### Read the Dev Labels... (For Final Evaluation, we will replace it with the Test Data1)
dev_label_file = './Data1/dev_labels.pickle'
dev_labels = pickle.load(open(dev_label_file, 'rb'))
# for key, value in dev_mentions.items():
#     print(value['mention'], value['candidate_entities'], '\t\t\n', dev_labels[key]['label'])
result = project_part2.disambiguate_mentions(train_mentions, train_labels, dev_mentions, men_docs, parsed_entity_pages)

## Here, we print out sample result of the model for illustration...
for key in list(result)[:5]:
    print('KEY: {} \t VAL: {}'.format(key,result[key]))


accuracy = compute_accuracy(result, dev_labels)
print("Accuracy = ", accuracy)
end_time = time.time()
print(end_time - start_time)