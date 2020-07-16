import numpy as np
import spacy as sp


def calculate_tf_idf(doc_entity_list, doc_token_dic):
    doc_key_token_dic = {}
    total_doc_num = len(doc_entity_list)
    for key, value in doc_entity_list.items():
        tf_idf_list = []
        for token in value:
            tf_idf = 0
            try:
                tf = 1.0 + np.log((1.0 + np.log(doc_token_dic[token][key])))
                idf = 1.0 + np.log(total_doc_num / (1 + len(doc_token_dic[token])))
                tf_idf = tf * idf
            except KeyError:
                pass
            tf_idf_list.append(tf_idf)
        # print(tf_idf_list)
        # max_tf_idf_token = [x for _, x in sorted(zip(tf_idf_list, value))]
        key_dict = dict(zip(value, tf_idf_list))
        value.sort(key=key_dict.get)
        doc_key_token_dic[key] = value[-15:]
    return doc_key_token_dic


def disambiguate_mentions(train_mentions, train_labels, dev_mentions, men_docs, parsed_entity_pages):
    pass
    dev_label = {}
    nlp = sp.load("en_core_web_sm")
    doc_token_dic = {}
    doc_entity_dic = {}
    doc_token_list = {}
    doc_entity_list = {}
    for key, value in men_docs.items():
        doc_title = str(key)
        doc = nlp(value)
        doc_token_list[doc_title] = []
        doc_entity_list[doc_title] = []
        for token in doc:
            if not token.is_stop and not token.is_punct and token.text.lower() != '\n ' and len(token.text) != 1:
                doc_token_list[doc_title].append(token.text.lower())
                if token.text.lower() in doc_token_dic.keys():
                    doc_token_dic[token.text.lower()][doc_title] = doc_token_dic[token.text.lower()].get(doc_title, 0) + 1
                else:
                    doc_token_dic[token.text.lower()] = {}
                    doc_token_dic[token.text.lower()][doc_title] = 1

        for entity in doc.ents:
            doc_entity_list[doc_title].extend(entity.text.lower().split())
            if entity.text in doc_entity_dic.keys():
                doc_entity_dic[entity.text][doc_title] = doc_entity_dic[entity.text].get(doc_title, 0) + 1
            else:
                doc_entity_dic[entity.text] = {}
                doc_entity_dic[entity.text][doc_title] = 1
    for key, value in doc_entity_list.items():
        doc_entity_list[key] = list(set(value))
    for key, value in doc_token_list.items():
        temp = []
        [temp.append(i) for i in value if i not in temp]
        doc_token_list[key] = temp

    # for key, value in doc_token_list.items():
    #     temp = []
    #     for token in value:
    #         if token not in temp:
    #             temp.append(token)
    #     doc_token_list[key] = temp



    # print(doc_token_dic)
    # print(doc_entity_dic)
    # print(doc_token_list)
    # print(doc_entity_list)
    # print(len(men_docs))

    parsed_pages_token_list = {}
    parsed_pages_token_dic = {}
    for key, value in parsed_entity_pages.items():
        parsed_pages_token_list[key] = []
        for t in value:
            if t[1].isalnum() and t[1] != r'\n\n':
                parsed_pages_token_list[key].append(t[1].lower())
            if t[1] in parsed_pages_token_dic.keys():
                parsed_pages_token_dic[t[1]][key] = parsed_pages_token_dic[t[1]].get(key, 0) + 1
            else:
                parsed_pages_token_dic[t[1]] = {}
                parsed_pages_token_dic[t[1]][key] = 1

    for key, value in parsed_pages_token_list.items():
        temp = []
        [temp.append(i) for i in value if i not in temp]
        parsed_pages_token_list[key] = temp

    doc_key_token_dic = calculate_tf_idf(doc_token_list, doc_token_dic)
    parsed_pages_key_token_dic = calculate_tf_idf(parsed_pages_token_list, parsed_pages_token_dic)

    print(doc_key_token_dic['Commonwealth_men'])
    # for key, value in parsed_pages_token_dic.items():
    #     print(key, value)
    # for key, value in parsed_pages_token_list.items():
    #     print(key, value)


    cos_list = []

    for key, value in dev_mentions.items():
        temp_list = []
        title = value['doc_title']
        mention = value['mention'].lower()
        candidate_entities = value['candidate_entities']
        offset = value['offset']
        doc_key_token = doc_key_token_dic[title]
        for entity in candidate_entities:
            parsed_key_token = parsed_pages_key_token_dic[entity]
            # union_key_token = list(set(doc_key_token + parsed_key_token + mention.split()))
            union_key_token = []
            for token in doc_key_token:
                if token not in union_key_token:
                    union_key_token.append(token)
            for token in parsed_key_token:
                if token not in union_key_token:
                    union_key_token.append(token)
            for token in mention.split():
                if token not in union_key_token:
                    union_key_token.append(token)
            print(union_key_token)
            # union_key_token.sort()
            # print(union_key_token)
            tf_1 = []
            tf_2 = []
            for token in union_key_token:
                try:
                    tf_1.append(doc_token_dic[token][title])
                except KeyError:
                    tf_1.append(0)
                try:
                    tf_2.append(parsed_pages_token_dic[token][entity])
                except KeyError:
                    tf_2.append(0)

            denominator_1 = 0
            denominator_2 = 0
            molecule = 0
            for index in range(0, len(tf_1)):
                i = tf_1[index]
                j = tf_2[index]
                molecule += i * j
                denominator_1 += i * i
                denominator_2 += j * j

            cos = molecule / (np.sqrt(denominator_1) * np.sqrt(denominator_2))

            temp_list.append(cos)
            index_list = [i for i, x in enumerate(temp_list) if x == max(temp_list)]
            dev_label[key] = candidate_entities[index_list[0]]
        cos_list.extend(temp_list)
        # print(temp_list)
    # print(cos_list)

    return dev_label
