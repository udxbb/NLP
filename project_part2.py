
import numpy as np
import xgboost as xgb
import spacy as sp


def transform_data(features, groups, labels=None):
    xgb_data = xgb.DMatrix(data=features, label=labels)
    xgb_data.set_group(groups)
    return xgb_data


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
        # # print(tf_idf_list)
        # max_tf_idf_token = [x for _, x in sorted(zip(tf_idf_list, value))]
        key_dict = dict(zip(value, tf_idf_list))
        value.sort(key=key_dict.get)
        doc_key_token_dic[key] = value[-15:]
    return doc_key_token_dic


def feature_entity_token_tf_idf(candidate_entities, token_dic, doc_num, doc_title):
    count_list = []
    for candidate_entity in candidate_entities:
        for i in candidate_entity:
            if not i.isalnum() and i != '_':
                candidate_entity = candidate_entity.replace(i, '_')
        # print(candidate_entity)
        tf_idf = 0
        for token in candidate_entity.split('_'):
            try:
                tf = 1 + np.log(1 + np.log(token_dic[token][doc_title]))
                idf = 1 + np.log(doc_num / (1 + len(token_dic[token])))
                tf_idf += tf * idf
            except KeyError:
                pass
        count_list.append(tf_idf)
    return count_list


def feature_entity_tf_idf(candidate_entities, entity_dic, doc_num, doc_title):
    count_list = []
    for candidate_entity in candidate_entities:
        tf_idf = 0
        for i in candidate_entity:
            if not i.isalnum() and i != '_':
                candidate_entity = candidate_entity.replace(i, '')
        candidate_entity = candidate_entity.replace('_', ' ')
        try:
            tf = 1 + np.log(entity_dic[candidate_entity][doc_title])
            idf = 1 + np.log(doc_num / (1 + len(entity_dic[candidate_entity])))
        except KeyError:
            tf = 0
            idf = 1 + np.log(doc_num)
        tf_idf += tf * idf
        count_list.append(tf_idf)
    return count_list


def feature_mention_token_tf_pages(candidate_entities, mention, parsed_tf_dic, parsed_doc):
    count_list = []
    for candidate_entity in candidate_entities:
        tf = 0
        idf = 0
        tf_idf = 0
        for word in mention.split():
            # normal_word = word[0].upper() + word[1:].lower()
            word = word.lower()
            try:
                idf = 1 + np.log(len(parsed_doc) / (len(parsed_tf_dic[word]) + 1))
                tf = parsed_tf_dic[word][candidate_entity]
            except KeyError:
                tf = 0
                idf = np.log(len(parsed_doc))
            tf_idf += tf * idf
        count_list.append(idf * tf)
    return count_list


def feature_same_token(candidate_entities, token_attr_dic, doc_title, new_parsed_pages):
    count_list = []
    for candidate_entity in candidate_entities:
        count = len([t for t in new_parsed_pages[candidate_entity] if t in token_attr_dic[doc_title]])
        count_list.append(count)
    return count_list


def feature_offset(candidate_entities, new_parsed_page1, token_offset_dic, doc_title, mention, offset):
    count_list = []
    token_list = mention.split()
    offset_list = []
    entity_attr_list = []
    entity_tag_list = []
    for i in range(0, len(token_list)):
        offset_list.append(offset)
        offset += len(token_list[i]) + 1
        entity_attr_list.append(token_offset_dic[doc_title][offset_list[i]])
        entity_tag_list.append(token_offset_dic[doc_title][offset_list[i]][3])
    for candidate_entity in candidate_entities:
        count = 0
        description_page = new_parsed_page1[candidate_entity]
        for i in range(0, len(description_page)):
            # the mention is an entity
            if description_page[i][3] in entity_tag_list:
                count += 1

            # if description_page[i] in entity_tag_list:
            #     count += 1

            # the mention is not an entity ('O')
            # test = description_page[i: i+len(entity_attr_list)]
            # if test == entity_attr_list:
            #     count += 1
        count_list.append(count)
    return count_list


def min_distance(word1, word2):
    if not word1:
        return len(word2 or '') or 0
    if not word2:
        return len(word1 or '') or 0
    size1 = len(word1)
    word2 = word2[0: size1]
    size2 = len(word2)
    last = 0
    tmp = list(range(size2 + 1))
    value = None
    for i in range(size1):
        tmp[0] = i + 1
        last = i
        # # print word1[i], last, tmp
        for j in range(size2):
            if word1[i] == word2[j]:
                value = last
            else:
                value = 1 + min(last, tmp[j], tmp[j + 1])
                # # print(last, tmp[j], tmp[j + 1], value)
            last = tmp[j+1]
            tmp[j+1] = value
        # # print tmp
    return value


def bm25(query, parsed_doc_dic, candidate_entities, parsed_tf_dic, avg_length):
    k1 = 1
    k2 = 1
    b = 0.75
    N = len(parsed_doc_dic)
    query = query.split()
    score_list = []
    for entity in candidate_entities:
        score = 0
        for q in query:
            try:
                W = np.log((N + 0.5) / (len(parsed_tf_dic[q]) + 0.5))
                fi = parsed_tf_dic[q][entity]
            except KeyError:
                fi = 0
                W = np.log((N + 0.5) / 0.5)
            q_fi = query.count(q)
            K = k1 * (1 - b + b * len(parsed_doc_dic[entity].split()) / avg_length)
            r = (fi * (k1 + 1)) / (fi + K) * q_fi * (k2 + 1) / (q_fi + k2)
            score += W * r
        score_list.append(score)
    return score_list


def feature_method(data_mentions, parsed_entity_page, new_parsed_pages, new_parsed_page1, token_dic, ent_dic,
                   token_attr_dic, token_offset_dic, doc_num, title_dic, parsed_doc, parsed_tf_dic, avg_length,
                   parsed_pages_token_dic, doc_key_token_dic, parsed_pages_key_token_dic):
    data_label = {}
    feature_0 = []
    feature_1 = []
    feature_2 = []
    feature_3 = []
    feature_4 = []
    feature_5 = []
    feature_6 = []
    feature_7 = []
    for mention_id in data_mentions.keys():
        value_dic = data_mentions[mention_id]
        doc_title = value_dic['doc_title']
        mention = value_dic['mention']
        candidate_entities = value_dic['candidate_entities']
        offset = value_dic['offset']
        query = title_dic[doc_title]
        doc_key_token = doc_key_token_dic[doc_title]
        # feature 1: number of entities appear in parsed_page that also appear in doc
        # Accuracy =  0.7429577464788732 Accuracy =  0.5217391304347826
        temp_list1 = feature_same_token(candidate_entities, token_attr_dic, doc_title, new_parsed_pages)
        index_list1 = [i for i, x in enumerate(temp_list1) if x == max(temp_list1)]
        # feature_1 list used for XGBoost
        feature_1.extend(temp_list1)
        # if max number of candidate entity is the same, compare the feature two
        # if len(index_list1) == 1:
        data_label[mention_id] = candidate_entities[index_list1[0]]
        #
        # feature 0: find the token that corresponds the location of the mention, check the entity type of the token
        # and compare with entity in parsed page
        # Accuracy =  0.5211267605633803 Accuracy =  0.3167701863354037
        # temp_list0 = feature_offset(candidate_entities, new_parsed_page1, token_offset_dic, doc_title, mention, offset)
        # index_list0 = [i for i, x in enumerate(temp_list0) if x == max(temp_list0)]
        # # feature_0 list used for XGBoost
        # feature_0.extend(temp_list0)
        # # if len(index_list0) == 1:
        # data_label[mention_id] = candidate_entities[index_list0[0]]
        #
        # feature_2: calculate mentions' token tf in candidate entity page
        # Accuracy =  0.6091549295774648 0.5093167701863354
        temp_list2 = feature_mention_token_tf_pages(candidate_entities, mention, parsed_tf_dic, parsed_doc)
        index_list2 = [i for i, x in enumerate(temp_list2) if x == max(temp_list2)]
        # feature_2 list used for XGBoost
        feature_2.extend(temp_list2)
        # if data_label[mention_id] is None and len(index_list2) == 1:
        data_label[mention_id] = candidate_entities[index_list2[0]]

        # feature_3 entities's tokens' tf_idf in doc
        # Accuracy =  0.19014084507042253   0.42857142857142855
        temp_list3 = feature_entity_token_tf_idf(candidate_entities, token_dic, doc_num, doc_title)
        index_list3 = [i for i, x in enumerate(temp_list3) if x == max(temp_list3)]
        # feature_3 list used for XGBoost
        feature_3.extend(temp_list3)
        # if data_label[mention_id] is None and len(index_list2) == 1:
        data_label[mention_id] = candidate_entities[index_list3[0]]

        # # feature_4 entity tf_idf in doc
        # # Accuracy =  0.5880281690140845  Accuracy =  0.3105590062111801
        # temp_list4 = feature_entity_tf_idf(candidate_entities, ent_dic, doc_num, doc_title)
        # index_list4 = [i for i, x in enumerate(temp_list4) if x == max(temp_list4)]
        # # feature_4 list used for XGBoost
        # feature_4.extend(temp_list4)
        # # if data_label[mention_id] is None and len(index_list4) == 1:
        # data_label[mention_id] = candidate_entities[index_list4[0]]

        # feature_5: min edit distance
        # Accuracy =  0.6725352112676056 Accuracy =  0.484472049689441
        temp_list5 = []
        for candidate_entity in candidate_entities:
            temp_list5.append(min_distance(mention, candidate_entity))
        index_list5 = [i for i, x in enumerate(temp_list5) if x == min(temp_list5)]
        feature_5.extend(temp_list5)
        data_label[mention_id] = candidate_entities[index_list5[0]]

        temp_list6 = bm25(query, parsed_doc, candidate_entities, parsed_tf_dic, avg_length)
        feature_6.extend(temp_list6)
        index_list6 = [i for i, x in enumerate(temp_list6) if x == max(temp_list6)]
        data_label[mention_id] = candidate_entities[index_list6[0]]

        temp_list7 = []
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
            # print(union_key_token)
            # union_key_token.sort()
            # # print(union_key_token)
            tf_1 = []
            tf_2 = []
            for token in union_key_token:
                try:
                    tf_1.append(token_dic[token][doc_title])
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

            temp_list7.append(cos)
            index_list = [i for i, x in enumerate(temp_list7) if x == max(temp_list7)]
            data_label[mention_id] = candidate_entities[index_list[0]]
        feature_7.extend(temp_list7)

    return feature_0, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, data_label


# men_docs: dic
def get_related_dic(men_docs):
    nlp = sp.load("en_core_web_sm")
    token_dic = {}
    entity_dic = {}
    token_attr_dic = {}
    token_offset_dic = {}
    doc_token_list = {}
    title_dic = {}
    for key, value in men_docs.items():
        title = str(key)
        doc = nlp(value)
        token_attr_dic[title] = []
        token_offset_dic[title] = {}
        subject = ''
        doc_token_list[title] = []
        for word in value:
            subject += word
            if word == '.':
                title_dic[key] = subject.lower()
                break
        for token in doc:
            if not token.is_stop and not token.is_punct:
                doc_token_list[title].append(token.text.lower())
                if token.text in token_dic.keys():
                    if title in token_dic[token.text].keys():
                        token_dic[token.text][title] += 1
                    else:
                        token_dic[token.text][title] = 1
                else:
                    token_dic[token.text] = {}
                    token_dic[token.text][title] = 1
            if token.ent_iob_ != 'O':
                token_attr_dic[title].append([token.text, token.lemma_, token.pos_, token.ent_type_])
                entity_tag = token.ent_iob_ + '-' + token.ent_type_
            else:
                entity_tag = token.ent_iob_
            token_offset_dic[title][token.idx] = (token.text, token.lemma_, token.pos_, token.ent_type_)
        for ent in doc.ents:
            if ent.text in entity_dic.keys():
                if title in entity_dic[ent.text].keys():
                    entity_dic[ent.text][title] += 1
                else:
                    entity_dic[ent.text][title] = 1
            else:
                entity_dic[ent.text] = {}
                entity_dic[ent.text][title] = 1
    return token_dic, entity_dic, token_attr_dic, token_offset_dic, title_dic, doc_token_list


def get_parsed_dic(parsed_entity_pages):
    parsed_doc = {}
    length = 0
    for key, value in parsed_entity_pages.items():
        parsed_doc[key] = ''
        for t in value:
            parsed_doc[key] += t[1].lower() + ' '

    parsed_tf_dic = {}
    for key, value in parsed_doc.items():
        length += len(value.split())
        for word in value.split():
            if word in parsed_tf_dic.keys():
                parsed_tf_dic[word][key] = parsed_tf_dic[word].get(key, 0) + 1
            else:
                parsed_tf_dic[word] = {}

    avg_length = length / len(parsed_doc)
    return parsed_doc, parsed_tf_dic, avg_length


def disambiguate_mentions(train_mentions, train_labels, dev_mentions, men_docs, parsed_entity_pages):
    pass
    # test code:
    count = 0
    for value in dev_mentions.values():
        count += len(value['candidate_entities'])
    # print(count)
    # get related information from the mention documents
    # token_dic: all documents' token   key: token, value: {doc_title: times}
    # entity_dic: all documents' entity  key: entity, value: {doc_title: time}
    # token_attr_dic: all documents' tokens' attribute
    # key: doc_title, value: [[token_id, token_text, token_lemma, post_tag, entity-tag (I-ORG)]]
    token_dic, entity_dic, token_attr_dic, token_offset_dic, title_dic, doc_token_list = get_related_dic(men_docs)
    parsed_doc, parsed_tf_dic, avg_length = get_parsed_dic(parsed_entity_pages)

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

    doc_key_token_dic = calculate_tf_idf(doc_token_list, token_dic)
    parsed_pages_key_token_dic = calculate_tf_idf(parsed_pages_token_list, parsed_pages_token_dic)

    # print(avg_length)

    # remove the id and first part of the entity_tag (I/B) and not-entity token
    new_parsed_page = {}
    for key, value in parsed_entity_pages.items():
        new_parsed_page[key] = []
        for t in value:
            if t[4] != 'O':
                convert_t = list(t)
                new_parsed_page[key].append([convert_t[1], convert_t[2], convert_t[3], convert_t[4][2:]])

    # only remove the id
    new_parsed_page_1 = {}
    for key, value in parsed_entity_pages.items():
        new_parsed_page_1[key] = []
        for t in value:
            if t[4] != 'O':
                new_parsed_page_1[key].append((t[1], t[2], t[3], t[4][2:]))

    test_feature_0, test_feature_1, test_feature_2, test_feature_3, test_feature_4, test_feature_5, test_featrue_6, test_feature_7, data_labels = \
        feature_method(dev_mentions, parsed_entity_pages, new_parsed_page, new_parsed_page_1, token_dic, entity_dic,
                       token_attr_dic, token_offset_dic, len(men_docs),  title_dic, parsed_doc, parsed_tf_dic,
                       avg_length, parsed_pages_token_dic, doc_key_token_dic, parsed_pages_key_token_dic)

    doc_num = len(men_docs)
    train_feature_0, train_feature_1, train_feature_2, train_feature_3, train_feature_4, train_feature_5, train_feature_6, train_feature_7, _ = \
        feature_method(train_mentions, parsed_entity_pages, new_parsed_page, new_parsed_page_1, token_dic, entity_dic,
                       token_attr_dic, token_offset_dic, len(men_docs), title_dic, parsed_doc,
                       parsed_tf_dic, avg_length, parsed_pages_token_dic, doc_key_token_dic, parsed_pages_key_token_dic)

    train_data = np.column_stack((train_feature_2, train_feature_3, train_feature_5, train_feature_6))
    test_data = np.column_stack((test_feature_2, test_feature_3, test_feature_5, test_featrue_6))
    # print(test_feature_7)
    train_groups = []
    test_groups = []
    for value in train_mentions.values():
        train_groups.append(len(value['candidate_entities']))
    for value in dev_mentions.values():
        test_groups.append(len(value['candidate_entities']))
    # print(sum(train_groups))
    # print(sum(test_groups))
    train_groups = np.array(train_groups)
    test_groups = np.array(test_groups)

    label_list = []
    for mention_id in train_mentions.keys():
        for entity in train_mentions[mention_id]['candidate_entities']:
            if entity == train_labels[mention_id]['label']:
                label_list.append(1)
            else:
                label_list.append(0)
    label_list = np.array(label_list)
    # print(label_list, len(label_list))

    xgboost_train = transform_data(train_data, train_groups, label_list)
    xgboost_test = transform_data(test_data, test_groups)
    param = {'max_depth': 2, 'eta': 0.5, 'silent': 1, 'objective': 'rank:pairwise',
             'min_child_weight': 2, 'lambda': 100}

    classifier = xgb.train(param, xgboost_train, num_boost_round=280)

    #  Predict test data...
    preds = classifier.predict(xgboost_test)
    preds = preds.tolist()

    begin = 0
    for i in range(1, len(dev_mentions) + 1):
        sub_list = preds[begin: begin + test_groups[i - 1]]
        index = sub_list.index(max(sub_list))
        begin += test_groups[i - 1]
        data_labels[i] = dev_mentions[i]['candidate_entities'][index]

    return data_labels
