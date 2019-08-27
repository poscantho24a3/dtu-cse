import json
import numpy as np
import re
import math
import os
from scipy import spatial


def get_data(path_input):
    with open(path_input, encoding='utf-8-sig') as json_file:
        input = json.load(json_file)
    return input

def word_segmentation(sentence,OnStopWord):
    rex = re.compile(r"[\\/&![\]#,+(\-–)$~%.…=:;*?<>{}.^]") 
    sentence = re.sub(rex, '', sentence).lower().strip()
    sentence = re.sub('(\tr|\r\n|\n|\r|\f)', '', sentence)
    sentence = re.sub('\s+/g', ' ', sentence)
    arrWord = sentence.split(" ")
    path_dict = "./data/lowercase_dict.json"
    dict_word = get_data(path_dict)
    gram = len(arrWord)
    stopword = {}
    # if OnStopWord : 
    stopword["tôi"] =1
    stopword["và"] =1
    stopword["của"] =1
    stopword["nhưng"] =1
    stopword["bị"] =1
    stopword["do"] =1
    stopword["vậy"] =1
    stopword["một"] =1
    stopword["về"] =1
    stopword["qua"] =1
    stopword["thích"] =1
    stopword["sau"] =1
    stopword["dạ"] =1
    stopword["vâng"] =1
    stopword["khi"] =1
    stopword["muốn"] =1
    stopword["cũng"] =1
    stopword["quá"] =1
    stopword["nhiều"] =1
    stopword["không"] =1
    stopword["có"] =1
    stopword["nên"] = 1
    result = []
    for i in range(gram):
        word = ""
        gram4 = ""
        gram3 = ""
        gram2 = ""
        gram1 = ""

        if len(arrWord) >= 4:
            gram4 = arrWord[0] + " " + arrWord[1] + " " + arrWord[2] + " " + arrWord[3]
        if len(arrWord) >= 3:
            gram3 = arrWord[0] + " " + arrWord[1] + " " + arrWord[2]
        if len(arrWord) >= 2:
            gram2 = arrWord[0] + " " + arrWord[1]
        if len(arrWord) >= 1:
            gram1 = arrWord[0]

        if not arrWord:
            break
        if (gram4 != "") & (gram4 in dict_word):
            word = gram4
            del arrWord[0:4]
        elif (gram3 != "") & (gram3 in dict_word):
            word = gram3
            del arrWord[0:3]
        elif (gram2 != "") & (gram2 in dict_word):
            word = gram2
            del arrWord[0:2]
        elif (gram1 != "") & (gram1 in dict_word):
            word = gram1
            del arrWord[0]
        else:
            word = gram1    
            del arrWord[0]
        if word not in stopword and word != "" :
            result.append(word)
            
    return result

def chunkSentence(segmentation):
    arrChunk = []
    if len(segmentation) < 2:
        return segmentation
    for i in range(len(segmentation) -1):
        chukI= segmentation[i]+"_"+ segmentation[i+1]
        arrChunk.append(chukI)
    return arrChunk    

def createTFSentence(sentence):
    tf = dict()
    if len(sentence) > 0:
        max = 1
        for s in sentence:
            if s in tf:
                tf[s] = tf[s] + 1
            else:
                tf[s] = 1
            # if tf[s] >= max :
            #     max =  tf[s]    
        for key in tf:
            tf[key] = tf[key]/len(sentence)

    return tf
def addKeyTojson():
    path_input = "./data/listAnserQ.json"
    list_data = get_data(path_input)
    listAnswerQuestion = []
    #  "tag": 0,
    # "content": "Kẹt xe",
    # "context_question": "Bạn có thể nói rõ hơn được không?"
    for data in list_data:
        listAnswerQuestion.append({"tag":data['tag'],"content":data['content'],"context_question":data['context_question'],"End":False})
    with open('./data/listAnserQ.json', 'w+', encoding='utf-8-sig') as json_file:
        json.dump(listAnswerQuestion, json_file, ensure_ascii=False)
    print('write: thêm key vào json thành công!')    
 
def createTFIDFWordForSentence(count_dict_word):
    path_input = "./data/listAnserQ.json"
    list_data = get_data(path_input)
    listAnswerQuestion = []
    N = len(list_data)
    for data in list_data:
        content = data['content']
        tag = data['tag']
        word_seg = word_segmentation(content,True)
        tf = createTFSentence(word_seg)
        tf_idf = dict()
        for key in tf:
            idf = math.log10(N/count_dict_word[key])
            tf_idf[key] = tf[key]*idf
        listAnswerQuestion.append({'content': tf_idf, 'tag': tag})
    with open('./data/TFIDFSentence.json', 'w+', encoding='utf-8-sig') as json_file:
        json.dump(listAnswerQuestion, json_file, ensure_ascii=False)
    print('write: TFIDFSentence.json thành công!')

def preProcessScropus():
    import time
    start_time = time.time()
    path_input = "./data/listAnserQ.json"
    list_data = get_data(path_input)
    dict_result = dict()
    for element in list_data:
        sentence = element['content']
        word_seg = word_segmentation(sentence,True)
        arr_temp = []
        for e in word_seg :
            if e not in arr_temp : #and bool(e.strip())
                arr_temp.append(e)
                if e in dict_result:
                    dict_result[e] = dict_result[e] + 1
                else:
                    dict_result[e] = 1
    with open('./data/CountWordinDocument.json', 'w+', encoding='utf-8-sig') as json_file:
        json.dump(dict_result, json_file, ensure_ascii=False)
    print('write: CountWordinDocument.json thành công!')
    print("--- %s Count N word ---" % (time.time() - start_time))
    createTFIDFWordForSentence(dict_result)
def RemoveDupplicate():
    import time
    start_time = time.time()
    path_input = "./data/listAnserQ.json"
    list_data = get_data(path_input)
    dict_result = dict()
    dict_result["nameFile"] = path_input
    for element in list_data:
        key = element["content"]
        if key not in dict_result:
            dict_result[key] = 1
        else:
            dict_result[key] = dict_result[key] + 1
    with open('./data/UpdatelistAnserQ.json', 'w+', encoding='utf-8-sig') as json_file:
        json.dump(dict_result, json_file, ensure_ascii=False)
        print("check done")
def FindNewQuestion():
    RemoveDupplicate()
    import time
    start_time = time.time()
    path_input = "./data/ListSentenceNotVaild.json"
    list_dataQ = get_data(path_input)
    path_input = "./data/listAnserQ.json"
    list_data = get_data(path_input)
    dict_result = dict()
    dict_result["nameFile"] = path_input
    for key in list_dataQ:
        match = False
        for item in list_data:
            if  key["content"].lower().strip() ==  item["content"].lower().strip():
                match = True
                continue
        if not match :
            print(key["content"].lower().strip())   
    #     key = element["content"]
    #     if key not in dict_result:
    #         dict_result[key] = 1
    #     else:
    #         dict_result[key] = dict_result[key] + 1
    # with open('./data/notInList.json', 'w+', encoding='utf-8-sig') as json_file:
    #     json.dump(dict_result, json_file, ensure_ascii=False)
    #     print("check done")


def initModel():
    path_input_model = "./data/TFIDFSentence.json"
    data_model = get_data(path_input_model)
    # print(data_model)
    path_input_count = "./data/CountWordinDocument.json"
    data_count = get_data(path_input_count)
    # print(data_count)
    newList = []
    for data in data_model:
        content = data['content']
        newJson = []
        for key in data_count:
            if key in content:
                newJson.append(content[key])
            else:
                newJson.append(0)
        newList.append({'data': newJson, 'label': data['tag']})
    with open('./model/Model.json', 'w+', encoding='utf-8-sig') as json_file:
        json.dump(newList, json_file, ensure_ascii=False)
    print('Khởi tạo model xong!')


def createNewTFIFOneSentence(sen):
    path_input = "./data/listAnserQ.json"
    list_data = get_data(path_input)
    N = len(list_data)
    path_input_count = "./data/CountWordinDocument.json"
    data_count = get_data(path_input_count)
    arr_word = word_segmentation(sen,False)
    tf = createTFSentence(arr_word)
    tf_idf = {}
    for key in tf:
        countN = 1
        idf = 0
        if key in data_count:
            countN = data_count[key]
            idf = math.log10(N/countN)
        tf_idf[key] = tf[key] #tf[key] * idf
    newJson = []
    for key in data_count:
        if key in tf_idf:
            newJson.append(tf_idf[key])
        else:
            newJson.append(0)
    with open('./data/NewTFIDFOne.json', 'w+', encoding='utf-8-sig') as json_file:
        json.dump(newJson, json_file, ensure_ascii=False)
    return newJson

def createVectorStan(senA, senB):
    vector_chuan = {}
    for key in senA:
        vector_chuan[key] = 0
    for key in senB:
        vector_chuan[key] = 0
    return vector_chuan

def cosin2Sentence(sentence1, sentence2):
    stopword = {}
    c = createVectorStan(sentence1, sentence2)
    if len(c) == len(sentence1) + len(sentence2):
        return 0
    A = 0
    B = 0
    powA = 0
    powB = 0
    totalAB = 0
    for key in c:
        if key not in stopword:
            if  key in sentence1:
                if sentence1[key] > 0:
                    A = sentence1[key]
                    powA = powA + A*A
                else:
                    A = 0
            else:
                A = 0
            if key in sentence2:
                if sentence2[key] > 0:
                    B = sentence2[key]
                    powB = powB + B*B
                else:
                    B = 0
            else:
                B = 0
            totalAB = totalAB + A*B 
    cosin_result = totalAB/(math.sqrt(powA)*math.sqrt(powB))
    return cosin_result

def similarListToken(sen, label):
    # indexMax = conSinNew(label)
    # print(label)
    path_input_model = "./data/TFIDFSentence.json"
    data_model = get_data(path_input_model)
    path_input_count = "./data/CountWordinDocument.json"
    data_count = get_data(path_input_count)
    # # print(data_count)
    path_input = "./data/listAnserQ.json"
    list_data = get_data(path_input)
    arr_word = word_segmentation(sen,False)
    # arr_word = chunkSentence(arr_word)# tạo chunk cho câu
    if len(arr_word) == 0:
        return {"context_question":"Tôi không hiểu câu nói của bạn","tag":False,"End":False }
    tf = createTFSentence(arr_word)
    tf_idf = {}
    for key in tf:
        if key not in data_count:
            data_count[key] = 1
        idf = math.log10(len(data_model)/data_count[key])
        tf_idf[key] = tf[key] #*idf
    indexMax = {}
    indexMax['max'] = 0
    indexMax['value'] = 0
    list_cos = []
    for i in range(len(data_model)):
        content = data_model[i]['content']
        lop = data_model[i]['tag']
        if lop == label:
            cos = cosin2Sentence(content, tf_idf)
            list_cos.append({"cosin":cos,"index":i})
            if cos > indexMax['value']:
                indexMax['value'] = cos
                indexMax['max'] = i
    print("Tương đồng với: ",word_segmentation(list_data[indexMax['max']]['content'],False),indexMax['value'])
    # change funtion here
    if indexMax['value'] < 0.3:
        path_listSentenceNotVaild = "./data/ListSentenceNotVaild.json"
        data = {"context_question":"Tôi không hiểu câu nói của bạn","tag":False,"End":False }
        if os.stat(path_listSentenceNotVaild).st_size == 0:
            arr_sentense = []
            arr_sentense.append({"content": sen, "tag": False})
            with open(path_listSentenceNotVaild, 'w+', encoding='utf-8-sig') as json_file:
                json.dump(arr_sentense, json_file, ensure_ascii=False)
        else:
            data_listSentenceNotVaild = get_data(path_listSentenceNotVaild)
            data_listSentenceNotVaild.append({"content": sen,"context_question":"Tôi không hiểu câu nói của bạn","tag": False,"End":True})
            with open(path_listSentenceNotVaild, 'w+', encoding='utf-8-sig') as json_file:
                json.dump(data_listSentenceNotVaild, json_file, ensure_ascii=False)
        return data
    else:
        print(list_data[indexMax['max']]['context_question'],indexMax['value'])
        if list_data[indexMax['max']]['context_question']:
            return list_data[indexMax['max']]
        else:
            return data

def conSinNew(label):
    print("cosin new",label)
    path_tfidfSen = "./data/NewTFIDFOne.json"
    dataSetSen = data_model = get_data(path_tfidfSen) 
    path_input_model = "./model/Model.json"
    data_model = get_data(path_input_model)
    indexMax = {}
    indexMax['max'] = 0
    indexMax['value'] = 0
    index = -1
    result = 0
    for element in data_model:
        index = index + 1
        dataSetI= element['data']
        labelI = element['label']
        if label == labelI :
            result = 1 - spatial.distance.cosine(dataSetSen, dataSetI)
            if result >= indexMax['value'] :
                indexMax['max'] = index
                indexMax['value'] = result
    print(indexMax)
    return indexMax

# preProcessScropus()
# initModel()
# FindNewQuestion()
# segment = word_segmentation("tôi đi tìm em qua muôn vạn đường tình",False)
# print(segment)
# chunk = chunkSentence(segment)
# print(chunk);
# addKeyTojson()