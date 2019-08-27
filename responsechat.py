import tensorflow as tf
import json
import numpy as np
from tensorflow import keras
from keras.models import load_model
import random
import time


from MachineSimilarAnswser import *

class_names = ["LIỆT_KÊ", "ĐẶT_MÓN", "THAY_ĐỔI_MÓN","END"]
path_input = './model/Model.json'
path_save_model = '.\data\ModelClass.bin'
path_test = './data/NewTFIDFOne.json'
pathMenuFood = './data/DataMenu.json'
orderItemTam = []

def train_model():
    with open(path_input, encoding='utf-8-sig') as json_file:
        input = json.load(json_file)

    random.shuffle(input)  # random mảng dữ liệu
    arr_train = []
    label_train = []
    arr_test = []
    label_test = []
    indexTest = []
    i = 0

    for element in input:
        length = len(element['data'])
        i = i + 1
        # print(element['label'])
        if i < 3000:
            arr_train.append(element['data'])
            label_train.append(element['label'])
        else:
            arr_test.append(element['data'])
            label_test.append(element['label'])

    train_sentences = np.array(arr_train)
    train_labels = np.array(label_train)
    test_sentences = np.array(arr_test)
    test_labels = np.array(label_test)
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(length,)),
        keras.layers.Dense(128, activation=tf.nn.relu),
        keras.layers.Dense(len(class_names), activation=tf.nn.softmax)
    ])

    # config model when train
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(train_sentences, train_labels,
              epochs=500)  # epochs: so lan train
    if len(test_sentences) > 0:
        test_loss, test_acc = model.evaluate(test_sentences, test_labels)
        print('\nTest accuracy:', test_acc, 'loss: ', test_loss)
        print(test_sentences, test_labels)
    # if test_acc > 0.8 :
    #     model.save(path_save_model)
    #     return
    # else:
    #     return train_model()

    return model.save(path_save_model)


def predictSentence(sen):
    sen = chuyen_doi_chu_so_thanh_so(sen)
    vectorInput = createNewTFIFOneSentence(sen)
    vectorInput = np.asarray(vectorInput)
    start_time = time.time()
    if 'model' not in globals():
        global model
        model = load_model(path_save_model)
    if 'orderItemTam' not in globals():
        global orderItemTam    
    predicted = model.predict(np.array([vectorInput, ]))
    index = np.argmax(predicted[0])
    print(word_segmentation(sen, False), ": dự đoán max ==>",class_names[index], predicted[0][index])
    weigth = predicted[0][index]
    data = {"context_question": "Vui lòng thử lại", "tag": False, "End": False}
    if weigth >= 0.7:
        action = "showList"  # hiển thị danh sách
        Food = "null"
        num_oder = 0
        data = similarListToken(sen, index)
        if index != 0 and index != 3:
            Food = detectionListFood(sen) # lấy danh sách các món ăn
            action  = detectionDetailAction(sen,Food) # lấy danh sách các hành động
            num_oder = detection_number(sen,Food)
            print("order",num_oder)
            print("action",action)
            FoodArraySort = [] 
            for i in sorted (Food.keys()):  
                FoodArraySort.append(Food[i])
            Food = FoodArraySort
            num = 0
            print("Food",Food)
            if len(Food) > 0 and len(num_oder):
                if len(Food) == len(num_oder):
                    for i in range(len(action)):
                        num = num_oder[i]
                        if action[i] == 2 : # xóa hết hóa đơn
                            if num == 0:
                                deleteFoodInOrder(Food[i],False)
                            else :
                                deleteFoodInOrder(Food[i],num)   
                                
                        if action[i] == 0 : # thêm vào hóa đơn
                            AddFoodInOrder(Food[i],num)
                        if action[i] == 1:  # giảm số lượng dữ thức uống hóa đơn
                            GiamFoodInOrder(Food[i],num)     
                else:
                    for i in range(len(action)):
                        num = num_oder[i]
                        if action[i] == 2 : # xóa dữ liệu
                            if num == 0:
                                deleteFoodInOrder(Food[i],False)
                            else :
                                deleteFoodInOrder(Food[i],num)   
                        if action[i] == 0: 
                            AddFoodInOrder(Food[i],num)
                        if action[i] == 1: 
                            GiamFoodInOrder(Food[i],num)    
            else: 
                return {"status":False, "End":False,"data":[]} # không tìm thấy món ăn
            print("Hóa đơn",orderItemTam)
            return {"status":True, "End":False,"data":orderItemTam}

        if index == 0:
            
            Food = data["context_question"] # Loại liệt kê
        if index == 3:
            return {"status":True, "End":True,"data":orderItemTam}
    else:
        print("Không hiểu")
        return {"status":False, "End":False,"data":[]} # trường hợp  không phân lớp


def UpdateNewQuestion():
    path_input = "./data/ListSentenceNotVaild.json"
    list_data = get_data(path_input)
    listAnswerQuestion = []
    #  "tag": 0,
    # "content": "Kẹt xe",
    # "context_question": "Bạn có thể nói rõ hơn được không?"
    for data in list_data:
        content = data['content']
        # listAnswerQuestion.append({"tag":data['tag'],"content":data['content'],"context_question":data['context_question'],"End":False})
        Class = predictSentence(content)
        lable = Class["label"]
        weigth = Class["weigth"]

        if weigth >= 0.9:
            data["newtag"] = int(lable)
            data["nameLabel"] = class_names[lable]
        if "context_question" not in data:
            data["context_question"] = "Tôi chưa hiểu câu nói của bạn"
        if "End" not in data:
            data["End"] = True
        listAnswerQuestion.append(data)
    with open('./data/UpdatelistAnserQ3.json', 'w+', encoding='utf-8-sig') as json_file:
        json.dump(listAnswerQuestion, json_file, ensure_ascii=False)
    print('write: thêm key vào json thành công!')

def deleteFoodInOrder(food,num):
    if num : 
        for i in range(len(orderItemTam)):
            item = orderItemTam[i]
            if food["name"] == item["name"]:
                orderItemTam[i]["num_order"] = orderItemTam[i]["num_order"] - num
    else:            
        for i in range(len(orderItemTam)):
            item = orderItemTam[i]
            if food["name"] == item["name"]:
                del  orderItemTam[i]

def AddFoodInOrder(food,num):
    if 'orderItemTam' not in globals():
        global orderItemTam 
    # print("Gọi hàm add food")
    if len(orderItemTam) == 0:
        orderItemTam.append({"name":food["name"],"price":food["price"],"num_order":num})
    else:
        print(orderItemTam)
        checkHave = False    
        for i in range(len(orderItemTam)):
            item = orderItemTam[i]
            if food["name"] == item["name"]:
                checkHave = True
                print("thêm  food",num,food)
                orderItemTam[i]["num_order"] = orderItemTam[i]["num_order"] + num 
        if not checkHave :
            orderItemTam.append({"name":food["name"],"price":food["price"],"num_order":num})

def GiamFoodInOrder(food,num):
    if 'orderItemTam' not in globals():
        global orderItemTam 
    # print("Gọi hàm add food")
    if len(orderItemTam) == 0:
        orderItemTam.append({"name":food["name"],"price":food["price"],"num_order":num})
    else:
        print(orderItemTam)
        checkHave = False    
        for i in range(len(orderItemTam)):
            item = orderItemTam[i]
            if food["name"] == item["name"]:
                print("giảm  food",num,food["name"])
                orderItemTam[i]["num_order"] = orderItemTam[i]["num_order"] - num

def detection_number(sentence,Food):
    sentence = word_segmentation(sentence,False)
    arrNumber = []
    if len(sentence) > 0:
        for word in sentence:
            if word.isdigit():
                arrNumber.append(int(word))
    print(arrNumber)            
    return arrNumber

def detectionDetailAction(sentence,Food):
    data = dict()
    # thêm số lương
    data["thêm"] = 0
    data["đặt"] = 0
    data["cho"] = 0
    data["tăng"] = 0
    data["chọn"] = 0
    data["mua"] = 0
    data["lấy"] = 0
    data["gọi"] = 0
    data["cập nhật"] = 0
    # giảm số lượng
    data["bớt"] = 1
    data["giảm"] = 1
    data["giảm bớt"] = 1
    data["ít"] = 1
    data["bỏ bớt"] = 1
    # xóa số lượng
    data["bỏ đi"] = 2
    data["xóa"] = 2
    data["bỏ"] = 2
    # print("số thức ăn trong câu là",len(Food))
    sentence = word_segmentation(sentence,False)
    arr = []
    countAction = 0
    for word in sentence:
        if word in data :
            if countAction < len(Food):
                countAction = countAction + 1
                arr.append(data[word])  
    # print("số action  trong câu là",len(arr))
    # if  len(arr) < len(Food):
    #     for target_list in expression_list:
    #         pass
    #     arr.append()   
    return arr

def chuyen_doi_chu_so_thanh_so(sen):
    dict_chu_so = {}
    dict_chu_so = {'hai mươi mốt': 21, 'hai mươi hai': 22, 'hai mươi ba': 23, 'hai mươi bốn': 24, 'hai mươi lăm': 25, 'hai mươi sáu': 26, 'hai mươi bảy': 27, 'hai mươi tám': 28, 'hai mươi chín': 29,
                   'mười một': 11, 'mười hai': 12, 'mười ba': 13, 'mười bốn': 14, 'mười lăm': 15, 'mười sáu': 16, 'mười bảy': 17, 'mười tám': 18, 'mười chín': 19, 'hai mươi': 20, 'ba mươi': 30,
                   'một': 1, 'hai': 2, 'ba': 3, 'bốn': 4, 'năm': 5, 'sáu': 6, 'bảy': 7, 'tám': 8, 'chín': 9, 'mười': 10}

    sen = " ".join(sen.split())  # Chuyển nhiều khoảng trắng về 1 khoảng trắng
    for key in dict_chu_so:
        if key in sen:
            sen = sen.replace(key, str(dict_chu_so[key]))
    return sen


def detectionListFood(sentence):
    sentence = sentence.lower()
    listFood = dict()
    # print("câu truy vấn ", sentence)
    with open(pathMenuFood, encoding='utf-8-sig') as json_file:
        input = json.load(json_file)
    for food in input:
        nameFood = food["name"].lower()
        index = sentence.find(nameFood)
        if index >= 0:
            listFood[index] = food

    return listFood


if __name__ == '__main__':
    # newSen = chuyen_doi_chu_so_thanh_so("Tôi muốn bớt mười lăm món Bánh phô mai cà phê")
    listFood = detectionListFood("Tôi muốn bớt món Bánh phô mai cà phê")
    print(listFood)
    # train_model()
    # num = detection_number("tôi đặt 5 phần bún thịt nướng")
    # print("số lượng ", num)

    # predictSentence("tôi bị bệnh")
    # ms.similarListToken("tôi đi vá bánh xe", 0)
    # UpdateNewQuestion()
