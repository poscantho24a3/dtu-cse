import numpy as np
from flask import Flask, render_template, Response, jsonify, redirect, url_for, request
import cv2
import os
import dlib
from keras_preprocessing.image import img_to_array
from sqlalchemy import engine
from sqlalchemy.orm import scoped_session, sessionmaker
from tensorflow.python.keras.models import load_model
from PIL import Image
import requests

import json
from FaceAligner import FaceAligner
from name_tracker import NamesTracker
import datetime
import threading
import listen_helper as lh
import speak_helper as sh
import glob
import random
from text import NNClassification
import config
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
import math

app = Flask(__name__)

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///Product.db?check_same_thread=False'
db = SQLAlchemy(app)
migrate = Migrate(app, db)

db.drop_all()
global detector,predictor,model_recognition,model_emotion
detector = dlib.simple_object_detector("Models/FaceDetection100.svm")
predictor = dlib.shape_predictor("Models/shape_predictor_68_face_landmarks.dat")
model_recognition = load_model("Models/alexnet_recognition.h5")
model_emotion = load_model("Models/model_alexnet_New_v6.h5")

class Product(db.Model):
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    menuid = db.Column(db.Integer)
    Name = db.Column(db.String(1000))
    Img = db.Column(db.String(1000))
    Category = db.Column(db.String(1000))
    Tag = db.Column(db.String(128))
    Price = db.Column(db.String(1000))


class Users(db.Model):
    customerid = db.Column(db.String(1000), primary_key=True, autoincrement=False)
    customername = db.Column(db.String(1000))
    customerclass = db.Column(db.String(1000))


class Vector(db.Model):
    IDs = db.Column(db.Integer, primary_key=True, autoincrement=True)
    id = db.Column(db.String(1000))
    v1 = db.Column(db.Integer)
    v2 = db.Column(db.Integer)
    v3 = db.Column(db.Integer)
    v4 = db.Column(db.Integer)
    v5 = db.Column(db.Integer)
    v6 = db.Column(db.Integer)
    v7 = db.Column(db.Integer)
    v8 = db.Column(db.Integer)
    v9 = db.Column(db.Integer)
    v10 = db.Column(db.Integer)
    v11 = db.Column(db.Integer)
    v12 = db.Column(db.Integer)
    v13 = db.Column(db.Integer)
    v14 = db.Column(db.Integer)
    v15 = db.Column(db.Integer)
    v16 = db.Column(db.Integer)
    v17 = db.Column(db.Integer)
    v18 = db.Column(db.Integer)
    v19 = db.Column(db.Integer)
    v20 = db.Column(db.Integer)
    v21 = db.Column(db.Integer)
    v22 = db.Column(db.Integer)
    v23 = db.Column(db.Integer)
    v24 = db.Column(db.Integer)
    v25 = db.Column(db.Integer)
    v26 = db.Column(db.Integer)
    v27 = db.Column(db.Integer)
    v28 = db.Column(db.Integer)
    v29 = db.Column(db.Integer)
    v30 = db.Column(db.Integer)
    v31 = db.Column(db.Integer)
    v32 = db.Column(db.Integer)
    v33 = db.Column(db.Integer)
    v34 = db.Column(db.Integer)
    v35 = db.Column(db.Integer)
    v36 = db.Column(db.Integer)

class Order(db.Model):
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    Name = db.Column(db.String(1000))
    Quantity = db.Column(db.String(1000))
    UserID = db.Column(db.String(128))
    Price = db.Column(db.String(1000))



def SelectVector(id, option=1):
    r_json = []
    if option == 1:
        vector = Vector.query.filter_by(id=id).all()
        for data in vector:
            v = [data.id, data.v1, data.v2, data.v3, data.v4, data.v5, data.v6, data.v7, data.v8, data.v9, data.v10,
                 data.v11, data.v12,
                 data.v13, data.v14, data.v15, data.v16, data.v17, data.v18, data.v19, data.v20, data.v21, data.v22,
                 data.v23, data.v24,
                 data.v25, data.v26, data.v27, data.v28, data.v29, data.v30, data.v31, data.v32, data.v33, data.v34,
                 data.v35, data.v36]
            r_json.append(v)

            # r_json.append({"id": data.id, "v1": data.v1, "v2": data.v2, "v3": data.v3, "v4": data.v4, "v5": data.v5,
            #                "v6": data.v6, "v7": data.v7, "v8": data.v8, "v9": data.v9, "v10": data.v10, "v11": data.v11,
            #                "v12": data.v12, "v13": data.v13, "v14": data.v14, "v15": data.v15, "v16": data.v16,
            #                "v17": data.v17, "v18": data.v18,
            #                "v19": data.v19, "v20": data.v20, "v21": data.v21, "v22": data.v22, "v23": data.v23,
            #                "v24": data.v24, "v25": data.v25,
            #                "v26": data.v26, "v27": data.v27, "v28": data.v28, "v29": data.v29, "v30": data.v30,
            #                "v31": data.v31,
            #                "v32": data.v32, "v33": data.v33, "v34": data.v34, "v35": data.v35, "v36": data.v36})
    else:
        vector = Vector.query.all()
        for data in vector:
            if data.id != id:
                v = [data.id, data.v1, data.v2, data.v3, data.v4, data.v5, data.v6, data.v7, data.v8, data.v9, data.v10,
                     data.v11, data.v12,
                     data.v13, data.v14, data.v15, data.v16, data.v17, data.v18, data.v19, data.v20, data.v21, data.v22,
                     data.v23, data.v24,
                     data.v25, data.v26, data.v27, data.v28, data.v29, data.v30, data.v31, data.v32, data.v33, data.v34,
                     data.v35, data.v36]
                r_json.append(v)
                # r_json.append({"id": data.id, "v1": data.v1, "v2": data.v2, "v3": data.v3, "v4": data.v4, "v5": data.v5,
                #                "v6": data.v6, "v7": data.v7, "v8": data.v8, "v9": data.v9, "v10": data.v10,
                #                "v11": data.v11,
                #                "v12": data.v12, "v13": data.v13, "v14": data.v14, "v15": data.v15, "v16": data.v16,
                #                "v17": data.v17, "v18": data.v18,
                #                "v19": data.v19, "v20": data.v20, "v21": data.v21, "v22": data.v22, "v23": data.v23,
                #                "v24": data.v24, "v25": data.v25,
                #                "v26": data.v26, "v27": data.v27, "v28": data.v28, "v29": data.v29, "v30": data.v30,
                #                "v31": data.v31,
                #                "v32": data.v32, "v33": data.v33, "v34": data.v34, "v35": data.v35, "v36": data.v36})
    return r_json


def SelectVectorDrink(id, option=1):
    r_json = []
    if option == 1:
        vector = Vector.query.filter_by(id=id).all()
        for data in vector:
            v = [data.id, data.v1, data.v2, data.v3, data.v4, data.v5, data.v6, data.v7, data.v8, data.v9, data.v10,
                 data.v11, data.v12,
                 data.v13, data.v14, data.v15, data.v16, data.v17, data.v18, data.v19]
            r_json.append(v)
            # r_json.append({"id": data.id, "v1": data.v1, "v2": data.v2, "v3": data.v3, "v4": data.v4, "v5": data.v5,
            #                "v6": data.v6, "v7": data.v7, "v8": data.v8, "v9": data.v9, "v10": data.v10, "v11": data.v11,
            #                "v12": data.v12, "v13": data.v13, "v14": data.v14, "v15": data.v15, "v16": data.v16,
            #                "v17": data.v17, "v18": data.v18,
            #                "v19": data.v19})
    else:
        vector = Vector.query.all()
        for data in vector:
            if data.id != id:
                v = [data.id, data.v1, data.v2, data.v3, data.v4, data.v5, data.v6, data.v7, data.v8, data.v9, data.v10,
                     data.v11, data.v12,
                     data.v13, data.v14, data.v15, data.v16, data.v17, data.v18, data.v19]
                r_json.append(v)
    return r_json


def SelectVectorFood(id, option=1):
    r_json = []
    if option == 1:
        vector = Vector.query.filter_by(id=id).all()
        for data in vector:
            v = [data.id, data.v20, data.v21, data.v22,
                 data.v23]
            r_json.append(v)
    else:
        vector = Vector.query.all()
        for data in vector:
            if data.id != id:
                v = [data.id, data.v20, data.v21, data.v22,
                     data.v23]
                r_json.append(v)
    return r_json


def SelectVectorCake(id, option=1):
    r_json = []
    if option == 1:
        vector = Vector.query.filter_by(id=id).all()
        for data in vector:
            v = [data.id, data.v24,
                 data.v25, data.v26, data.v27, data.v28, data.v29, data.v30, data.v31, data.v32]
            r_json.append(v)
    else:
        vector = Vector.query.all()
        for data in vector:
            if data.id != id:
                v = [data.id, data.v24,
                     data.v25, data.v26, data.v27, data.v28, data.v29, data.v30, data.v31, data.v32]
                r_json.append(v)
    return r_json


def SelectVectorOther(id, option=1):
    r_json = []
    if option == 1:
        vector = Vector.query.filter_by(id=id).all()
        for data in vector:
            v = [data.id, data.v33, data.v34,
                 data.v35, data.v36]
            r_json.append(v)
    else:
        vector = Vector.query.all()
        for data in vector:
            if data.id != id:
                v = [data.id, data.v33, data.v34,
                     data.v35, data.v36]
                r_json.append(v)
    return r_json


def InsertProduct(Name, Img, Price, Category, Tag, menuid):
    db.create_all()
    product = Product(menuid=menuid, Name=Name, Img=Img,
                      Price=Price, Category=Category, Tag=Tag)
    db.session.add(product)
    db.session.commit()


def InsertUser(Customerid, Customername, Customerclass):
    db.create_all()
    user = Users(customerid=Customerid, customername=Customername, customerclass=Customerclass)
    db.session.add(user)
    db.session.commit()


def InsertVector(id, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18, v19, v20, v21,
                 v22, v23, v24, v25, v26, v27, v28, v29, v30, v31, v32, v33, v34, v35, v36):
    db.create_all()
    vector = Vector(id=id, v1=v1, v2=v2, v3=v3, v4=v4, v5=v5, v6=v6, v7=v7, v8=v8, v9=v9, v10=v10, v11=v11, v12=v12,
                    v13=v13, v14=v14, v15=v15, v16=v16, v17=v17, v18=v18, v19=v19, v20=v20,
                    v21=v21, v22=v22, v23=v23, v24=v24, v25=v25, v26=v26, v27=v27, v28=v28, v29=v29, v30=v30, v31=v31,
                    v32=v32, v33=v33, v34=v34, v35=v35, v36=v36)
    db.session.add(vector)
    db.session.commit()


def UpdateVector(columsearchupdate=None, colunmUpdate=None, value=None, valueColunmSearchUpdate=None):
    if columsearchupdate is not None and valueColunmSearchUpdate is not None:
        vector = Vector.query.filter_by(columsearchupdate=valueColunmSearchUpdate).all()
        if colunmUpdate is not None:
            vector.colunmUpdate = value
            db.session.commit()


# json = [{"menuid": 13,"Name" : "FREEZE TRÀ XANH", "Img" : "https://www.highlandscoffee.com.vn/vnt_upload/product/03_2018/GTF.png","Price" : "49,000","Category" : "FREEZE KHÔNG CÀ PHÊ","Tag":"Freeze"},
#             {"menuid": 14,"Name" : "COOKIES & CREAM", "Img" : "https://www.highlandscoffee.com.vn/vnt_upload/product/05_2018/COOKIES-CREAM.png" ,"Price" : "49,000","Category" :"FREEZE KHÔNG CÀ PHÊ","Tag":"Freeze"},
#             {"menuid": 15,"Name" : "FREEZE SÔ-CÔ-LA", "Img" : "https://www.highlandscoffee.com.vn/vnt_upload/product/05_2018/CHOCOLATE-FREEZE.png" ,"Price" : "49,000","Category" :"FREEZE KHÔNG CÀ PHÊ","Tag":"Freeze"},
#             {"menuid": 10,"Name" : "CARAMEL PHIN FREEZE", "Img" : "https://www.highlandscoffee.com.vn/vnt_upload/product/03_2018/.png" ,"Price" : "49,000","Category" :"FREEZE CÀ PHÊ PHIN ","Tag":"Freeze"},
#             {"menuid": 11,"Name": "CLASSIC PHIN FREEZE", "Img": "https://www.highlandscoffee.com.vn/vnt_upload/product/05_2018/CLASSIC-FREEZE.png","Price": "49.000", "Category": "FREEZE CÀ PHÊ PHIN ","Tag":"Freeze"},
#             {"menuid": 3,"Name": "PHIN SỮA ĐÁ","Img": "https://www.highlandscoffee.com.vn/vnt_upload/product/03_2018/PHIN-SUA-DA.png", "Price": "29,000", "Category": "CÀ PHÊ PHIN ","Tag" : "Cà Phê"},
#             {"menuid": 4,"Name": "PHIN ĐEN ĐÁ", "Img": "https://www.highlandscoffee.com.vn/vnt_upload/product/05_2018/CFD.png","Price": "29,000", "Category": "CÀ PHÊ PHIN " , "Tag" : "Cà Phê"},
#             {"menuid": 1,"Name": "PHIN ĐEN NÓNG", "Img": "https://www.highlandscoffee.com.vn/vnt_upload/product/03_2018/AMERICANO.png","Price": "29,000", "Category": "CÀ PHÊ PHIN ","Tag" : "Cà Phê"},
#             {"menuid": 2,"Name": "PHIN SỮA NÓNG","Img": "https://www.highlandscoffee.com.vn/vnt_upload/product/06_2018/PHIN-SUA-NONG.png","Price": "29,000", "Category": "CÀ PHÊ PHIN ","Tag": "Cà Phê"},
#             {"menuid": 19,"Name": "TRÀ THANH ĐÀO","Img": "https://www.highlandscoffee.com.vn/vnt_upload/product/03_2018/TRATHANHDAO.png", "Price": "39,000", "Category": "TRÀ" , "Tag" : "Trà"},
#             {"menuid": 17,"Name": "TRÀ THẠCH VẢI","Img": "https://www.highlandscoffee.com.vn/vnt_upload/product/03_2018/TRATHACHVAI.png","Price": "39,000", "Category": "TRÀ", "Tag" : "Trà"},
#             {"menuid": 16,"Name": "TRÀ SEN VÀNG", "Img": "https://www.highlandscoffee.com.vn/vnt_upload/product/03_2018/TRASENVANG.png","Price": "39,000", "Category": "TRÀ", "Tag" : "Trà"},
#             {"menuid": 29,"Name": "BÁNH PHÔ MAI CHANH DÂY","Img": "https://www.highlandscoffee.com.vn/vnt_upload/product/03_2018/PHOMAICHANHDAY.jpg","Price": "29,000", "Category": "BÁNH NGỌT" , "Tag" : "Bánh Ngọt"},
#             {"menuid": 24,"Name": "BÁNH PHÔ MAI CÀ PHÊ", "Img": "https://www.highlandscoffee.com.vn/vnt_upload/product/03_2018/PHOMAICAPHE.jpg","Price": "29,000", "Category": "BÁNH NGỌT","Tag" : "Bánh Ngọt"},
#             {"menuid": 26,"Name": "BÁNH MOUSSE ĐÀO","Img": "https://www.highlandscoffee.com.vn/vnt_upload/product/03_2018/MOUSSEDAO.png","Price": "29,000", "Category": "BÁNH NGỌT","Tag" : "Bánh Ngọt"},
#             {"menuid": 22,"Name": "CHẢ LỤA XÁ XÍU","Img": "https://www.highlandscoffee.com.vn/vnt_upload/product/03_2018/BMCHALUAXAXIU.png","Price": "19,000", "Category": "BÁNH MÌ","Tag" : "Bánh Mì"},
#             {"menuid": 23,"Name": "GÀ XÉ NƯỚC TƯƠNG","Img": "https://www.highlandscoffee.com.vn/vnt_upload/product/03_2018/BMGAXE.png", "Price": "19,000", "Category": "BÁNH MÌ","Tag" : "Bánh Mì"},
#             {"menuid": 20,"Name": "THỊT NƯỚNG","Img": "https://www.highlandscoffee.com.vn/vnt_upload/product/03_2018/BMTHITNUONG.png", "Price": "19,000","Category": "BÁNH MÌ","Tag" : "Bánh Mì"},
#         {"menuid": 7,"Name" : "CARAMEL MACCHIATO", "Img" : "https://www.highlandscoffee.com.vn/vnt_upload/product/05_2018/CARAMEL-MACCHIATO.png","Price" : " 59,000","Category" : "Cà Phê Espresso","Tag":"Cà Phê"},
#         {"menuid": 9,"Name": "LATTE","Img": "https://www.highlandscoffee.com.vn/vnt_upload/product/03_2018/LATTE.png","Price": "54,000", "Category": "Cà Phê Espresso", "Tag": "Cà Phê" },
#         {"menuid": 8,"Name": "MOCHA MACCHIATO","Img": "https://www.highlandscoffee.com.vn/vnt_upload/product/05_2018/MOCHA.png","Price": " 59,000", "Category": "Cà Phê Espresso", "Tag": "Cà Phê" },
#         {"menuid": 12,"Name": "CAPPUCCINO", "Img": "https://www.highlandscoffee.com.vn/vnt_upload/product/03_2018/CAPPUCINO.png", "Price": " 54,000", "Category": "Cà Phê Espresso", "Tag": "Cà Phê"},
#         {"menuid": 6,"Name": "AMERICANO", "Img": "https://www.highlandscoffee.com.vn/vnt_upload/product/03_2018/AMERICANO.png","Price": " 44,000", "Category": "Cà Phê Espresso", "Tag": "Cà Phê"},
#         {"menuid": 5,"Name": "ESPRESSO", "Img": "https://www.highlandscoffee.com.vn/vnt_upload/product/05_2018/ESPRESSO.png","Price": " 44,000", "Category": "Cà Phê Espresso", "Tag": "Cà Phê"},
#         {"menuid": 17,"Name": "TRÀ THẠCH VÃI", "Img": "https://www.highlandscoffee.com.vn/vnt_upload/product/03_2018/TRATHACHVAI.png","Price": "39,000", "Category": "Trà", "Tag": "Trà"},
#         {"menuid": 18,"Name": "TRÀ THẠCH ĐÀO","Img": "https://www.highlandscoffee.com.vn/vnt_upload/product/03_2018/TRATHACHDAO.png", "Price": "39,000","Category": "Trà", "Tag": "Trà"},
#         {"menuid": 21,"Name": "XÍU MẠI","Img": "https://www.highlandscoffee.com.vn/vnt_upload/product/03_2018/BMXIUMAI.png", "Price": "19,000","Category": "Bánh Mì", "Tag": "Bánh mì"},
#         {"menuid": 28,"Name": "BÁNH SÔ-CÔ-LA HIGHLANDS", "Img": "https://www.highlandscoffee.com.vn/vnt_upload/product/03_2018/SOCOLAHL.png","Price": "29,000", "Category": "Bánh Ngọt", "Tag": "Bánh Ngọt"},
#         {"menuid": 30,"Name": "BÁNH PHÔ MAI TRÀ XANH", "Img": "https://www.highlandscoffee.com.vn/vnt_upload/product/03_2018/PHOMAITRAXANH.jpg","Price": "29,000", "Category": "Bánh Ngọt", "Tag": "Bánh Ngọt"},
#         {"menuid": 31,"Name": "BÁNH CARAMEL PHÔ MAI", "Img": "https://www.highlandscoffee.com.vn/vnt_upload/product/03_2018/CARAMELPHOMAI.jpg","Price": "29,000", "Category": "Bánh Ngọt", "Tag": "Bánh Ngọt"},
#         {"menuid": 32,"Name": "BÁNH TIRAMISU","Img": "https://www.highlandscoffee.com.vn/vnt_upload/product/03_2018/TIRAMISU.jpg","Price": "19,000", "Category": "Bánh Ngọt", "Tag": "Bánh Ngọt"},
#         {"menuid": 25,"Name": "BÁNH CHUỐI", "Img": "https://www.highlandscoffee.com.vn/vnt_upload/product/03_2018/BANHCHUOI.jpg","Price": "19,000", "Category": "Bánh Ngọt", "Tag": "Bánh Ngọt"},
#         {"menuid": 27,"Name": "BÁNH MOUSSE CACAO", "Img": "https://www.highlandscoffee.com.vn/vnt_upload/product/03_2018/MOUSSECACAO.png","Price": "29,000", "Category": "Bánh Ngọt", "Tag": "Bánh Ngọt"},
#         {"menuid": 26,"Name": "BÁNH MOUSSE ĐÀO", "Img": "https://www.highlandscoffee.com.vn/vnt_upload/product/03_2018/MOUSSEDAO.png","Price": "29,000", "Category": "Bánh Ngọt", "Tag": "Bánh Ngọt"},
#         {"menuid": 33,"Name": "PHIN INOX", "Img": "https://www.highlandscoffee.com.vn/vnt_upload/product/04_2018/Screen_Shot_2018-04-04_at_09.18.39.png","Price": " 79,000", "Category": "MERCHANDISE", "Tag": "MERCHANDISE"},
#         {"menuid": 34,"Name": "LY SỨ MOSAIC", "Img": "https://www.highlandscoffee.com.vn/vnt_upload/product/04_2018/Screen_Shot_2018-04-04_at_09.16.18.png","Price": "149,000", "Category": "MERCHANDISE", "Tag": "MERCHANDISE"},
#         {"menuid": 35,"Name": "TRUYỀN THỐNG 200GR", "Img": "https://www.highlandscoffee.com.vn/vnt_upload/product/04_2018/TT.png","Price": "47,000", "Category": "Cà phê đóng gói", "Tag": "Cà phê đóng gói"},
#         {"menuid": 36,"Name": "TRUYỀN THỐNG 1KG","Img": "https://www.highlandscoffee.com.vn/vnt_upload/product/05_2018/Traditional_1kg.png", "Price": "235,000", "Category": "Cà phê đóng gói", "Tag": "Cà phê đóng gói"}]
# for data in json:
#     InsertProduct(menuid= data['menuid'],Name = data['Name'],Img = data['Img'],Price = data['Price'],Category = data['Category'],Tag=data['Tag'])
# jsonUser = [{"customerid": "C02082019001", "customername": "Lê Hoàng Nhí", "customerclass": "VIP"}, {"customerid": "C02082019002", "customername": "Trần Bảo Toàn", "customerclass": "VIP"}, {"customerid": "C02082019003", "customername": "Hồ Quốc An", "customerclass": "VIP"},{"customerid": "C02082019004", "customername": "Đinh Mẩu Minh", "customerclass": "VIP"},{"customerid": "C02082019005", "customername": "Trương Tuấn Anh", "customerclass": "VIP"}]
# for data in jsonUser:
#     print(data["customerid"])
#     InsertUser(Customerid=data["customerid"],Customername=data["customername"], Customerclass=data["customerclass"])
# jsonvector = [{"id": "1.0", "1": 0.0, "2": 0.0, "3": 0.0, "4": 0.0, "5": 0.0, "6": 0.0, "7": 0.0, "8": 0.0, "9": 0.0, "10": 0.0, "11": 0.0, "12": 0.0, "13": 3.0, "14": 2.0, "15": 2.0, "16": 0.0, "17": 0.0, "18": 0.0, "19": 0.0, "20": 0.0, "21": 0.0, "22": 0.0, "23": 0.0, "24": 0.0, "25": 0.0, "26": 0.0, "27": 0.0, "28": 0.0, "29": 0.0, "30": 0.0, "31": 0.0, "32": 0.0, "33": 0.0, "34": 0.0, "35": 0.0, "36": 0.0},
#               {"id": "2.0", "1": 0.0, "2": 0.0, "3": 0.0, "4": 0.0, "5": 0.0, "6": 0.0, "7": 0.0, "8": 0.0, "9": 0.0, "10": 0.0, "11": 0.0, "12": 0.0, "13": 3.0, "14": 0.0, "15": 0.0, "16": 0.0, "17": 0.0, "18": 0.0, "19": 0.0, "20": 0.0, "21": 0.0, "22": 0.0, "23": 0.0, "24": 0.0, "25": 0.0, "26": 1.0, "27": 0.0, "28": 0.0, "29": 0.0, "30": 0.0, "31": 0.0, "32": 0.0, "33": 0.0, "34": 0.0, "35": 0.0, "36": 0.0},
#               {"id": "3.0", "1": 5.0, "2": 0.0, "3": 0.0, "4": 0.0, "5": 0.0, "6": 0.0, "7": 0.0, "8": 0.0, "9": 0.0, "10": 0.0, "11": 0.0, "12": 0.0, "13": 0.0, "14": 0.0, "15": 0.0, "16": 0.0, "17": 0.0, "18": 0.0, "19": 0.0, "20": 1.0, "21": 0.0, "22": 0.0, "23": 0.0, "24": 0.0, "25": 0.0, "26": 0.0, "27": 0.0, "28": 0.0, "29": 0.0, "30": 0.0, "31": 0.0, "32": 0.0, "33": 0.0, "34": 0.0, "35": 0.0, "36": 0.0},
#               {"id": "4.0", "1": 5.0, "2": 0.0, "3": 0.0, "4": 0.0, "5": 0.0, "6": 0.0, "7": 0.0, "8": 0.0, "9": 0.0, "10": 0.0, "11": 0.0, "12": 0.0, "13": 0.0, "14": 0.0, "15": 0.0, "16": 0.0, "17": 0.0, "18": 0.0, "19": 0.0, "20": 1.0, "21": 0.0, "22": 0.0, "23": 0.0, "24": 0.0, "25": 0.0, "26": 0.0, "27": 0.0, "28": 0.0, "29": 0.0, "30": 0.0, "31": 0.0, "32": 0.0, "33": 0.0, "34": 0.0, "35": 0.0, "36": 0.0},
#               {"id": "5.0", "1": 5.0, "2": 0.0, "3": 0.0, "4": 0.0, "5": 0.0, "6": 0.0, "7": 0.0, "8": 0.0, "9": 0.0, "10": 0.0, "11": 0.0, "12": 0.0, "13": 0.0, "14": 0.0, "15": 0.0, "16": 0.0, "17": 0.0, "18": 0.0, "19": 0.0, "20": 1.0, "21": 0.0, "22": 0.0, "23": 0.0, "24": 0.0, "25": 0.0, "26": 0.0, "27": 0.0, "28": 0.0, "29": 0.0, "30": 0.0, "31": 0.0, "32": 0.0, "33": 0.0, "34": 0.0, "35": 0.0, "36": 0.0},
#               {"id": "6.0", "1": 0.0, "2": 0.0, "3": 4.0, "4": 0.0, "5": 0.0, "6": 2.0, "7": 0.0, "8": 0.0, "9": 2.0, "10": 0.0, "11": 1.0, "12": 0.0, "13": 0.0, "14": 0.0, "15": 0.0, "16": 1.0, "17": 0.0, "18": 0.0, "19": 0.0, "20": 0.0, "21": 0.0, "22": 0.0, "23": 0.0, "24": 0.0, "25": 0.0, "26": 0.0, "27": 0.0, "28": 0.0, "29": 0.0, "30": 0.0, "31": 0.0, "32": 0.0, "33": 0.0, "34": 0.0, "35": 0.0, "36": 0.0},
#               {"id": "8.0", "1": 1.0, "2": 1.0, "3": 1.0, "4": 0.0, "5": 0.0, "6": 1.0, "7": 1.0, "8": 1.0, "9": 1.0, "10": 0.0, "11": 0.0, "12": 1.0, "13": 0.0, "14": 0.0, "15": 0.0, "16": 0.0, "17": 0.0, "18": 0.0, "19": 0.0, "20": 0.0, "21": 0.0, "22": 0.0, "23": 0.0, "24": 0.0, "25": 0.0, "26": 0.0, "27": 0.0, "28": 0.0, "29": 0.0, "30": 0.0, "31": 0.0, "32": 0.0, "33": 0.0, "34": 0.0, "35": 0.0, "36": 0.0},
#               {"id": "9.0", "1": 0.0, "2": 0.0, "3": 0.0, "4": 0.0, "5": 0.0, "6": 0.0, "7": 0.0, "8": 2.0, "9": 0.0, "10": 0.0, "11": 0.0, "12": 0.0, "13": 0.0, "14": 0.0, "15": 0.0, "16": 1.0, "17": 0.0, "18": 0.0, "19": 0.0, "20": 0.0, "21": 1.0, "22": 1.0, "23": 0.0, "24": 0.0, "25": 0.0, "26": 0.0, "27": 0.0, "28": 0.0, "29": 0.0, "30": 0.0, "31": 0.0, "32": 0.0, "33": 0.0, "34": 0.0, "35": 0.0, "36": 0.0},
#               {"id": "12.0", "1": 4.0, "2": 0.0, "3": 0.0, "4": 0.0, "5": 0.0, "6": 0.0, "7": 0.0, "8": 0.0, "9": 0.0, "10": 0.0, "11": 0.0, "12": 0.0, "13": 0.0, "14": 0.0, "15": 0.0, "16": 0.0, "17": 0.0, "18": 0.0, "19": 0.0, "20": 1.0, "21": 0.0, "22": 0.0, "23": 0.0, "24": 0.0, "25": 0.0, "26": 0.0, "27": 0.0, "28": 0.0, "29": 0.0, "30": 0.0, "31": 0.0, "32": 0.0, "33": 0.0, "34": 0.0, "35": 0.0, "36": 0.0},
#               {"id": "13.0", "1": 0.0, "2": 0.0, "3": 0.0, "4": 0.0, "5": 0.0, "6": 0.0, "7": 0.0, "8": 2.0, "9": 0.0, "10": 0.0, "11": 0.0, "12": 0.0, "13": 0.0, "14": 0.0, "15": 0.0, "16": 1.0, "17": 0.0, "18": 0.0, "19": 0.0, "20": 0.0, "21": 1.0, "22": 1.0, "23": 0.0, "24": 0.0, "25": 0.0, "26": 0.0, "27": 0.0, "28": 0.0, "29": 0.0, "30": 0.0, "31": 0.0, "32": 0.0, "33": 0.0, "34": 0.0, "35": 0.0, "36": 0.0},
#               {"id": "14.0", "1": 0.0, "2": 0.0, "3": 2.0, "4": 0.0, "5": 0.0, "6": 0.0, "7": 5.0, "8": 0.0, "9": 0.0, "10": 0.0, "11": 0.0, "12": 0.0, "13": 4.0, "14": 0.0, "15": 0.0, "16": 0.0, "17": 0.0, "18": 0.0, "19": 0.0, "20": 0.0, "21": 0.0, "22": 0.0, "23": 0.0, "24": 0.0, "25": 0.0, "26": 1.0, "27": 0.0, "28": 0.0, "29": 0.0, "30": 0.0, "31": 0.0, "32": 1.0, "33": 0.0, "34": 0.0, "35": 0.0, "36": 0.0},
#               {"id": "18.0", "1": 0.0, "2": 0.0, "3": 3.0, "4": 0.0, "5": 0.0, "6": 0.0, "7": 0.0, "8": 1.0, "9": 0.0, "10": 0.0, "11": 0.0, "12": 0.0, "13": 0.0, "14": 0.0, "15": 0.0, "16": 0.0, "17": 0.0, "18": 2.0, "19": 0.0, "20": 0.0, "21": 1.0, "22": 0.0, "23": 0.0, "24": 2.0, "25": 0.0, "26": 0.0, "27": 0.0, "28": 0.0, "29": 1.0, "30": 0.0, "31": 0.0, "32": 0.0, "33": 0.0, "34": 0.0, "35": 0.0, "36": 0.0},
#               {"id": "19.0", "1": 0.0, "2": 0.0, "3": 2.0, "4": 0.0, "5": 0.0, "6": 0.0, "7": 0.0, "8": 0.0, "9": 0.0, "10": 0.0, "11": 0.0, "12": 0.0, "13": 1.0, "14": 0.0, "15": 0.0, "16": 0.0, "17": 0.0, "18": 1.0, "19": 0.0, "20": 0.0, "21": 0.0, "22": 0.0, "23": 0.0, "24": 0.0, "25": 0.0, "26": 0.0, "27": 0.0, "28": 0.0, "29": 0.0, "30": 0.0, "31": 0.0, "32": 0.0, "33": 0.0, "34": 0.0, "35": 0.0, "36": 0.0},
#               {"id": "21.0", "1": 1.0, "2": 2.0, "3": 1.0, "4": 0.0, "5": 0.0, "6": 0.0, "7": 1.0, "8": 0.0, "9": 0.0, "10": 0.0, "11": 2.0, "12": 1.0, "13": 0.0, "14": 0.0, "15": 0.0, "16": 0.0, "17": 0.0, "18": 0.0, "19": 0.0, "20": 2.0, "21": 0.0, "22": 0.0, "23": 0.0, "24": 2.0, "25": 0.0, "26": 0.0, "27": 0.0, "28": 1.0, "29": 0.0, "30": 0.0, "31": 0.0, "32": 1.0, "33": 0.0, "34": 0.0, "35": 0.0, "36": 0.0},
#               {"id": "24.0", "1": 0.0, "2": 2.0, "3": 4.0, "4": 0.0, "5": 0.0, "6": 0.0, "7": 0.0, "8": 0.0, "9": 0.0, "10": 0.0, "11": 0.0, "12": 1.0, "13": 2.0, "14": 1.0, "15": 3.0, "16": 2.0, "17": 0.0, "18": 1.0, "19": 0.0, "20": 0.0, "21": 3.0, "22": 0.0, "23": 0.0, "24": 0.0, "25": 0.0, "26": 0.0, "27": 0.0, "28": 1.0, "29": 0.0, "30": 1.0, "31": 0.0, "32": 2.0, "33": 0.0, "34": 0.0, "35": 0.0, "36": 0.0},
#               {"id": "25.0", "1": 0.0, "2": 0.0, "3": 1.0, "4": 0.0, "5": 0.0, "6": 0.0, "7": 3.0, "8": 0.0, "9": 0.0, "10": 2.0, "11": 0.0, "12": 0.0, "13": 0.0, "14": 2.0, "15": 0.0, "16": 0.0, "17": 0.0, "18": 0.0, "19": 0.0, "20": 0.0, "21": 0.0, "22": 0.0, "23": 3.0, "24": 2.0, "25": 0.0, "26": 0.0, "27": 0.0, "28": 0.0, "29": 3.0, "30": 0.0, "31": 0.0, "32": 0.0, "33": 0.0, "34": 0.0, "35": 0.0, "36": 0.0},
#               {"id": "27.0", "1": 0.0, "2": 0.0, "3": 5.0, "4": 0.0, "5": 0.0, "6": 0.0, "7": 0.0, "8": 0.0, "9": 0.0, "10": 5.0, "11": 0.0, "12": 0.0, "13": 0.0, "14": 0.0, "15": 5.0, "16": 0.0, "17": 5.0, "18": 0.0, "19": 0.0, "20": 5.0, "21": 0.0, "22": 0.0, "23": 0.0, "24": 0.0, "25": 5.0, "26": 5.0, "27": 0.0, "28": 0.0, "29": 0.0, "30": 0.0, "31": 0.0, "32": 0.0, "33": 0.0, "34": 0.0, "35": 0.0, "36": 0.0},
#               {"id": "29.0", "1": 0.0, "2": 0.0, "3": 0.0, "4": 0.0, "5": 0.0, "6": 0.0, "7": 0.0, "8": 0.0, "9": 0.0, "10": 0.0, "11": 0.0, "12": 0.0, "13": 0.0, "14": 0.0, "15": 0.0, "16": 0.0, "17": 0.0, "18": 5.0, "19": 0.0, "20": 0.0, "21": 0.0, "22": 0.0, "23": 0.0, "24": 0.0, "25": 0.0, "26": 0.0, "27": 0.0, "28": 0.0, "29": 0.0, "30": 0.0, "31": 0.0, "32": 1.0, "33": 0.0, "34": 0.0, "35": 0.0, "36": 0.0},
#               {"id": "30.0", "1": 0.0, "2": 0.0, "3": 0.0, "4": 0.0, "5": 0.0, "6": 0.0, "7": 0.0, "8": 0.0, "9": 0.0, "10": 0.0, "11": 0.0, "12": 0.0, "13": 0.0, "14": 0.0, "15": 0.0, "16": 5.0, "17": 0.0, "18": 0.0, "19": 0.0, "20": 0.0, "21": 0.0, "22": 0.0, "23": 0.0, "24": 0.0, "25": 0.0, "26": 0.0, "27": 0.0, "28": 5.0, "29": 0.0, "30": 0.0, "31": 0.0, "32": 0.0, "33": 0.0, "34": 0.0, "35": 0.0, "36": 0.0},
#               {"id": "32.0", "1": 1.0, "2": 0.0, "3": 0.0, "4": 0.0, "5": 0.0, "6": 1.0, "7": 0.0, "8": 0.0, "9": 0.0, "10": 0.0, "11": 0.0, "12": 0.0, "13": 0.0, "14": 0.0, "15": 0.0, "16": 0.0, "17": 0.0, "18": 0.0, "19": 0.0, "20": 0.0, "21": 0.0, "22": 0.0, "23": 0.0, "24": 0.0, "25": 0.0, "26": 0.0, "27": 0.0, "28": 0.0, "29": 0.0, "30": 0.0, "31": 0.0, "32": 1.0, "33": 0.0, "34": 0.0, "35": 0.0, "36": 0.0},
#               {"id": "33.0", "1": 0.0, "2": 0.0, "3": 5.0, "4": 0.0, "5": 0.0, "6": 0.0, "7": 0.0, "8": 0.0, "9": 0.0, "10": 0.0, "11": 0.0, "12": 0.0, "13": 0.0, "14": 0.0, "15": 0.0, "16": 5.0, "17": 5.0, "18": 5.0, "19": 5.0, "20": 0.0, "21": 5.0, "22": 0.0, "23": 0.0, "24": 0.0, "25": 0.0, "26": 0.0, "27": 0.0, "28": 0.0, "29": 0.0, "30": 0.0, "31": 0.0, "32": 5.0, "33": 0.0, "34": 0.0, "35": 0.0, "36": 0.0},
#               {"id": "35.0", "1": 0.0, "2": 1.0, "3": 4.0, "4": 0.0, "5": 0.0, "6": 2.0, "7": 0.0, "8": 0.0, "9": 0.0, "10": 0.0, "11": 0.0, "12": 0.0, "13": 3.0, "14": 3.0, "15": 4.0, "16": 0.0, "17": 0.0, "18": 4.0, "19": 4.0, "20": 2.0, "21": 1.0, "22": 0.0, "23": 0.0, "24": 0.0, "25": 0.0, "26": 3.0, "27": 2.0, "28": 0.0, "29": 0.0, "30": 0.0, "31": 0.0, "32": 3.0, "33": 0.0, "34": 0.0, "35": 0.0, "36": 0.0},
#               {"id": "37.0", "1": 1.0, "2": 5.0, "3": 5.0, "4": 0.0, "5": 0.0, "6": 0.0, "7": 0.0, "8": 0.0, "9": 0.0, "10": 0.0, "11": 1.0, "12": 0.0, "13": 1.0, "14": 1.0, "15": 1.0, "16": 1.0, "17": 1.0, "18": 1.0, "19": 1.0, "20": 2.0, "21": 1.0, "22": 1.0, "23": 0.0, "24": 0.0, "25": 1.0, "26": 1.0, "27": 0.0, "28": 0.0, "29": 1.0, "30": 0.0, "31": 0.0, "32": 1.0, "33": 0.0, "34": 0.0, "35": 0.0, "36": 0.0},
#               {"id": "38.0", "1": 0.0, "2": 0.0, "3": 0.0, "4": 0.0, "5": 0.0, "6": 0.0, "7": 0.0, "8": 0.0, "9": 0.0, "10": 2.0, "11": 0.0, "12": 0.0, "13": 0.0, "14": 1.0, "15": 0.0, "16": 0.0, "17": 0.0, "18": 0.0, "19": 0.0, "20": 0.0, "21": 0.0, "22": 0.0, "23": 0.0, "24": 0.0, "25": 0.0, "26": 0.0, "27": 0.0, "28": 0.0, "29": 0.0, "30": 0.0, "31": 0.0, "32": 0.0, "33": 0.0, "34": 0.0, "35": 0.0, "36": 0.0},
#               {"id": "39.0", "1": 0.0, "2": 0.0, "3": 0.0, "4": 0.0, "5": 0.0, "6": 0.0, "7": 0.0, "8": 0.0, "9": 0.0, "10": 1.0, "11": 0.0, "12": 1.0, "13": 4.0, "14": 2.0, "15": 4.0, "16": 4.0, "17": 0.0, "18": 0.0, "19": 0.0, "20": 0.0, "21": 0.0, "22": 0.0, "23": 0.0, "24": 0.0, "25": 0.0, "26": 0.0, "27": 2.0, "28": 1.0, "29": 1.0, "30": 1.0, "31": 0.0, "32": 1.0, "33": 0.0, "34": 0.0, "35": 0.0, "36": 0.0},
#               {"id": "40.0", "1": 0.0, "2": 0.0, "3": 5.0, "4": 0.0, "5": 0.0, "6": 0.0, "7": 0.0, "8": 0.0, "9": 0.0, "10": 0.0, "11": 0.0, "12": 0.0, "13": 5.0, "14": 0.0, "15": 0.0, "16": 0.0, "17": 0.0, "18": 0.0, "19": 5.0, "20": 2.0, "21": 0.0, "22": 0.0, "23": 0.0, "24": 0.0, "25": 0.0, "26": 0.0, "27": 0.0, "28": 0.0, "29": 0.0, "30": 3.0, "31": 0.0, "32": 0.0, "33": 0.0, "34": 0.0, "35": 0.0, "36": 0.0},
#               {"id": "41.0", "1": 0.0, "2": 0.0, "3": 0.0, "4": 0.0, "5": 0.0, "6": 0.0, "7": 0.0, "8": 0.0, "9": 0.0, "10": 0.0, "11": 0.0, "12": 0.0, "13": 1.0, "14": 0.0, "15": 0.0, "16": 3.0, "17": 0.0, "18": 1.0, "19": 4.0, "20": 1.0, "21": 3.0, "22": 0.0, "23": 0.0, "24": 0.0, "25": 1.0, "26": 0.0, "27": 0.0, "28": 1.0, "29": 0.0, "30": 1.0, "31": 0.0, "32": 0.0, "33": 0.0, "34": 0.0, "35": 0.0, "36": 0.0},
#               {"id": "42.0", "1": 0.0, "2": 0.0, "3": 0.0, "4": 0.0, "5": 0.0, "6": 0.0, "7": 0.0, "8": 0.0, "9": 0.0, "10": 0.0, "11": 0.0, "12": 0.0, "13": 0.0, "14": 0.0, "15": 0.0, "16": 5.0, "17": 0.0, "18": 4.0, "19": 4.0, "20": 3.0, "21": 0.0, "22": 0.0, "23": 0.0, "24": 0.0, "25": 0.0, "26": 0.0, "27": 0.0, "28": 0.0, "29": 5.0, "30": 0.0, "31": 0.0, "32": 3.0, "33": 0.0, "34": 0.0, "35": 0.0, "36": 0.0},
#               {"id": "43.0", "1": 0.0, "2": 0.0, "3": 2.0, "4": 0.0, "5": 0.0, "6": 0.0, "7": 0.0, "8": 0.0, "9": 0.0, "10": 0.0, "11": 0.0, "12": 0.0, "13": 0.0, "14": 0.0, "15": 0.0, "16": 4.0, "17": 0.0, "18": 1.0, "19": 0.0, "20": 0.0, "21": 0.0, "22": 0.0, "23": 0.0, "24": 0.0, "25": 0.0, "26": 0.0, "27": 0.0, "28": 0.0, "29": 0.0, "30": 1.0, "31": 0.0, "32": 1.0, "33": 0.0, "34": 0.0, "35": 0.0, "36": 0.0},
#               {"id": "44.0", "1": 2.0, "2": 3.0, "3": 2.0, "4": 0.0, "5": 0.0, "6": 0.0, "7": 0.0, "8": 0.0, "9": 0.0, "10": 0.0, "11": 0.0, "12": 0.0, "13": 2.0, "14": 0.0, "15": 3.0, "16": 0.0, "17": 0.0, "18": 0.0, "19": 0.0, "20": 0.0, "21": 0.0, "22": 0.0, "23": 0.0, "24": 2.0, "25": 0.0, "26": 0.0, "27": 2.0, "28": 0.0, "29": 4.0, "30": 0.0, "31": 0.0, "32": 0.0, "33": 0.0, "34": 0.0, "35": 0.0, "36": 0.0},
#               {"id": "45.0", "1": 0.0, "2": 0.0, "3": 5.0, "4": 0.0, "5": 0.0, "6": 0.0, "7": 0.0, "8": 0.0, "9": 0.0, "10": 0.0, "11": 0.0, "12": 0.0, "13": 0.0, "14": 0.0, "15": 0.0, "16": 0.0, "17": 0.0, "18": 0.0, "19": 0.0, "20": 3.0, "21": 0.0, "22": 0.0, "23": 0.0, "24": 0.0, "25": 0.0, "26": 0.0, "27": 0.0, "28": 0.0, "29": 0.0, "30": 0.0, "31": 0.0, "32": 0.0, "33": 0.0, "34": 0.0, "35": 0.0, "36": 0.0},
#               {"id": "C02082019001", "1": 0.0, "2": 0.0, "3": 5.0, "4": 0.0, "5": 0.0, "6": 0.0, "7": 0.0, "8": 0.0, "9": 0.0,
#                "10": 0.0, "11": 0.0, "12": 0.0, "13": 0.0, "14": 3.0, "15": 5.0, "16": 5.0, "17": 5.0, "18": 5.0,
#                "19": 0.0, "20": 3.0, "21": 0.0, "22": 0.0, "23": 0.0, "24": 4.0, "25": 4.0, "26": 4.0, "27": 4.0,
#                "28": 0.0, "29": 0.0, "30": 0.0, "31": 0.0, "32": 0.0, "33": 4.0, "34": 5.0, "35": 5.0, "36": 5.0},
#
#               {"id": "C02082019002", "1": 0.0, "2": 0.0, "3": 0.0, "4": 0.0, "5": 0.0, "6": 0.0, "7": 0.0, "8": 0.0,
#                "9": 0.0,
#                "10": 0.0, "11": 0.0, "12": 0.0, "13": 0.0, "14": 0.0, "15": 5.0, "16": 0.0, "17": 5.0, "18": 5.0,
#                "19": 0.0, "20": 3.0, "21": 0.0, "22": 0.0, "23": 0.0, "24": 0.0, "25": 4.0, "26": 0.0, "27": 0.0,
#                "28": 0.0, "29": 0.0, "30": 0.0, "31": 0.0, "32": 0.0, "33": 0.0, "34": 0.0, "35": 0.0, "36": 0.0},
#
#               {"id": "C02082019003", "1": 0.0, "2": 0.0, "3": 0.0, "4": 0.0, "5": 0.0, "6": 0.0, "7": 0.0, "8": 0.0,
#                "9": 0.0,
#                "10": 0.0, "11": 0.0, "12": 0.0, "13": 0.0, "14": 5.0, "15": 5.0, "16": 0.0, "17": 0.0, "18": 0.0,
#                "19": 0.0, "20": 3.0, "21": 0.0, "22": 0.0, "23": 0.0, "24": 0.0, "25": 4.0, "26": 0.0, "27": 0.0,
#                "28": 0.0, "29": 0.0, "30": 0.0, "31": 0.0, "32": 0.0, "33": 0.0, "34": 0.0, "35": 0.0, "36": 0.0},
#
#               {"id": "C02082019004", "1": 0.0, "2": 0.0, "3": 0.0, "4": 0.0, "5": 0.0, "6": 0.0, "7": 0.0, "8": 0.0,
#                "9": 0.0,
#                "10": 0.0, "11": 0.0, "12": 0.0, "13": 0.0, "14": 5.0, "15": 5.0, "16": 0.0, "17": 0.0, "18": 0.0,
#                "19": 0.0, "20": 3.0, "21": 0.0, "22": 3.0, "23": 0.0, "24": 0.0, "25": 4.0, "26": 5.0, "27": 0.0,
#                "28": 0.0, "29": 5.0, "30": 0.0, "31": 0.0, "32": 0.0, "33": 0.0, "34": 0.0, "35": 0.0, "36": 0.0},
#
#               {"id": "C02082019005", "1": 5.0, "2": 4.0, "3": 6.0, "4": 7.0, "5": 0.0, "6": 0.0, "7": 0.0, "8": 0.0,
#                "9": 0.0,
#                "10": 0.0, "11": 0.0, "12": 0.0, "13": 0.0, "14": 0.0, "15": 0.0, "16": 0.0, "17": 0.0, "18": 0.0,
#                "19": 0.0, "20": 3.0, "21": 0.0, "22": 0.0, "23": 0.0, "24": 0.0, "25": 4.0, "26": 5.0, "27": 0.0,
#                "28": 0.0, "29": 5.0, "30": 0.0, "31": 0.0, "32": 0.0, "33": 0.0, "34": 0.0, "35": 0.0, "36": 0.0}
#               ]
# for data in jsonvector:
#     InsertVector(id= data["id"],v1=data["1"],v2=data["2"],v3=data["3"],v4=data["4"],v5=data["5"],v6=data["6"],v7=data["7"],v8=data["8"],v9=data["9"],
#                  v10=data["10"],v11=data["11"],v12=data["12"],v13=data["13"],v14=data["14"],v15=data["15"],v16=data["16"],v17=data["17"],v18=data["18"],
#                  v19=data["19"],v20=data["20"],v21=data["21"],v22=data["22"],v23=data["23"],v24=data["24"],v25=data["25"],v26=data["26"],
#                  v27=data["27"],v28=data["28"],v29=data["29"],v30=data["30"],v31=data["31"],v32=data["32"],v33=data["33"],v34=data["34"],v35=data["35"],v36=data["36"])
text_classification = NNClassification(config.DEFAULT_PATH_PROCESSING)
listener = lh.Listen_Helper()
speaker = sh.Speak_Helper()

is_say_hi = True
user_id = 0


def cosine_cumpute(vector_a, vector_b):
    if len(vector_a) == len(vector_b):
        a_squared = 0
        for a in vector_a:
            a_squared += a * a
        b_squared = 0
        for b in vector_b:
            b_squared += b * b
        sum_ab = 0
        for i in range(len(vector_a)):
            a = vector_a[i]
            b = vector_b[i]
            sum_ab += a * b
        if sum_ab == 0 or a_squared == 0 or b_squared == 0:
            return 0
        else:
            cosine = sum_ab / (math.sqrt(a_squared) * math.sqrt(b_squared))
        return cosine
    else:
        return 0


def get_vector_removeid(vector):
    vector_new = []
    for i in range(1, len(vector)):
        vector_new.append(vector[i])
    return vector_new


def recommender_menuid(vector, option):
    vector = get_vector_removeid(vector)
    sum = 0
    if option == 0:
        get_index = \
            {
                0: 1,
                1: 2,
                2: 3,
                3: 4,
                4: 5,
                5: 6,
                6: 7,
                7: 8,
                8: 9,
                9: 10,
                10: 11,
                11: 12,
                12: 13,
                13: 14,
                14: 15,
                15: 16,
                16: 17,
                17: 18,
                18: 19,
                19: 20,
                20: 21,
                21: 22,
                22: 23,
                23: 24,
                24: 25,
                25: 26,
                26: 27,
                27: 28,
                28: 29,
                29: 30,
                30: 31,
                31: 32,
                32: 33,
                33: 24,
                34: 35,
                35: 36
            }
    elif option == 1:
        get_index = \
            {
                0: 1,
                1: 2,
                2: 3,
                3: 4,
                4: 5,
                5: 6,
                6: 7,
                7: 8,
                8: 9,
                9: 10,
                10: 11,
                11: 12,
                12: 13,
                13: 14,
                14: 15,
                15: 16,
                16: 17,
                17: 18,
                18: 19
            }
    elif option == 2:
        get_index = \
            {
                0: 20,
                1: 21,
                2: 22,
                3: 23
            }
    elif option == 3:
        get_index = \
            {
                0: 24,
                1: 25,
                2: 26,
                3: 27,
                4: 28,
                5: 29,
                6: 30,
                7: 31,
                8: 32,
                9: 10
            }
    else:
        get_index = \
            {
                0: 33,
                1: 34,
                2: 35,
                3: 36
            }
    for i in vector:
        sum += i
    if sum > 0:
        return get_index[vector.index(max(vector))]
    else:
        return 0


def get_nearesy_vector(vector, vectorS):
    max_consine = 0
    get_vector = []
    for v in vectorS:
        v1 = get_vector_removeid(v)
        cs = cosine_cumpute(vector, v1)
        if max_consine <= cs:
            max_consine = cs
            get_vector = v
    return get_vector

def get_for_menu_id(id_user):
    a = SelectVectorDrink(id_user, 1)[0]
    b = get_vector_removeid(a)
    array_v = SelectVectorDrink(id_user, 2)
    v_output = get_nearesy_vector(b, array_v)
    vector_id = v_output[0]
    vector_food = SelectVectorFood(vector_id, 1)[0]
    vector_cake = SelectVectorCake(vector_id, 1)[0]
    vector_other = SelectVectorOther(vector_id, 1)[0]
    menu_id_food = recommender_menuid(vector_food, 2)
    menu_id_cake = recommender_menuid(vector_cake, 3)
    menu_id_other = recommender_menuid(vector_other, 4)

    return menu_id_food, menu_id_cake, menu_id_other

def get_rating(id_user):
    a = SelectVectorDrink(id_user, 1)[0]
    id_raitng = recommender_menuid(a, 1)
    return id_raitng


def get_menu_id_rating(vectorS, option): # 1 drink #food # cake
    max_v = 0
    get_vector = []
    for v in vectorS:
        v1 = get_vector_removeid(v)
        max_v1 = max(v1)
        if max_v <= max_v1:
            max_v = max_v1
            get_vector = v
    menu_id = recommender_menuid(get_vector, option)
    return menu_id

@app.route('/')
def Index():
    return render_template('index.html')
@app.route('/load', methods=["POST"])
def load():
    global Menu
    print("load")
    if Menu == 1 :
        return jsonify({"Return" : "OK"})
    else:
        return jsonify({"Return" : "NO"
                                   ""})


@app.route('/get')
def get_bot_response():
    DataReturn = []
    global JsonOrder
    global Text
    print(Text + "aaaa")
    if request.args.get('msg') != "a":
        datajson = GetJson(url="http://dtuct.ddns.net:8088/classification", User_input=request.args.get('msg'),
                           Keyword="key_word",numberKey="numberKey")
        print(datajson)
        if datajson['type'] == 'pay' and datajson['status'] == True:
            return jsonify({"Data": "Order"})
        if datajson['type'] == 'Find' and datajson['status'] == False:
            if len(datajson['data']) > 0:
                return jsonify({"Data": datajson['data']})

        else:
            if len(datajson['data']) > 0:
                JsonOrder = []
                for data in datajson['data']:

                    print(data["name"])
                    if "CÀ PHÊ" in data["name"].upper():
                        data["name"] =  data["name"].upper().split("CÀ PHÊ")[1]
                    if "BÁNH MÌ" in  data["name"].upper():
                        data["name"] =  data["name"].upper().split("BÁNH MÌ")[1]
                    if "TRÀ" in  data["name"].upper():
                        data["name"] = data["name"].split("Trà")[1]
                        data["name"] = data["name"].lstrip()
                        data["name"] = "Trà " + data["name"]
                    if "BÁNH" in  data["name"].upper():
                        data["name"] = data["name"].split("Bánh")[1]
                        data["name"] = data["name"].lstrip()
                        data["name"] = "Bánh " + data["name"]
                    user_input = data["name"].lstrip()
                    user_input = user_input.upper()
                    product = Product.query.filter_by(Name=user_input).all()
                    if len(product) > 0:
                        DataReturn.append(user_input)
                        JsonOrder.append({"name": user_input, "num_order": data["num_order"], "price": data["price"]})
                    else:
                        return jsonify({"Data": "Món quý khách vừa gọi không có"})
                return jsonify({"Data": DataReturn})
            else:
                return jsonify({"Data": datajson['mes']})
    else:
        if Text == "Finish":
            return jsonify({"Data": "Return"})
        elif Text != "":
            print("aaaa" + Text)
            datajson = GetJson(url="http://dtuct.ddns.net:8088/classification", User_input=Text,
                               Keyword="key_word",numberKey="numberKey")
            Text = ""
            print(datajson)
            if datajson['type'] == 'pay' and datajson['status'] == True:
                return jsonify({"Data": "Order"})
            if datajson['type'] == 'Find' and datajson['status'] == False:
                if len(datajson['data']) > 0:
                    return jsonify({"Data": datajson['data']})

            else:
                if len(datajson['data']) > 0:
                    JsonOrder = []
                    for data in datajson['data']:
                        print( data["name"].lower())
                        if "Cà Phê" in data["name"]:
                            data["name"] = data["name"].split("Cà Phê")[1]
                        if "Bánh Mì" in data["name"]:
                            data["name"] = data["name"].split("Bánh Mì")[1]
                        if "Trà" in data["name"]:
                            data["name"] = data["name"].split("Trà")[1]
                            data["name"] = data["name"].lstrip()
                            data["name"] = "Trà " + data["name"]
                        if "Bánh" in data["name"]:
                            data["name"] = data["name"]
                            data["name"] = data["name"].split("Bánh")[1]
                            data["name"] = data["name"].lstrip()
                            data["name"] = "Bánh " + data["name"]

                        user_input = data["name"].lstrip()
                        print(user_input + "5")
                        user_input = user_input.upper()
                        print(data["name"] + "6")
                        print(str(user_input) +"aaaaaa")
                        product = Product.query.filter_by(Name=user_input).all()
                        if len(product) > 0:
                            DataReturn.append(user_input)
                            JsonOrder.append(
                                {"name": user_input, "num_order": data["num_order"], "price": data["price"]})
                        else:
                            return jsonify({"Data": "Món quý khách vừa gọi không có"})

                    return jsonify({"Data": DataReturn})
                else:
                    return jsonify({"Data": datajson['mes']})
        else:
            return jsonify({"Data" : "NoReturn"})

@app.route('/process', methods=['POST'])
def process():
    global JsonOrder
    global userOrder
    user_input1 = request.form['dataJson']

    if user_input1 == "getdata":

        NameFooddrink = []
        user_input = request.form['user_input']
        user_input = user_input.split("[")
        user_input = user_input[1].split("]")
        user_input = user_input[0].split(',')
        Menu = []
        for data in user_input:
            NameFooddrink.append(data.split('"')[1])
        for user_input in NameFooddrink:
            user_input = user_input.upper()
            JsonFood = ["BÁNH MÌ", "TRÀ", "BÁNH NGỌT", "FREEZE", "CÀ PHÊ", "MERCHANDISE", "CÀ PHÊ ĐÓNG GÓI"]
            if user_input in JsonFood:
                user_input = user_input.lower()
                print(user_input)
                JsonCafe = ["cà phê", "ca phe", "cà phe", "ca phê", "cafe", "ca fe"]
                JsonBanhNgot = ["bánh ngọt", "banh ngot", "bánh ngot", "banh ngọt"]
                JsonBanhMi = ["bánh mì", "banh mi", "bánh mi", "banh mì"]
                JsonFreeze = ["freeze"]
                JsonTea = ["trà", "tra", "tea"]
                JsonMenu = ["menu"]
                if user_input in JsonMenu:
                    product = Product.query.all()
                    for data in product:
                        Menu.append(
                            {"Name": data.Name, "Img": data.Img, "Price": data.Price, "Category": data.Category})
                    return jsonify({"Menu": Menu})
                if user_input in JsonCafe:
                    product = Product.query.filter_by(Tag="Cà Phê").all()
                    for data in product:
                        Menu.append(
                            {"Name": data.Name, "Img": data.Img, "Price": data.Price, "Category": data.Category})
                if user_input in JsonBanhNgot:
                    product = Product.query.filter_by(Tag="Bánh Ngọt").all()
                    for data in product:
                        Menu.append(
                            {"Name": data.Name, "Img": data.Img, "Price": data.Price, "Category": data.Category})
                if user_input in JsonBanhMi:
                    product = Product.query.filter_by(Tag="Bánh Mì").all()
                    for data in product:
                        Menu.append(
                            {"Name": data.Name, "Img": data.Img, "Price": data.Price, "Category": data.Category})
                if user_input in JsonFreeze:
                    product = Product.query.filter_by(Tag="Freeze").all()
                    for data in product:
                        Menu.append(
                            {"Name": data.Name, "Img": data.Img, "Price": data.Price, "Category": data.Category})
                if user_input in JsonTea:
                    product = Product.query.filter_by(Tag="Trà").all()
                    for data in product:
                        Menu.append(
                            {"Name": data.Name, "Img": data.Img, "Price": data.Price, "Category": data.Category})

            else:
                product = Product.query.filter_by(Name=user_input).all()
                if len(product) > 0:
                    Menu.append({"Name": product[0].Name, "Img": product[0].Img, "Price": product[0].Price,
                                 "Category": product[0].Category})

        if len(Menu) > 0:
            return jsonify({"Menu": Menu})
    if user_input1 == "Order":

        user_input = request.form['user_input']
        user_input = user_input.lower()
        print(user_input)
        Jsonorder = ["order"]
        Jsonfinish = ["finish"]
        JsonCafe = ["cà phê", "ca phe", "cà phe", "ca phê", "cafe", "ca fe"]
        JsonBanhNgot = ["bánh ngọt", "banh ngot", "bánh ngot", "banh ngọt"]
        JsonBanhMi = ["bánh mì", "banh mi", "bánh mi", "banh mì"]
        JsonFreeze = ["freeze"]
        JsonTea = ["trà", "tra", "tea"]
        JsonMenu = ["menu"]
        if user_input in JsonMenu:
            product = Product.query.all()
            for data in product:
                Menu.append({"Name": data.Name, "Img": data.Img, "Price": data.Price, "Category": data.Category})
            return jsonify({"Menu": Menu})
        if user_input in JsonCafe:
            product = Product.query.filter_by(Tag="Cà Phê").all()
            for data in product:
                Menu.append({"Name": data.Name, "Img": data.Img, "Price": data.Price, "Category": data.Category})
            return jsonify({"Menu": Menu})
        if user_input in JsonBanhNgot:
            product = Product.query.filter_by(Tag="Bánh Ngọt").all()
            print(product)
            for data in product:
                Menu.append({"Name": data.Name, "Img": data.Img, "Price": data.Price, "Category": data.Category})
            return jsonify({"Menu": Menu})
        if user_input in JsonBanhMi:
            product = Product.query.filter_by(Tag="Bánh Mì").all()
            print(product)
            for data in product:
                Menu.append({"Name": data.Name, "Img": data.Img, "Price": data.Price, "Category": data.Category})
            return jsonify({"Menu": Menu})
        if user_input in JsonFreeze:
            product = Product.query.filter_by(Tag="Freeze").all()
            for data in product:
                Menu.append({"Name": data.Name, "Img": data.Img, "Price": data.Price, "Category": data.Category})
            return jsonify({"Menu": Menu})
        if user_input in JsonTea:
            product = Product.query.filter_by(Tag="Trà").all()
            for data in product:
                Menu.append({"Name": data.Name, "Img": data.Img, "Price": data.Price, "Category": data.Category})
            return jsonify({"Menu": Menu})
        if user_input in Jsonorder:
            for data in JsonOrder:
                db.create_all()
                order = Order(Name=data['name'], Quantity=data['num_order'], UserID=userOrder,
                                  Price=data['price'])
                db.session.add(order)
                db.session.commit()

            return jsonify({"Order": JsonOrder})
        if user_input in Jsonfinish:
            JsonOrder = []
            return jsonify({"Menu": "OK"})
session = scoped_session(sessionmaker(bind=engine))
@app.teardown_request
def remove_session(ex=None):
    session.remove()
@app.route('/video_feed')
def video_feed():
    image = ImageProgressSing()
    return Response(image.gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
#@app.route('/Menu/<userid>/<emotions>')
@app.route('/Menu/<userid>/<emotions>')
def Menu(userid, emotions):
    # voice = ImageProgressSing()
    # threading.Thread(target=voice.Voice).start()
    global userOrder
    userOrder = userid
    Menu = []
    if userid != "000000000":
        menuid = get_for_menu_id(userid)
        menuidRating = get_rating(userid)
        print(str(menuid) + "menuid")
        if len(menuid) > 0:
            if menuidRating > 0:
                product = Product.query.filter_by(menuid=menuidRating).all()
                for data in product:
                    Menu.append(
                        {"Name": data.Name, "Img": data.Img, "Price": data.Price, "Category": data.Category})

        if len(menuid) > 0:
            for id in menuid:
                product = Product.query.filter_by(menuid=id).all()
                for data in product:
                    Menu.append(
                        {"Name": data.Name, "Img": data.Img, "Price": data.Price, "Category": data.Category})
        return render_template('chatbot.html', Menu=Menu,Title = "MÓN ĐƯỢC BÌNH CHỌN CAO VÀ THƯỜNG XUYÊN SỬ DỤNG")
    else:
        S = SelectVectorDrink(id=0, option=2)
        menuid_drink = get_menu_id_rating(S, 1)
        menuid_food = get_menu_id_rating(S, 2)
        menuid_cake = get_menu_id_rating(S, 3)
        if menuid_drink > 0 :
            product = Product.query.filter_by(menuid=menuid_drink).all()
            for data in product:
                Menu.append(
                    {"Name": data.Name, "Img": data.Img, "Price": data.Price, "Category": data.Category})
        if menuid_food > 0:
            product = Product.query.filter_by(menuid=menuid_food).all()
            for data in product:
                Menu.append(
                    {"Name": data.Name, "Img": data.Img, "Price": data.Price, "Category": data.Category})
        if menuid_cake > 0:
            product = Product.query.filter_by(menuid=menuid_cake).all()
            for data in product:
                Menu.append(
                    {"Name": data.Name, "Img": data.Img, "Price": data.Price, "Category": data.Category})
        return render_template('chatbot.html', Menu=Menu,Title = "MÓN ĐƯỢC BÌNH CHỌN CAO")
@app.route('/GoMenu/')
def GoMenu():
    print("eeeeee")
    global Menu
    print(Menu)
    if Menu == 1:
        name = NameFace[0]
        emotions = NameFace[1]
        Menu = 0
        print(name, Menu, emotions)
        return redirect(url_for('Menu', userid=name, emotions=emotions))
    else:
        return redirect('/')
# def GoMenu():
#     userid = ""
#     emotions = ""
#     global Menu
#     if Menu == 1:
#         print("dddd" +  userid )
#         userid = NameFace[0]
#         emotions = NameFace[1]
#         Menu = 0
#         return redirect(url_for('Menu', userid=userid, emotions=emotions))
#     else:
#         return redirect('/')
@app.route('/GoIndex')
def GoIndex():
    user_id
    return redirect(url_for('Index'))
def GetJson(User_input, url, Keyword,numberKey):
    data ={Keyword: User_input,numberKey:8888}
    try:
        reponse = requests.post(url=url, data=data)
        data = reponse.json()
    except requests.exceptions.ConnectionError:
        data = [{'End': False, 'data': [], 'mes': 'Kết nói với server đang bị gián đoạn quý khách vui lòng thông cảm',
                 'status': True, 'type': 'order'}]
    return data
class ImageProgressSing:
    Check = False


    def __init__(self):
        self.a = 0
        self.detector = dlib.simple_object_detector("Models/FaceDetection100.svm")
        self.predictor = dlib.shape_predictor("Models/shape_predictor_68_face_landmarks.dat")
        self.model_recognition = load_model("Models/alexnet_recognition.h5")
        self.model_emotion = load_model("Models/model_alexnet_New_v6.h5")

    def create_tree_folder(self):
        listData = ["Data_Attendance", "Data_All_Frame"]
        listTime = ["AM", "PM"]
        listName = self.get_label_name("Models/LabelName.txt")
        now = datetime.datetime.now()
        date = now.strftime("%d%m%Y")
        # Thư mục gốc
        if not os.path.exists("Database"):
            os.mkdir("Database")
            for dataFolder in listData:
                os.mkdir("Database/" + str(dataFolder))
                os.mkdir("Database/" + str(dataFolder) + "/" + str(date))
                for time in listTime:
                    os.mkdir("Database/" + str(dataFolder) + "/" + str(date) + "/" + str(time))
                    for name in listName:
                        os.mkdir(
                            "Database/" + str(dataFolder) + "/" + str(date) + "/" + str(time) + "/" + str(name))
        else:
            for dataFolder in listData:
                if not os.path.exists("Database/" + str(dataFolder)):
                    os.mkdir("Database/" + str(dataFolder))
                    os.mkdir("Database/" + str(dataFolder) + "/" + str(date))
                    for time in listTime:
                        os.mkdir("Database/" + str(dataFolder) + "/" + str(date) + "/" + str(time))
                        for name in listName:
                            os.mkdir(
                                "Database/" + str(dataFolder) + "/" + str(date) + "/" + str(time) + "/" + str(
                                    name))
                else:
                    if not os.path.exists("Database/" + str(dataFolder) + "/" + str(date)):
                        os.mkdir("Database/" + str(dataFolder) + "/" + str(date))
                        for time in listTime:
                            os.mkdir("Database/" + str(dataFolder) + "/" + str(date) + "/" + str(time))
                            for name in listName:
                                os.mkdir(
                                    "Database/" + str(dataFolder) + "/" + str(date) + "/" + str(time) + "/" + str(
                                        name))
                    else:
                        for time in listTime:
                            if not os.path.exists(
                                    "Database/" + str(dataFolder) + "/" + str(date) + "/" + str(time)):
                                os.mkdir("Database/" + str(dataFolder) + "/" + str(date) + "/" + str(time))
                                for name in listName:
                                    os.mkdir(
                                        "Database/" + str(dataFolder) + "/" + str(date) + "/" + str(
                                            time) + "/" + str(
                                            name))
                            else:
                                for name in listName:
                                    if not os.path.exists(
                                            "Database/" + str(dataFolder) + "/" + str(date) + "/" + str(
                                                time) + "/" + str(name)):
                                        os.mkdir("Database/" + str(dataFolder) + "/" + str(date) + "/" + str(
                                            time) + "/" + str(name))
                                    else:
                                        break

    def get_label_name(self, path_label_name):
        file = open(path_label_name, "r")
        MSCB = []
        for line in file:
            line = line.strip()
            MSCB.append(line)
        return MSCB

    def gen(self):

        fa = FaceAligner(self.predictor, desiredFaceWidth=150, desiredLeftEye=(0.28, 0.28))
        nt = NamesTracker()
        label = self.get_label_name("Models/LabelName.txt")
        full_name = {'000000000': "Unknown", 'C02082019002': "Tran Bao Toan", 'C02082019005': "Truong Tuan Anh",
                     'C02082019001': "Le Hoang Nhi", 'C02082019004': "Dinh Mau Minh", 'C02082019003': "Ho Quoc An"}
        emotions = ['Angry', 'Happy', 'Neutral']
        now = datetime.datetime.now()
        date_folder = now.strftime("%d%m%Y")
        date = now.strftime("%d-%m-%Y %H-%M-%S-%MS")
        p = now.strftime("%p")
        id_img = 0
        id_frame = 0
        count_no_face = 0
        check = True
        cap = cv2.VideoCapture(cv2.CAP_DSHOW)
        while check:
            red, frame = cap.read()
            image1 = frame.copy()
            if not red:
                print("Error: failed to capture image")
                break
            rects = self.detector(frame, 0)
            if len(rects) == 0:
                nt.update_no_faces()
                count_no_face += 1
                if count_no_face >= 3 * 60:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + open('static/b.jpg', 'rb').read() + b'\r\n')
                else:
                    cv2.imwrite('demo.jpg', image1)
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + open('demo.jpg', 'rb').read() + b'\r\n')
            else:

                count_no_face = 0
                face_max = 0
                face_res = 0
                for i, rect in enumerate(rects):
                    freq = rect.right() - rect.left()
                    if freq > face_max:
                        face_max = freq
                        face_res = i

                faceAligned = fa.align(frame, rects[face_res])
                faceImage = cv2.cvtColor(faceAligned, cv2.COLOR_BGR2RGB)
                arrayImage = Image.fromarray(np.uint8(faceImage))
                imageResize = arrayImage.resize((227, 227))
                array = img_to_array(imageResize)
                feature = np.expand_dims(array, axis=0)

                result_recognition = self.model_recognition.predict(feature)
                ind_label = int(np.argmax(result_recognition[0]))
                name_recognition = label[ind_label]
                proba_recognition = round(result_recognition[0][ind_label] * 100, 0)

                result_emotion = self.model_emotion.predict(feature)
                ind_emotion = int(np.argmax(result_emotion[0]))
                name_emotion = emotions[ind_emotion]
                proba_emotion = round(result_emotion[0][ind_emotion] * 100, 0)

                if name_recognition == "000000000":
                    cv2.imwrite(
                        "Database/Data_Attendance/" + str(date_folder) + "/" + str(p) + "/" + str(name_recognition)
                        + "/" + str(date) + "_" + str(id_img) + ".jpg", faceAligned)
                    id_img += 1
                else:
                    if proba_recognition < 100:
                        cv2.imwrite(
                            "Database/Data_Attendance/" + str(date_folder) + "/" + str(p) + "/" + str(name_recognition)
                            + "/" + str(date) + "_" + str(id_img) + ".jpg", faceAligned)
                        id_img += 1
                # Set nguong cho du lieu unknow
                if proba_recognition < 99:
                    name_recognition = "000000000"

                # cv2.putText(image1, u'Xin Chao ' + str(full_name[name_recognition]) + "-" + str(proba_recognition),
                #             (rects[face_res].left(), rects[face_res].top()),
                #             cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 1, cv2.LINE_AA)
                # cv2.putText(image1, name_emotion + "__" + str(proba_emotion), (rects[face_res].left(),
                #                                                                rects[face_res].bottom()),
                #             cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 1, cv2.LINE_AA)
                cv2.rectangle(image1, (rects[face_res].left(), rects[face_res].top()),
                              (rects[face_res].right(), rects[face_res].bottom()), (0, 255, 0), 1)
                cv2.imwrite(
                    "Database/Data_All_Frame/" + str(date_folder) + "/" + str(p) + "/" + str(
                        name_recognition) + "/" + str(
                        date) + "_" + str(id_frame) + ".jpg", image1)

                nt.update(name_recognition, name_emotion)
                label_objects, label_emotions = nt.attendance()
                if len(label_objects) > 0:
                    self.object_name = label_objects[0]
                    user_id = self.object_name
                    if self.Check == False:
                        self.Check = True
                    if self.Check == True:
                        global NameFace, Menu
                        Menu = 1
                        NameFace = [self.object_name, label_emotions[0]]
                        print(Menu)
                        cv2.imwrite('demo.jpg', image1)
                        self.Check = None
                        threading.Thread(target=self.Voice).start()

                    # cv2.imwrite('/static/Image/demo.jpg', image1)
                    # yield (b'--frame\r\n'
                    #        b'Content-Type: image/jpeg\r\n\r\n' + open( 'demo.jpg', 'rb').read() + b'\r\n')
                    # cv2.imwrite('demo.jpg', image1)
                    # yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + open('demo.jpg', 'rb').read() + b'\r\n')

                    # print(object_name)
                    # cap.release()

                else:
                    cv2.imwrite('demo.jpg', image1)
                    yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + open('demo.jpg', 'rb').read() + b'\r\n')

    def Voice(self):
        global user_id
        global is_say_hi
        global Text
        Text = ""

        print("Name" + self.object_name)
        print(str(user_id) + "user_id")
        name = '000000000'
        if is_say_hi:
            if self.object_name != '000000000':
                pass
                # current_user = self.get_current_user(self.object_name)
                # print(self.object_name)
                # name = current_user[0]["name"]

            greetings_text = 'Xin chào {}'.format(name)
            speaker.speak(greetings_text)
            is_say_hi = False

        file_path = glob.glob("{}/*.mp3".format(config.DEFAULT_FOLDER_AUDIO_GREETINGS_WELCOME))
        file = random.choice(file_path)
        speaker.play_audio_file(file)

        count = 0
        while True:
            Text = listener.listen()
            if Text != "" :
                count = 0
            if Text == "":
                speaker.speak("Tôi chưa hiểu yêu cầu của bạn, xin vui lòng lặp lại")
                count += 1
                if count == 5:
                    Text =  "Finish"
                    break
                continue
            else:
                pass
                # classes = text_classification.classify(Text)
                # print("else", classes)
                # if len(classes) > 0:
                #     if classes[0] == "Order-Processing":
                #         # write file json order
                #         print("aaaa")
                #         # return place_orders()
                #     elif classes[0] == "Complains-Bad-FB" or classes[0] == "Complains-Bad-Services":
                #         print("ccccc")
                #         # write file json complain text
                #         # return complain(text)
                #     else:
                #         count += 1

            print(count)
            if count == 5:
                Text = "Finish"
                break

        speaker.speak("Tôi chưa hiểu yêu cầu của bạn!")
        print("goes here")
        # return jsonify([text, "Tôi chưa hiểu yêu cầu của bạn!"])
        if self.Check is None:
            self.Check = False

    # def get_current_user(self, user_id):
    #     with open(config.DEFAULT_USER_INFORMATION, encoding='utf-8') as data_file:
    #         user_data = json.load(data_file)
    #         current_user = [item for item in user_data if item.get('id') == int(user_id)]
    #         return current_user


if __name__ == '__main__':
    app.run()
