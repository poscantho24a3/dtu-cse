import json


Jsondata = []

def JsonRead (dir):
    with open(dir,encoding="utf8") as data:
        dataMerg = json.load(data)
    return  dataMerg
Data = JsonRead("DataMenu.json")

for data in Data:
    datas = data['unit']
    if 'ly' in datas:
         pass
        # print(datas)
        # content = data['species'] + " " + data['name']
        # Jsondata.append({'Tag' : 'Nước','Name' : content})
    else:
        # pass
        content = data['species'] + " " + data['name']
        Jsondata.append({'Tag': 'Thức ăn', 'Name': content})

with open('MenuThucAn.json', 'w+', encoding='utf-8-sig') as json_file:
    json.dump(Jsondata, json_file, ensure_ascii=False)