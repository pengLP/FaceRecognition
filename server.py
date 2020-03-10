from flask import Flask, render_template, request
app = Flask(__name__)
from face_train import Model
from skimage import io,transform
import json
import os
import csv
import time,hashlib
import datetime
import pandas as pd

def create_id():
    m = hashlib.md5()
    m.update(bytes(str(time.clock()),encoding='utf-8'))
    return m.hexdigest()

class SaveCSV(object):

    def save(self, keyword_list,path, item):
        """
        保存csv方法
        :param keyword_list: 保存文件的字段或者说是表头
        :param path: 保存文件路径和名字
        :param item: 要保存的字典对象
        :return:
        """
        try:
            # 第一次打开文件时，第一行写入表头
            if not os.path.exists(path):
                with open(path, "w", newline='', encoding='utf-8') as csvfile:  # newline='' 去除空白行
                    writer = csv.DictWriter(csvfile, fieldnames=keyword_list)  # 写字典的方法
                    writer.writeheader()  # 写表头的方法

            # 接下来追加写入内容
            with open(path, "a", newline='', encoding='utf-8') as csvfile:  # newline='' 一定要写，否则写入数据有空白行
                writer = csv.DictWriter(csvfile, fieldnames=keyword_list)
                writer.writerow(item)  # 按行写入数据
                print("^_^ write success")

        except Exception as e:
            print("write error==>", e)
            # 记录错误数据
            with open("error.txt", "w") as f:
                f.write(json.dumps(item) + ",\n")
            pass

@app.route('/')
def hello_world():
    df = pd.read_csv('log.csv')
    res = df.to_dict(orient='records')

    return render_template("index.html", data=res)

@app.route('/upload',methods=['POST'])
def upload():
    model = Model()
    model.load_model(file_path='./model/face.model')
    file = request.files.get('file')
    f = io.imread(file)
    probability, name_number = model.face_predict(transform.resize(f,(160,160,3)))
    keyword_list = ['id','name', 'time']
    path = "log.csv"
    s = SaveCSV()
    with open('contrast_table', 'r') as f:
        contrast_table = json.loads(f.read())
    res = {}
    res['id'] = create_id()
    res['name'] = contrast_table[str(name_number)]
    res['time'] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    s.save(keyword_list , path , res)

    # res = "识别结果：" + contrast_table[str(name_number)]
    return json.dumps(res, ensure_ascii=False)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080)