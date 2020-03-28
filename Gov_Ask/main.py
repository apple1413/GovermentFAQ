from extract_feature import BertVector
from flask import Flask
from flask import request
from sklearn.metrics.pairwise import cosine_similarity
import  requests
import numpy as np
import pymysql
from predict_GPU import department
from bert_serving.client import BertClient
bc = BertClient()
app = Flask(__name__)
conn = pymysql.connect(user='root', password='liujie', database='faq', charset='utf8mb4')
cursor = conn.cursor()
#id2deaprt
depart = {'交通运输厅': '1', '人力资源和社会保障厅': '2', '住房城乡建设厅': '3', '体育局': '4', '公安厅': '5', '农业农村厅': '6', '医疗保障局': '7',
                    '卫生健康委': '8', '发展改革委': '9', '司法厅': '10', '商务厅': '11', '国土资源厅': '12', '国家安全厅': '13', '工业和信息化厅': '14',
                    '市场监管局': '15', '广播电视局': '16'
        , '教育厅': '17', '文化厅': '18', '文物局': '19', '新闻出版局': '20', '林业厅': '21', '档案局': '22', '民政厅': '23', '气象局': '24',
                    '水利厅': '25', '环保厅': '26'
        , '生态环境厅': '27', '畜牧局': '28', '科技厅': '29', '粮食和储备局': '30', '能源局': '31', '自然资源厅': '32', '药品监督管理局': '33',
                    '财政厅': '34', '食品药品监管局': '35'
                    }
#get
def get(url,params):
    res = requests.get(url=url,params=params)
    print(res.text)
    return res.text

#  AI question
@app.route('/', methods = ['POST'])
def getAnswer():
    question = request.form.get("question")
    user_que_vectors = bc.encode([question])
    departme= department(question)
    depatnum = depart[str(departme)]
    query = (
        'select  question_vector,answer from que_ans_vec where departmentId=%s')
    cursor.execute(query, str(depatnum))
    cont = cursor.fetchall()
    max=0.0
    bestans=''
    for k in cont:
        bertcos = cosine_similarity(user_que_vectors, np.load(k[0] + '.npy'))
        if bertcos>max:
            max = bertcos
            bestans = k[1]
    print(bestans)
    return  bestans
app.run(port='1001')

