import pandas as pd
import json
import requests
import time

pd_all = pd.read_csv('../data/baoxian_right.csv', sep='\t')
best_title = pd_all['best_title'].tolist()


def translate(word, ip):
    # 有道词典 api
    url = 'http://fanyi.youdao.com/translate?smartresult=dict&smartresult=rule&smartresult=ugc&sessionFrom=null'
    # 传输的参数，其中 i 为需要翻译的内容
    key = {
        'type': "AUTO",
        'i': word,
        "doctype": "json",
        "version": "2.1",
        "keyfrom": "fanyi.web",
        "ue": "UTF-8",
        "action": "FY_BY_CLICKBUTTON",
        "typoResult": "true"
    }

    # 代理
    proxies = {'http': ip, 'https': ip}
    # key 这个字典为发送给有道词典服务器的内容
    response = requests.post(url, data=key, proxies=proxies)
    # 判断服务器是否相应成功
    if response.status_code == 200:
        # 然后相应的结果
        return response.text
    else:
        print("有道词典调用失败")
        # 相应失败就返回空
        return None


def get_ip(url):
    ip_text = requests.get(url)
    ip_text = ip_text.text
    while ip_text.find('提取太频繁') != -1:
        time.sleep(5)
        print('提取太频繁')
        ip_text = requests.get(url)
        ip_text = ip_text.text
    return ip_text.strip()


num_thread_all = 15
num_thread_ing = 0
ip = None
flag = True  # 表示线程可前进
synonymous = []
synonymous.append("best_title" + '\t' + "translated" + '\n')
from threading import Lock
import threading

lock = Lock()


def get_synonymous_thread(line, index):
    global num_thread_ing
    global ip
    try:
        list_trans = translate(line, ip)
        #     print('test')
        if list_trans == None:
            if index == 0:
                ip = get_ip(
                    'http://api.xdaili.cn/xdaili-api//greatRecharge/getGreatIp?spiderId=d4980dea2ab74a35907e9534fc146246&orderno=YZ2019840424LhDCX9&returnType=1&count=1')

        elif list_trans.find('来自您ip的请求异常频繁') == -1:
            result = json.loads(list_trans)
            en_result = result['translateResult'][0][0]['tgt']
            list_trans = translate(en_result, ip)
        else:
            flag = False

        if list_trans == None:
            if index == 0:
                ip = get_ip(
                    'http://api.xdaili.cn/xdaili-api//greatRecharge/getGreatIp?spiderId=d4980dea2ab74a35907e9534fc146246&orderno=YZ2019840424LhDCX9&returnType=1&count=1')


        elif list_trans.find('来自您ip的请求异常频繁') == -1:
            result = json.loads(list_trans)
            cn_result = result['translateResult'][0][0]['tgt']

            print(line + '\t' + cn_result)
            # lock.acquire()
            synonymous.append(line + '\t' + cn_result + '\n')
            # lock.release()
        else:
            flag = False
    except Exception:
        pass
    num_thread_ing -= 1


ip = get_ip(
    'http://api.xdaili.cn/xdaili-api//greatRecharge/getGreatIp?spiderId=d4980dea2ab74a35907e9534fc146246&orderno=YZ2019840424LhDCX9&returnType=1&count=1')

for idx, line in enumerate(best_title):
    while True:
        if num_thread_ing < num_thread_all:

            num_thread_ing += 1
            threading.Thread(target=get_synonymous_thread, args=(line, num_thread_ing)).start()

            idx = idx + 1
            if idx % 500 == 0:
                print(idx)
                ip = get_ip(
                    'http://api.xdaili.cn/xdaili-api//greatRecharge/getGreatIp?spiderId=d4980dea2ab74a35907e9534fc146246&orderno=YZ2019840424LhDCX9&returnType=1&count=1')

            break
        else:
            time.sleep(1)

with open('../data/baoxian_synonymous.csv', 'w', encoding='utf-8') as file:
    file.writelines(synonymous)

translated = pd.read_csv('../data/baoxian_synonymous.csv', sep='\t', header=0)
merged = pd_all.merge(translated, left_on='best_title', right_on='best_title')
merged[['best_title', 'translated', 'reply', 'is_best']].drop_duplicates(inplace=True)
merged[['best_title', 'translated', 'reply', 'is_best']].to_csv('../data/baoxian_preprocessed_synonymous.csv',
                                                                index=False, sep='\t')