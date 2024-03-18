import os
os.environ["PYTHONIOENCODING"] = "utf-8"

import requests
import json
import pandas as pd
import hashlib
import time

def get_baidu_translate(text, from_lang, to_lang, appid, secretKey):
    url = 'http://api.fanyi.baidu.com/api/trans/vip/translate'
    salt = '12345'
    sign = make_md5(appid + text + salt + secretKey)

    headers = {'Content-Type': 'application/x-www-form-urlencoded'}
    payload = {'appid': appid, 'q': text, 'from': from_lang, 'to': to_lang, 'salt': salt, 'sign': sign}

    try:
        response = requests.post(url, params=payload, headers=headers)
        result = json.loads(response.content)
        return result['trans_result'][0]['dst']
    except KeyError:
        print("Error in translation:", response.text)
        return ""

def make_md5(s, encoding='utf-8'):
    return hashlib.md5(s.encode(encoding)).hexdigest()

# The appid and secretKey for the Baidu Translation API.
appid = 'your appid'
secretKey = 'your secretKey'

for i in range(2, 6):
    # Read the corresponding file, here use f-string to insert the value of variable i
    file_name = f'..\\FoldData\\fold{i}.tsv'
    df = pd.read_csv(file_name, sep='\t')
    # selected_df = df[df['myLabel'].isin([0,1,2,3,5, 6,7， 8, 9,10])]
    selected_df = df[df['myLabel'].isin([3])]

    # language to translate
    languages = ['en', 'spa', 'fra', 'ara', 'jp']
    # Create new DataFrame
    new_df = pd.DataFrame(columns=['comment', 'myLabel'])

    # Process each text
    for index, row in selected_df.iterrows():
        original_text = row['comment']
        for lang in languages:
            translated_text = get_baidu_translate(original_text, 'zh', lang, appid, secretKey)
            time.sleep(1)  # increase delay
            back_translated_text = get_baidu_translate(translated_text, lang, 'zh', appid, secretKey)
            time.sleep(1)  # increase delay
            #print(back_translated_text.encode('utf-8').decode('utf-8'))
            new_df = new_df._append({'comment': back_translated_text, 'myLabel': row['myLabel']}, ignore_index=True)

    save_file_name = f'baiduHuiyidata-{i}-3.tsv'
    new_df.to_csv(save_file_name, sep='\t', index=False)
    print("success!")
print("Completed!")
