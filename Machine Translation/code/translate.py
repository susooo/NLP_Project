# %%
from transformers import AutoTokenizer, pipeline
from nltk.tokenize import sent_tokenize
from konlpy.tag import Kkma

import os, time
import sys
import urllib.request

import openai
openai.api_key = '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'
gpt2_tokenizer = AutoTokenizer.from_pretrained('gpt2')

import googletrans
gtranslator = googletrans.Translator()

en2ko = 'alphahg/mbart-large-50-finetuned-en-to-ko-8603428-finetuned-en-to-ko-9914408'
ko2en = 'alphahg/opus-mt-ko-en-finetuned-ko-to-en-2780616'

kkma = Kkma()
#en_pipe = pipeline('translation', model=en2ko, tokenizer=en2ko, src_lang = "en", tgt_lang = "ko", device_map="auto")
en_pipe = pipeline('translation', model=en2ko, tokenizer=en2ko, src_lang = "en_XX", tgt_lang = "ko_KR", device="cuda:0")
ko_pipe = pipeline('translation', model=ko2en, tokenizer=ko2en, device="cuda:0")

def papago_translate(text, src, dst):
    time.sleep(0.1)
    client_id = "imOVQK47mF3cTddTpKHY" # 개발자센터에서 발급받은 Client ID 값
    client_secret = "GS0Goaqb24" # 개발자센터에서 발급받은 Client Secret 값
    encText = urllib.parse.quote(text)
    data = f"source={src}&target={dst}&text={encText}"
    url = "https://openapi.naver.com/v1/papago/n2mt"
    request = urllib.request.Request(url)
    request.add_header("X-Naver-Client-Id",client_id)
    request.add_header("X-Naver-Client-Secret",client_secret)
    response = urllib.request.urlopen(request, data=data.encode("utf-8"))
    rescode = response.getcode()
    if(rescode==200):
        response_body = response.read()
        return response_body.decode('utf-8')
    else:
        return "Error Code:" + rescode

def len_tokens(text, pipe):
    return len(pipe.tokenizer(text)['input_ids'])


def split_sent(sentences, pipe, max_len=256):
    if not sentences:
        return []

    paragraphs = []
    example = sentences[0]
    for i in range(1, len(sentences)):
        if len_tokens(example + ' ' + sentences[i], pipe) > max_len:
            paragraphs.append(example)
            example = sentences[i]
        else:
            example += ' ' + sentences[i]
    
    paragraphs.append(example)

    return paragraphs

# %%
def translate(text, lang, gpt_fix=False):
    from_en = False if lang == '한영' else True
    sentences = sent_tokenize(text) if from_en else kkma.sentences(text)
    #print(sentences)
    if not sentences:
        return ''

    paragraphs = split_sent(sentences, en_pipe, max_len=180) if from_en else split_sent(sentences, ko_pipe)
    #print(paragraphs)

    ret = []
    for text in paragraphs:
        result = en_pipe(text) if from_en else ko_pipe(text)
        ret.append(result[0]['translation_text'])

    translated = ' '.join(ret)

    if gpt_fix:
        if lang == '한영':
            prompt = 'Improve given formal article without adding:'
        elif lang == '영한':
            prompt = "추가적인 내용없이 주어진 글을 개선해:"

        def fix_sent(sent):
            number_of_tokens = len(gpt2_tokenizer(sent)['input_ids'])

            response = openai.Completion.create(
                model="text-davinci-003",
                prompt=prompt+'\n'+sent,
                temperature=0,
                max_tokens=number_of_tokens+128,
                top_p=1.0,
                frequency_penalty=0.0,
                presence_penalty=0.0
            )

            return response.choices[0].text.strip()

        # def fix_sent(sent):
        #     generated = kogpt_api.generate(prompt+'\n'+sent, max_tokens=256)
        #     return generated['generations'][0]['text']
        
        translated = fix_sent(translated)

    return translated

#%%
import chardet
import pandas as pd
from pathlib import Path

trans_en = pd.DataFrame(columns=['Original', 'Model', 'Google', 'Papago', 'Bing'])

path = Path('./en-paper')
for file in path.iterdir():
    if file.is_file():
        if file.suffix == '.csv':
            print(file.name)
            with open(file, 'rb') as f:
                result = chardet.detect(f.read())

            with open(file, 'r', encoding=result['encoding']) as f:
                df = pd.read_csv(file)

            trans_en = trans_en.append(df)

for i, text in enumerate(trans_en['Original']):
    trans_en['Model'].iloc[i] = translate(text, lang='영한')
    trans_en['Google'].iloc[i] = gtranslator.translate(text, src='en', dest='ko').text
    #trans_en['Papago'].iloc[i] = papago_translate(text, 'en', 'ko')
    time.sleep(0.1)

print(trans_en.head())
trans_en.to_csv('trans_en.csv')

# %%
trans_ko = pd.DataFrame(columns=['Original', 'Model', 'Google', 'Papago', 'Bing'])

path = Path('./ko-paper')
for file in path.iterdir():
    if file.is_file():
        if file.suffix == '.csv':
            print(file.name)
            with open(file, 'rb') as f:
                result = chardet.detect(f.read())

            with open(file, 'r', encoding=result['encoding']) as f:
                df = pd.read_csv(file)

            trans_ko = trans_ko.append(df)

for i, text in enumerate(trans_ko['Original']):
    trans_ko['Model'].iloc[i] = translate(text, lang='한영')
    trans_ko['Google'].iloc[i] = gtranslator.translate(text, src='ko', dest='en').text
    #trans_ko['Papago'].iloc[i] = papago_translate(text, 'ko', 'en')
    time.sleep(0.1)

print(trans_ko.head())
trans_ko.to_csv('trans_ko.csv')
# %%
