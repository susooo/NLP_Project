# %%
import gradio as gr
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from nltk.tokenize import sent_tokenize
from konlpy.tag import Kkma
import gc

import nltk
nltk.download('punkt')
# from PyKakao import KoGPT
# kogpt_api = KoGPT(service_key = "")
import openai
openai.api_key = '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'
gpt2_tokenizer = AutoTokenizer.from_pretrained('gpt2')

#en2ko = 'alphahg/m2m100_418M-finetuned-en-to-ko-4770260'#'alphahg/mbart-large-50-finetuned-en-to-ko-8603428-finetuned-en-to-ko-9914408'
en2ko = 'alphahg/mbart-large-50-finetuned-en-to-ko-8603428-finetuned-en-to-ko-9914408'
ko2en = 'alphahg/opus-mt-ko-en-finetuned-ko-to-en-2780616'
ensum = 'allenai/led-large-16384-arxiv'
kosum = 'alphahg/pko-t5-small-finetuned-paper-4564652' #'lcw99/t5-base-korean-text-summary'

kkma = Kkma()
#en_pipe = pipeline('translation', model=en2ko, tokenizer=en2ko, src_lang = "en", tgt_lang = "ko", device_map="auto")
en2ko_model = AutoModelForSeq2SeqLM.from_pretrained(en2ko)

en_pipe = pipeline('translation', model=en2ko_model, tokenizer=en2ko, src_lang = "en_XX", tgt_lang = "ko_KR", device="cuda:0")
ko_pipe = pipeline('translation', model=ko2en, tokenizer=ko2en, device="cuda:0")
style_pipe = pipeline('translation', model=en2ko_model, tokenizer=en2ko, src_lang = "ko_KR", tgt_lang = "ko_KR", device="cuda:0")

en_sum = pipeline('summarization', model=ensum, tokenizer=ensum, device="cuda:1")
ko_sum = pipeline('summarization', model=kosum, tokenizer=kosum, device="cuda:1")

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

# chatbot = Chatbot({
#   #"session_token": "eyJhbGciOiJkaXIiLCJlbmMiOiJBMjU2R0NNIn0..hV_ujfbYLwBgI-g6.zQW0evUrpYfli2cujTFp1ie5PhthUZayoSY2Chb1Eb8Ow3t6l2-NUwGJcYyxVKQS0aITN3-ph-KzPysnu7dCF9KrC-22DZzs1zMFm3PHEkjb4jD69qndcFEGGH8y4SejfYwvdj4wfKmVnGo3XTNZ31qPSDA2PBoaOMpxWABSqWULMJbS-_Y--wd0YhsqMFlkCQpXyfxSf9yxXlPYt0HR_NgupoBXP-WVYbODSUYFVqa3IsScbSPS-mUY0YrAb8AJUkvej7HiSCN9onyThgTtZgZklwqpB4FesJaC6R3nSXSg5cKDDupGVBUQlPRiemacCV6tXnSC-bCtN9a-l9RRqtX_FJNP8T7Kb75ktuedKKrXXTmk3x7hz_RhYhZ4wXFkbqXexZQXTfQoI2vKlLN73EHBlJOqsDLnOP7zT4Vr2RpBbk1HK5D_uHh5x1X3aBHslHfEQpjjZKiMs0to9DwKzSNHXlNmCeGjT9ZzKVsWYCiseO20IlKxQ63Q_nIbi-6y8e6LWw9O82ESKkkRe8kN-CzGxakKJegmHKQGRPZu9ZEIqwWYqlnahVyWRFOtjfMNN3ncGbQGi54VMyfSqmSvPXecaxsVNzl00gHvmCFBFJBDXM2GTsvEzQsJi1MLyopXtSuiU1anL_kC1eMvew61vd3TtC97ZwlQLaWc6dT14p5NdJpN9ihFpgtxMP1rcQhNTc5fo9BBoKrO4yGOuy0wixJs6ORVdY2o3c653X8PFmrso1XaV3KpBaSmcvszIhL1anJTA8SpnPNRmksEFONcX8AfpQ-4WbckCS737TZYCDulVtvukyVAbtq9cEQP5kXXAGOWKg5lX4nRFynM83f8P-XGIg-XUGE99NRIOCBo28cr04fWFOaJOyHf9eP6Rx5zjNv1qwp4FxhVVP9jlmSTfu97CZSR91L-k8V6jVgbj8F6YUZ6iiu51kaOAqf5de4EUncSFyGLuJfCGTJTPSYYl1lnR6bSfTVHKwP28YzzcU2myMM5B0ZXDwydD900TYXZOCxxLPUbu5-G3roR2KZnuWLXFOiafAvDx-LHYUHSWQZ9ouWcDaQBNsXmfTZtIWHQ8aTZwlNEnN4-uFdlk2Lm35qp1v-8Fp_3aXGQ3CrTy-ryMV0rUPTSMCEA8gVA_mD40zV6Wcb4asc3zsYAuomQ3Iu4iB5wyWGxUIJVzl1C9QaPpAx7vp5u7w-0_rtocVVXFRTZ8aSxNS3QAd62TbVyToIOrsvp4kOWDcqhNp5QBAsJtES9pbO9fiy_SJS83SFMliSFd-jhXfKu0kUYIUb9yaN5QC6eEpgJ7KzhwTcNDtoqyBKMyVTSdUXA9P2Yv2e4r-BVnxlW0RxknQdesK-wZrwuAZt_bnLaHSqFzyWz5AE7pukTBQ2QdVoity_tVURzhTcINh6rvPywm2IVl1gC3FjfhQVfTvHWFtUzNqLN4yUfI0Tc1mGHQlYuxZ_yux2B8HeYb_cyb1rR_mwDiOs3PKOnhfNRdXqXf6RWr7KdjNc0k-CMm13DAYQggmFCmEZW20FiwalKqVq3nFTFDhfp5mxtr0sLCVxGA3eTqC6_i2TAVGqDLjxzfz5WiK7J3FAN2_kmEZLBVXHabwa9kKyCzcgCx6FrxaFidskO6t4dWu3wok95hXMae0Z4ZFs7HVNisM1pkRm7LE3XdvnslKHAJkPr57HsFdlQITJRSx3Tg0EN1LUt80hKx8VGXPv7zBeXP5lni2ixpglMQmiKLiszowGoqu2oJPwougueu5Bj4BLhmoqK8DCtdxl3MYAyxLWWStXQqcEJQw7koYmPNwr4BzI9cQVk81LbPwrXBbJfR7G14e5qV0lULfuU5qVfNt7DU6FbwXmzv6qFI-jOClLzSTKpFzp51wQQ5fh2REs6CPJlL-kiyomJPXcqSeezDCLLwWjI_vIyODFkzt91l-dmFriu3HMkMC1v29AJlfPA_avSiJzJDEI7rb6AbEyT6piqp1TYlWMkI_rJCsCZXIb10Rjd2y9sR-Dz3_FZzRvJUA7BfRlP7Bf04HnYsyMRoJilbuyQ5fB0B2L2nxjYY2zoHJ_x6HTS6tcrAijOO4FSSQngWD9iTKCm6pjW3aZjFyXyjmP82S3VnhEyON390aIL7j9Y0wGnHzOkn54OfyxxGeo2mFAIv9kthL_Fi8d9G_rvvQOBUM2a7kjF5-n8wby0YDujoRl0ETg379HyMVf7F2BHWQ8nAbICRxWZ7EzPLwzrVjPiQVPZklkrVYgEmGxZDrgEG8IeNi7FMgGruaQ1tENczRMXzaApK8k6-FXKhfFIV7dN95tP4k6tnnxRFoMAUWcXwQCzRH8YhID36TAUFdBQ-c52MTogPo1Rki1N49j_e7Mph1OABQX2Fw9-CukT6reQkp3nGrwi0IKnKoyGhHOBHK3kzQwINfjOBbNpOjP-6MX_9kiRTstN2GLte8w0QJQVl84o8ACTjV8N4rhI6xyLKIvqoyZ6jNO3SYs8fEutmZO8-qB0iksIiQHupxQgcmgbAyM.KxAbJWqMvwm0PYtq7MuR6A"

# }, conversation_id=None, parent_id=None) # You can start a custom conversation
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
    gc.collect()

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
def translate_with_sum(text, lang, gpt_fix=False):
    from_en = False if lang == '한영' else True

    if lang == '영한':
        summary = en_sum(text, max_length=int(len_tokens(text, en_sum) * 0.6)+8)
        text = summary[0]['summary_text']

    sentences = sent_tokenize(text) if from_en else kkma.sentences(text)
    #print(sentences)
    if not sentences:
        return ''

    paragraphs = split_sent(sentences, en_pipe if from_en else ko_pipe)
    #print(paragraphs)

    ret = []
    for text in paragraphs:
        result = en_pipe(text) if from_en else ko_pipe(text)
        ret.append(result[0]['translation_text'])

    summarized = ' '.join(ret)
    if lang == '한영':
        summary = en_sum(summarized, max_length=int(len_tokens(summarized, en_sum) * 0.6)+8)
        return summary[0]['summary_text']

    gc.collect()
    return summarized

def summarize(text, lang):
    if lang == 'Korean':
        summarizer = ko_sum
    elif lang == 'English':
        summarizer = en_sum

    summary = summarizer(text, max_length=int(len_tokens(text, summarizer) * 0.6)+8)[0]['summary_text']
    return summary

def translate_styleonly(text):
    sentences = kkma.sentences(text)
    paragraphs = split_sent(sentences, style_pipe, max_len=180)
    #print(paragraphs)

    ret = []
    for text in paragraphs:
        result = style_pipe(text)
        ret.append(result[0]['translation_text'])

    translated = ' '.join(ret)
    gc.collect()

    return translated

# %%
interface1 = gr.Interface(fn=translate, inputs=["text", gr.Radio(["영한", "한영"], value='영한'), 'checkbox'], outputs="text")
interface2 = gr.Interface(fn=translate_with_sum, inputs=["text", gr.Radio(["영한", "한영"], value='영한')], outputs="text")
parallel_interface = gr.Parallel(interface1, interface2)

summarize_interface = gr.Interface(fn=summarize, inputs=["text", gr.Radio(["Korean", "English"], value='Korean')], outputs="text")
style_interface = gr.Interface(fn=translate_styleonly, inputs=["text"], outputs="text")
#%%
demo = gr.TabbedInterface([parallel_interface, summarize_interface, style_interface], ['번역 및 요약', '요약', '스타일 번역'], css="footer {visibility: hidden}") # '요약'
demo.launch(share=True) # Share the demo
# %%