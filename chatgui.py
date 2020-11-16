import logging

import boto3
import nltk
from botocore.exceptions import ClientError
from nltk.stem import WordNetLemmatizer
import pandas as pd
import pickle
import numpy as np
from keras.models import load_model
import json
import random
from fuzzywuzzy import process
from bs4 import BeautifulSoup as BS
import requests
import csv
import tkinter
from tkinter import *
import time

file_list = ['chatbot_model.h5', 'countries.txt', 'abbrevations', 'intents.json', 'words.pkl', 'classes.pkl', 'eur',
             'classifier_model.h5']

# uncomment to use s3
# s3 = boto3.client('s3')
# for file in file_list:
#     s3.download_file('BUCKET_NAME', 'OBJECT_NAME', file)

model = load_model('chatbot_model.h5')
lemmatizer = WordNetLemmatizer()
countries = []
abbs = []
with open('countries.txt', newline='') as f:
    lines = f.readlines()
    for line in lines:
        countries.append(line.strip('\n'))
with open('abbrevations', newline='') as f:
    lines = f.readlines()
    for line in lines:
        abbs.append(line.strip('\n'))
print(abbs)
intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
print(classes)
context_list = None
ctx_i = None
err_msg = "Sorry I didn't understand, please kindly answer following\n"
err_msg2 = "You can only select yes, no or 1,2\n"
res_msg = "Do you want to restart process? \n"
symptoms = ["exposure", "travel", "fever", "tired", "cough", "breathing",
            "sorethroat", "pain", "nasal", "runnynose", "diarrhea"]
severity = ["mild", "moderate", "severe"]

columns = ["fever", "tired", "cough", "breathing", "sorethroat", "pain", "nasal",
           "runnynose", "diarrhea", "Age_0-9", "Age_10-19", "Age_20-24", "Age_25-59", "Age_60+", "Female",
           "Male", "Other", "Mild", "Moderate", "None", "Severe"
    , "DontKnow", "Contact_No", "Contact_Yes", "China", "Italy", "Iran", "Republic of Korean", "France",
           "Spain", "Germany", "UAE", "Other-EUR", "Other"]
df = pd.DataFrame(columns=columns)
df.loc[len(df)] = 0
eur_list = []
other_list = ["China", "Italy", "Iran", "Republic of Korean", "France",
              "Spain", "Germany"]

with open('eur', 'r') as rd:
    lines = rd.readlines()
    for line in lines:
        eur_list.append(line.strip('\n'))


def find_best_match(misspelled, correct_names):
    closest, ratio = process.extractOne(misspelled, correct_names)
    return closest, ratio


def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words


# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence

def bow(sentence, words, show_details=True):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words - matrix of N words, vocabulary matrix
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print("found in bag: %s" % w)
    return (np.array(bag))


def predict_class(sentence, model):
    # filter out predictions below a threshold
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    print(res)
    ERROR_THRESHOLD = 0.1
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list


def getResponse(ints, intents_json, cur_ctx):
    # context = context
    prev_q = ''
    step_success = False
    stop = False
    probs = [float(p['probability']) for p in ints]
    if max(probs) <= 0.3:
        return err_msg, step_success, stop
    list_of_intents = intents_json['intents']

    for i in ints:
        tag = i['intent']
        for j in list_of_intents:
            # print('j tag ' + str(j['tag']) + 'tag ' + str(tag) + ' ' + str(cur_ctx) + ' ' + str(j['context'][0]))
            if tag == j['tag'] and cur_ctx in j['context']:
                # print('success ' + str(tag) + ' ' + str(cur_ctx) + ' ' + str(j['context'][0]))
                if cur_ctx not in symptoms:
                    result = random.choice(j['responses'])
                else:
                    result = j['responses'][symptoms.index(cur_ctx)]
                if tag == 'severity' or str(j['context'][0]) == 'severity':
                    stop = True
                step_success = True
                break

    if not step_success:
        return err_msg, step_success, stop
    else:
        return result, step_success, stop


def chatbot_response(msg, ctx):
    msg = msg.lower()
    ints = predict_class(msg, model)
    print(ints)
    res, status, stop = getResponse(ints, intents, ctx)
    return res, status, stop


# Creating GUI with tkinter


def send():
    global prev_q
    global ctx_i
    global context_list
    msg = EntryBox.get("1.0", 'end-1c').strip()
    print('msg' + str(msg))
    EntryBox.delete("0.0", END)

    if msg != '':
        ChatLog.config(state=NORMAL)
        ChatLog.insert(END, "You: " + msg + '\n\n')
        ChatLog.config(foreground="#442265", font=("Verdana", 8))
        cur_ctx = context_list[ctx_i]
        res, stat, stop = chatbot_response(msg, cur_ctx)
        if ctx_i > 1:
            if stat:
                prev_q = res
                # if there is a successful step record all the user responses in to a dataframe
                if str(cur_ctx) in columns:
                    if str(msg) == 'yes':
                        df.iloc[0, df.columns.get_loc(str(cur_ctx))] = 1
                    elif str(msg) == 'no':
                        df.iloc[0, df.columns.get_loc(str(cur_ctx))] = 0
                if cur_ctx == 'age':
                    age = int(msg)
                    if 0 < age <= 9:
                        df.iloc[0, df.columns.get_loc('Age_0-9')] = 1
                    elif 9 < age <= 19:
                        df.iloc[0, df.columns.get_loc('Age_10-19')] = 1
                    elif 19 < age <= 24:
                        df.iloc[0, df.columns.get_loc('Age_20-24')] = 1
                    elif 24 < age <= 59:
                        df.iloc[0, df.columns.get_loc('Age_25-59')] = 1
                    else:
                        df.iloc[0, df.columns.get_loc('Age_60+')] = 1

                if cur_ctx == 'severity':
                    match, _ = find_best_match(str(msg), ['Mild', 'Moderate', 'Severe', 'None'])
                    print('match ' + str(match))
                    df.iloc[0, df.columns.get_loc(str(match))] = 1
                    df.to_csv('result.csv', index=False)
                if cur_ctx == 'gender':
                    g_match, _ = find_best_match(str(msg), ['Male', 'Female', 'Other'])
                    df.iloc[0, df.columns.get_loc(str(g_match))] = 1
                if cur_ctx == 'exposure':
                    ex_match, _ = find_best_match(str(msg), ['Yes', 'No'])
                    if ex_match == 'Yes':
                        df.iloc[0, df.columns.get_loc('Contact_Yes')] = 1
                    else:
                        df.iloc[0, df.columns.get_loc('Contact_No')] = 1
                if cur_ctx == 'country':
                    if str(msg).lower() in eur_list:
                        df.iloc[0, df.columns.get_loc('Other-EUR')] = 1
                    elif str(msg) in other_list:
                        df.iloc[0, df.columns.get_loc(str(msg))] = 1
                    else:
                        df.iloc[0, df.columns.get_loc('Other')] = 1
            else:
                res = str(res) + '\n' + str(prev_q)
        if stop:
            ctx_i = 0
            clasifier = load_model('classifier_model.h5')
            prediction = clasifier.predict(df)
            print('preditcion ' + str(prediction))
            prediction = prediction[0][0] * 100
            prediction = str(round(prediction, 2))
            res = 'you have ' + prediction + '% chance of having covid.\n**This is just a prediction, ' \
                                             'do not consider this as the best result. As per expert ' \
                                             'doctors, to know whether you have covid or not is only by ' \
                                             'testing at the nearby hospitals \n\n' + res + '\n\n Do you want to restart the process?'
        if cur_ctx == 'country':
            cnt = str(msg)
            country, ratio = find_best_match(cnt, countries)
            country = str(country).strip('\r')
            print('country ' + str(country) + 'ratio ' + str(ratio))
            if ratio != 100:
                for abc in abbs:
                    conts = abc.split(',')[:-1]
                    print('conts ' + str(conts))
                    clos, ratio2 = find_best_match(cnt, conts)
                    print('closest ' + str(clos) + ' ratios ' + str(ratio2))
                    if ratio2 > 90:
                        country = abc.split(',')[-1]
                        country = country.strip('\r')
                        print('clean ' + str(country) + 'ratio ' + str(ratio2))
                        break

            info = get_info(country)
            res = '\nstats for country\n 1.Total Cases = ' + str(info['Total Cases']) + '\n2.Recovered Cases =' + str(
                info['Recovered Cases']) + '\n3.Total Deaths= ' + str(info['Total Deaths']) + '\n' + str(res)

        if stat:
            ctx_i = ctx_i + 1
        # if cur_ctx == 'severity':
        #     res = res_msg

        with open('chat_history.txt', 'a') as fw:
            fw.write('Bot: ' + str(res) + '\nYou: ' + str(msg) + '\n')
        # uncomment to incorporate s3
        # if stop:
        #     s3_client = boto3.client('s3')
        #     timestr = time.strftime("%Y%m%d-%H%M%S")
        #     try:
        #         response = s3_client.upload_file('chat_history.txt', 'chathistory', timestr)
        #     except ClientError as e:
        #         logging.error(e)
        #         return False
        #     return True
        ChatLog.insert(END, "Bot: " + res + '\n\n')

        ChatLog.config(state=DISABLED)
        ChatLog.yview(END)


def create_global_variables():
    global context_list
    global ctx_i
    # context list to define flow of questioning
    context_list = ["start", "begin", "age", "gender", "exposure", "travel", "country", "fever", "tired", "cough",
                    "breathing",
                    "sorethroat", "pain", "nasal", "runnynose", "diarrhea", "severity", "restart"]
    ctx_i = 0

    # method to get the info


def get_info(country_name):
    #
    # if b_match == 'USA':
    #     country_name = 'United States of America'
    # creating url using country name
    url = 'https://www.worldometers.info/coronavirus/country/' + str(country_name) + '/'
    print('url' + str(url))
    # getting the request from url
    data = requests.get(url)

    # converting the text
    soup = BS(data.text, 'html.parser')

    # finding meta info for cases
    cases = soup.find_all("div", class_="maincounter-number")

    # getting total cases number
    total = cases[0].text

    # filtering it
    total = total[1: len(total) - 2]

    # getting recovered cases number
    recovered = cases[2].text

    # filtering it
    recovered = recovered[1: len(recovered) - 1]

    # getting death cases number
    deaths = cases[1].text

    # filtering it
    deaths = deaths[1: len(deaths) - 1]

    # saving details in dictionary
    ans = {'Total Cases': total, 'Recovered Cases': recovered,
           'Total Deaths': deaths}

    # returnng the dictionary
    return ans


create_global_variables()
base = Tk()
base.title("Covid Bot")
base.geometry("400x500")
base.resizable(width=FALSE, height=FALSE)

# Create Chat window
ChatLog = Text(base, bd=0, bg="white", height="8", width="50", font="Arial", )
ChatLog.config(foreground="#442265", font=("Verdana", 8))

ChatLog.config(state=NORMAL)
ChatLog.insert(END, "Bot:Hello, I am CoviBot. I am your digital assistant. I was designed to provide info about "
                    "COVID-19.I hope you are doing good, and I’m here to answer questions.\n\nShall we Start?\n\n")

# Bind scrollbar to Chat window
scrollbar = Scrollbar(base, command=ChatLog.yview, cursor="heart")
ChatLog['yscrollcommand'] = scrollbar.set

# Create Button to send message
SendButton = Button(base, font=("Verdana", 12, 'bold'), text="Send", width="12", height=5,
                    bd=0, bg="#32de97", activebackground="#3c9d9b", fg='#ffffff',
                    command=send)

# Create the box to enter message
EntryBox = Text(base, bd=0, bg="white", width="29", height="5", font="Arial")
# EntryBox.bind("<Return>", send)


# Place all components on the screen
scrollbar.place(x=376, y=6, height=386)
ChatLog.place(x=6, y=6, height=386, width=370)
EntryBox.place(x=6, y=401, height=90, width=240)
SendButton.place(x=250, y=401, height=90)

base.mainloop()
