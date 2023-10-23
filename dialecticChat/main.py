import os
import logging
import sys
from flask import Flask, request, render_template, session
from flask_session import Session
from datetime import datetime
from urllib.parse import urlparse

import openaiChat

BASEDIR = os.getcwd()
print('BASEDIR ', BASEDIR)
DUMPDIR = os.path.join(BASEDIR, 'runs/openai')
BASENAME = 'lincoln'

# logging.basicConfig(filename='/tmp/factCheck.log', level=logging.INFO)
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

app = Flask(__name__)
app.secret_key = 'hungryRats'

@app.route("/", methods=['GET'])
def initForm():
    session['convo'] = []
    cstr = "assistant: What do you want to talk about?"
    print('Zeroed convo')
    return render_template('index.html', convo=cstr)

@app.route("/", methods=["POST"])
def processText():
    print('ProcessText convo:', session['convo'])
    userInput = request.form['userbox']
    convo = session['convo']
    convo.append({'role': 'user', 'content': userInput})
    if userInput.strip().lower() == 'bye':
        outFname = os.path.join(DUMPDIR, f"{BASENAME}_{datetime.now().strftime('%y%m%d_%H%M')}.log")
        convo.append({'role': 'assistant', 'content': 'Bye'})
        cstr = '\n'.join(f"{t['role']}: {t['content']}" for t in convo)
        with open(outFname, 'w') as wx:
            wx.write(cstr)
        session['convo'] = []
        cstr = "assistant: Bye!\n\nWhat do you want to talk about next?"
    else:
        response = openaiChat.getResponse(convo, rtype=1)
        convo.append({'role': 'assistant', 'content': response})
        cstr = '\n'.join(f"{t['role']}: {t['content']}" for t in convo)
        session['convo'] = convo
    return render_template('index.html', convo=cstr)

if __name__ == '__main__':
    app.run(debug=True, host="127.0.0.1", port=8080)