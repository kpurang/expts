"""
This is to get chat responses from openai
A simple instruction works well for this purpose
"""
import os
import logging
from typing import TextIO
import openai

oMODEL = "gpt-3.5-turbo"

BASEDIR = os.getcwd()
logging.basicConfig(filename=os.path.join(BASEDIR, 'test.log'),
                    level=logging.DEBUG)

INSTR_0 = """You are a philosopher who uses dialectic techniques to get people to think more deeply about issues.

Given the conversation below, generate a reponse or counterargument which can be statements or open-ended questions.
"""

def queryOpenai(query: list[dict] = [],
               instr: str = 'You are a helpful assistant.',
               temperature: float= 0.0,
               fobj: TextIO = None) -> (str, list[str]):
    qObj = [{"role": "system", "content": instr}]
    qObj.extend(query)
    response = openai.ChatCompletion.create(
                   model = oMODEL,
                   messages = qObj,
                   temperature = temperature)
    answers = [x['message']['content'] for x in response['choices']]
    if fobj is not None:
        msgStr = '\n'.join([f"{x['role']}: {x['content']}" for x in qObj])
        fobj.write(f"\n{'-'*10}\nSYSTEM: {instr}\nmsgStr\nTEMP: {temperature}")
        for c in response['choices']:
            fobj.write(f"\n--\n{c['message']['content']}")
        fobj.flush()
    return response, answers

# called from the ui controller
def getResponse(convo: list[dict] = [], rtype: int=0) -> dict:
    if rtype == 0:
        response, answers = queryOpenai(convo, INSTR_0, temperature=0.4)
    return answers[0]
