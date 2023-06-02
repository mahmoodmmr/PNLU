from pickletools import pyunicode
from django.shortcuts import render
from django.http import HttpResponse
from rest_framework.decorators import api_view
from rest_framework.response import Response
from .train_chatbot import train_intents
from .chat_chatbot import chat
from .persian import runP, GPT2Persian
from rest_framework.authentication import get_authorization_header
from rest_framework import exceptions, status
from django.conf import settings
from bson.objectid import ObjectId

import jwt
import pymongo


client = pymongo.MongoClient(   
    host = settings.MONGO_DB["HOST"],
    serverSelectionTimeoutMS = 3000,
    username=settings.MONGO_DB["USERNAME"],
    password=settings.MONGO_DB["PASSWORD"])

botiDb = client["boti"]

def getTokenPayload(request):
    auth_heaer = get_authorization_header(request)
    auth_data = auth_heaer.decode('utf-8')
    auth_token = auth_data.split(" ")

    if(len(auth_token) != 2):
        raise exceptions.AuthenticationFailed('invalid token')

    token = auth_token[1]

    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=['HS256'])
        return payload
    except jwt.ExpiredSignatureError as ex:
        raise exceptions.AuthenticationFailed('token sig error: ' + ex)
    except jwt.DecodeError as ex:
        raise exceptions.AuthenticationFailed('token has been expired: ' + ex.__str__())
    except:
        raise exceptions.AuthenticationFailed('token is invalid')

def getUser(request):
    tokenPayload = getTokenPayload(request)
    user = botiDb["users"].find_one({ '_id': ObjectId(tokenPayload['id'])})

    if user is None:    
        return Response({'success': 'false', 'message': 'user not found'}, status=status.HTTP_404_NOT_FOUND)
      
    return user

# apis
def say_hello(request):
    # runP();
    # return HttpResponse('Hello, Ai-Service is running :)')
    gpt2_model = GPT2Persian('media/model')
    # input_text = request.GET.get('text', '')
    response_text = gpt2_model.forward(' نیروهای قزلباش ایران نزدیک به چهل‌هزار تن بود؟')
    return HttpResponse(response_text)


@api_view(['POST'])
def add_intents_json(request):
    user = getUser(request) 

    ai_id = request.data['aiId']
    intents = request.data['intents']

    ai = botiDb["ais"].find_one({ '_id': ObjectId(ai_id)})

    if ai is None:
        return Response({'success': 'false', 'message': 'ai not found'}, status=status.HTTP_404_NOT_FOUND)

    projectId = ai['project']

    project = botiDb["projects"].find_one({ '_id': projectId})

    if project is None:
        return Response({'success': 'false', 'message': 'project not found'}, status=status.HTTP_404_NOT_FOUND)

    if user['_id'] != project['owner'] and ("users" not in project or user['_id'] not in project['users']):
        return Response({'success': 'false', 'message': 'no access found'}, status=status.HTTP_400_BAD_REQUEST)

    try:
        train_intents(intents, ai_id)
        return Response({'success': 'true', 'message': 'training done'})
    except:
        return Response({'success': 'false', 'message': 'training got error'}, status=status.HTTP_400_BAD_REQUEST)

@api_view(['POST'])
def chat_with_me(request):
    try:
        if "token" not in request.data:
            return Response({'success': 'false', 'message': 'token is invalid'}, status=status.HTTP_400_BAD_REQUEST)

        token = request.data["token"]

        if token != settings.GENERAL_TOKEN:
            return Response({'success': 'false', 'message': 'token is invalid'}, status=status.HTTP_400_BAD_REQUEST)

        ai_id = request.data['aiId']
        sentence = request.data["message"]

        answer = chat(sentence, ai_id)

        return Response({'success': 'true', 'message': answer})
    except Exception as ex:
        print(ex)
        return Response({'success': 'false', 'message': 'invalid data'}, status=status.HTTP_400_BAD_REQUEST)
