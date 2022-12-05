#!/usr/bin/env python
# coding: utf-8

# Esse código visa usar modelo de emoções musicais já treinado para falar mood de uma musica, associar isso com os tópicos mais relevantes da letra da música e criar imagens a partir disso.
# 
# Alguns sites de referência:
# 
# https://towardsdatascience.com/extracting-song-data-from-the-spotify-api-using-python-b1e79388d50
# 
# https://towardsdatascience.com/predicting-the-music-mood-of-a-song-with-deep-learning-c3ac2b45229e
# 
# https://sites.tufts.edu/eeseniordesignhandbook/2015/music-mood-classification/
# 
# https://scholar.smu.edu/cgi/viewcontent.cgi?article=1197&context=datasciencereview


import pandas as pd
import numpy as np
import librosa as lr
import matplotlib.pyplot as plt
import os
import pickle

import lyricsgenius

# Features do spotify
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

from craiyon import Craiyon
from PIL import Image # pip install pillow
from io import BytesIO
import base64

from statistics import mode

# Processamento de texto
from nltk.corpus import stopwords
from nltk.cluster.util import cosine_distance
import numpy
import networkx as nx
import contractions

# Funções para puxar tópicos principais do texto da letra
 
def expand_contractions(text):
    expanded_words = []
    for word in text.split():
        expanded_words.append(contractions.fix(word)) 
    return ' '.join(expanded_words)

def clean_sentence(sentence):
    sentence = sentence.replace("[^a-zA-Z]+", " ")
    sentence = expand_contractions(sentence)
    return sentence.lower().split(" ")

def sentence_similarity(sent1, sent2, stopwords=None):
    if stopwords is None:
        stopwords = []
 
    sent1 = [w.lower() for w in sent1]
    sent2 = [w.lower() for w in sent2]
 
    all_words = list(set(sent1 + sent2))
 
    vector1 = [0] * len(all_words)
    vector2 = [0] * len(all_words)
 
    # build the vector for the first sentence
    for w in sent1:
        if w in stopwords:
            continue
        vector1[all_words.index(w)] += 1
 
    # build the vector for the second sentence
    for w in sent2:
        if w in stopwords:
            continue
        vector2[all_words.index(w)] += 1
 
    return 1 - cosine_distance(vector1, vector2)
 
def build_similarity_matrix(sentences, stop_words):
    # Create an empty similarity matrix
    similarity_matrix = np.zeros((len(sentences), len(sentences)))
 
    for idx1 in range(len(sentences)):
        for idx2 in range(len(sentences)):
            if idx1 == idx2: #ignore if both are same sentences
                continue 
            similarity_matrix[idx1][idx2] = sentence_similarity(sentences[idx1], sentences[idx2], stop_words)

    return similarity_matrix

def generate_summary(file_name, top_n=5):
    stop_words = stopwords.words('english')
    summarize_text = []

    # Step 1 - Read text anc split it
    sentences =  read_article(file_name)

    # Step 2 - Generate Similary Martix across sentences
    sentence_similarity_martix = build_similarity_matrix(sentences, stop_words)

    # Step 3 - Rank sentences in similarity martix
    sentence_similarity_graph = nx.from_numpy_array(sentence_similarity_martix)
    scores = nx.pagerank(sentence_similarity_graph)

    # Step 4 - Sort the rank and pick top sentences
    ranked_sentence = sorted(((scores[i],s) for i,s in enumerate(sentences)), reverse=True)    
    #print("Indexes of top ranked_sentence order are ", ranked_sentence)    

    for i in range(top_n):
        summarize_text.append(" ".join(ranked_sentence[i][1]))

    # Step 5 - Offcourse, output the summarize texr
    #print("Summarize Text: \n", ". ".join(summarize_text))
    print("Tópico principal da música é: ",summarize_text[0].split(",")[0])
    #if summarize_text[0].split(",")[0] == summarize_text[0].split(",")[1].lstrip(' '):
    #    return summarize_text[0].split(",")[0]
    #else:
    #    return summarize_text[0].split(",")
    
    return summarize_text[0].split(",")[0]
    
def read_article(file_name):
    file = open(file_name, "r", encoding="utf-8")
    article = file.readlines()
    sentences = []
    
    for sentence in article:
        sentences.append(clean_sentence(sentence))
    sentences.pop() 
    
    return sentences

# Funções do código

def letra(title, artist):
    try:
        api_key='ORBJc-e5kH3LA_H5KohP8Grgp_YCbaBZgqz536TachvT1_iX7mPPwU3WrADiJ8Nw'
        genius= lyricsgenius.Genius(api_key)
        #title="Clube Da Esquina Nº 2"
        #artist="Milton Nascimento"
        song = genius.search_song(title=title, artist=artist)
        #print(song.lyrics)
        full_song = song.lyrics
        os.chdir(r'C:\Users\sara-\OneDrive\Área de Trabalho\UNICAMP\Computação afetiva\Projeto final\Letras')
        with open('{}_{}.txt'.format(title,artist), 'w', encoding="utf-8") as f:
            f.write(full_song)
        print("Música salva em: {}_{}.txt".format(title,artist))
        return "Certo"
        
    except AttributeError: #ValueError:
        print("Oops! A música não tem letra. Sua imagem será gerada apenas com base no nome e na emoção da música")
        return "Erro"
        

def features_musicais():
    artist = input("Digite o nome da banda/artista: ")
    title = input("Digite o nome da música (ou palavra chave): ")
    
    df_nova_mus = pd.DataFrame(columns=['name','artist','uri','danceability', 'acousticness', 'energy', 'instrumentalness','liveness', 'valence', 'loudness', 'speechiness', 'tempo'])
    
    # Make your own Spotify app at https://beta.developer.spotify.com/dashboard/applications
    client_id = '0cbd7765b9634995877c5d78e483bf07'
    client_secret = 'f5e3b60f8471408eb6f195734ff8a775'

    client_credentials_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
    sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)
    sp.trace=False
    search_querry = artist + ' ' + title
    result = sp.search(search_querry)

    # Pega o primeiro resultado da busca
    prim_resp = result['tracks']['items'][0]
    uri = prim_resp['uri']
    features = sp.audio_features(uri)
    #print(features[0])

    print("Nome do artista: ",prim_resp['artists'][0]['name'])
    print("Nome da música selecionada: ",prim_resp['name'])
    #print(uri)

    # Informações da música
    df_nova_mus.loc[0,'name'] = prim_resp['name']
    df_nova_mus.loc[0,'artist'] = prim_resp['artists'][0]['name']
    df_nova_mus.loc[0,'uri'] = uri

    # Puxar as features: 'danceability', 'acousticness', 'energy', 'instrumentalness','liveness', 'valence', 'loudness', 
    # 'speechiness', 'tempo'
    df_nova_mus.loc[0,'danceability'] = features[0]['danceability']
    df_nova_mus.loc[0,'acousticness'] = features[0]['acousticness']
    df_nova_mus.loc[0,'energy'] = features[0]['energy']
    df_nova_mus.loc[0,'instrumentalness'] = features[0]['instrumentalness']
    df_nova_mus.loc[0,'liveness'] = features[0]['liveness']
    df_nova_mus.loc[0,'valence'] = features[0]['valence']
    df_nova_mus.loc[0,'loudness'] = features[0]['loudness']
    df_nova_mus.loc[0,'speechiness'] = features[0]['speechiness']
    df_nova_mus.loc[0,'tempo'] = features[0]['tempo']
    
    title_novo = prim_resp['artists'][0]['name']
    artist_novo = prim_resp['name']
    
    return df_nova_mus, title, artist

def carregar_modelo(nome_modelo):
    os.chdir(r'C:\Users\sara-\OneDrive\Área de Trabalho\UNICAMP\Computação afetiva\Projeto final')
    with open("{}.pkl".format(nome_modelo), "rb") as f:
        model = pickle.load(f)
    
    return model

def classificar_musica(modelo, df_nova_mus):
    new = df_nova_mus[['danceability', 'acousticness', 'energy', 'instrumentalness',
                       'liveness', 'valence', 'loudness', 'speechiness', 'tempo']]
    m = modelo.predict(new)[0]
    dic = {0:"Calm",1:"Energetic",2:"Happy",3:"Sad"}
    mood = dic[m]
    print("Sua música foi classificada como: ", mood,"\n")
    proba = modelo.predict_proba(new)
    #print(proba)
    print("Probabilidade: ",proba[0][m])
    
    return mood

def gerar_imagens(title,artist,frase):
    nome_pasta = title + "_" + artist
    os.chdir(r'C:\Users\sara-\OneDrive\Área de Trabalho\UNICAMP\Computação afetiva\Projeto final')
    cwd = os.getcwd()
    target_dir = cwd + '/imagens' + '/{}'.format(nome_pasta)
    if os.path.exists(target_dir):
        print("Pasta já existe")
    else:
        os.mkdir(target_dir)

    os.chdir(target_dir)
    
    generator = Craiyon() # Instantiates the api wrapper
    result = generator.generate(frase)
    result.save_images(target_dir) 
    # Saves the generated images to 'current working directory/generated', you can also provide a custom pa
    
    print("Imagens salvas em: {}".format(target_dir))

# # Main

if __name__ == "__main__":
    # Escolher música de preferência
    df_feat, title, artist = features_musicais()
    print("\n")
    
    # Carregar modelo escolhido
    mod = carregar_modelo("modelo")
    
    # Classificar música escolhida
    mood = classificar_musica(mod,df_feat)
    
    # Pegar tópico da letra da música
    retorno = letra(title, artist)
    os.chdir(r'C:\Users\sara-\OneDrive\Área de Trabalho\UNICAMP\Computação afetiva\Projeto final\Letras')
    if retorno == "Certo":
        topico = generate_summary("{}_{}.txt".format(title,artist), 1)
    else:
        topico = ""
    
    # Criar frase para geração de imagem
    frase = title + ' ' + mood + ' ' + topico
    print("A frase que irá gerar a imagem é: ", frase)
    
    # Gerar imagens na pasta específica
    gerar_imagens(title,artist,frase)
    
    # Salva frase que gerou imagem
    with open('frase_final.txt', 'w', encoding="utf-8") as f:
        f.write(frase)

