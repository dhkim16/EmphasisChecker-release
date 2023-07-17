from flask import Flask, request, jsonify, make_response
import pandas as pd
from file_cache import retrieve_or_save_data
from prominent_feature_finder import _get_prominent_features
import time
import sys

from pipeline import ref_extract
from pipeline import save

from datetime import datetime 

import os
import csv
import json
import spacy

app = Flask(__name__)
FLASK_RUN_PORT = 7777

@app.route('/get-prominent-features', methods = ['OPTIONS', 'POST'])
def get_prominent_features():
    if request.method == 'OPTIONS': 
        return build_preflight_response()
    elif request.method == 'POST': 
        req = request.get_json()
        # query user with req['id']
        session_id = req['sessionId']
        data_id = req['dataId']
        data = req['data']
        metadata = req['metadata']

        start_time = time.time()
        data = retrieve_or_save_data(data_id, data, False)

        prominent_features = _get_prominent_features(data, metadata)
        end_time = time.time()

        # for demonstration, we assume the username to be Eric
        return build_actual_response(jsonify({ 'features': prominent_features }))

@app.route('/get-references', methods = ['OPTIONS', 'POST'])
def get_references():
    if request.method == 'OPTIONS': 
        return build_preflight_response()
    elif request.method == 'POST': 
        req = request.get_json()
        # query user with req['id']
        session_id = req['sessionId']
        data_id = req['dataId']
        data = req['data'] # List[Dict[str:str, str:int]]
        text = req['text']
        data = retrieve_or_save_data(data_id, data, True)

        ## Parse sentences
        nlp = spacy.load('en_core_web_sm')
        sentences = nlp(text).sents
        sents = [each.text for each in sentences]
        
        total_ref_results = []
        for each_sentence in sents:
            # Initiate the 'extractor' class
            extractor = ref_extract.ref_extract(each_sentence, data) # text type: List[str]
            
            # Put a date when the article was published.
            now = datetime.now() # when the article was published.

            # Extract the trend features and the time features
            trend_dict, trend_df = extractor.trend_extract()
            time_dict = extractor.time_extract(now)

            data = pd.DataFrame(data)

            # Find each pair of reference.
            extractor.find_sdp()
            
            # Extract the plot data for the time range extracted from the text.
            extractor.fit_chart_data(data)
            total_ref_results.append(extractor.ref_result[0])

        for i, each in enumerate(total_ref_results):
            if each['references'] != []:
                for each_sent in each['references']:
                    if each_sent['text']['time'] == None or each_sent['text']['feature'] == None:
                        extractor.ref_result[i]['references'] = []

        ## Delete when they has no matching chart data.
        for i, each in enumerate(total_ref_results):
            if each['references'] != []:
                for j, each_sent in enumerate(each['references']):
                    if total_ref_results[i]['references'][j]['chart'] == "" or total_ref_results[i]['references'][j]['chart'] == []:
                        total_ref_results[i]['references'].remove(total_ref_results[i]['references'][j])

        ## Resetting the indices of results !! and Revising index related things.
        for i, each_result in enumerate(total_ref_results):
            total_ref_results[i]['sentenceIdx'] = i
            if i == 0:
                prev_idx_start = total_ref_results[i]['charIdx'][1] + 1
                
                if total_ref_results[i]['references'] != []:
                    ## Convert it nested list format.
                    for k, ref_result in enumerate(total_ref_results[i]['references']):
                        if type(total_ref_results[i]['references'][k]['text']['time'][0]) == int :
                            total_ref_results[i]['references'][k]['text']['time'] = [total_ref_results[i]['references'][k]['text']['time']]
            else:
                ## Revise char_idx of sentence
                total_ref_results[i]['charIdx'] = list(total_ref_results[i]['charIdx'])
                total_ref_results[i]['charIdx'][0] += prev_idx_start
                total_ref_results[i]['charIdx'][1] += prev_idx_start
                
                if total_ref_results[i]['references'] != []:
                    total_ref_results[i]['references'][0]['text']['feature'] = list(total_ref_results[i]['references'][0]['text']['feature'])
                    total_ref_results[i]['references'][0]['text']['feature'][0] += prev_idx_start
                    total_ref_results[i]['references'][0]['text']['feature'][1] += prev_idx_start
                    
                    for k, ref_result in enumerate(total_ref_results[i]['references']):
                        if type(total_ref_results[i]['references'][k]['text']['time'][0]) == int :
                            total_ref_results[i]['references'][k]['text']['time'] = [total_ref_results[i]['references'][k]['text']['time']]
                    
                    for j, each in enumerate(total_ref_results[i]['references'][0]['text']['time']):
                        total_ref_results[i]['references'][0]['text']['time'][j] = list(total_ref_results[i]['references'][0]['text']['time'][j])
                        total_ref_results[i]['references'][0]['text']['time'][j][0] += prev_idx_start
                        total_ref_results[i]['references'][0]['text']['time'][j][1] += prev_idx_start
                
                ## Update prev_idx_start
                prev_idx_start = (each_result['charIdx'][1] + 1)
                
            if each_result['references'] != [] and each_result['references'][0] != []:
                for j, each_sent in enumerate(each_result['references']):
                    total_ref_results[i]['references'][j]['referenceIdx'] = f"{i}-{j}"   

        return build_actual_response(jsonify(total_ref_results))
        
def build_preflight_response():
    response = make_response()
    response.headers.add("Access-Control-Allow-Origin", "*")
    response.headers.add('Access-Control-Allow-Headers', "*")
    response.headers.add('Access-Control-Allow-Methods', "*")
    return response
    
def build_actual_response(response):
    response.headers.add("Access-Control-Allow-Origin", "*")
    return response
    

if __name__ == '__main__':
    app.run(host='0.0.0.0',debug = False, port = FLASK_RUN_PORT)