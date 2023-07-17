from multiprocessing.sharedctypes import Value
import os
import re
from time import time
import torch
import stanza
import spacy
import networkx as nx
import pickle
import pandas as pd
import numpy as np
from stanza.server import CoreNLPClient

from datetime import datetime
from matplotlib import pyplot as plt
from scipy.spatial.distance import cosine
from dateutil.relativedelta import relativedelta
from typing import Iterable, Optional, List, Tuple, Dict, Union

from transformers import BertTokenizer, BertModel

from pipeline.save import count_in_match, check_year, manual_date_detection

# stanza.install_corenlp()
# stanza.download_corenlp_models(model='english-kbp', version='4.5.0') # model='english-kbp'

client = CoreNLPClient(
                annotators=['tokenize','pos','lemma','ner','mwt','entitymentions'],
                timeout=30000,
                memory='6G',
                quite=True)
                
nlp = spacy.load('en_core_web_sm')
stanza_nlp = stanza.Pipeline(lang='en', processors='tokenize, mwt, pos, lemma, depparse')

model = BertModel.from_pretrained("bert-base-uncased", output_hidden_states = True)
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

class ref_extract():
    def __init__(self, input_texts:str, chart_data=None):
        # stanza.install_corenlp()
        # stanza.download_corenlp_models(model='english-kbp', version='4.5.0') # model='english-kb
        self.chart_data =  pd.DataFrame(chart_data)
        self.raw_text = input_texts
        self.texts = input_texts.lower().replace(";", ".")#.replace(",", "")#.replace(".", "")
        print(self.texts)
        self.figure_names = "Figure Names"
        
        # Time extraction
        self.anns = client.annotate(self.texts)

        sentences = nlp(input_texts).sents
        self.sets = [each.text for each in sentences]

        # Trend extraction
        self.model = BertModel.from_pretrained("bert-base-uncased", output_hidden_states = True)
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        
        self.emb_dict = []
        self.lemma_tg_li = {}

        # Reference
        self.ref = [] # for only single paragraph
        self.ref_result = []


    def get_date_range(self, date: Dict) -> Tuple[str, str]:
        min_date = {"year": None, "month": None, "day": None}
        max_date = {"year": None, "month": None, "day": None}
        
        if 'year' in date:
            if re.match("\d+[/]\d+", date['year']):
                range_year = re.match("\d+[/]\d+", date['year']).group()
                min_date["year"], max_date["year"] = range_year.split('/')[0], range_year.split('/')[1]
            else: 
                min_date["year"], max_date["year"] = date['year'], date['year']
        
        if 'month' in date:
            if re.match("\d+[/]\d+", date['month']):
                range_month =re.match("\d+[/]\d+", date['month']).group() #"\d+[/]?\d+?"
                # print("range_month", range_month)
                min_date["month"], max_date["month"] = range_month.split('/')[0], range_month.split('/')[1]
            elif re.match("\d+", date['month']):
                range_month =re.match("\d+", date['month']).group()
                min_date["month"], max_date["month"] = range_month, range_month
        else:
            min_date["month"], max_date["month"] = 1, 12

        if 'day' in date:
            min_date["day"], max_date["day"] = date['day'], date['day']
        else:
            min_date["day"] = 1
            if max_date['month'] == 2: max_date["day"] = 28
            else: max_date["day"] = (30 if max_date['month'] in [4, 6, 9, 11] else 31)
            
        return min_date, max_date

    def cal_year(self, y: str, timex_str: str, c_year: str) -> int:
        this_year_match = re.compile("THIS P1[YMDW]")
        prev_year_match = re.compile("OFFSET [P][-]\d+[YMDW]") # DATE에서 사용되는 PXY는 보통 몇년전 언제를 의미, 즉 음수만 나옴.
        
        if y == "XXXX" or this_year_match.search(timex_str):
            year = int(c_year)
        elif y[2:] == "XX": year = "/".join([int(y[:2])*100, int(y[:2])*100+99])
        elif y[3] == "X": year = "/".join([str(int(y[:3])*10), str(int(y[:3])*10+9)])
        elif re.match('\d{4}', y): year = int(y)
        else: year = int(c_year)

        # offset
        if prev_year_match.search(timex_str):
            year -= 1

        return year

    def time_str_to_date(self, timex_str: str, current_date):
        ''' Convert normalized time expression into absolute time expression.
        '''

        ## Save current year, month, day
        p = re.compile("(\d{4})[-]\d{2}[-]\d{2}")
        date = p.match(str(current_date)).group()
        
        if self.chart_data.empty:
            c_year, c_month, c_day = date.split('-')[0], date.split('-')[1], date.split('-')[2]
        else:
            date_df = self.chart_data["date"].str.split('-', expand=True).rename(columns={0:'year', 1:'month', 2:'day'})
            date_df = date_df.astype("int")
            c_year, c_month, c_day = date_df.iloc[-1, 0], date_df.iloc[-1, 1], date_df.iloc[-1, 2]
        
        ### DURATION
        offset = 5 # the offset for 'few' in 'for few months(=> PXM)'
        X_match= re.compile("[P][X][YMDW]") # PXM
        num_match = re.compile("[P]\d+[YMDW]") # P30Y, 무조건 양수 왜냐하면 duration이니까

        if num_match.match(timex_str):
            if timex_str[-1] == 'Y':
                duration = re.search("\d+", timex_str).group(0)
                year = int(c_year)-int(duration)
                return {"mode": "year", "year": "/".join([str(year), c_year])}
            elif timex_str[-1] == 'M':
                duration = re.search("\d+", timex_str).group(0)
                current_date = datetime.strptime(f'{c_year}-{c_month}-{c_day}', '%Y-%m-%d')
                before_datetime = current_date - relativedelta(months=int(duration))
                before_date = p.match(str(before_datetime)).group()
                b_year, b_month, b_day = before_date.split('-')[0], before_date.split('-')[1], before_date.split('-')[2]

                c_year, c_month = str(c_year), str(c_month)
                return {"mode": "month", "year": str(c_year), "month": "/".join([b_month.strip("0"), c_month.strip("0")])} if c_year==b_year \
                    else {"mode": "month", "year": "/".join([b_year, c_year]), "month": "/".join([b_month.strip("0"), c_month.strip("0")])}
        elif X_match.match(timex_str):
            if timex_str[-1] == 'Y':
                year = int(c_year) - offset
                return {"mode": "year", "year": str(year)+'/'+c_year}
            elif timex_str[-1] == 'M':
                if c_month != 1 : month = int(c_month)-offset
                else: month = 13 - offset
                return {"mode": "month", "year": str(c_year), "month": "/".join([str(month), c_month.strip("0")])}

        ### DATE: 1960/1980, 2000, 2020-SU, 199X, XXXX-SU, THIS P1Y OFFSET P-1Y INTERSECT XXXX-11, ******05/******06
        timex_str = timex_str.replace('*', 'X')
        
        between_match = re.compile("\w+[/]\w+")
        between_day_match = re.compile("\w{4}[-]\w{2}[-]\w{2}[/]\w{4}[-]\w{2}[-]\w{2}")
        between_month_match = re.compile("[X]*\d{2}[/][X]*\d{2}")
        between_month_match_2 = re.compile("\w+[-]\d+[/]\w+[-]\d+") #
        between_month_match_3 = re.compile("\w+[-]\w+[/]\w+[-]\w+")
        day_match = re.compile("\w{4}[-]\w{2}[-]\d{2}") # XXXX-XX-XX
        month_match = re.compile("\w{4}[-]\w{2}") # XXXX-XX
        year_match = re.compile("(\d+[X]*)") # XXXX
        this_year_match = re.compile("THIS P1[YMDW]")
        prev_year_match = re.compile("OFFSET [P][-]\d+[YMDW]") # DATE에서 사용되는 PXY는 보통 몇년전 언제를 의미, 즉 음수만 나옴.
        prev_imm_year_match1, prev_imm_year_match2 = re.compile("PREV P\d+[YMDW]"), re.compile("PREV_IMMEDIATE P\d+[YMDW]")
        
        season_dict = {"SP":"3/5", 'WI':"12/2", 'SU':"6/8", 'FA':"9/11"}
        if between_day_match.match(timex_str):
            s_d, e_d = timex_str.split("/")[0], timex_str.split("/")[1]
            s_year, s_month, s_day = s_d.split("-")[0], s_d.split("-")[1], s_d.split("-")[2]
            e_year, e_month, e_day = e_d.split("-")[0], e_d.split("-")[1], e_d.split("-")[2]

            if s_year == "XXXX" and e_year == "XXXX": s_year, e_year = c_year, c_year
            elif s_year == "XXXX" and e_year != "XXXX": s_year = e_year
            elif s_year != "XXXX" and e_year == "XXXX": e_year = s_year

            if s_month == "XX" and e_month == "XX": s_month, e_month = c_month, c_month
            elif s_month == "XX" and e_month != "XX": s_month = e_month
            elif s_month != "XX" and e_month == "XX": e_month = s_month

            if s_day == "XX" and e_day == "XX":
                return {"mode": "month", "year": str(s_year) + "/" + str(e_year), \
                    "month": str(s_month) + "/" + str(e_month)}
            elif s_day == "XX" and e_day != "XX": s_day = e_day
            elif s_day != "XX" and e_day == "XX": e_day = s_day 
            
            return {"mode": "day", "year": s_year + "/" + e_year, \
                    "month": s_month + "/" + e_month,
                    "day": s_day + "/" + e_day}
        elif between_match.match(timex_str): # 2018/2020, 2018-05/2020-12
            if between_month_match.match(timex_str): # ******05/******06
                timex_str = timex_str.replace("X", "")
                return {"mode": "month", "year": c_year, "month": between_match.match(timex_str).group()}
            elif between_match.match(timex_str): # 2018/2020
                return {"mode": "year", "year": between_match.match(timex_str).group()}
        elif between_month_match_2.match(timex_str):#2018-05/2020-12
            return {"mode": "month", "year": timex_str.split("/")[0].split("-")[0] + "/" + timex_str.split("/")[1].split("-")[0], \
                    "month": timex_str.split("/")[0].split("-")[1] + "/" + timex_str.split("/")[1].split("-")[1]}
        elif between_month_match_3.match(timex_str):
            return {"mode": "month", "year": timex_str.split("/")[0].split("-")[0] + "/" + timex_str.split("/")[1].split("-")[0], \
                    "month": season_dict[timex_str.split("/")[0].split("-")[1]] + "/" + season_dict[timex_str.split("/")[1].split("-")[1]]}
        elif day_match.search(timex_str):
            d = day_match.search(timex_str).group()
            year, month, day = d.split('-')[0], d.split('-')[1], d.split('-')[2]

            if year == "XXXX": year = c_year
            if month == "XX": month = c_month
            
            return {"mode": "day", "year": str(year), "month": month, "day": day}
        elif month_match.search(timex_str): # only the case XXXX-XX
            date_pair = month_match.search(timex_str).group(0)
            y, m = date_pair.split('-')[0], date_pair.split('-')[1]

            # Month calculation
            month = m
            if m == 'SP': month = "3/5"
            elif m == 'WI': month = "12/2"
            elif m == 'SU': month = "6/8"
            elif m == 'FA': month = "9/11"

            elif m == 'Q1': month = "10/12"
            elif m == 'Q2': month = "1/3"
            elif m == 'Q3': month = "4/6"
            elif m == 'Q4': month = "7/9"

            elif m[0] == 'W': return {"mode": "Not found", "year": str(y), "month": str(m)}
            else: month = m

            # Year calculation
            year = self.cal_year(y, timex_str, c_year)

            return {"mode": "month", "year": str(year), "month": month}
        elif year_match.match(timex_str): # only the case XXXX
            y =  year_match.match(timex_str).group(0)
            year = self.cal_year(y, timex_str, c_year)
            return {"mode": "year", "year": str(year)}
        elif this_year_match.search(timex_str):
            this_P = this_year_match.search(timex_str).group()
            if this_P[-1] == "Y": return {"mode": "year", "year": str(c_year)}
            elif this_P[-1] == "M": return {"mode": "month", "year": str(c_year), "month": str(c_month)}
            elif this_P[-1] == "D": return {"mode": "day", "year": str(c_year), "month": str(c_month), "day": str(c_day)}
        elif prev_imm_year_match1.search(timex_str) or prev_imm_year_match2.search(timex_str):
            prev_P = prev_imm_year_match1.search(timex_str).group() if prev_imm_year_match1.search(timex_str) else prev_imm_year_match2.search(timex_str).group()
            p = re.search("[P]\d+[YMDW]", prev_P).group()
            timeline, n = p[-1], p[1]
            if timeline == "Y": return {"mode": "year", "year": str(int(c_year)-int(n))}
            elif timeline == "M": return {"mode": "month", "year": str(c_year), "month": str(int(c_month)-int(n))}
            elif timeline == "D": return {"mode": "day", "year": str(c_year), "month": str(int(c_month)), "day": str(c_day-int(n))}
        
        else: return {"mode": "Not found", "year": str(c_year), "month": str(c_month), "day": str(c_day)}

    def bert_text_preparation(self, text: str, tokenizer) -> Tuple:
        """Preparing the input for BERT
        
        Takes a string argument and performs
        pre-processing like adding special tokens,
        tokenization, tokens to ids, and tokens to
        segment ids. All tokens are mapped to seg-
        ment id = 1.
        """
        marked_text = "[CLS] " + text + " [SEP]"
        tokenized_text = tokenizer.tokenize(marked_text)
        indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
        # print("indexed_tokens:", indexed_tokens, len(tokenized_text), len(indexed_tokens))
        segments_ids = [1]*len(indexed_tokens)

        # Convert inputs to PyTorch tensors
        tokens_tensor = torch.tensor([indexed_tokens])
        segments_tensors = torch.tensor([segments_ids])

        return tokenized_text, tokens_tensor, segments_tensors

    def get_bert_embeddings(self, tokens_tensor, segments_tensors, model):
        """Get embeddings from an embedding model
        
        Args:
            tokens_tensor (obj): Torch tensor size [n_tokens]
                with token ids for each token in text
            segments_tensors (obj): Torch tensor size [n_tokens]
                with segment ids for each token in text
            model (obj): Embedding model to generate embeddings
                from token and segment ids
        
        Returns:
            list: List of list of floats of size
                [n_tokens, n_embedding_dimensions]
                containing embeddings for each token
        
        """
        
        # Gradient calculation id disabled
        # Model is in inference mode
        with torch.no_grad():
            outputs = model(tokens_tensor, segments_tensors)
            # Removing the first hidden state
            # The first state is the input state
            hidden_states = outputs[2][1:]

        # Getting embeddings from the final BERT layer
        token_embeddings = hidden_states[-1]
        # Collapsing the tensor into 1-dimension
        token_embeddings = torch.squeeze(token_embeddings, dim=0)
        # Converting torchtensors to lists
        list_token_embeddings = [token_embed.tolist() for token_embed in token_embeddings]

        return list_token_embeddings

    def weird_time_extract(self, sent_idx, single_sent, curr_time): # 2018-05/2020-12

        count_dict = {}
        # Time expressions
        if re.search("between\s\w*[-]*\s?\w*[-]*\s?\w*[-]*\s?\w*[-]*\sand\s\w*[-]*\s?\w*[-]*\s?\w*[-]*\s?\w*[-]*", single_sent):# between 2020 and 2021, between 2007 and 2010
            for each in re.finditer("between\s\w+\sand\s\w+", single_sent):
                if  each.group() in count_dict: count_dict[each.group()] +=1 
                else: count_dict[each.group()] = 0

                prev_y, curr_y = each.group().split(' ')[1], each.group().split(' ')[3]
                new_normalized_NER = prev_y + '/' + curr_y
                p1 = self.time_str_to_date(new_normalized_NER, curr_time)
                range = self.get_date_range(p1)

                # Save into a reference list.
                self.ref[sent_idx]['references']["index"]["time"].append(find_ch_span(each.group().lower(), self.texts, self.ref[sent_idx]['charIdx'], count_dict))
                self.ref[sent_idx]['references']["text"]["time"].append(each.group())
                self.ref[sent_idx]['references']["range"].append(range)
                self.ref[sent_idx]['references']["mention"].append([True, True])

                single_sent = single_sent.replace(each.group(), "TIME")
        ## The time normalizer doesn't work well on 'late 1920s'.
        if re.search("late \d{4}s", single_sent):
            for each in re.finditer("late \d{4}s", single_sent):
                if  each.group() in count_dict: count_dict[each.group()] +=1 
                else: count_dict[each.group()] = 0

                new_normalized_NER = each.group()[-5:-2] + "X"
                p1 = self.time_str_to_date(new_normalized_NER, curr_time)
                range = self.get_date_range(p1)

                self.ref[sent_idx]['references']["index"]["time"].append(find_ch_span(each.group(), self.texts, self.ref[sent_idx]['charIdx'], count_dict))
                self.ref[sent_idx]['references']["text"]["time"].append(each.group())
                self.ref[sent_idx]['references']["range"].append(range)
                self.ref[sent_idx]['references']["mention"].append([False, False])

                single_sent = single_sent.replace(each.group(), "TIME")
        if re.finditer("from\s\d{4}\sto\s\d{4}", single_sent.lower()):
            for each in re.finditer("from\s\d{4}\sto\s\d{4}", single_sent.lower()):
                if  each.group() in count_dict: count_dict[each.group()] +=1 
                else: count_dict[each.group()] = 0
                
                from_text = re.findall("from\s\d{4}\sto", each.group())
                from_norm = manual_date_detection(from_text[0][5:-3])

                to_text = re.findall("to\s\d{4}", each.group())
                to_norm = manual_date_detection(to_text[0][2:])
                
                new_normalized_NER = from_norm + '/' + to_norm

                p1 = self.time_str_to_date(new_normalized_NER, curr_time)
                range = self.get_date_range(p1)

                # Save into a reference list.
                self.ref[sent_idx]['references']["index"]["time"].append(find_ch_span(each.group(), self.texts, self.ref[sent_idx]['charIdx'], count_dict))
                self.ref[sent_idx]['references']["text"]["time"].append(each.group())
                self.ref[sent_idx]['references']["range"].append(range)
                self.ref[sent_idx]['references']["mention"].append([True, True])

                single_sent = single_sent.replace(each.group(), "TIME")
        
        if re.finditer("from\s\w*[-]*\s?\w*[-]*\s?\d{,4}[-]*\sto\s\w*[-]*\s?\w*[-]*\s?\d{,4}", single_sent.lower()):
            for each in re.finditer("from\s\w*[-]*\s?\w*[-]*\s?\d{,4}[-]*\sto\s\w*[-]*\s?\w*[-]*\s?\d{,4}", single_sent.lower()):
                if  each.group() in count_dict: count_dict[each.group()] +=1 
                else: count_dict[each.group()] = 0
                
                from_text = re.findall("from\s\w*[-]*\s?\w*[-]*\s?\w*[-]*\s?\w*[-]*\sto", each.group())
                from_norm = manual_date_detection(from_text[0][5:-3])

                to_text = re.findall("to\s\w*[-]*\s?\w*[-]*\s?\w*[-]*\s?\w*[-]*", each.group())
                to_norm = manual_date_detection(to_text[0][2:])

                new_normalized_NER = from_norm + '/' + to_norm

                p1 = self.time_str_to_date(new_normalized_NER, curr_time)
                range = self.get_date_range(p1)

                # Save into a reference list.
                self.ref[sent_idx]['references']["index"]["time"].append(find_ch_span(each.group(), self.texts, self.ref[sent_idx]['charIdx'], count_dict))
                self.ref[sent_idx]['references']["text"]["time"].append(each.group())
                self.ref[sent_idx]['references']["range"].append(range)
                self.ref[sent_idx]['references']["mention"].append([True, True])

                single_sent = single_sent.replace(each.group(), "TIME")
        return single_sent

    def time_phrase_extract(self, temp_text, sent_idx, norm_ner_dict, time_ch_dict, curr_time):
            count_dict = {}
            for each in re.finditer("between\stime\d{1,2}\sand\stime\d{1,2}", temp_text):
                prev, curr = each.group().split(' ')[1], each.group().split(' ')[3]
                p1 = self.time_str_to_date(norm_ner_dict[prev], curr_time)
                p2 = self.time_str_to_date(norm_ner_dict[curr], curr_time)
                p1_range = self.get_date_range(p1)
                p2_range = self.get_date_range(p2)

                final_range = (p1_range[0], p2_range[1])
                between_text = each.group()
                for e in re.finditer("time\d{1,2}", between_text):
                    temp_text = temp_text.replace(e.group(), norm_ner_dict[e.group()])
                    between_text = between_text.replace(e.group(), time_ch_dict[norm_ner_dict[e.group()]])
                
                if  each.group() in count_dict: count_dict[between_text] +=1 
                else: count_dict[between_text] = 0

                self.ref[sent_idx]['references']["index"]["time"].append(find_ch_span(between_text, self.texts, self.ref[sent_idx]['charIdx'], count_dict))
                self.ref[sent_idx]['references']["text"]["time"].append(between_text)
                self.ref[sent_idx]['references']["range"].append(final_range)
                self.ref[sent_idx]['references']["mention"].append([True, True])
            for each in re.finditer("since\stime\d{1,2}", temp_text):
                prev = each.group().split(' ')[-1]
                p1 = self.time_str_to_date(norm_ner_dict[prev], curr_time)
                p1_range = self.get_date_range(p1)
                final_range = (p1_range[0], "TBD")

                from_text = each.group() # e.g. from_text = "since time2"
                for e in re.finditer('time\d{1,2}', from_text):
                    temp_text = temp_text.replace(e.group(), norm_ner_dict[e.group()])
                    from_text = from_text.replace(e.group(), time_ch_dict[norm_ner_dict[e.group()]])

                self.ref[sent_idx]['references']["index"]["time"].append(find_ch_span(from_text.strip().lower(), self.texts, self.ref[sent_idx]['charIdx']))
                self.ref[sent_idx]['references']["text"]["time"].append(from_text)
                self.ref[sent_idx]['references']["range"].append(final_range)
                self.ref[sent_idx]['references']["mention"].append([True, False])
            
            for each in re.finditer("after time\d{1,2}", temp_text):
                prev = each.group().split(' ')[-1]
                p1 = self.time_str_to_date(norm_ner_dict[prev], curr_time)
                p1_range = self.get_date_range(p1)

                final_range = (p1_range[0], "TBD")

                from_text = each.group()
                
                for e in re.finditer("time\d{1,2}", from_text):
                    temp_text = temp_text.replace(e.group(), norm_ner_dict[e.group()])
                    from_text = from_text.replace(e.group(), time_ch_dict[norm_ner_dict[e.group()]])

                self.ref[sent_idx]['references']["index"]["time"].append(find_ch_span(from_text.strip().lower(), self.texts, self.ref[sent_idx]['charIdx']))
                self.ref[sent_idx]['references']["text"]["time"].append(from_text)
                self.ref[sent_idx]['references']["range"].append(final_range)
                self.ref[sent_idx]['references']["mention"].append([True, False])

            ## CASE: mention [False, True]
            for each in re.finditer("until\stime\d{1,2}", temp_text):
                prev = each.group().split(' ')[-1]
                p1 = self.time_str_to_date(norm_ner_dict[prev], curr_time)
                p1_range = self.get_date_range(p1)

                final_range = ("TBD", p1_range[1])

                from_text = each.group()
                for e in re.finditer("time\d{1,2}", from_text):
                    temp_text = temp_text.replace(e.group(), norm_ner_dict[e.group()])
                    from_text = from_text.replace(e.group(), time_ch_dict[norm_ner_dict[e.group()]])
                
                self.ref[sent_idx]['references']["index"]["time"].append(find_ch_span(from_text.strip().lower(), self.texts, self.ref[sent_idx]['charIdx']))
                self.ref[sent_idx]['references']["text"]["time"].append(from_text)
                self.ref[sent_idx]['references']["range"].append(final_range)
                self.ref[sent_idx]['references']["mention"].append([False, True])
            for each in re.finditer("to\stime\d{1,2}", temp_text):
                prev = each.group().split(' ')[-1]
                p1 = self.time_str_to_date(norm_ner_dict[prev], curr_time)
                p1_range = self.get_date_range(p1)

                final_range = ("TBD", p1_range[1])

                from_text = each.group()
                for e in re.finditer("time\d{1,2}", from_text):
                    temp_text = temp_text.replace(e.group(), norm_ner_dict[e.group()])
                    from_text = from_text.replace(e.group(), time_ch_dict[norm_ner_dict[e.group()]])

                self.ref[sent_idx]['references']["index"]["time"].append(find_ch_span(from_text.strip().lower(), self.texts, self.ref[sent_idx]['charIdx']))
                self.ref[sent_idx]['references']["text"]["time"].append(from_text)
                self.ref[sent_idx]['references']["range"].append(final_range)
                self.ref[sent_idx]['references']["mention"].append([False, True])
            
            for each in re.finditer("till\s\w{,10}\s{,1}time\d{1,2}", temp_text):
                prev = each.group().split(' ')[-1]
                p1 = self.time_str_to_date(norm_ner_dict[prev], curr_time)

                p1_range = self.get_date_range(p1)
                final_range = ("TBD", p1_range[1])

                from_text = each.group()
                for e in re.finditer("time\d{1,2}", from_text):
                    temp_text = temp_text.replace(e.group(), norm_ner_dict[e.group()])
                    from_text = from_text.replace(e.group(), time_ch_dict[norm_ner_dict[e.group()]])

                self.ref[sent_idx]['references']["index"]["time"].append(find_ch_span(from_text.strip().lower(), self.texts, self.ref[sent_idx]['charIdx']))
                self.ref[sent_idx]['references']["text"]["time"].append(from_text)
                self.ref[sent_idx]['references']["range"].append(final_range)
                self.ref[sent_idx]['references']["mention"].append([False, True])
            for each in re.finditer("preceding\s\w{,10}\s{,1}time\d{1,2}", temp_text):
                prev = each.group().split(' ')[-1]
                p1 = self.time_str_to_date(norm_ner_dict[prev], curr_time)

                p1_range = self.get_date_range(p1)
                final_range = ("TBD", p1_range[1])

                from_text = each.group()
                for e in re.finditer("time\d{1,2}", from_text):
                    temp_text = temp_text.replace(e.group(), norm_ner_dict[e.group()])
                    from_text = from_text.replace(e.group(), time_ch_dict[norm_ner_dict[e.group()]])

                self.ref[sent_idx]['references']["index"]["time"].append(find_ch_span(from_text.strip().lower(), self.texts, self.ref[sent_idx]['charIdx']))
                self.ref[sent_idx]['references']["text"]["time"].append(from_text)
                self.ref[sent_idx]['references']["range"].append(final_range)
                self.ref[sent_idx]['references']["mention"].append([False, True])
            for each in re.finditer("prior to\s\w{,10}\s{,1}time\d{1,2}", temp_text):
                prev = each.group().split(' ')[-1]
                p1 = self.time_str_to_date(norm_ner_dict[prev], curr_time)

                p1_range = self.get_date_range(p1)
                final_range = ("TBD", p1_range[1])

                from_text = each.group()
                for e in re.finditer("time\d{1,2}", from_text):
                    temp_text = temp_text.replace(e.group(), norm_ner_dict[e.group()])
                    from_text = from_text.replace(e.group(), time_ch_dict[norm_ner_dict[e.group()]])

                self.ref[sent_idx]['references']["index"]["time"].append(find_ch_span(from_text.strip().lower(), self.texts, self.ref[sent_idx]['charIdx']))
                self.ref[sent_idx]['references']["text"]["time"].append(from_text)
                self.ref[sent_idx]['references']["range"].append(final_range)
                self.ref[sent_idx]['references']["mention"].append([False, True])
            for each in re.finditer("previous to\s\w{,10}\s{,1}time\d{1,2}", temp_text):
                prev = each.group().split(' ')[-1]
                p1 = self.time_str_to_date(norm_ner_dict[prev], curr_time)

                p1_range = self.get_date_range(p1)
                final_range = ("TBD", p1_range[1])

                from_text = each.group()
                for e in re.finditer("time\d{1,2}", from_text):
                    temp_text = temp_text.replace(e.group(), norm_ner_dict[e.group()])
                    from_text = from_text.replace(e.group(), time_ch_dict[norm_ner_dict[e.group()]])

                self.ref[sent_idx]['references']["index"]["time"].append(find_ch_span(from_text.strip().lower(), self.texts, self.ref[sent_idx]['charIdx']))
                self.ref[sent_idx]['references']["text"]["time"].append(from_text)
                self.ref[sent_idx]['references']["range"].append(final_range)
                self.ref[sent_idx]['references']["mention"].append([False, True])
            for each in re.finditer("end\s\w{,10}\s{,1}time\d{1,2}", temp_text):
                prev = each.group().split(' ')[-1]
                p1 = self.time_str_to_date(norm_ner_dict[prev], curr_time)

                p1_range = self.get_date_range(p1)
                final_range = ("TBD", p1_range[1])

                from_text = each.group()
                for e in re.finditer("time\d{1,2}", from_text):
                    temp_text = temp_text.replace(e.group(), norm_ner_dict[e.group()])
                    from_text = from_text.replace(e.group(), time_ch_dict[norm_ner_dict[e.group()]])

                self.ref[sent_idx]['references']["index"]["time"].append(find_ch_span(from_text.strip().lower(), self.texts, self.ref[sent_idx]['charIdx']))
                self.ref[sent_idx]['references']["text"]["time"].append(from_text)
                self.ref[sent_idx]['references']["range"].append(final_range)
                self.ref[sent_idx]['references']["mention"].append([False, True])
            for each in re.finditer("finish\s\w{,10}\s{,1}time\d{1,2}", temp_text):
                prev = each.group().split(' ')[-1]
                p1 = self.time_str_to_date(norm_ner_dict[prev], curr_time)

                p1_range = self.get_date_range(p1)
                final_range = ("TBD", p1_range[1])

                from_text = each.group()
                for e in re.finditer("time\d{1,2}", from_text):
                    temp_text = temp_text.replace(e.group(), norm_ner_dict[e.group()])
                    from_text = from_text.replace(e.group(), time_ch_dict[norm_ner_dict[e.group()]])

                self.ref[sent_idx]['references']["index"]["time"].append(find_ch_span(from_text.strip().lower(), self.texts, self.ref[sent_idx]['charIdx']))
                self.ref[sent_idx]['references']["text"]["time"].append(from_text)
                self.ref[sent_idx]['references']["range"].append(final_range)
                self.ref[sent_idx]['references']["mention"].append([False, True])
            for each in re.finditer("stop\s\w{,10}\s{,1}time\d{1,2}", temp_text):
                prev = each.group().split(' ')[-1]
                p1 = self.time_str_to_date(norm_ner_dict[prev], curr_time)

                p1_range = self.get_date_range(p1)
                final_range = ("TBD", p1_range[1])

                from_text = each.group()
                for e in re.finditer("time\d{1,2}", from_text):
                    temp_text = temp_text.replace(e.group(), norm_ner_dict[e.group()])
                    from_text = from_text.replace(e.group(), time_ch_dict[norm_ner_dict[e.group()]])

                self.ref[sent_idx]['references']["index"]["time"].append(find_ch_span(from_text.strip().lower(), self.texts, self.ref[sent_idx]['charIdx']))
                self.ref[sent_idx]['references']["text"]["time"].append(from_text)
                self.ref[sent_idx]['references']["range"].append(final_range)
                self.ref[sent_idx]['references']["mention"].append([False, True])
            for each in re.finditer("terminate\s\w{,10}\s{,1}time\d{1,2}", temp_text):
                prev = each.group().split(' ')[-1]
                p1 = self.time_str_to_date(norm_ner_dict[prev], curr_time)

                p1_range = self.get_date_range(p1)
                final_range = ("TBD", p1_range[1])

                from_text = each.group()
                for e in re.finditer("time\d{1,2}", from_text):
                    temp_text = temp_text.replace(e.group(), norm_ner_dict[e.group()])
                    from_text = from_text.replace(e.group(), time_ch_dict[norm_ner_dict[e.group()]])

                self.ref[sent_idx]['references']["index"]["time"].append(find_ch_span(from_text.strip().lower(), self.texts, self.ref[sent_idx]['charIdx']))
                self.ref[sent_idx]['references']["text"]["time"].append(from_text)
                self.ref[sent_idx]['references']["range"].append(final_range)
                self.ref[sent_idx]['references']["mention"].append([False, True])
            for each in re.finditer("up\s{,1}to\stime\d{1,2}", temp_text):
                prev = each.group().split(' ')[-1]
                p1 = self.time_str_to_date(norm_ner_dict[prev], curr_time)
                p1_range = self.get_date_range(p1)

                final_range = ("TBD", p1_range[1])

                from_text = each.group()
                for e in re.finditer("time\d{1,2}", from_text):
                    temp_text = temp_text.replace(e.group(), norm_ner_dict[e.group()])
                    from_text = from_text.replace(e.group(), time_ch_dict[norm_ner_dict[e.group()]])
                
                self.ref[sent_idx]['references']["index"]["time"].append(find_ch_span(from_text.strip().lower(), self.texts, self.ref[sent_idx]['charIdx']))
                self.ref[sent_idx]['references']["text"]["time"].append(from_text)
                self.ref[sent_idx]['references']["range"].append(final_range)
                self.ref[sent_idx]['references']["mention"].append([False, True])

            for each in re.finditer("start\w{,4}\s\w{,9}\s?time\d{1,2}", temp_text):
                prev = each.group().split(' ')[-1]
                p1 = self.time_str_to_date(norm_ner_dict[prev], curr_time)
                p1_range = self.get_date_range(p1)

                final_range = (p1_range[0], "TBD")

                from_text = each.group()
                for e in re.finditer("time\d{1,2}", from_text):
                    temp_text = temp_text.replace(e.group(), norm_ner_dict[e.group()])
                    from_text = from_text.replace(e.group(), time_ch_dict[norm_ner_dict[e.group()]])
                
                self.ref[sent_idx]['references']["index"]["time"].append(find_ch_span(from_text.strip().lower(), self.texts, self.ref[sent_idx]['charIdx']))
                self.ref[sent_idx]['references']["text"]["time"].append(from_text)
                self.ref[sent_idx]['references']["range"].append(final_range)
                self.ref[sent_idx]['references']["mention"].append([True, False])

            for each in re.finditer("from\s?time\d{1,2}", temp_text):
                prev = each.group().split(' ')[-1]
                p1 = self.time_str_to_date(norm_ner_dict[prev], curr_time)
                p1_range = self.get_date_range(p1)
                final_range = (p1_range[0], "TBD")

                from_text = each.group()
                for e in re.finditer("time\d{1,2}", from_text):
                    temp_text = temp_text.replace(e.group(), norm_ner_dict[e.group()])
                    from_text = from_text.replace(e.group(), time_ch_dict[norm_ner_dict[e.group()]])
                
                self.ref[sent_idx]['references']["index"]["time"].append(find_ch_span(from_text.strip().lower(), self.texts, self.ref[sent_idx]['charIdx']))
                self.ref[sent_idx]['references']["text"]["time"].append(from_text)
                self.ref[sent_idx]['references']["range"].append(final_range)
                self.ref[sent_idx]['references']["mention"].append([True, False])
                
            for each in re.finditer("commence\s?time\d{1,2}", temp_text):
                prev = each.group().split(' ')[-1]
                p1 = self.time_str_to_date(norm_ner_dict[prev], curr_time)
                p1_range = self.get_date_range(p1)

                final_range = (p1_range[0], "TBD")

                from_text = each.group()
                for e in re.finditer("time\d{1,2}", from_text):
                    temp_text = temp_text.replace(e.group(), norm_ner_dict[e.group()])
                    from_text = from_text.replace(e.group(), time_ch_dict[norm_ner_dict[e.group()]])
                
                self.ref[sent_idx]['references']["index"]["time"].append(find_ch_span(from_text.strip().lower(), self.texts, self.ref[sent_idx]['charIdx']))# re.search(from_text.strip(), self.texts, re.IGNORECASE).span())
                self.ref[sent_idx]['references']["text"]["time"].append(from_text)
                self.ref[sent_idx]['references']["range"].append(final_range)
                self.ref[sent_idx]['references']["mention"].append([True, False])
            for each in re.finditer("following\s?time\d{1,2}", temp_text):
                prev = each.group().split(' ')[-1]
                p1 = self.time_str_to_date(norm_ner_dict[prev], curr_time)
                p1_range = self.get_date_range(p1)

                final_range = (p1_range[0], "TBD")

                from_text = each.group()
                for e in re.finditer("time\d{1,2}", from_text):
                    temp_text = temp_text.replace(e.group(), norm_ner_dict[e.group()])
                    from_text = from_text.replace(e.group(), time_ch_dict[norm_ner_dict[e.group()]])
                
                self.ref[sent_idx]['references']["index"]["time"].append(find_ch_span(from_text.strip().lower(), self.texts, self.ref[sent_idx]['charIdx']))# re.search(from_text.strip(), self.texts, re.IGNORECASE).span())
                self.ref[sent_idx]['references']["text"]["time"].append(from_text)
                self.ref[sent_idx]['references']["range"].append(final_range)
                self.ref[sent_idx]['references']["mention"].append([True, False])
            
            return temp_text

    def time_extract(self, curr_time):
        print("\n\nExtracting the time features ...")

        time_features = []
        norm_ner_dict = {}
        time_ch_dict = {}
        idx = 1
        doc = nlp(self.texts)

        count_dict = {}
        for sent_idx, (single_sent, sent) in enumerate(zip(self.anns.sentence, doc.sents)):
            temp_text = sent.text
            temp_text = self.weird_time_extract(sent_idx, temp_text, curr_time)
            
            # Convert time expressions into {time{}.format(i)} -> Using library
            for k, mention in enumerate(single_sent.mentions):
                flag = 0
                if mention.entityType == "DATE" or mention.entityType == "DURATION":
                    if not re.search("\d+[/]\d+", mention.normalizedNER) and mention.timex.text not in [' ', '', '  ']:
                        norm_ner_dict["time{}".format(idx)] = mention.normalizedNER
                        time_ch_dict[mention.normalizedNER] = mention.timex.text
                        
                        temp_text = temp_text.replace(mention.timex.text, "time{}".format(idx))
                        idx += 1
            # Find time expressions some formats that are not detected by library such as "March 1th 2020" -> Manual detection
            temp_text = self.time_phrase_extract(temp_text, sent_idx, norm_ner_dict, time_ch_dict, curr_time)

            # Detect normal time feature
            for e in re.finditer("time\d{1,2}", temp_text):
                
                p1 = self.time_str_to_date(norm_ner_dict[e.group()], curr_time)
                range = self.get_date_range(p1)

                time_text = time_ch_dict[norm_ner_dict[e.group()]]
                if  time_text in count_dict: count_dict[time_text] +=1 
                else: count_dict[time_text] = 0

                if p1['mode'] != "Not found":
                    time_features.append({"type":"TIME/DURATION", "text":mention.timex.text, "range": range})

                    # Check this 'timex.text' is already in manually made reference list.
                    self.ref[sent_idx]['references']["index"]["time"].append(find_ch_span(time_ch_dict[norm_ner_dict[e.group()]], self.texts, self.ref[sent_idx]['charIdx'], count_dict)) 
                    self.ref[sent_idx]['references']["text"]["time"].append(time_ch_dict[norm_ner_dict[e.group()]])
                    self.ref[sent_idx]['references']["range"].append(range)
                    self.ref[sent_idx]['references']["mention"].append([False, False])

        print("Time >>>> ", time_features)
        return time_features

    def trend_extract(self):
        print("Extracting the trend features ...")

        sim_thresh = 0.7
        with open("pipeline/emb_dict.pickle", "rb") as f:
            self.emb_dict = pickle.load(f)

        # Load stopwords
        f = open("../../reference/plotdigitizer/data/englishST.txt")
        st = f.read().split('\n')

        # Compare each word in the paragraphs with each words of contextual embedding dictionary.
        list_of_distances = []
        temp_tok_list = []
        sim_word_li_idx = 0
        
        max_extremum = ["maximum", "max", "peak", 'greatest', 'high', "apex", "biggest", "cap", "ceiling", "largest", "max", "maximum",\
                        "most", "pinnacle", "top", "topmost", "zenith"]
        min_extremum = ["minimum", "min", "least", 'lowest', 'low', "base", "bottom", "fewest", "floor", \
                        "smallest", "tinest", "trough"]
        increase_trend_lemma_word = ["boost", "climb", "gain", "grow", "heighten", "higher", "increment", \
                                    "increase", "rise", "jump", "spike", "spiked", "add", "grow", "high", "inflation", \
                                    "rebounding", "uptick", 'soar', 'surge', 'raise', 'up', 'go', 'upward', "skyrocket", "recover", "restore"]
        decrease_trend_lemma_word = ["contract", "decrease", "drop", "decline", "down", "negative", "low", "sink", \
                                    "plum", "fall", 'dip', 'downward', 'weak', 'weaken', "diminish", "dwindle", "lessen", "lower", \
                                    "falloff", "reduce", "shrink", "subside", "wane"]

        doc = nlp(self.texts)
        self.lemma_tg_li = [token.lemma_ for (w, _) in self.emb_dict.items() for token in nlp(w) ]
        self.lemma_tg_li += max_extremum
        self.lemma_tg_li += min_extremum
        self.lemma_tg_li += increase_trend_lemma_word
        self.lemma_tg_li += decrease_trend_lemma_word

        tokenized_text, tokens_tensor, segments_tensors = self.bert_text_preparation(self.texts, tokenizer)

        count_dict = {}
        for sent_idx, (sent, ann) in enumerate(zip(doc.sents, self.anns.sentence)): # doc.sents: spacy, self.anns.sentence: stanza
            single_sent = sent.text

            # Tokens per sentence
            sing_doc = nlp(single_sent)
            sent_tokenized_text = [str(token) for token in sing_doc]
            sent_tokenized_text_wo_st = [token for token in sent_tokenized_text if token not in st]

            # Initiate
            self.ref.append({'sentenceIdx':sent_idx, \
                            "charIdx": (sent.start_char, sent.end_char), \
                            "references":{"index":{"time":[], 'feature':[]}, \
                                        'text':{'time':[], 'feature':[]}, \
                                        'chart':[], 
                                        'range':[],
                                        'mention': []}})
            self.ref_result.append({'sentenceIdx':sent_idx, \
                                    'charIdx': (sent.start_char, sent.end_char), \
                                    'references': []})


            # Preprocessing the paragraph
            sent_span = (sent.start_char, sent.end_char)
            single_sent = single_sent.replace(",", "")
            
            tokenized_text, tokens_tensor, segments_tensors = self.bert_text_preparation(single_sent, self.tokenizer)
            list_token_embeddings = self.get_bert_embeddings(tokens_tensor, segments_tensors, self.model)
            tokenized_text_wo_st = [token for token in tokenized_text if token not in st] # just for comparison
            
            for token_wo_st in sent_tokenized_text_wo_st:
                if token_wo_st == '[CLS]': continue
                word_index = sent_tokenized_text.index(token_wo_st)
                word_embedding = list_token_embeddings[word_index]

                if token_wo_st in count_dict: count_dict[token_wo_st] += 1
                else: count_dict[token_wo_st] = 0
                
                # Comparing lemmas
                if nlp(token_wo_st)[0].lemma_ in self.lemma_tg_li:
                    ## Find the span and the index of trend feature.
                    self.ref[sent_idx]['references']["index"]["feature"].append(find_ch_span(token_wo_st, self.texts, sent_span, count_dict))
                    self.ref[sent_idx]['references']["text"]["feature"].append(token_wo_st)
                
                # Calculate cosine similarity
                flag = 0
                sim_thresh = 0.7
                
                for trend, trend_emb in self.emb_dict.items():
                    cos_dist = 1 - cosine(word_embedding, trend_emb) # emb_dim = 768

                    if flag == 0 and cos_dist >= sim_thresh:
                        flag = 1
                        list_of_distances.append([token_wo_st, self.figure_names , trend, cos_dist, sent_idx])
                        temp_tok_list.append(token_wo_st)

                        ## Developing a reference list
                        # Detokenized text
                        temp_tok_list = detokenize(temp_tok_list, self.texts)
                        
                        # Finding the range of reference.
                        if re.search(temp_tok_list[sim_word_li_idx], single_sent.lower()):
                            span = find_ch_span(temp_tok_list[sim_word_li_idx], self.texts, sent_span, count_dict)
                        else: ## e.g. temp_tok_list[sim_word_li_idx] = "##wind" OR "##tick"
                            sim_word_li_idx += 1
                            break

                        self.ref[sent_idx]['references']["index"]["feature"].append(span)
                        self.ref[sent_idx]['references']["text"]["feature"].append(temp_tok_list[sim_word_li_idx])
                        sim_word_li_idx += 1
        # Convert list into dataframe 
        trend_df = pd.DataFrame(list_of_distances, columns = ['trend_token', 'tokenized_text', 'trend_embedding', 'cos_dist', 'sentence_idx'])   
        trend_dict = []
        for each in trend_df['trend_token'].unique():
            trend_dict.append({"text":each})

        return trend_dict, trend_df

    def dist_find_sdp(self, sent_idx, ref, sentence, each_time, ref_idx, j): # sdp: Shortest Dependency Path
        sentence = sentence.replace("–", " ").lower()
        doc = nlp(sentence) # spacy nlp
        print(sentence)

        # Load spacy's dependency tree into a networkx graph
        edges = []; toks = []
        temp_matching = []

        for token in doc:
            for child in token.children:
                toks.append(token.lower_); toks.append(child.lower_)
                
                edges.append(('{0}'.format(token.lower_),
                            '{0}'.format(child.lower_)))
        graph = nx.Graph(edges)

        init_min_len = 6
        min_len = 6
        sem_dependent_word = -1

        trend, time = ref['references']['text']['feature'], ref['references']['text']['time']
        if ref['references'] == {} or trend == [] or time == []:
            return 0, []
        
        path_len_info = {}
        minpath_trend_count = {}

        for k, each_trend in enumerate(trend):
            try:
                curr_path_len = nx.shortest_path_length(graph, source=each_time.replace('-', ' ').split(' ')[0].lower(), target=each_trend.lower())
            except:
                print("Errors!!\n", toks)
                continue

            if each_trend in minpath_trend_count:
                minpath_trend_count[each_trend] += 1
            else:
                minpath_trend_count[each_trend] = 0

            # Save all path information
            if curr_path_len in path_len_info:
                path_len_info[curr_path_len].append([each_trend+"-"+str(minpath_trend_count[each_trend]), ref['references']['index']['feature'][k]])
            else:
                path_len_info[curr_path_len] = [[each_trend+"-"+str(minpath_trend_count[each_trend]), ref['references']['index']['feature'][k]]]

            if curr_path_len < min_len: 
                min_len = curr_path_len
                sem_dependent_word = k

        if min_len < init_min_len: # If there's no path the shortest path is less than 6, don't save them.
            if len(path_len_info[min_len]) == 1:
                self.add_pair(sent_idx, ref_idx, ref['references']['index']['time'][j], ref['references']['index']['feature'][sem_dependent_word], \
                        each_time, each_trend, ref['references']['range'][j], self.check_ft_type(trend[k]), ref['references']['mention'][j])
                ref_idx += 1
            elif len(path_len_info[min_len]) > 1:
                # Compare distance between a time featuere word and trend words.
                toks = sentence.replace(".", "").replace(",", "").split() # Replace ".," because it is preprocessed by CoreNLP library.
                dist_list = {}
                dist_list_for_time = {}

                count_dict = {}
                for seq, trend in enumerate(toks):
                    if trend in count_dict:
                        count_dict[trend] += 1
                        dist_list[trend + "-" + str(count_dict[trend])] = seq
                    else:
                        count_dict[trend] = 0
                        dist_list[trend + "-" + str(count_dict[trend])] = seq
                    dist_list_for_time[trend] = seq
                final_trend = ""
                min = 100

                for (each_trend, span) in path_len_info[min_len]: # Consider an actual distance in the sentence
                    time = each_time.replace('-', ' ').split(' ')[0].lower()

                    if abs(dist_list[each_trend] - dist_list_for_time[time]) < min:
                        final_trend = each_trend.split("-")[0]
                        final_span = span
                        min = abs(dist_list[each_trend] - dist_list_for_time[time])

                self.ref_result[sent_idx]['references'].append({'referenceIdx': "{}-{}".format(sent_idx, ref_idx), \
                                                'text':{"time": ref['references']['index']['time'][j], \
                                                        'feature': final_span, \
                                                        'timeText': each_time, 'featureText': final_trend}, \
                                                'chart': "", 'range': ref['references']['range'][j], 'factCheck': False, \
                                                'featureType': self.check_ft_type(trend[k]), \
                                                "mentions": ref['references']['mention'][j]})
                temp_matching.append({'referenceIdx': "{}-{}".format(sent_idx, ref_idx), \
                                                'text':{"time": ref['references']['index']['time'][j], \
                                                        'feature': final_span, \
                                                        'timeText': each_time, 'featureText': final_trend}, \
                                                'chart': "", 'range': ref['references']['range'][j], 'factCheck': False, \
                                                'featureType': self.check_ft_type(trend[k]), \
                                                "mentions": ref['references']['mention'][j]})
                ref_idx += 1

        return ref_idx, temp_matching

    def create_defrl_dict(self):
        defrl_dict, rev_defrl_dict = {}, {}

        doc = stanza_nlp(self.texts)
        for sent_idx, sent in enumerate(doc.sentences):
            defrl_dict[sent_idx], rev_defrl_dict[sent_idx] = {}, {}
            for word in sent.words:
                defrl_dict[sent_idx][word.text] = {"head_text":  sent.words[word.head-1].text, "deprel": word.deprel, "xpos": word.xpos}
                
                if sent.words[word.head-1].text not in rev_defrl_dict[sent_idx]:
                    rev_defrl_dict[sent_idx][sent.words[word.head-1].text] = [{"child_text": word.text, "deprel": word.deprel, "xpos": word.xpos}]
                else:  rev_defrl_dict[sent_idx][sent.words[word.head-1].text].append({"child_text": word.text, "deprel": word.deprel, "xpos": word.xpos})

        return defrl_dict, rev_defrl_dict
    
    def check_ft_type(self, feature):
        max_extremum = ["maximum", "max", "peak", 'greatest', 'highest']
        min_extremum = ["minimum", "min", "least", 'lowest', 'low']
        increase_trend_lemma_word = ["boost", "climb", "gain", "grow", "heighten", "higher", "increment", \
                                    "increase", "rise", "jump", "spike", "spiked", "add", "grow", "high", "inflation", \
                                    "rebounding", "uptick", 'soar', 'surge', 'raise', 'up', 'go', 'upward', "skyrocket", "recover", "restore"]
        decrease_trend_lemma_word = ["contract", "decrease", "drop", "decline", "down", "negative", "low", "sink", \
                                    "plum", "fall", 'dip', 'downward', 'weak', 'weaken', "diminish", "dwindle", "lessen", "lower", \
                                    "falloff", "reduce", "shrink", "subside", "wane"]

        if feature == 'highest':
            return "extremum:+"
        elif feature == 'lowest' or feature == 'greatest':
            return "extremum:-"
        elif [word.lemma for sent in stanza_nlp(feature).sentences for word in sent.words][0] in increase_trend_lemma_word:
            return "trend:+"
        elif [word.lemma for sent in stanza_nlp(feature).sentences for word in sent.words][0] in decrease_trend_lemma_word:
            return "trend:-"
        elif [word.lemma for sent in stanza_nlp(feature).sentences for word in sent.words][0] in max_extremum:
            return "extremum:+"
        elif [word.lemma for sent in stanza_nlp(feature).sentences for word in sent.words][0] in min_extremum:
            return "extremum:+"
        else: return "trend:0"

    def remove_item(self, matching, idx_to_remove):
        matching = [each_match for each_match in matching if each_match['referenceIdx'] != idx_to_remove]
        return matching
     
    def add_new_item(self, matching, start, end):
        new_mentions = [True, True]
     
        if start['range'][1] == "TBD":
            new_chart = [start['range'][0], end['range'][1]]
            new_range = (start['range'][0], end['range'][1])
        elif start['range'][0] == "TBD":
            new_chart = [end['range'][0], start['range'][1]]
            new_range = (end['range'][0], start['range'][1])
            
        new_text = {'feature': start['text']['feature'], \
            'featureText': start['text']['featureText'], \
            'time': [list(start['text']['time']), list(end['text']['time'])], \
            'timeText': [start['text']['timeText'], end['text']['timeText']]}
        
        matching.append({'referenceIdx': start['referenceIdx'], \
                        'text' : new_text, \
                        'chart': new_chart, \
                        'range': new_range, 'factCheck': False, \
                        'featureType': start['featureType'], "mentions": new_mentions})
        
        return matching
    
    def compare_real_dist(self, min_time, compare_time, trend, sentence):
        sent_toks = sentence.split()
        
        min_time = min_time.split()
        compare_time = compare_time.split()
        
        min_time_idx = sent_toks.index(min_time[0])
        compare_time_idx = sent_toks.index(compare_time[0])
        trend_idx = sent_toks.index(trend)
        
        dist_min_time = abs(trend_idx - min_time_idx)
        dist_compare_time = abs(trend_idx - compare_time_idx)
        
        if dist_min_time <= dist_compare_time:
            return False
        elif dist_min_time > dist_compare_time:
            return True

    def compare_depdency_dist(self, pair, sentence):
        ## Extract dependency tree
        nlp = spacy.load('en_core_web_sm')
        sentence = sentence.replace("–", " ").lower()
        doc = nlp(sentence) # spacy nlp
        
        toks, edges = [], []
        for token in doc:
            for child in token.children:
                toks.append(token.lower_); toks.append(child.lower_)
                
                edges.append(('{0}'.format(token.lower_),
                            '{0}'.format(child.lower_)))
        graph = nx.Graph(edges)

        ## Calculate the shortest distance
        trend = pair['text']['featureText'].lower()
        time = pair['text']['timeText'].replace('-', ' ').split(' ')[0].lower()
        
        while True:
            try:
                curr_path_len = nx.shortest_path_length(graph, source = trend, target = time)
                print(curr_path_len, "::: ", time, "-", trend)
                break
            except:
                print("Errors!!\n", toks)
                continue
            
        return curr_path_len
    
    def find_sdp(self):         
        defrl_dict, rev_defrl_dict = self.create_defrl_dict()   
        
        # Time features 
        pos = ["JJR", "JJS", "JJ"]
        temp_match = []

        for sent_idx, (ref, sentence) in enumerate(zip(self.ref, self.sets)):
            time_fts = ref['references']['text']['feature']

            ref_idx = 0
            trend_feat_count_dict = {} 

            for j, each_time in enumerate(ref['references']['text']['time']):
                trend_feat_count_dict = {} 
                each_time = each_time.replace('-', ' ') # The deprel dictionary is lowercase and there's a time feature like '20-year'.
                for each in each_time.strip().split(' '):
                    try:
                        each_dict = defrl_dict[sent_idx][each]
                        head_each_dict = defrl_dict[sent_idx][each_dict["head_text"]]
                    except KeyError:
                        continue

                    has_rule = False
                    deprel = ["advcl", "obl", "nummod", "nmod", "amod", "compound", "nsubj", "obj"]

                    try:
                        if each_dict["deprel"] == "advcl" and each_dict["head_text"] in time_fts: # The most common case.
                            has_rule = True; print("1")
                            vocab_idx, trend_feat_count_dict = count_in_match(defrl_dict[sent_idx][each]["head_text"], trend_feat_count_dict, time_fts)
                            temp_match.append({'referenceIdx': "{}-{}".format(sent_idx, j), \
                                                                            'text':{"time":ref['references']['index']['time'][j], 'feature':ref['references']['index']['feature'][vocab_idx],\
                                                                                    'timeText': each_time, 'featureText': each_dict["head_text"]}, \
                                                                            'chart': [], 'range': ref['references']['range'][j],\
                                                                            'factCheck': False, 'featureType': self.check_ft_type(each_dict["head_text"]), 'mentions': ref['references']['mention'][j]})
                        
                        if each_dict["deprel"] == "obl" and each_dict["head_text"] in time_fts: # The most common case.
                            has_rule = True; print("2")
                            vocab_idx, trend_feat_count_dict = count_in_match(defrl_dict[sent_idx][each]["head_text"], trend_feat_count_dict, time_fts)
                            temp_match.append({'referenceIdx': "{}-{}".format(sent_idx, j), \
                                                                            'text':{"time":ref['references']['index']['time'][j], 'feature':ref['references']['index']['feature'][vocab_idx], #time_fts.index(defrl_dict[sent_idx][each]["head_text"])],\
                                                                                    'timeText': each_time, 'featureText': each_dict["head_text"]}, \
                                                                            'chart': [], 'range': ref['references']['range'][j],\
                                                                            'factCheck': False, 'featureType': self.check_ft_type(each_dict["head_text"]), 'mentions': ref['references']['mention'][j]})
                            
                        if each_dict["deprel"] == "nmod" and each_dict["head_text"] in time_fts and head_each_dict['xpos'] in pos:
                            has_rule = True
                            vocab_idx, trend_feat_count_dict = count_in_match(defrl_dict[sent_idx][each]["head_text"], trend_feat_count_dict, time_fts)
                            temp_match.append({'referenceIdx': "{}-{}".format(sent_idx, j), \
                                                                            'text':{"time":ref['references']['index']['time'][j], 'feature':ref['references']['index']['feature'][vocab_idx],#time_fts.index(defrl_dict[sent_idx][each]["head_text"])],\
                                                                                    'timeText': each_time, 'featureText': each_dict["head_text"]}, \
                                                                            'chart': [], 'range': ref['references']['range'][j], \
                                                                            'factCheck': False, 'featureType': self.check_ft_type(each_dict["head_text"]), 'mentions': ref['references']['mention'][j]})
                        
                        if each_dict["deprel"] == "nmod" and (head_each_dict["xpos"] == "NN" or head_each_dict["xpos"] == "VB"):
                            if each_dict["head_text"] in time_fts:
                                vocab_idx, trend_feat_count_dict = count_in_match(defrl_dict[sent_idx][each]["head_text"], trend_feat_count_dict, time_fts)
                                has_rule = True
                                temp_match.append({'referenceIdx': "{}-{}".format(sent_idx, j), \
                                                                                'text':{"time":ref['references']['index']['time'][j], 'feature':ref['references']['index']['feature'][vocab_idx],
                                                                                        'timeText': each_time, 'featureText': each_dict["head_text"]}, \
                                                                                'chart': [], 'range': ref['references']['range'][j], \
                                                                                'factCheck': False, 'featureType': self.check_ft_type(each_dict["head_text"]), 'mentions': ref['references']['mention'][j]})
                            for each_rev in rev_defrl_dict[sent_idx][each_dict["head_text"]]:
                                if each_rev['child_text'] in time_fts and each_rev['deprel'] == 'amod':
                                    has_rule = True
                                    vocab_idx, trend_feat_count_dict = count_in_match(each_rev['child_text'], trend_feat_count_dict, time_fts)
                                    temp_match.append({'referenceIdx': "{}-{}".format(sent_idx, j), \
                                                                                    'text':{"time":ref['references']['index']['time'][j], 'feature':ref['references']['index']['feature'][vocab_idx],#time_fts.index(each_rev['child_text'])],\
                                                                                            'timeText': each_time, 'featureText': each_rev['child_text']}, \
                                                                                    'chart': [], 'range': ref['references']['range'][j], \
                                                                                    'factCheck': False, 'featureType': self.check_ft_type(each_rev['child_text']), 'mentions': ref['references']['mention'][j]})
                        
                        if each_dict["deprel"] == "obl" and (head_each_dict["xpos"] == "VBD" or head_each_dict["xpos"] == "VB"): #  e.g. It reached its lowest point in the 1920s and again in the 1940s
                            if head_each_dict["deprel"] == 'advcl' and  head_each_dict["head_text"] in time_fts:
                                has_rule = True
                                vocab_idx, trend_feat_count_dict = count_in_match(head_each_dict["head_text"], trend_feat_count_dict, time_fts)
                                temp_match.append({'referenceIdx': "{}-{}".format(sent_idx, j), \
                                                                                'text':{"time": ref['references']['index']['time'][j], 'feature':ref['references']['index']['feature'][vocab_idx],#time_fts.index(head_each_dict["head_text"])],\
                                                                                        'timeText': each_time, 'featureText': head_each_dict["head_text"]}, \
                                                                                'chart': [], 'range': ref['references']['range'][j], \
                                                                                'factCheck': False, 'featureType': self.check_ft_type(head_each_dict["head_text"]), 'mentions': ref['references']['mention'][j]})
                    
                            for each_rev in rev_defrl_dict[sent_idx][each_dict["head_text"]]:
                                if each_rev['xpos'] == 'NN' and each_rev['deprel'] == 'obj':
                                    for each_rrev in rev_defrl_dict[sent_idx][each_rev['child_text']]:
                                        if each_rrev['xpos'] in pos and each_rrev['deprel'] == 'amod' and each_rrev['child_text'] in time_fts:
                                            
                                            has_rule = True
                                            vocab_idx, trend_feat_count_dict = count_in_match(each_rrev['child_text'], trend_feat_count_dict, time_fts)
                                            temp_match.append({'referenceIdx': "{}-{}".format(sent_idx, j), \
                                                                                        'text':{"time": ref['references']['index']['time'][j], 'feature':ref['references']['index']['feature'][vocab_idx],#time_fts.index(each_rrev['child_text'])],\
                                                                                        'timeText': each_time, 'featureText': each_rrev['child_text']}, \
                                                                                        'chart': [], 'range': ref['references']['range'][j], \
                                                                                        'factCheck': False, 'featureType': self.check_ft_type(each_rrev['child_text']), 'mentions': ref['references']['mention'][j]})
                        
                        if each_dict["deprel"] == "obl": #and head_each_dict["xpos"] == "VBD": # It went up after 1942.
                            for each_rev in rev_defrl_dict[sent_idx][each_dict["head_text"]]:
                                if each_rev['child_text'] in time_fts and ((each_rev['deprel'] == "compound:prt") or (each_rev['deprel'] == "advmod") or (each_rev['deprel'] == "xcomp")):# or time_fts and each_rev['deprel'] == "advmod"):
                                    has_rule = True; print("8")
                                    vocab_idx, trend_feat_count_dict = count_in_match(each_rev['child_text'], trend_feat_count_dict, time_fts)
                                    temp_match.append({'referenceIdx': "{}-{}".format(sent_idx, j), \
                                                                                    'text':{"time":ref['references']['index']['time'][j], 'feature':ref['references']['index']['feature'][vocab_idx],#time_fts.index(each_rev['child_text'])],\
                                                                                    'timeText': each_time, 'featureText': each_rev['child_text']}, \
                                                                                    'chart': [], 'range': ref['references']['range'][j], \
                                                                                    'factCheck': False, 'featureType': self.check_ft_type(each_rev['child_text']), 'mentions': ref['references']['mention'][j]})

                        if each_dict["deprel"] == "nsubj" and head_each_dict["xpos"] == "VBD":
                            if head_each_dict["deprel"] == "obl" and head_each_dict["head_text"] in time_fts :
                                has_rule = True; print("9")
                                vocab_idx, trend_feat_count_dict = count_in_match(defrl_dict[sent_idx][defrl_dict[sent_idx][each]["head_text"]]["head_text"], trend_feat_count_dict, time_fts)
                                temp_match.append({'referenceIdx': "{}-{}".format(sent_idx, j), \
                                                                            'text':{"time":ref['references']['index']['time'][j], 'feature':ref['references']['index']['feature'][vocab_idx],#time_fts.index(defrl_dict[sent_idx][defrl_dict[sent_idx][each]["head_text"]]["head_text"])],\
                                                                            'timeText': each_time, 'featureText': each_dict["head_text"]}, \
                                                                            'chart': [], 'range': ref['references']['range'][j], \
                                                                            'factCheck': False, 'featureType': self.check_ft_type(each_dict["head_text"]), 'mentions': ref['references']['mention'][j]})
                    
                    except:
                        break
                    if has_rule:
                        ref_idx += 1 
                        break
                 
                if not has_rule: 
                    ref_idx, temp_match = self.dist_find_sdp(sent_idx, ref, sentence, each_time, ref_idx, j); 
                    has_rule = True
            ## If the first stage for finidng temp_match is completed, starting to overlapped cases.
            final_match = temp_match.copy()
            
            time, trend = [], []
            for each in final_match:
                time.append(each['text']['timeText'])
                trend.append(each['text']['featureText'])
            trend = set(trend)
            
            # e.g., Since until, From until.
            # Find if there is overlapping
            curr_time_idx = {}
            anal_features = []
            ref_idx_li = {}
            rep_pairs = []
            trend_centered_rep_pair = {}

            for i, each in enumerate(temp_match):
                ref_idx = each['referenceIdx'] #ref_idx = 1-2
                curr_time_idx[ref_idx] = each #curr_time_idx[3] = { ..... }
                
                if each['text']['featureText'] in anal_features:
                    rep1 = curr_time_idx[ref_idx_li[anal_features.index(each['text']['featureText'])]] # first
                    rep2 = each # second
                    rep_pairs.append((rep1, rep2))
                    
                anal_features.append(each['text']['featureText']) # 
                ref_idx_li[i] = ref_idx # {3: 0-2}
                
                if each['text']['featureText'] in trend_centered_rep_pair:
                    trend_centered_rep_pair[each['text']['featureText']].append(each)
                else: 
                    trend_centered_rep_pair[each['text']['featureText']] = [each]
            
            used_pair, used_trend = [], []

            if len(rep_pairs) == 0:
                self.ref_result[sent_idx]['references'] = final_match
                print("There is no repeated cases. Return to main function!")
                return
            else:
                for idx, (each_a, each_b) in enumerate(rep_pairs):
                    print(f"{idx} repetition ==>",  each_a['text']['timeText'].lower(), ", ", each_b['text']['timeText'].lower())
                    
                    # If two time features are converged, combine them.
                    if (re.findall("\s?\w{,20}\s?from\s\w{,20}\W", each_a['text']['timeText'].lower()) and re.findall("\s?\w{,20}\s?until\s\w{,20}", each_b['text']['timeText'].lower())) or \
                        (re.findall("\s?\w{,20}\s?since\s\w{,20}", each_a['text']['timeText'].lower()) and re.findall("\s?\w{,20}\s?until\s\w{,20}", each_b['text']['timeText'].lower())) or \
                        (re.findall("\s?\w{,20}\s?until\s\w{,20}", each_a['text']['timeText'].lower()) and re.findall("\s?\w{,20}\s?from\s\w{,20}", each_b['text']['timeText'].lower())) or \
                        (re.findall("\s?\w{,20}\s?from\s\w{,20}", each_a['text']['timeText'].lower()) and re.findall("\s?\w{,20}\s?to\s\w{,20}", each_b['text']['timeText'].lower())) or \
                        (re.findall("\s?\w{,20}\s?to\s\w{,20}", each_a['text']['timeText'].lower()) and re.findall("\s?\w{,20}\s?from\s\w{,20}", each_b['text']['timeText'].lower())):

                        print("Starting combining two...")
                        ## Remove two references in final_match
                        final_match = self.remove_item(final_match, each_a["referenceIdx"])
                        final_match = self.remove_item(final_match, each_b["referenceIdx"])
                        
                        ## Add new combined one
                        final_match = self.add_new_item(final_match, each_a, each_b)
                        
                        ## Save used features
                        used_pair.append(each_a['text']['timeText'].lower())
                        used_pair.append(each_b['text']['timeText'].lower())
                        used_trend.append(each_a['text']['featureText'].lower())

                used_trend = set(used_trend)
                
            ## If two of the time features are not related, choose just one pair based on their distances in a dependency tree.
            if (len(used_pair) != len(time)) and (len(used_trend) != len(trend)):## time들이 전부 매칭된 게 아니면, time을 기준으로 각각의 trend와 거리 계산
                print("Starting comparing the distances...")
                
                ## Find the most minimum distances between a trend word and time features.
                min_dist, min_dist_time = 100, ""
                for trend, times in trend_centered_rep_pair.items():
                    print(len(times))
                    for idx, time in enumerate(times):
                        print("==> ", time['text']['timeText'])
                        if time['text']['timeText'] not in used_pair:
                            ## Compare the distance between trend and time..
                            dist = self.compare_depdency_dist(time, sentence)
                            print(f"Distance between [{time['text']['timeText']}] and [{trend}] : ", dist)
                            if min_dist > dist:     
                                min_ref = time
                                min_dist = dist; 
                                min_dist_time_refidx = time['referenceIdx']
                            elif min_dist == dist:
                                change_min = self.compare_real_dist(min_ref['text']['timeText'], time['text']['timeText'], trend, sentence) 
                                if change_min:
                                    min_ref = time
                                    min_dist = dist; 
                                    min_dist_time_refidx = time['referenceIdx']
                        final_match = self.remove_item(final_match, time['referenceIdx'])
                        
                    # Remove all of the pairs and add the one whose distance is the minimum.
                    final_match.append(min_ref)
            else:
                print("There is no conflicted case in their overlapped cases.")
                
            ## Removing remaining cases.
            time_sub_used_time = [x for x in time if x not in used_pair]

            for each_time in time_sub_used_time:
                final_match = [each for each in final_match if each['text']['timeText'] != each_time]
            
            self.ref_result[sent_idx]['references'] = final_match
            print("Finish matching pairs.")

    def add_pair(self, sent_idx, ref_idx, time, feature, timeText, featureText, range_, feature_type, mentions):
        self.ref_result[sent_idx]['references'].append({'referenceIdx': "{}-{}".format(sent_idx, ref_idx), \
                                                    'text':{"time": time, 'feature': feature, \
                                                            'timeText': timeText, 'featureText': featureText}, \
                                                    'chart': "", 'range': range_, 'factCheck': False, \
                                                    'featureType':feature_type, \
                                                    "mentions": mentions})

    def fit_chart_data(self, plots):
        """ Finding the real 'chart' value that fits into the calculated date 'range'.
        If the range is (TBD, "XXXX-XX-XX") since its time feature is like 'until', 'upto', set the min date based on the real chart data.
        """
        ref_date = []
        
        # Matching the reference dictionary with the plot data.
        date_df = plots["date"].str.split('-', expand=True).rename(columns={0:'year', 1:'month', 2:'day'})
        date_df = date_df.astype("int")
        
        import copy
        remove_list = []
        for i, sent in enumerate(self.ref_result):
            if sent['references'] != []: 
                sent_ref = copy.deepcopy(sent['references'])
            else: continue

            for j, each_pair in enumerate(sent_ref):
                try:
                    dates, feature = each_pair["range"], each_pair["featureType"]

                    date_df["plot_{}".format(i)] = [False]*len(date_df)

                    if dates[0] == "TBD": # If the 'dates' value is "TBD", set its start date as the first date of chart data. 
                        start_y, start_m, start_d = date_df.iloc[0, 0], date_df.iloc[0, 1], date_df.iloc[0, 2]
                    else:
                        start_y, start_m, start_d = int(dates[0]['year']), int(dates[0]['month']), int(dates[0]['day'])

                    if dates[1] == "TBD":
                        end_y, end_m, end_d =  date_df.iloc[-1, 0], date_df.iloc[-1, 1], date_df.iloc[-1, 2]
                    else:
                        end_y, end_m, end_d = int(dates[1]['year']), int(dates[1]['month']), int(dates[1]['day'])

                    # Deleting a data when the date is more than max date and less than min date. 
                    if dates[0] == "TBD":
                        p_max_y, p_max_m, p_max_d = int(dates[1]['year']), int(dates[1]['month']), int(dates[1]['day'])
                        r_max_y, r_max_m, r_max_d = date_df.iloc[-1, 0], date_df.iloc[-1, 1], date_df.iloc[-1, 2]
                        if (r_max_y < p_max_y):
                            remove_list.append((i, j))
                            continue
                    elif dates[1] == "TBD":
                        p_min_y, p_min_m, p_min_d = int(dates[0]['year']), int(dates[0]['month']), int(dates[0]['day'])
                        r_min_y, r_min_m, r_min_d = date_df.iloc[0, 0], date_df.iloc[0, 1], date_df.iloc[0, 2]
                        if (r_min_y > p_min_y): 
                            remove_list.append((i, j))
                            continue
                    else:
                        p_min_y, p_min_m, p_min_d = int(dates[0]['year']), int(dates[0]['month']), int(dates[0]['day'])
                        p_max_y, p_max_m, p_max_d = int(dates[1]['year']), int(dates[1]['month']), int(dates[1]['day'])
                        r_min_y, r_min_m, r_min_d = date_df.iloc[0, 0], date_df.iloc[0, 1], date_df.iloc[0, 2]
                        r_max_y, r_max_m, r_max_d = date_df.iloc[-1, 0], date_df.iloc[-1, 1], date_df.iloc[-1, 2]

                        if (r_min_y > p_min_y) or (r_max_y < p_max_y):
                            remove_list.append((i, j))
                            continue

                    # Filtering a Date data with time reference.
                    date_df.loc[((date_df['year'] >= start_y)&(date_df['year']<=end_y)), "plot_{}".format(i)] = True
                    date_df.loc[((date_df['year'] == start_y)&(date_df['month']<start_m)), "plot_{}".format(i)] = False
                    date_df.loc[((date_df['year'] == end_y)&(date_df['month']>end_m)), "plot_{}".format(i)] = False
                    date_df.loc[((date_df['year'] == start_y)&(date_df['month']==start_m)&(date_df['day']<start_d)), "plot_{}".format(i)] = False
                    date_df.loc[((date_df['year'] == end_y)&(date_df['month']==end_m)&(date_df['day']>end_d)), "plot_{}".format(i)] = False

                    # Get the row index that want to highlight
                    tmp_plots = plots[date_df["plot_{}".format(i)]==True]
                    tmp_plots['index'] = tmp_plots.index
                    
                    # Sorting by 'value'
                    sort_tmp_plots = tmp_plots.sort_values(by="value", ascending=True)

                    # Filtering a date data with trend reference.
                    ## If the trend feature is in 'increasing trend', find the increasing range of data.
                    max_idx, min_idx = sort_tmp_plots.iloc[-1, 2], sort_tmp_plots.iloc[0, 2]
                    # print("max, min\n", max_idx, min_idx)
                    ## For the case mention: [False, True], [True, False]
                    if dates[0] == "TBD" and feature == 'trend:+': # until
                        max_idx = tmp_plots.iloc[-1, 2]
                        max_idx = tmp_plots.loc[date_df['year'] == end_y].sort_values(by = "value", ascending = True).iloc[-1, 2]
                    elif dates[0] == "TBD" and feature == 'trend:-':
                        min_idx = tmp_plots.iloc[-1, 2]
                        min_idx = tmp_plots.loc[date_df['year'] == end_y].sort_values(by = "value", ascending = True).iloc[0, 2]
                    elif dates[1] == "TBD" and feature == 'trend:+': # since
                        min_idx = tmp_plots.iloc[0, 2] # max point should be a start point
                        min_idx = tmp_plots.loc[date_df['year'] == start_y].sort_values(by = "value", ascending = True).iloc[0, 2]
                    elif dates[1] == "TBD" and feature == 'trend:-':
                        max_idx = tmp_plots.iloc[0, 2] # min point should be a start point
                        max_idx = tmp_plots.loc[date_df['year'] == start_y].sort_values(by = "value", ascending = True).iloc[-1, 2]

                    if feature == 'trend:+':
                        if min_idx < max_idx: 
                            self.ref_result[i]['references'][j]['factCheck'] = True
                            fin_tmp_plots = tmp_plots[(tmp_plots['index']>=min_idx) & (tmp_plots['index']<=max_idx)]

                            if len(fin_tmp_plots) == 1:
                                self.ref_result[i]['references'][j]['chart'] = fin_tmp_plots.iloc[0]["date"]
                            else: self.ref_result[i]['references'][j]['chart'] = [fin_tmp_plots.iloc[0, 0], fin_tmp_plots.iloc[-1, 0]]
                        else:
                            if len(tmp_plots) == 1: 
                                self.ref_result[i]['references'][j]['chart'] = tmp_plots.iloc[0]['date']
                            else: self.ref_result[i]['references'][j]['chart'] = [tmp_plots.iloc[0, 0], tmp_plots.iloc[-1, 0]]
                    elif feature == 'trend:-':
                        if max_idx < min_idx: 
                            self.ref_result[i]['references'][j]['factCheck'] = True
                            fin_tmp_plots = tmp_plots[(tmp_plots['index']<=min_idx) & (tmp_plots['index']>=max_idx)]

                            if len(fin_tmp_plots) == 1:
                                self.ref_result[i]['references'][j]['chart'] = fin_tmp_plots.iloc[0]["date"]
                            else: self.ref_result[i]['references'][j]['chart'] = [fin_tmp_plots.iloc[0, 0], fin_tmp_plots.iloc[-1, 0]]
                        else:
                            if len(tmp_plots) == 1: 
                                self.ref_result[i]['references'][j]['chart'] = tmp_plots.iloc[0]['date']
                            else: self.ref_result[i]['references'][j]['chart'] = [tmp_plots.iloc[0, 0], tmp_plots.iloc[-1, 0]]
                    elif feature == "extremum:+":
                        self.ref_result[i]['references'][j]['factCheck'] = True
                        self.ref_result[i]['references'][j]['chart'] = sort_tmp_plots.iloc[-1, 0]
                    elif feature == "extremum:-":
                        self.ref_result[i]['references'][j]['factCheck'] = True
                        self.ref_result[i]['references'][j]['chart'] = sort_tmp_plots.iloc[0, 0]
                    else:
                        if len(tmp_plots) == 1: 
                                self.ref_result[i]['references'][j]['chart'] = tmp_plots.iloc[0]['date']
                        else: self.ref_result[i]['references'][j]['chart'] = [tmp_plots.iloc[0, 0], tmp_plots.iloc[-1, 0]]
                except IndexError: print("IndexError"); continue
                except KeyError: print("KeyError!"); continue
                except ValueError: print("ValueError"); continue

        for (i, j) in remove_list:
            self.ref_result[i]['references'].remove(self.ref_result[i]['references'][j])

        print("---\nFinal result")
        print(self.ref_result)

def detokenize(all_tokens: List[str], all_texts: List[str])-> List[str]: 
    for i, (token, texts) in enumerate(zip(all_tokens, all_texts)):
        if i==0: continue
        if re.search("[#]{2}\w+", token):
            agg_word = all_tokens[i-1] + all_tokens[i].replace("#", "")

            if re.search(agg_word, texts):
                all_tokens[i-1], all_tokens[i] = agg_word, agg_word
    return all_tokens

def find_ch_span(word, sentence, sent_span=None, count_dict=None):
    word = word.lower().strip()
    try:
        if sent_span != None and count_dict != None : # for token

            for idx, each_match in enumerate(re.finditer(rf"{word}\W?\s", sentence.replace(".", " "))):
                each_span = each_match.span()

                if len(re.findall(rf"{word}\W?\s", sentence.replace(".", " "))) > count_dict[word]: ## January 2008, 2008 sequentially
                    start_idx = each_span[0]; end_idx = each_span[1]
                    if (each_span[0] >= sent_span[0] and each_span[1] <= sent_span[1]+1):
                        if sentence[start_idx:end_idx-1][-1] == "." or sentence[start_idx:end_idx-1][-1] == ",":
                            each_span = (start_idx, end_idx-2)
                        else:
                            each_span = (start_idx, end_idx-1)

                        return each_span
                
                if word in count_dict: # If it is token, not a phrase(e.g. between A and B)
                    start_idx = each_span[0]; end_idx = each_span[1]
                    if idx == count_dict[word] and (each_span[0] >= sent_span[0] and each_span[1] <= sent_span[1]+1): # when 'peaked' in first sentence but want to detect 'peak' in second sentence.
                        
                        start_idx = each_span[0]; end_idx = each_span[1]
                        if sentence[start_idx:end_idx-1][-1] == "." or sentence[start_idx:end_idx-1][-1] == ",":
                            each_span = (start_idx, end_idx-2)
                        else:
                            each_span = (start_idx, end_idx-1)
                        return each_span
                else:
                    return each_span
        else:
            match = re.search(word, sentence)
            return match.span()
    except: 
        print("Can't find span")
        return None
    
if __name__ == '__main__':
    print("This is the main function for testing.")