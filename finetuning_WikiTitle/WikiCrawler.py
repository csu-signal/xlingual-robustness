#!/usr/bin/env python
# encoding: utf-8
import requests
import urllib.request
import urllib
import json
import argparse
import sys
from wikiextractor.extract import Extractor as ex
from io import StringIO
import re
import pandas as pd
import random
import numpy as np
import pickle
import csv
import os
import warnings
import concurrent.futures
import string
ex.to_json = True


def load_page(page,base):
    try:
        base_url=base[:base.index('/w/api.php')]+'/wiki?curid={}'
        pageid=str(page['pageid'])

        title=page['title']
#         print(title)
        
        pretext=requests.get(base,params={'format':'json','action':'query','prop':'revisions','titles':title,'rvprop':'content'}).json()
        pretext=pretext['query']['pages'][pageid]['revisions'][0]['*']

        ex.HtmlFormatting=True
        tmp_txt=ex(pageid, '', base, title,pretext).extract()
        res = json.loads(tmp_txt)
        text=res['text']
        html_pattern="<h\d>(.*?)</h[2,3]>"
        html_titles=re.findall(html_pattern, text)
#         print(html_titles)
        
        ex.HtmlFormatting=False
        tmp_txt=ex(pageid, '', base, title,pretext).extract()
        res = json.loads(tmp_txt)
        text=res['text']

        lines=text.split("\n")
        title_ids=[]
        section_titles=[]
        for t in html_titles:
            if(t+'.' in lines):
                section_titles.append(t)
                title_ids.append(lines.index(t+'.'))
#         print(section_titles)
        title_ids.append(-1)
        sections_df=pd.DataFrame(columns=['title','text'])
        for t_id,t in enumerate(section_titles):
            section_text="".join(lines[title_ids[t_id]+1:title_ids[t_id+1]])
            if(len(section_text)>200):
                sections_df.loc[len(sections_df.index)]=[t,section_text]
        
        u_num=len(np.unique(np.array(sections_df['title'])))
        if(len(np.unique(np.array(section_titles)))>=4 and u_num>0):
            infos=[]
            
            random_list=list(range(u_num))
            for potential_section in range(min(u_num,4)):
                selected_id=random.choice(random_list)
                selected_title,selected_section=sections_df.loc[selected_id]
                info=[selected_section]
                non_correct_titles=random.sample(set(section_titles)-{selected_title},3)
                non_correct_titles.append(selected_title)
                random.shuffle(non_correct_titles)
                info.extend(non_correct_titles)
                info.append(string.ascii_letters[26+non_correct_titles.index(selected_title)])
                info.append(base_url.format(pageid))
                random_list=list(set(random_list)-{selected_id})
                infos.append(info)
            return infos
    except:
        print("An exception occurred")


class Crawl:
    def __init__(self,lang,limit,dataset_size,directory,start_id):
        self.lang=lang
        self.limit=limit
        self.dataset_size=dataset_size
        self.directory=directory
        self.start_id=start_id
        self.base= 'https://{}.wikipedia.org/w/api.php'.format(lang)
    
    def pages_crawl(self):
        if not os.path.isfile(self.directory+self.lang+"/pages.pickle"):
            
            base_url=self.base[:self.base.index('/w/api.php')]+'/wiki?curid={}'
            request={'list':'allpages','apminsize':self.limit,'aplimit':'max'}
            request['action'] = 'query'
            request['format'] = 'json'
            lastContinue = {}
            
            pages=[]
            while True:
                # Clone original request
                req = request.copy()
                # Modify it with the values returned in the 'continue' section of the last result.
                req.update(lastContinue)
                result=requests.get(self.base,params=req).json()
                if 'error' in result:
                    raise Error(result['error'])
#                 if 'warnings' in result:
#                     print(result['warnings'])
        #         if 'query' in result:
        #             yield result['query']
                if 'continue' not in result:
                    break
                pages.extend(result['query']['allpages'])
#                 print(len(pages))
                lastContinue = result['continue']

            with open(self.directory+self.lang+'/pages.pickle', 'wb') as f:
                pickle.dump(pages, f)
        else:
            with open(self.directory+self.lang+"/pages.pickle",'rb') as f:
                pages = pickle.load(f)
                print("pages exist")
        return pages




    def crawl(self):
        base_url=self.base[:self.base.index('/w/api.php')]+'/wiki?curid={}'
        lastContinue = {}
        dataset=pd.DataFrame(columns=['sectionText','titleA','titleB','titleC','titleD','correctTitle','url'])

        pages=self.pages_crawl()
#         print(len(pages))

        dataset_len=0
        if os.path.isfile(self.directory+self.lang+"/title_dataset.csv"):
            dataset_file = csv.writer(open(self.directory+self.lang+"/title_dataset.csv",'a'))
            pre_dataset = pd.read_csv(self.directory+self.lang+"/title_dataset.csv", sep=',', encoding='utf-8')
            dataset_len=len(pd.DataFrame(pre_dataset).index)

        else:
            dataset_file = csv.writer(open(self.directory+self.lang+"/title_dataset.csv",'w'))
            dataset_file.writerow(dataset.columns)

        with concurrent.futures.ProcessPoolExecutor() as executor:

            for infos in(executor.map(load_page,pages[self.start_id:],[self.base]*len(pages[self.start_id:]))):
                if(infos!=None):
                    for info in infos:
                        dataset_file.writerow(info)
                        dataset.loc[len(dataset.index)]=info

                
                if(len(dataset.index)>=(self.dataset_size-dataset_len)):
                    break

        return(dataset)

