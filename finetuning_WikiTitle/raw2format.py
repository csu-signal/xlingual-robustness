#!/usr/bin/env python
# encoding: utf-8

import json
import io
import re
import argparse

def raw2format(lang,directory):
    inFile = directory+'/' + lang + '_raw.txt'
    outFile = directory+'/'  + lang + '.txt'
    fo = open(outFile, 'w')
    category_count = {}
    pattern = re.compile('.*:.*')
    with io.open(inFile, 'r', encoding='utf-8') as fi:
        lines = fi.readlines()
        for i, line in enumerate(lines):
            data = json.loads(line.rstrip('\n'))
            #for i in xrange(len(data)):
            count = 0
            for d in data:
                #print d["title"]
                if not pattern.match(data[d]['title']):
                    if data[d]["category"] not in category_count:
                        category_count[data[d]["category"]] = 0
                    category_count[data[d]["category"]] += 1
                    fo.write(str(data[d]["category"]).decode('utf-8'))
                    fo.write("\t".decode('utf-8'))
                    fo.write(data[d]["title"].encode('utf8'))
                    fo.write("\n".decode('utf-8'))
                    #pid.add(d["pageid"])
                    count += 1
    print("category distribution for " + lang + " ")
    print(category_count)
    fo.close() 