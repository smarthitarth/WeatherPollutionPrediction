# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 22:09:59 2020

@author: hitarth
"""


# -*- coding: utf-8 -*-
"""
Created on Mon Aug 17 01:54:18 2020

@author: hitarth
"""
import os
import time
import requests
import sys

def retrieve_csv():
    for year in range(2012, 2017):
        #This URL will give html format files
        #url="http://www.airqualityontario.com/history/index.php?s=29000&y={}&p=124&m=1&e=12&t=html&submitter=Search&i=1".format(year)
        #this url is for getting direct CSV files for PM2 AQI
        url="http://www.airqualityontario.com/history/index.php?s=29000&y={}&p=124&m=1&e=12&t=csv&submitter=Search&i=1".format(year)
        texts = requests.get(url)
        text_utf = texts.text.encode('utf=8')
        
        if not os.path.exists("Data/html_data/hamilton/AQI/"):
            os.makedirs("Data/html_data/hamilton/AQI/")
        with open("Data/html_data/hamilton/AQI/{}.csv".format(year),"wb") as output:
            output.write(text_utf)
            
    sys.stdout.flush()
    
if __name__=="__main__":
    start_time = time.time()
    retrieve_csv()
    stop_time = time.time()
    print("Time taken {}".format(stop_time-start_time))