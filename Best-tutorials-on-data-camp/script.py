# In this tutorial, I am going to scrape the tutorials section of the DataCamp website 
# and try to get some insights.

# import important libraries

from bs4 import BeautifulSoup
from urllib.request import urlopen
import matplotlib.ticker as ticker
import matplotlib.dates as mdates
from datetime import datetime
import time
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import itertools

# specifiying the URL to scrape

url = 'https://www.datacamp.com/community/tutorials?page=2'

# we need to identify how many pages we can query 
# we loop over and find all a-tags and return their number.
html = urlopen(url)
soup = BeautifulSoup(html, 'html.parser') # add html.parser feature to surpass warning in the terminal

page = [i.text for i in soup.find_all('a') if 'community/tutorials?page=' in str(i)]
last_page = page[-1]

print(last_page)
# 25

# for each card, we have 

# tag, a, description, author, upvote, social-media, date
# I will initialize then as an empty array

tag = []
link  = []
title =  []
description = []
author = []
date = []
upvotes = []

for page in np.arange(1, int(last_page)+1):
    url = 'https://www.datacamp.com/community/tutorials?page=2' + str(page)
    html = urlopen(url)
    soup = BeautifulSoup(html, 'html.parser')
    tag.append([i.text for i in soup.find_all(class_='jsx-1764811326 title')])
    title.append([i.text for i in soup.find_all(class_='jsx-379356511 blue')])
    link.append(i.text for i in soup.find_all('href'))
    description.append([i.text for i in soup.find_all(class_='jsx-379356511 blocText description')])
    author.append([i.text for i in soup.find_all(class_='jsx-566588255 name')])
    date.append([i.text for i in soup.find_all(class_='jsx-566588255 date')])
    upvotes.append([i.text for i in soup.find_all(class_='jsx-1972554161 count')])

chain_object = itertools.chain.from_iterable(description)
desc_flatted = list(chain_object)
print(desc_flatted)
