import requests
import urllib.request
from bs4 import BeautifulSoup
from urllib.request import urlopen
import textwrap
import re


search_term = input('Please enter the search term for Wikipedia: ')
url = 'https://en.wikipedia.org/wiki/' + search_term.replace(' ','_')
print(url) 

html = urlopen(url)
soup = BeautifulSoup(html, 'html.parser')
toc_tag=soup.find("div", {"id": "toc"})

dic={}
for e in toc_tag.descendants:
  if (e.name == "li"):
    link =e.find("a")
    tocurl = link.get("href")
    tocnum = link.find("span", {"class", "tocnumber"})
    toctext = link.find("span", {"class", "toctext"})
    print(tocnum.contents, end=" ")
    print(toctext.contents, end=" ")
    print(tocurl)
    dic[tocnum.contents[0]]=toctext.contents


subsecno = input("Enter section number: ")


def print_sec(subsec):
  subsec = subsec.replace(" ", "_")
  subspan = soup.find("span", {"id": subsec})
  hed = subspan.parent

  para_text = hed.next_siblings

  start = 0
  for i in para_text:
    if (i.name == "p"):
      my_wrap = textwrap.TextWrapper(width = 70)
      wrapped = my_wrap.wrap(text=i.getText())
      for j in wrapped:
        print(j)
      start = 1
    else:
      if (start ==1):
        break
      else:
        continue 

lis=dic.keys()
arr=[]
sorted(lis)
if subsecno not in lis:
  print('Wrong section number')
  exit()
for l in lis:
  if re.match(subsecno,l):
    print('\n'+l+': '+dic[l][0])
    print_sec(dic[l][0])
  else:
    break
