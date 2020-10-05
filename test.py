
import requests
from bs4 import BeautifulSoup

code = "028300"

url = 'https://navercomp.wisereport.co.kr/v2/company/ajax/cF1001.aspx?cmp_cd=004990&fin_typ=0&freq_typ=Q&encparam=VG5ybFRrbFBRRDFhQ1BORG1sWHZyZz09&id=QmZIZ20rMn'

res = requests.get(url)

bs = BeautifulSoup(res.text, 'html.parser')
data = bs.select("tbody tr")
# print(data)
issued = data[-1]
print(issued)
# title = issued.select("tr th")[0].text
_1906 = issued.select("td")[0].text
_1909 = issued.select("td")[1].text
_1912 = issued.select("td")[2].text
_2003 = issued.select("td")[3].text
_2006 = issued.select("td")[4].text

# print(title)
print(_1906)
print(_1909)
print(_1912)
print(_2003)
print(_2006)

