from collections import Counter

from konlpy.corpus import kolaw
from konlpy.tag import Okt

okt = Okt();
doc = kolaw.open('constitution.txt').read()
# print(doc)
result = okt.nouns(doc)
print(result)
cnt = Counter(result)
mstr = cnt.most_common(20)
dstr = dict(mstr)
print(dstr)