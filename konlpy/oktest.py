from konlpy.tag import Okt

okt = Okt()

text = '한글 자연어 처리는 재밌어 이제부터 시작해 볼까나'

print('-----------------------------------------------')
print(okt.morphs(text));
print(okt.morphs(text, stem = True));
print('-----------------------------------------------')
print(okt.nouns(text));
print(okt.phrases(text));
print('-----------------------------------------------')
print(okt.pos(text));
print(okt.pos(text, join = True));
print(okt.pos(text, stem=True, join = True));
