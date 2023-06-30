# AIFFEL Campus Online 4th Code Peer Review Templete
- 코더 : 김다인
- 리뷰어 : 소용현


# PRT(PeerReviewTemplate)
각 항목을 스스로 확인하고 토의하여 작성한 코드에 적용합니다.
- [o] 1.코드가 정상적으로 동작하고 주어진 문제를 해결했나요?
```
# 명사 추출
okt = Okt()
tokenized = []    # corpus 저장
with open(os.getenv('HOME')+'/aiffel/weat/synopsis.txt', 'r') as file:
    while True:
        line = file.readline()
        if not line: break
        words = okt.pos(line, stem=True, norm=True)
        res = []
        for w in words:
            if w[1] in ["Noun"]:      # "Adjective", "Verb" 등을 포함할 수도 있습니다.
                res.append(w[0])    # 명사일 때만 tokenized 에 저장하게 됩니다. 
        tokenized.append(res)
```
```
model = Word2Vec(tokenized, vector_size=100, window=5, min_count=3, sg=0)
model_result = model.wv.most_similar(positive=['여행'])
print(model_result)
```
형태소분석으로 명사를 추출하고, Word2Vec 임베딩을 수행했다.   
  
tfidf결과 중복제거하여 대표단어를 추출하고, weat를 구했다.  


- [o] 2.주석을 보고 작성자의 코드가 이해되었나요?
```
# 텍스트 데이터를 TF-IDF 벡터로 변환
vectorizer = TfidfVectorizer()
# 텍스트 데이터에 대한 학습과 변환
X = vectorizer.fit_transform([art, gen])
```
tfidf를 이용하여 대표단어를 추출한 과정이 주석으로 잘 기록되어 있다.
- [x] 3.코드가 에러를 유발할 가능성이 있나요?
```
# w1에만 있고 w2에는 없는, 예술영화를 잘 대표하는 단어를 15개 추출
target_art, target_gen = [], []
for i in range(100):
    if (w1_[i] not in w2_) and (w1_[i] in model.wv): 
        target_art.append(w1_[i])
        
    if len(target_art) == n: 
        break 
```
word2vec모델에 해당단어가 있는지 확인하여 에러 유발을 막았다.
- [o] 4.코드 작성자가 코드를 제대로 이해하고 작성했나요?
```
def read_token(file_name):
    okt = Okt()
    result = []
    with open(os.getenv('HOME')+'/aiffel/weat/'+file_name, 'r') as fread: 
        print(file_name, '파일을 읽고 있습니다.')
        while True:
            line = fread.readline() 
            if not line: break 
            tokenlist = okt.pos(line, stem=True, norm=True) 
            for word in tokenlist:
                if word[1] in ["Noun"]:#, "Adjective", "Verb"]:
                    result.append((word[0])) 
    return ' '.join(result)
```
형태소 분석 후 토크나이즈를 이해하고 있다.
- [o] 5.코드가 간결한가요?
```
for i in range(len(genre_name)-1):
    for j in range(i+1, len(genre_name)):
        print(genre_name[i], genre_name[j],matrix[i][j])
```
포문을 활용하여 간결하게 표현하였다.
