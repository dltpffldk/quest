# Code Peer Review Templete
- 코더 : 김다인
- 리뷰어 : 김동규


# PRT(PeerReviewTemplate)
각 항목을 스스로 확인하고 체크하고 확인하여 작성한 코드에 적용하세요.
- [ ] 1.코드가 정상적으로 동작하고 주어진 문제를 해결했나요?
- [ ] 2.주석을 보고 작성자의 코드가 이해되었나요?
- [ ] 3.코드가 에러를 유발한 가능성이 있나요?
- [ ] 4.코드 작성자가 코드를 제대로 이해하고 작성했나요?
- [ ] 5.코드가 간결한가요?


# basis for evaluation
## 1st section
### Preprocessing is done.
missing data -> duplicated -> length-based filter
```py
# 중복 및 결측치 제거
train_data.drop_duplicates(subset=['document'], inplace=True)
train_data = train_data.dropna(how = 'any') 
test_data.drop_duplicates(subset=['document'], inplace=True)
test_data = test_data.dropna(how = 'any') 

print('train 데이터 사이즈:', len(train_data))
print('test 데이터 사이즈:', len(test_data))
```

This one is not perfact but it is okay.
```py
filtered_corpus = [s for s in cleaned_corpus if (len(s) < max_len) & (len(s) >= min_len)]
```

### Using sentencepiece
She selected model. 
It means she already understand the command of sentencepiece

```py
# 디폴트 --model_type = 'unigram' <-- from reviewer: this one
spm.SentencePieceTrainer.Train(
    '--input={} --model_prefix=naver_review_spm --vocab_size={}'.format(temp_file, vocab_size)    
)

# --model_type = 'bpe' <-- from reviewer: this one
spm.SentencePieceTrainer.Train(
    '--input={} --model_type=bpe --model_prefix=naver_review_spm_bpe --vocab_size={}'.format(temp_file, vocab_size)    
)
```
### tokenization was done

```py
# Unigram 타입
def sp_tokenize_uni(s_uni, corpus):

    tensor = []

    for sen in corpus:
        tensor.append(s_uni.EncodeAsIds(sen))

    with open("./naver_review_spm.vocab", 'r') as f: # unigram 타입 vocab
        vocab = f.readlines()

    word_index = {}
    index_word = {}

    for idx, line in enumerate(vocab):
        word = line.split("\t")[0]

        word_index.update({idx:word})
        index_word.update({word:idx})

    tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor, padding='pre', maxlen = 100)

    return tensor, word_index, index_word

my_corpus = ['나는 밥을 먹었습니다.', '그러나 여전히 ㅠㅠ 배가 고픕니다...']
tensor_uni, word_index_uni, index_word_uni = sp_tokenize_uni(s_uni, my_corpus)
print(tensor_uni)

tensor_uni, word_index_uni, index_word_uni = sp_tokenize_uni(s_uni, data)
print(tensor_uni.shape)
```

### train was also done, and accuracy is okay

the log of train
```
Epoch 26/30
188/188 [==============================] - 2s 8ms/step - loss: 0.0920 - accuracy: 0.9724 - val_loss: 0.7903 - val_accuracy: 0.8159
Epoch 27/30
188/188 [==============================] - 2s 8ms/step - loss: 0.0892 - accuracy: 0.9737 - val_loss: 0.8032 - val_accuracy: 0.8148
Epoch 28/30
188/188 [==============================] - 2s 8ms/step - loss: 0.1011 - accuracy: 0.9688 - val_loss: 0.7633 - val_accuracy: 0.8169
Epoch 29/30
188/188 [==============================] - 2s 8ms/step - loss: 0.0790 - accuracy: 0.9783 - val_loss: 0.8364 - val_accuracy: 0.8166
Epoch 30/30
188/188 [==============================] - 2s 8ms/step - loss: 0.0737 - accuracy: 0.9800 - val_loss: 0.8743 - val_accuracy: 0.8159
```

1537/1537 - 4s - loss: 0.9056 - accuracy: 0.8090

### At last, she visualized the result from the history of training.
See the ipynb

## 2nd section

Comment is nice and kind.

```py
temp_file = os.getenv('HOME')+'/aiffel/Going_Deeper_NLP/GD_NLP_1/sp_tokenizer/data/korean-english-park.train.ko.temp'

vocab_size = 8000

with open(temp_file, 'w') as f:
    for row in filtered_corpus:   # 이전 스텝에서 정제했던 corpus를 활용
        f.write(str(row) + '\n')

# 디폴트 --model_type = 'unigram'
spm.SentencePieceTrainer.Train(
    '--input={} --model_prefix=naver_review_spm --vocab_size={}'.format(temp_file, vocab_size)    
)

# --model_type = 'bpe'
spm.SentencePieceTrainer.Train(
    '--input={} --model_type=bpe --model_prefix=naver_review_spm_bpe --vocab_size={}'.format(temp_file, vocab_size)    
)

!ls -l naver_review_spm*
```

## 3rd possibility of error

She blocked path issue by using os.getenv function.

```py
temp_file = os.getenv('HOME')+'/aiffel/Going_Deeper_NLP/GD_NLP_1/sp_tokenizer/data/korean-english-park.train.ko.temp'

```

# 예시
1. 코드의 작동 방식을 주석으로 기록합니다.
2. 코드의 작동 방식에 대한 개선 방법을 주석으로 기록합니다.
3. 참고한 링크 및 ChatGPT 프롬프트 명령어가 있다면 주석으로 남겨주세요.
```
```

# 참고 링크 및 코드 개선 여부
```python
#
#
#
#
```
