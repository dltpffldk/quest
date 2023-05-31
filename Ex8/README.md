# 아이펠캠퍼스 온라인4기 피어코드리뷰

- 코더 : 김다인
- 리뷰어 : 부석경

---------------------------------------------
## **PRT(PeerReviewTemplate)**

### **[⭕] 코드가 정상적으로 동작하고 주어진 문제를 해결했나요?**
![image](https://github.com/JeJuBOO/Aiffel_Nodes/assets/71332005/03e2fa12-1aa0-47ab-b2a8-b8b921597532)   
* Epoch 40번을 실행한 결과로 질문에 대한 분위기를 파악한 느낌이 듭니다.

### **[⭕] 주석을 보고 작성자의 코드가 이해되었나요?**
```python
def decoder_inference(sentence):
  sentence = preprocess_sentence(sentence)#훈련셋과 맞춰주기 위해 입력 문장도 전처리를 거칩니다. 

  # 입력된 문장을 정수 인코딩 후, 시작 토큰과 종료 토큰을 앞뒤로 추가.
  # ex) Where have you been? → [[8331   86   30    5 1059    7 8332]]
  sentence = tf.expand_dims(
      START_TOKEN + tokenizer.encode(sentence) + END_TOKEN, axis=0)

  # 디코더의 현재까지의 예측한 출력 시퀀스가 지속적으로 저장되는 변수.
  # 처음에는 예측한 내용이 없음으로 시작 토큰만 별도 저장. ex) 8331
  output_sequence = tf.expand_dims(START_TOKEN, 0)

  # 디코더의 인퍼런스 단계
  for i in range(MAX_LENGTH):
    # 디코더는 최대 MAX_LENGTH의 길이만큼 다음 단어 예측을 반복합니다.
    predictions = model(inputs=[sentence, output_sequence], training=False)
    predictions = predictions[:, -1:, :]

    # 현재 예측한 단어의 정수
    predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

    # 만약 현재 예측한 단어가 종료 토큰이라면 for문을 종료
    if tf.equal(predicted_id, END_TOKEN[0]):
      break

    # 예측한 단어들은 지속적으로 output_sequence에 추가됩니다.
    # 이 output_sequence는 다시 디코더의 입력이 됩니다.
    output_sequence = tf.concat([output_sequence, predicted_id], axis=-1)

  return tf.squeeze(output_sequence, axis=0)
```
위와 같이 모든 행동에 대한 주석이 달려있어 코드를 한 줄씩 이해하기 쉬웠습니다.
### **[❌] 코드가 에러를 유발할 가능성이 있나요?**
못 찾았습니다.
### **[⭕] 코드 작성자가 코드를 제대로 이해하고 작성했나요?** (직접 인터뷰해보기)
* 네. 데이터 프레임을 shuffle할 때 `BUFFER_SIZE`가 설정되는데 이에 대해 질문했고, 잘 답변해 주었습니다.
```python
  inputs = tf.keras.Input(shape=(None,), name='inputs')
  enc_outputs = tf.keras.Input(shape=(None, d_model), name='encoder_outputs')
  look_ahead_mask = tf.keras.Input(
      shape=(1, None, None), name='look_ahead_mask')
```
* 모델 뒤에 name으로 레이어의 이름을 붙여주는것이 꼭 필요한 작업이냐에 대하여 질문했고 ,[tensorflow](https://www.tensorflow.org/guide/keras/sequential_model?hl=ko)를 보며 설명해 주었습니다. 레이어에서 한레이어의 특성을 추출하거나 할때 쓰인다고 답변해주었습니다.
### **[⭕] 코드가 간결한가요?**
```python
def load_conversations():
    inputs, outputs = [], []
    for sentence in train_data['Q']:
        inputs.append(preprocess_sentence(sentence))
    for sentence in train_data['A']:
        outputs.append(preprocess_sentence(sentence))
    return inputs, outputs 
```
```python
# 질문과 답변의 쌍인 데이터셋을 구성하기 위한 데이터 로드 함수
def load_conversations(data):
    
    question = list(map(preprocess_sentence,data['Q']))
    answer = list(map(preprocess_sentence,data['A']))

    return question, answer
```
* `map`함수도 적극 활용하셨으면 좋을 것 같습니다.
----------------------------------------------
### **참고 링크 및 코드 개선**
* 코드 리뷰 시 참고한 링크가 있다면 링크와 간략한 설명을 첨부합니다.
* 코드 리뷰를 통해 개선한 코드가 있다면 코드와 간략한 설명을 첨부합니다.

----------------------------------------------
