아이펠캠퍼스 온라인4기 피어코드리뷰 [23.05.25]

- 코더 : 김다인
- 리뷰어 : 이동익

----------------------------------------------

**PRT(PeerReviewTemplate)**

* [O] 코드가 정상적으로 동작하고 주어진 문제를 해결했나요?
  - 네 정상적으로 동작합니다.
  - 텍스트 전처리 및 lstm+attention 모델 설계를 진행하였습니다.
  - early stopping을 주고 모델을 학습하고 loss감소를 확인했습니다.
  - 핵심단어가 포함된 예측 요약을 생성해냈습니다.

* [O] 주석을 보고 작성자의 코드가 이해되었나요?
  - 스텝별로 주석이 달려 있어서 이해하기에 편했습니다.

* \[X\] 코드가 에러를 유발할 가능성이 있나요?
  - 없습니다.
  
* [O] 코드 작성자가 코드를 제대로 이해하고 작성했나요? (직접 인터뷰해보기)
  - lstm의 구조와 어텐션 층의 연결에 대해 이해하고, keras 라이브러리 이외에 타 github의 attention함수를 적용해보셨습니다.

* [O] 코드가 간결한가요?
  - 대체적으로 간결합니다.


----------------------------------------------

참고 링크 및 코드 개선
```python
#동일한 방법으로 테스트
for i in range(50,53):
    _text = data.iloc[i,1]
    print("원문 :", _text)
    print("실제 요약 : ", data.iloc[i,0])
    print("예측 요약 : ", summarize(_text, ratio=0.35))
    print("\n")

```
> 원문 : former finance minister yashwant sinha tuesday demanded probe alleged diversion loans worth crore dewan housing finance agencies including regulators government failed track nefarious deals said comes media report tuesday accused dhfl controlling shareholders diverting funds shell companies buy assets
> 실제 요약 :  yashwant sinha demands probe into alleged fund diversion by dhfl
> 예측 요약 :
> - 원문의 문장을 하나로 인식해서 summa의 출력이 나오지 않는 것 같습니다. 전처리 되지 않은 텍스트를 사용하거나
>  print(summarize(_text, words=20))와 같은 형태로 변경하면 될 것 같습니다.


