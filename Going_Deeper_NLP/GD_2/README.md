# Code Peer Review Templete
- 코더 : 김다인
- 리뷰어 : 김용석



# PRT(PeerReviewTemplate)
각 항목을 스스로 확인하고 체크하고 확인하여 작성한 코드에 적용하세요.
- [O] 1.코드가 정상적으로 동작하고 주어진 문제를 해결했나요?
  : 코드는 정상적으로 동작하고 있고 퀘스트에서 주어진 문제들도 잘 해결되었습니다.
- [O] 2.주석을 보고 작성자의 코드가 이해되었나요?
  : 기본 노드를 바탕으로 코드가 작성되어 이해하기 쉬웠으며,제가 벤치마킹할 필요성을 느낀 부분을 찾았습니다. 
- [x] 3.코드가 에러를 유발한 가능성이 있나요?
  : 없었습니다. 
- [O] 4.코드 작성자가 코드를 제대로 이해하고 작성했나요?
  : 몇 가지 질문에 정확하게 답변을 해주셨으며, 코드에 대한 정확한 동작을 이해하고 있었습니다. 
- [O] 5.코드가 간결한가요?
  : 매우 간결하고 이해하기 쉽게 작성되었습니다.



# 참고 링크 및 코드 개선 여부

# 보팅

print("[보팅]")

voting_classifier = VotingClassifier(estimators=[
('lr', LogisticRegression(C=10000, penalty='l2')),
('cb', ComplementNB()),
('grbt', GradientBoostingClassifier(random_state=0))
], voting='soft', n_jobs=-1)


voting_classifier.fit(tfidfv, y_train)


predicted = voting_classifier.predict(tfidfv_test) #테스트 데이터에 대한 예측
print("정확도:", accuracy_score(y_test, predicted)) #예측값과 실제값 비교
print("\n")
print("[보팅의 Classification Report]")
print(classification_report(y_test, voting_classifier.predict(tfidfv_test)))

▶ 보팅 정확도: 0.808993766696349
▶ 모델 정확도가 가장 높게 작성되어 있어 좋았습니다. 
