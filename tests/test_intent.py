from generation.intent_classifier import IntentClassifier, RequestType

def test_intents():
    classifier = IntentClassifier()
    assert classifier.classify("یک سناریوی آموزشی بساز") is RequestType.SCENARIO
    assert classifier.classify("سوال ارزیابی چهارگزینه‌ای بساز") is RequestType.QUIZ
    assert classifier.classify("کمک‌های اولیه چیست") is RequestType.QA
