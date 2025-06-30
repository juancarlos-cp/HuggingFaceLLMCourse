import torch
from transformers import pipeline, pipelines
# what if we want to do it without a pipeline?
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification


checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
classifier = pipeline("sentiment-analysis", model=checkpoint)

# response = classifier("I've been waiting for a Huggingface course my whole life.")
# print(response)

raw_inputs = [
    "I've been waiting for a HuggingFace course my whole life.", 
    "I hate this so much!",
    ]
my_inputs = [
    "Listen Jesus I don't like what I see.", 
    "All I ask is that you listen to me.",
    ]

responses = classifier(raw_inputs)
print(responses)
responses = classifier(my_inputs)
print(responses)


def sentiment_analysis_no_pipeline(checkpoint, raw_inputs):

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    inputs = tokenizer(raw_inputs, padding=True, truncation=True, return_tensors="pt")
    # print(inputs)

    model = AutoModel.from_pretrained(checkpoint)

    outputs = model(**inputs)
    print(outputs.last_hidden_state.shape)

    model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
    print(model.config.id2label)

    outputs = model(**inputs)
    print(outputs.logits.shape)
    print(outputs.logits)

    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    print(predictions)

sentiment_analysis_no_pipeline(checkpoint, raw_inputs)
sentiment_analysis_no_pipeline(checkpoint, my_inputs)

'''
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
responses = classifier(
    "This is a course about the Transformers library",
    candidate_labels=["education", "politics", "business"],
)

print(responses)

responses = classifier(
    "My physical therapist's appointments are all booked.",
    candidate_labels=["literature", "medicine", "sports"],
)

print(responses)

generator = pipeline("text-generation", model="HuggingFaceTB/SmolLM2-360M")
responses = generator(
    "In this course, we will teach you how to",
    max_length=30,
    truncation=True,
    num_return_sequences=2,
    do_sample=True,
)

print(responses)

responses = generator(
    "For breakfast, we are serving",
    max_length=60,
    truncation=True,
    num_return_sequences=2,
    do_sample=True,
)

print(responses)


unmasker = pipeline("fill-mask", model="distilroberta-base")
response = unmasker("This course will teach you all about <mask> models.", top_k=2)

print(response)

response = unmasker("I am going to need a <mask> to take care of this leak.", top_k=2)

print(response)


ner = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english",
               aggregation_strategy=pipelines.token_classification.AggregationStrategy.SIMPLE)
responses = ner("My name is George. I'm unemployed and I live with my parents.")

print(responses)

text = """

"I've talked to a lot of CISOs at Fortune 500 companies, and nearly every one 
that I've spoken to about the North Korean IT worker problem has admitted 
they've hired at least one North Korean IT worker, if not a dozen or a few 
dozen," Charles Carmakal, chief technology officer at Google Cloud's Mandiant, 
said during a recent media briefing.

In almost a dozen interviews with top security experts across the cyber sector,
 the prolific scheme was cited as a major threat, with many admitting that 
 their companies had fallen victim and were struggling to stop the spread. Iain 
 Mulholland, Cloud CISO at Google Cloud, said during the same media briefing 
 that Google had seen North Korean IT workers “in our pipeline,” but declined 
 to specify if this meant the applicants had been caught in the screening 
 process or had actually been hired.

Cybersecurity firm SentinelOne is one of the companies that have gone public 
about being targeted by the scheme. Last month, it released a report revealing 
it had received around 1,000 job applications linked to the North Korean IT 
workers program. Brandon Wales, the former executive director of the 
Cybersecurity and Infrastructure Security Agency and current vice president of 
cybersecurity strategy at SentinelOne, said the “scale and speed” of the North 
Korean government's use of this strategy to amass funding for its weapons 
program had not been seen before.
"""

responses = ner(text)

print(responses)


question_answerer = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")
answer = question_answerer(
    question="Where do I work?",
    context="My name is Elon Musk and I work for DOGE in Washington, DC",
)

print(answer)

answer = question_answerer(
    question="What type of bread did I eat for lunch?",
    context="For lunch I ate a chef salad and an English muffin.",
)

print(answer)
'''

text = """
    America has changed dramatically during recent years. Not only has the number of 
    graduates in traditional engineering disciplines such as mechanical, civil, 
    electrical, chemical, and aeronautical engineering declined, but in most of 
    the premier American universities engineering curricula now concentrate on 
    and encourage largely the study of engineering science. As a result, there 
    are declining offerings in engineering subjects dealing with infrastructure, 
    the environment, and related issues, and greater concentration on high 
    technology subjects, largely supporting increasingly complex scientific 
    developments. While the latter is important, it should not be at the expense 
    of more traditional engineering.

    Rapidly developing economies such as China and India, as well as other 
    industrial countries in Europe and Asia, continue to encourage and advance 
    the teaching of engineering. Both China and India, respectively, graduate 
    six and eight times as many traditional engineers as does the United States. 
    Other industrial countries at minimum maintain their output, while America 
    suffers an increasingly serious decline in the number of engineering graduates 
    and a lack of well-educated engineers.
"""

'''
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
summary = summarizer(text)

print(summary)
'''
