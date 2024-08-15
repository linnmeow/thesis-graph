from transformers import T5ForConditionalGeneration, T5Tokenizer, BertForQuestionAnswering, BertTokenizer, pipeline
import spacy
import re
from datasets import load_metric
from transformers import pipeline

# load spaCy model for named entity recognition and noun phrase extraction
nlp = spacy.load('en_core_web_sm')

# load T5 model and tokenizer for question generation
t5_model_name = 'valhalla/t5-small-qg-hl'
t5_model = T5ForConditionalGeneration.from_pretrained(t5_model_name)
t5_tokenizer = T5Tokenizer.from_pretrained(t5_model_name)

# # load BERT model and tokenizer for answer extraction
# bert_model_name = 'bert-base-uncased'
# bert_model = BertForQuestionAnswering.from_pretrained(bert_model_name)
# bert_tokenizer = BertTokenizer.from_pretrained(bert_model_name)

# load metric for F1 score computation
f1_metric = load_metric("f1")

def mask_entities_and_phrases(text):
    """
    Create masked text with highlighted spans to use with T5 model.
    Wrap only the first entity or noun phrase with <hl> tokens and add </s> at the end.
    """
    doc = nlp(text)

    # extract entities and noun phrases
    entities_and_phrases = [ent.text for ent in doc.ents] + [chunk.text for chunk in doc.noun_chunks]

    # sort by start position to handle the first occurrence
    spans = sorted(set(entities_and_phrases), key=lambda span: text.find(span))

    if not spans:
        return text + " </s>"

    # choose the first entity or phrase
    first_span = spans[0]

    # find start and end indices for the first span
    start_idx = text.find(first_span)
    end_idx = start_idx + len(first_span)

    # replace the first span with <hl> tokens
    masked_text = text[:start_idx] + f'<hl> {first_span} <hl>' + text[end_idx:]

    # add end-of-sequence token
    masked_text = f"{masked_text} </s>"

    return masked_text

def generate_questions(masked_sentence):
    """Generate questions from the masked sentence using T5."""
    inputs = t5_tokenizer(masked_sentence, return_tensors='pt', max_length=1024, truncation=True)
    question_ids = t5_model.generate(inputs['input_ids'], max_length=64, num_beams=4, early_stopping=True)
    questions = [t5_tokenizer.decode(qid, skip_special_tokens=True) for qid in question_ids]
    return questions

def generate_qa_pairs(questions, masked_sentence):
    """
    Generate QA pairs from the summary sentence.
    """
    return [(q, masked_sentence.split('<hl> ')[1].split(' <hl>')[0]) for q in questions]

def get_answers_from_document(qa_pairs, document, model_name="deepset/roberta-base-squad2"):
    """
    Use a powerful pretrained QA model to get answers from the source document.

    Args:
    - qa_pairs (list of tuples): List of (question, _) pairs.
    - document (str): The document to extract answers from.
    - model_name (str): Name of the pretrained model to use.

    Returns:
    - answers (list): List of extracted answers.
    """
    qa_pipeline = pipeline("question-answering", model=model_name)
    
    answers = []
    for question, _ in qa_pairs:
        result = qa_pipeline(question=question, context=document)
        answers.append(result['answer'])
    
    return answers

def compute_f1_score(predictions, references):
    """
    Compute F1 score for the given predictions and references.
    """
    results = []
    for pred, ref in zip(predictions, references):
        results.append(f1_metric.compute(predictions=[pred], references=[ref]))

    # averaging the F1 score results
    avg_f1 = sum([result['f1'] for result in results]) / len(results)
    return avg_f1

def main():
    # Example sentence and document
    summary_sentence = "Sally was born in 1958."
    document = "Sally was born in 1958 in a small town, which was a remarkable event."

    # Generate masked sentence
    masked_sentence = mask_entities_and_phrases(summary_sentence)

    # Generate questions
    questions = generate_questions(masked_sentence)

    # Generate QA pairs
    qa_pairs = generate_qa_pairs(questions, masked_sentence)

    # Get answers from the document
    predictions = get_answers_from_document(qa_pairs, document)

    # Define the "gold" answers
    gold_answers = [answer for _, answer in qa_pairs]

    # Compute the F1 score
    f1_score = compute_f1_score(predictions, gold_answers)

    print("Masked Sentence:", masked_sentence)
    print("Generated Questions:", questions)
    print("Generated QA Pairs:", qa_pairs)
    print("Predictions:", predictions)
    print("Gold Answers:", gold_answers)
    print("F1 Score:", f1_score)

if __name__ == "__main__":
    main()
