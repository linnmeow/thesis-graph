from transformers import T5ForConditionalGeneration, T5Tokenizer, pipeline
import spacy
import re
import string
from sklearn.metrics import precision_recall_fscore_support

# Load spaCy model for named entity recognition and noun phrase extraction
nlp = spacy.load('en_core_web_sm')

# Load T5 model and tokenizer for question generation
t5_model_name = 'valhalla/t5-small-qg-hl'
t5_model = T5ForConditionalGeneration.from_pretrained(t5_model_name)
t5_tokenizer = T5Tokenizer.from_pretrained(t5_model_name)

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
    return [(q, masked_sentence.split('<hl> ')[1].split(' <hl>')[0].strip()) for q in questions]

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

def normalize_answer(s):
    """Lower text and remove punctuation, articles, and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def remove_punctuation(text):
        return text.translate(str.maketrans('', '', string.punctuation))

    def lower(text):
        return text.lower()

    def remove_whitespace(text):
        return ' '.join(text.split())

    def normalize(text):
        """Apply all normalization steps to a single string."""
        return remove_whitespace(remove_articles(remove_punctuation(lower(text))))

    # check if the input is a list, if so, apply normalization to each element
    if isinstance(s, list):
        return [normalize(item) for item in s]
    else:
        return normalize(s)

def compute_f1_score(predictions, references):
    """
    Compute F1 score for the given predictions and references, considering token order.
    Assumes predictions and references are lists of strings.
    """
    all_preds = []
    all_refs = []
    
    for pred, ref in zip(predictions, references):
        # normalize the predicted and reference answers
        pred_tokens = normalize_answer(pred).split()
        ref_tokens = normalize_answer(ref).split()
        
        # append tokens for F1 score computation
        all_preds.extend(pred_tokens)
        all_refs.extend(ref_tokens)
    
    # compute token-level precision, recall, and F1 score
    precision, recall, f1, _ = precision_recall_fscore_support(all_refs, all_preds, average='macro', zero_division=0)
    
    return f1

def main():
    summary_text = "Sally was born in a small town. Sally moved to the city after graduating."
    document = "Sally was born in a small town. She grew up in a close-knit community and attended the local school. After graduating, she moved to the city to pursue her dreams."

    # split the summary text into individual sentences
    summary_sentences = summary_text.split('. ')
    if summary_sentences[-1].endswith('.'):
        summary_sentences[-1] = summary_sentences[-1][:-1]  # Remove the trailing period from the last sentence

    all_qa_pairs = []
    all_gold_answers = []
    all_predictions = []

    for sentence in summary_sentences:
        # Generate masked sentence
        masked_sentence = mask_entities_and_phrases(sentence)

        # Generate questions
        questions = generate_questions(masked_sentence)

        # Generate QA pairs
        qa_pairs = generate_qa_pairs(questions, masked_sentence)

        # Extract answers from the document
        answers = get_answers_from_document(qa_pairs, document)

        all_qa_pairs.extend(qa_pairs)
        all_gold_answers.extend([answer for _, answer in qa_pairs])
        all_predictions.extend(answers)

    # Compute the F1 score
    f1_score = compute_f1_score(all_predictions, all_gold_answers)

    print("All QA Pairs:", all_qa_pairs)
    print("All Gold Answers:", all_gold_answers)
    print("Predictions:", all_predictions)
    print("F1 Score:", f1_score)

if __name__ == "__main__":
    main()
