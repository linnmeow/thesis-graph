from transformers import T5ForConditionalGeneration, T5Tokenizer
import spacy

# load spaCy model for named entity recognition and noun phrase extraction
nlp = spacy.load('en_core_web_sm')

# load T5 model and tokenizer for question generation
t5_model_name = 'valhalla/t5-small-qg-hl'
t5_model = T5ForConditionalGeneration.from_pretrained(t5_model_name)
t5_tokenizer = T5Tokenizer.from_pretrained(t5_model_name)

def mask_entities_and_phrases(text):
    """
    Create masked text with highlighted spans to use with T5 model.
    Wrap the answer spans with <hl> tokens and add </s> at the end.
    """
    doc = nlp(text)
    
    # concatenate entities and noun phrases into a list
    entities_and_phrases = [ent.text for ent in doc.ents] + [chunk.text for chunk in doc.noun_chunks]
    
    # sort entities and phrases by length to avoid partial replacements
    entities_and_phrases = sorted(entities_and_phrases, key=len, reverse=True)
    
    masked_text = text
    for item in entities_and_phrases:
        masked_text = masked_text.replace(item, f'<hl> {item} <hl>')
    
    # add end-of-sequence token
    masked_text = f"{masked_text} </s>"
    
    return masked_text

def generate_questions(masked_sentence):
    """Generate questions from the masked sentence using T5."""
    inputs = t5_tokenizer(masked_sentence, return_tensors='pt', max_length=1024, truncation=True)
    question_ids = t5_model.generate(inputs['input_ids'], max_length=64, num_beams=4, early_stopping=True)
    questions = [t5_tokenizer.decode(qid, skip_special_tokens=True) for qid in question_ids]
    return questions

# Example sentence
summary_sentence = "Sally was born in 1958 in a small town."

# Generate masked sentence
masked_sentence = mask_entities_and_phrases(summary_sentence)

# Generate questions
questions = generate_questions(masked_sentence)

# Output results
print("Masked Sentence:", masked_sentence)
print("Generated Questions:", questions)


