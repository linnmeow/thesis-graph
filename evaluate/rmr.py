import spacy
import json
import argparse

# Load spacy's small English model
nlp = spacy.load("en_core_web_sm")

def extract_relational_triples(text):
    """
    Extract (subject, relation, object) triples using spaCy's dependency parsing.
    
    Args:
        text (str): The input text from which to extract triples.
    
    Returns:
        List[Tuple]: A list of extracted (subject, relation, object) triples.
    """
    doc = nlp(text)
    triples = []
    
    for sent in doc.sents:
        for token in sent:
            # find a verb to use as the relation
            if token.pos_ == "VERB":
                # look for a subject dependent and object dependent
                subject = [child for child in token.children if child.dep_ == "nsubj"]
                objects = [child for child in token.children if child.dep_ in ("dobj", "pobj")]
                
                if subject and objects:
                    for obj in objects:
                        triples.append((subject[0].text, token.lemma_, obj.text))
    
    return triples

def classify_tuples(article_triples, summary_triples):
    """
    Classify each triple in the summary based on its presence and correctness in the article.
    
    Args:
        article_triples (List[Tuple]): List of triples (subject, relation, object) from the article.
        summary_triples (List[Tuple]): List of triples (subject, relation, object) from the summary.
    
    Returns:
        Tuple: (Correct Hits (C), Wrong Hits (W), Misses (M))
    """
    correct_hits = 0
    wrong_hits = 0
    misses = 0
    
    for s_triple in summary_triples:
        if s_triple in article_triples:
            correct_hits += 1
        else:
            # check if it's a wrong hit: subject or object is different, but relation is correct
            subject, relation, obj = s_triple
            partial_match = any(
                (a_subject == subject or a_obj == obj) and a_relation == relation 
                for (a_subject, a_relation, a_obj) in article_triples
            )
            
            if partial_match:
                wrong_hits += 1
            else:
                misses += 1
    
    return correct_hits, wrong_hits, misses

def calculate_rmr(correct_hits, wrong_hits, misses):
    """
    Calculate RMR_1 and RMR_2.
    
    Args:
        correct_hits (int): The count of correct hits (C).
        wrong_hits (int): The count of wrong hits (W).
        misses (int): The count of misses (M).
    
    Returns:
        Tuple[float, float]: RMR_1 and RMR_2 scores.
    """
    if correct_hits + wrong_hits == 0:
        rmr_1 = 0  
    else:
        rmr_1 = 100 * correct_hits / (correct_hits + wrong_hits)
    
    if correct_hits + wrong_hits + misses == 0:
        rmr_2 = 0  
    else:
        rmr_2 = 100 * correct_hits / (correct_hits + wrong_hits + misses)
    
    return rmr_1, rmr_2

def evaluate_factual_consistency(json_file_path):
    """
    Evaluates factual consistency by calculating relation matching rate (RMR).
    
    Args:
        json_file_path (str): Path to the JSON file with articles and summaries.
    
    Returns:
        Tuple[float, float]: Average RMR_1 and RMR_2 across all samples.
    """
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    
    total_rmr_1 = 0
    total_rmr_2 = 0
    count = 0
    
    for entry in data:
        article_text = entry['source_text']
        summary_text = entry['generated_summary']
        
        # extract triples from article and summary
        article_triples = extract_relational_triples(article_text)
        summary_triples = extract_relational_triples(summary_text)
        
        # classify the triples
        correct_hits, wrong_hits, misses = classify_tuples(article_triples, summary_triples)
        
        # calculate RMR scores
        rmr_1, rmr_2 = calculate_rmr(correct_hits, wrong_hits, misses)
        
        total_rmr_1 += rmr_1
        total_rmr_2 += rmr_2
        count += 1
    
    # average RMR scores across all samples
    avg_rmr_1 = total_rmr_1 / count if count > 0 else 0
    avg_rmr_2 = total_rmr_2 / count if count > 0 else 0
    
    return avg_rmr_1, avg_rmr_2

def main():
    parser = argparse.ArgumentParser(description="Evaluate factual consistency using Relation Matching Rate (RMR).")
    parser.add_argument("json_file", type=str, help="Path to the JSON file containing articles and generated summaries.")
    
    args = parser.parse_args()
    
    avg_rmr_1, avg_rmr_2 = evaluate_factual_consistency(args.json_file)
    
    print(f"Average RMR_1: {avg_rmr_1:.2f}")
    print(f"Average RMR_2: {avg_rmr_2:.2f}")

if __name__ == "__main__":
    main()
