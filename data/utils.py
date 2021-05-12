import os

def process_document(raw_doc):
    TRIVIAL_SENTS = [
        'Share this with',
        'Copy this link',
        'These are external links and will open in a new window',
    ]
    
    raw_doc = raw_doc.strip()
    raw_doc_sents = raw_doc.split('\n')
    
    start_signal = False
    filtered_sentences = []
    for s in raw_doc_sents: 
        if start_signal:
            filtered_sentences.append(s)
        elif len(s.split()) > 1 and s not in TRIVIAL_SENTS:
            start_signal = True
            filtered_sentences.append(s)
            
    return ' '.join(filtered_sentences)


def read_document(bbcid, folder):
    file_path = folder + '{}.document'.format(bbcid)
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            return process_document(f.read())
    else:
        return None