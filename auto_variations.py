import json
import re

with open('simpler_extended.json') as f:
    data = json.load(f)

syn_subjects = {
    'objects': ['objects', 'items', 'entities', 'bodies', 'elements'],
    'collisions': ['collisions', 'impacts', 'contact events', 'collision events', 'crashes'],
}


def capitalize_first(text):
    return text[0].upper() + text[1:] if text else text


def ensure_question(text):
    text = text.strip()
    if not text.endswith('?'):
        text += '?'
    return capitalize_first(text)


def variants_how_many_objects(question):
    base = question.rstrip('?')
    rest = base[len('How many objects '):]
    subject_syns = syn_subjects['objects']
    sentences = []

    rest_variants = [
        rest,
        rest.replace('remain', 'stay'),
        rest.replace('become', 'end up'),
        rest.replace('are', 'happen to be'),
        rest.replace('are within', 'fall within'),
        rest.replace('are positioned', 'sit'),
        rest.replace('bend', 'flex'),
        rest.replace('undergo', 'experience'),
        rest.replace('rotate', 'spin'),
        rest.replace('are set in motion', 'get set in motion'),
    ]
    rest_variants = list(dict.fromkeys(rest_variants))

    templates = [
        'What is the total count of {subject} that {rest}?',
        'How many {subject} {rest} overall?',
        'How many {subject} {rest} in this sequence?',
        'Could you determine how many {subject} {rest}?',
        'How many {subject} are observed to {rest_without_to}?',
        'Please state the number of {subject} that {rest}.',
        'What number of {subject} manage to {rest_without_to}?',
        'Can you tally the {subject} that {rest}?',
        'Report how many {subject} {rest}.',
        'Approximately how many {subject} {rest}?',
    ]

    cleaned = []
    for version in rest_variants:
        version = version.strip()
        if version.startswith('are '):
            cleaned.append('are ' + version[4:])
        else:
            cleaned.append(version)
    rest_variants = cleaned or [rest]

    results = set()
    for i, template in enumerate(templates):
        rest_variant = rest_variants[i % len(rest_variants)]
        subject_variant = subject_syns[i % len(subject_syns)]
        rest_without_to = rest_variant
        if rest_variant.startswith('to '):
            rest_without_to = rest_variant[3:]
        elif rest_variant.startswith('are '):
            rest_without_to = rest_variant[4:]
        results.add(ensure_question(template.format(subject=subject_variant, rest=rest_variant, rest_without_to=rest_without_to)))
        if len(results) == 10:
            break

    return list(results)



def variants_default(question):
    base = question.rstrip('?')
    rest = base[len('What is the '):]
    templates = [
        'How would you characterize the {rest}?',
        'What best describes the {rest}?',
        'Which option corresponds to the {rest}?',
        'Could you identify the {rest}?',
        'Can you specify the {rest}?',
        'Which choice represents the {rest}?',
        'What do you determine to be the {rest}?',
        'Would you indicate the {rest}?',
        'What answer points to the {rest}?',
        'Please state the {rest}.',
    ]
    return [ensure_question(t.format(rest=rest)) for t in templates]


all_vars = {}
for category, entries in data.items():
    for qid, info in entries.items():
        q = info['question']
        if q.startswith('How many objects '):
            all_vars[qid] = variants_how_many_objects(q)
        elif q.startswith('What is the '):
            all_vars[qid] = variants_default(q)
        else:
            all_vars[qid] = []

print({k: len(v) for k, v in all_vars.items()})
