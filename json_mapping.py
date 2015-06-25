#! /usr/bin/python

from collections import defaultdict
import json

""" senses (extracted from the PDTB training data)
"""

senses = ['Temporal', 'Temporal.Asynchronous', 'Temporal.Asynchronous.Precedence', 'Temporal.Asynchronous.Succession', 'Temporal.Synchrony', 'Contingency', 'Contingency.Cause', 'Contingency.Cause.Reason', 'Contingency.Cause.Result', 'Contingency.Condition', 'Contingency.Condition.Factual past', 'Contingency.Condition.Factual present', 'Contingency.Condition.General', 'Contingency.Condition.Hypothetical', 'Contingency.Condition.Unreal past', 'Contingency.Condition.Unreal present', 'Contingency.Pragmatic cause', 'Contingency.Pragmatic cause.Justification', 'Contingency.Pragmatic condition', 'Contingency.Pragmatic condition.Relevance', 'Contingency.Pragmatic condition.Implicit assertion', 'Comparison', 'Comparison.Concession', 'Comparison.Concession.Contra-expectation', 'Comparison.Concession.Expectation', 'Comparison.Contrast', 'Comparison.Contrast.Juxtaposition', 'Comparison.Contrast.Opposition', 'Comparison.Pragmatic concession', 'Comparison.Pragmatic contrast', 'Expansion', 'Expansion.Alternative', 'Expansion.Alternative.Chosen alternative', 'Expansion.Alternative.Conjunctive', 'Expansion.Alternative.Disjunctive', 'Expansion.Conjunction', 'Expansion.Exception', 'Expansion.Instantiation', 'Expansion.List', 'Expansion.Restatement', 'Expansion.Restatement.Equivalence', 'Expansion.Restatement.Generalization', 'Expansion.Restatement.Specification']



""" mapping 3a: basic operation (temporal, causal, additive, non-causal)
"""

"""
basic_operation = ['temporal', 'temporal', 'temporal', 'temporal', 'temporal', 'causal', 'causal', 'causal', 'causal', 'causal', 'causal', 'causal', 'causal', 'causal', 'causal', 'causal', 'causal', 'causal', 'causal', 'causal', 'causal', 'n.a.', 'causal', 'causal', 'causal', 'additive', 'additive', 'additive', 'additive', 'additive', 'n.a.', 'n.a.', 'non-causal', 'additive', 'non-causal', 'additive', 'non-causal', 'additive', 'additive', 'additive', 'additive', 'additive', 'additive']
"""

""" mapping 3b: basic operation (temporal, causal, additive (without non-causal; 
    all non-causals => additive))
"""


basic_operation = ['temporal', 'temporal', 'temporal', 'temporal', 'temporal', 'causal', 'causal', 'causal', 'causal', 'causal', 'causal', 'causal', 'causal', 'causal', 'causal', 'causal', 'causal', 'causal', 'causal', 'causal', 'causal', 'n.a.', 'causal', 'causal', 'causal', 'additive', 'additive', 'additive', 'additive', 'additive', 'n.a.', 'n.a.', 'additive', 'additive', 'additive', 'additive', 'additive', 'additive', 'additive', 'additive', 'additive', 'additive', 'additive']




""" labels for order (forward: arg1 > arg2,  backward: arg2 > arg1)
    for each sense label
"""

order = ['n.a.', 'n.a.', 'forward', 'backward', 'n.a.', 'n.a.', 'n.a.', 'backward', 'forward', 'backward', 'backward', 'backward', 'backward', 'backward', 'backward', 'backward', 'n.a.', 'n.a.', 'n.a.', 'n.a.', 'n.a.', 'n.a.', 'n.a.', 'forward', 'backward', 'n.a.', 'n.a.', 'n.a.', 'n.a.', 'n.a.', 'n.a.', 'n.a.', 'n.a.', 'n.a.', 'n.a.', 'n.a.', 'n.a.', 'n.a.', 'n.a.', 'n.a.', 'n.a.', 'n.a.', 'n.a.'] 


""" labels for semantic/pragmatic dimension for each sense label
"""
sem_prag = ['sem', 'sem', 'sem', 'sem', 'sem', 'sem', 'sem', 'sem', 'sem', 'sem', 'sem', 'sem', 'sem', 'sem', 'sem', 'sem', 'prag', 'prag', 'prag', 'prag', 'prag', 'sem', 'sem', 'sem', 'sem', 'sem', 'sem', 'sem', 'prag', 'prag', 'sem', 'sem', 'sem', 'sem', 'sem', 'sem', 'sem', 'sem', 'sem', 'sem', 'sem', 'sem', 'sem']



""" polarity labels for each sense label
"""
polarity = ['pos', 'pos', 'pos', 'pos', 'pos', 'pos', 'pos', 'pos', 'pos', 'pos', 'pos', 'pos', 'pos', 'pos', 'pos', 'pos', 'pos', 'pos', 'pos', 'pos', 'pos', 'neg', 'neg', 'neg', 'neg', 'neg', 'neg', 'neg', 'neg', 'neg', 'n.a.', 'n.a.', 'neg', 'pos', 'neg', 'pos', 'neg', 'pos', 'pos', 'pos', 'pos', 'pos', 'pos']


""" now create json object and write to file
"""


mydict = lambda: defaultdict(mydict)
mapping = mydict()

for c in range(len(senses)):
    mapping[senses[c]]['basic'] = basic_operation[c]
    mapping[senses[c]]['order'] = order[c]
    mapping[senses[c]]['sem_prag'] = sem_prag[c]
    mapping[senses[c]]['polarity'] = polarity[c]


with open("mapping3b.json", "w") as fp:
    json.dump(mapping, fp, indent=2)

