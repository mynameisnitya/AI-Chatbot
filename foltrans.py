import nltk
from nltk.sem import Expression

from nltk.inference import ResolutionProverCommand

from nltk.sem import logic
import csv
from nltk.sem.logic import *
from nltk.inference import *
from sympy import *
from sympy.abc import x, y, z
from nltk.sem.logic import NegatedExpression
from nltk.inference import Prover9
from nltk.sem.logic import Expression, NegatedExpression, AndExpression

read_expr = Expression.fromstring
def translate_to_fol(sentence):
    tokens = nltk.word_tokenize(sentence)
    tagged = nltk.pos_tag(tokens)
    predicates = []
    for tag in tagged:
        if tag[1].startswith('N'):
            predicates.append(tag[0])
        elif tag[1].startswith('V'):
            predicates.append(tag[0] + "(x)")
        elif tag[1].startswith('JJ'):
            predicates.append(tag[0])
        elif tag[1].startswith('IN'):
            predicates.append("preposition(" + tag[0] + ")")
        elif tag[1].startswith('DT'):
            predicates.append("determiner(" + tag[0] + ")")
        else:
            predicates.append(tag[0])
    if "->" in predicates:
        predicates[predicates.index("->")] = ","
        return " ".join(predicates)
    else:
        return " -> ".join(predicates)
    
def load_knowledge(file_path):
    kb = []

    with open(file_path) as f:
        reader = csv.reader(f)
        for row in reader:
            if row:
                expr = read_expr(row[0])
                kb.append(expr)

    return kb


def count_matching_parts(formula, statement):
    parts1 = str(formula.simplify()).split("->")
    parts2 = str(statement.simplify()).split("->")
    count = 0
    for i, part1 in enumerate(parts1):
        if i >= len(parts2):
            break
        part2 = parts2[i]
        if part1 == part2:
            count += 1
            #print("part1: ",part1)
            #print("part2: ",part2)
        else:
            break
        
    return count



def add_to_kb(kb, sentence):
    if sentence=="":
        return "What do you know?"
    fol = translate_to_fol(sentence)
    formula = read_expr(fol)
    

    for statement in kb:
        parts_of_formula = str(formula.simplify()).split("->")
        count_of_formula=len(parts_of_formula)
       
       
        if formula.simplify() == statement.simplify() :
            return "That is true"
        elif count_matching_parts(formula, statement) >= 2  and count_matching_parts(formula, statement)<=count_of_formula:
            #print(count_matching_parts(statement, formula))
            return "That may not be true"
       

    kb.append(formula)
    with open('kb.csv', 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([fol])
    return "OK, I will remember that " + str(formula)



def check_kb(kb, sentence):
    if sentence=="":
        return "What do you want me to check?"
    fol = translate_to_fol(sentence)
    formula = read_expr(fol)
    
    for statement in kb:
        parts_of_formula = str(formula.simplify()).split("->")
        count_of_formula=len(parts_of_formula)
        if formula.simplify() == statement.simplify() :
             return "That is true"
        elif formula.simplify() != statement.simplify():
            if count_matching_parts(formula, statement) >= 2  and count_matching_parts(formula, statement)<=count_of_formula:
                print(count_matching_parts(statement, formula))
                return "That may not be true"
            
    #print(count_matching_parts(statement, formula))
    return "I don't know"

def display_kb(kb):
   
    return "\n".join([str(formula) for formula in kb])


kb = load_knowledge("kb.csv")
"""while True:
    user_input = input("> ")
    if user_input == "quit":
        break
    elif user_input.startswith("I know that"):
        response = add_to_kb(kb, user_input[len("I know that"):])
        print(response)
    elif user_input.startswith("Check that"):
        response = check_kb(kb, user_input[len("Check that"):])
        print(response)
    elif user_input == "What do you know?":
        print(display_kb(kb))"""

