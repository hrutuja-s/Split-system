"""Evaluate the categoriser (use these numbers/tables in your CA4 report).
Run:  python train_eval.py"""
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from classifier import ITEMS, CATEGORIES, ItemClassifier, _pipeline

X = [i[0].lower() for i in ITEMS]
y = [i[1] for i in ITEMS]
pred = cross_val_predict(_pipeline(), X, y, cv=StratifiedKFold(5, shuffle=True, random_state=42))

print(f"Samples: {len(X)}   Classes: {len(CATEGORIES)}")
print(f"5-fold cross-validated accuracy: {accuracy_score(y, pred):.3f}\n")
print(classification_report(y, pred, digits=3, zero_division=0))
cm = pd.DataFrame(confusion_matrix(y, pred, labels=CATEGORIES), index=CATEGORIES, columns=CATEGORIES)
print("Confusion matrix (rows = actual, columns = predicted):\n", cm)
cm.to_csv("docs/evaluation_confusion_matrix.csv")


# ---- Evaluation 2: realistic entries (plurals, typos, quantities, phrases).
# Replace/extend this with REAL entries from your own flat for a stronger CA4 result.
TEST = {
 "Vegetables": ["onions 2kg","tomatoes","potatos","bhindi half kilo","green chillies","carrots","palak bunch","garlic ginger","cabbages","brinjals"],
 "Fruits": ["bananas","apples","mangoes","oranges","grape","papayas","pomegranates","watermelon half","lemons","strawberries"],
 "Dairy": ["milk packet","doodh","curd 500g","paneer pack","butter amul","eggs dozen","ghee","cheese slices","buttermilk","dahi"],
 "Grocery": ["rice 5kg","atta","toor dal","sugar 1kg","salt","cooking oil","tea powder","coffee","masala packet","bread loaf"],
 "Snacks": ["sweets","chips packet","biscuit","chocolates","cake","ice creams","samosas","cold drinks","namkeen","pizza"],
 "Household": ["gas cylinder","detergent powder","soap","wifi recharge","tissue paper","bulb","dishwash liquid","garbage bags","shampoo","water can"],
 "Other": ["taxi","medicine","gift","printout","repair"],
}
m = ItemClassifier().fit_seed_plus([])
yt, yp = [], []
for c, xs in TEST.items():
    for x in xs:
        yt.append(c); yp.append(m.predict(x)[0])
print(f"\nEvaluation 2 - realistic entries: {sum(a==b for a,b in zip(yt,yp))}/{len(yt)} correct "
      f"({accuracy_score(yt, yp):.3f})")
print("Misclassified:", [(x, a, b) for x, a, b in zip([x for xs in TEST.values() for x in xs], yt, yp) if a != b])
print("\nNote: Evaluation 1 holds out whole unseen words (hard case, no shared letters to learn from).")
print("Fix in the app: users correct a category once and the model learns it on the next run.")
