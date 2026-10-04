"""Item categoriser: TF-IDF character n-grams + logistic regression.
Trained on a seed vocabulary plus every entry the flat has already logged
(and corrected), so it keeps learning the flat's own shopping habits."""
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

SEED = {
    "Vegetables": "onion tomato potato aloo brinjal baingan bhindi okra capsicum carrot gajar spinach palak cabbage cauliflower gobi peas matar garlic ginger chilli coriander dhaniya curry leaves cucumber radish beetroot pumpkin lauki sabzi vegetables".split(),
    "Fruits": "apple banana mango orange grapes papaya watermelon pomegranate anar guava pineapple lemon kiwi strawberry pear chikoo sweet lime fruits dates".split(),
    "Dairy": "milk doodh curd dahi paneer butter cheese ghee cream buttermilk chaas eggs egg lassi yogurt khoya".split(),
    "Grocery": "rice dal atta flour maida sugar salt oil tea coffee bread masala turmeric haldi poha rava suji besan jeera mustard pulses spices noodles maggi pasta cereal oats".split(),
    "Snacks": "sweets mithai chips biscuits chocolate namkeen cake ice cream juice cold drink samosa pizza burger pastry cookies popcorn soda bhujia".split(),
    "Household": "gas cylinder soap detergent tissue cleaner bulb wifi internet recharge garbage bags broom shampoo toothpaste dishwash phenyl mop water can sponge bucket toilet paper handwash battery".split(),
    "Other": "gift taxi auto repair medicine printout stationery charger plumber electrician rent party decoration".split(),
}
ITEMS = [(w, c) for c, ws in SEED.items() for w in ws]
CATEGORIES = list(SEED)


def _pipeline():
    return make_pipeline(
        TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), lowercase=True),
        LogisticRegression(max_iter=2000, C=10),
    )


class ItemClassifier:
    def __init__(self, threshold: float = 0.30):
        self.threshold = threshold
        self.pipe = None

    def fit(self, items, labels):
        self.pipe = _pipeline().fit([i.lower() for i in items], labels)
        return self

    def fit_seed_plus(self, extra):
        """extra: iterable of (item, category) already logged by the flat."""
        rows = ITEMS + [(i, c) for i, c in extra if c in CATEGORIES]
        return self.fit([r[0] for r in rows], [r[1] for r in rows])

    def predict(self, item: str):
        """Returns (category, confidence). Low confidence -> 'Other'."""
        p = self.pipe.predict_proba([item.lower()])[0]
        k = int(np.argmax(p))
        label, conf = self.pipe.classes_[k], float(p[k])
        return (label if conf >= self.threshold else "Other"), conf
