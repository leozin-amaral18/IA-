# decision_trees_from_scratch.py
# Implementação do zero de ID3, C4.5 e CART (árvores de decisão)
# - Inclui: TreeNode, ID3, C45, CART
# - Funções utilitárias: entropia, gini, info gain
# - Rotina de experimento com Play Tennis (14 linhas) e Titanic (tenta carregar 'titanic.csv'; se não existir gera amostra sintética)
# - Gera: árvore em texto (pretty), regras, métricas (accuracy/precision/recall/f1) e baseline sklearn

import os
import math
import random
from collections import Counter, defaultdict
from pprint import pprint

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.tree import DecisionTreeClassifier

# ---------------------------- Utilitários ----------------------------

def entropy(y):
    counts = Counter(y)
    total = len(y)
    ent = 0.0
    for c in counts.values():
        p = c / total
        ent -= p * math.log2(p) if p>0 else 0
    return ent

def gini(y):
    counts = Counter(y)
    total = len(y)
    g = 1.0
    for c in counts.values():
        p = c/total
        g -= p*p
    return g

# ---------------------------- Árvores (estrutura) ----------------------------

class TreeNode:
    def __init__(self, is_leaf=False, prediction=None, feature=None, threshold=None, children=None):
        self.is_leaf = is_leaf
        self.prediction = prediction  # for leaf
        self.feature = feature        # feature name
        self.threshold = threshold    # for continuous splits (numeric threshold)
        self.children = children or {} # dict: for categorical splits mapping value->node; or {'le':node,'gt':node} for binary

    def pretty(self, depth=0):
        indent = "  " * depth
        if self.is_leaf:
            return f"{indent}Leaf: predict={self.prediction}\n"
        if self.threshold is not None:
            s = f"{indent}[{self.feature} <= {self.threshold:.5f}]?\n"
            s += self.children['le'].pretty(depth+1)
            s += f"{indent}else:\n"
            s += self.children['gt'].pretty(depth+1)
            return s
        else:
            s = f"{indent}[{self.feature}] categorical split\n"
            for val, node in self.children.items():
                s += f"{indent} if {self.feature} == {val}:\n"
                s += node.pretty(depth+1)
            return s

    def rules(self, prefix=None):
        prefix = prefix or []
        if self.is_leaf:
            cond = " AND ".join(prefix) if prefix else "(always)"
            return [cond + f" => Predict {self.prediction}"]
        out = []
        if self.threshold is not None:
            out += self.children['le'].rules(prefix + [f"{self.feature} <= {self.threshold:.5f}"])
            out += self.children['gt'].rules(prefix + [f"{self.feature} > {self.threshold:.5f}"])
        else:
            for v, child in self.children.items():
                out += child.rules(prefix + [f"{self.feature} == {v}"])
        return out

# ---------------------------- Implementações ----------------------------

class ID3:
    # ID3: entropy + information gain, categorical only. For continuous used discretization into bins.
    def __init__(self, max_depth=10, min_samples_split=2, n_bins_for_continuous=5):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_bins = n_bins_for_continuous
        self.tree = None
        self.feature_types = {}

    def fit(self, X: pd.DataFrame, y: pd.Series):
        # determine feature types
        self.feature_types = {}
        for col in X.columns:
            if pd.api.types.is_numeric_dtype(X[col]):
                self.feature_types[col] = 'continuous'
            else:
                self.feature_types[col] = 'categorical'
        # discretize continuous features into bins (simple approach)
        Xb = X.copy()
        for col, t in self.feature_types.items():
            if t == 'continuous':
                # create n_bins equal-frequency
                try:
                    Xb[col] = pd.qcut(Xb[col].rank(method='first'), q=self.n_bins, duplicates='drop').astype(str)
                except Exception:
                    Xb[col] = Xb[col].astype(str)
        data = Xb.copy()
        data['__y__'] = y.values
        self.tree = self._build(data, depth=0)
        return self

    def _majority(self, y):
        return Counter(y).most_common(1)[0][0]

    def _build(self, data: pd.DataFrame, depth):
        y = data['__y__'].values
        if len(set(y)) == 1:
            return TreeNode(is_leaf=True, prediction=y[0])
        if depth >= self.max_depth or len(data) < self.min_samples_split:
            return TreeNode(is_leaf=True, prediction=self._majority(y))
        # compute best attribute by information gain
        best_gain = -1
        best_attr = None
        base_ent = entropy(y)
        for col in data.columns:
            if col == '__y__': continue
            values = data[col].unique()
            subsets_y = [data[data[col]==v]['__y__'].values for v in values]
            gain = base_ent - sum((len(s)/len(y))*entropy(s) for s in subsets_y)
            if gain > best_gain:
                best_gain = gain
                best_attr = col
        if best_attr is None or best_gain <= 0:
            return TreeNode(is_leaf=True, prediction=self._majority(y))
        node = TreeNode(is_leaf=False, feature=best_attr, threshold=None, children={})
        for v in data[best_attr].unique():
            subset = data[data[best_attr]==v]
            node.children[v] = self._build(subset.drop(columns=[best_attr]), depth+1)
        return node

    def predict_row(self, row):
        node = self.tree
        while not node.is_leaf:
            if node.threshold is not None:
                if row[node.feature] <= node.threshold:
                    node = node.children['le']
                else:
                    node = node.children['gt']
            else:
                val = str(row.get(node.feature, None))
                # if unseen category, choose majority of children
                if val in node.children:
                    node = node.children[val]
                else:
                    # fallback: choose child with most training rows (approx by majority prediction)
                    preds = [c.prediction for c in node.children.values() if c.is_leaf]
                    if preds:
                        return Counter(preds).most_common(1)[0][0]
                    node = list(node.children.values())[0]
        return node.prediction

    def predict(self, X: pd.DataFrame):
        return [self.predict_row(X.iloc[i]) for i in range(len(X))]

class C45:
    # C4.5 simplified: uses gain ratio; handles continuous by finding best threshold (like original)
    def __init__(self, max_depth=10, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None

    def fit(self, X: pd.DataFrame, y: pd.Series):
        data = X.copy()
        data['__y__'] = y.values
        self.tree = self._build(data, depth=0)
        return self

    def _majority(self, y):
        return Counter(y).most_common(1)[0][0]

    def _best_split_for_continuous(self, data, col):
        # consider midpoints between sorted unique values
        vals = sorted(data[col].dropna().unique())
        if len(vals) <= 1:
            return None, -1
        candidates = [(vals[i]+vals[i+1])/2.0 for i in range(len(vals)-1)]
        best_thr, best_gainratio = None, -1
        parent_entropy = entropy(data['__y__'].values)
        for thr in candidates:
            left = data[data[col] <= thr]['__y__'].values
            right = data[data[col] > thr]['__y__'].values
            if len(left)==0 or len(right)==0: continue
            ig = parent_entropy - (len(left)/len(data))*entropy(left) - (len(right)/len(data))*entropy(right)
            # split info (intrinsic value)
            si = 0.0
            for subset in [left, right]:
                p = len(subset)/len(data)
                si -= p * math.log2(p) if p>0 else 0
            if si == 0: continue
            gainratio = ig / si
            if gainratio > best_gainratio:
                best_gainratio = gainratio
                best_thr = thr
        return best_thr, best_gainratio

    def _build(self, data: pd.DataFrame, depth):
        y = data['__y__'].values
        if len(set(y)) == 1:
            return TreeNode(is_leaf=True, prediction=y[0])
        if depth >= self.max_depth or len(data) < self.min_samples_split:
            return TreeNode(is_leaf=True, prediction=self._majority(y))
        # find best attribute (cat or cont) by gain ratio
        best_attr = None
        best_info = -1
        best_threshold = None
        base_entropy = entropy(y)
        for col in data.columns:
            if col == '__y__': continue
            if pd.api.types.is_numeric_dtype(data[col]):
                thr, gr = self._best_split_for_continuous(data, col)
                score = gr if gr is not None else -1
                if score > best_info:
                    best_info = score
                    best_attr = col
                    best_threshold = thr
            else:
                # categorical: compute gain ratio
                values = data[col].unique()
                ig = base_entropy - sum((len(data[data[col]==v])/len(data))*entropy(data[data[col]==v]['__y__'].values) for v in values)
                si = 0.0
                for v in values:
                    p = len(data[data[col]==v])/len(data)
                    si -= p * math.log2(p) if p>0 else 0
                gr = ig / si if si>0 else -1
                if gr > best_info:
                    best_info = gr
                    best_attr = col
                    best_threshold = None
        if best_attr is None or best_info <= 0:
            return TreeNode(is_leaf=True, prediction=self._majority(y))
        # split
        if best_threshold is not None:
            node = TreeNode(is_leaf=False, feature=best_attr, threshold=best_threshold)
            left = data[data[best_attr] <= best_threshold].copy()
            right = data[data[best_attr] > best_threshold].copy()
            node.children['le'] = self._build(left, depth+1)
            node.children['gt'] = self._build(right, depth+1)
        else:
            node = TreeNode(is_leaf=False, feature=best_attr, threshold=None, children={})
            for v in data[best_attr].unique():
                subset = data[data[best_attr]==v].copy()
                node.children[v] = self._build(subset.drop(columns=[best_attr]), depth+1)
        return node

    def predict_row(self, row):
        node = self.tree
        while not node.is_leaf:
            if node.threshold is not None:
                if row[node.feature] <= node.threshold:
                    node = node.children['le']
                else:
                    node = node.children['gt']
            else:
                val = row.get(node.feature, None)
                if val in node.children:
                    node = node.children[val]
                else:
                    # fallback majority
                    preds = [c.prediction for c in node.children.values() if c.is_leaf]
                    if preds:
                        return Counter(preds).most_common(1)[0][0]
                    node = list(node.children.values())[0]
        return node.prediction

    def predict(self, X: pd.DataFrame):
        return [self.predict_row(X.iloc[i]) for i in range(len(X))]

class CART:
    # CART: binary splits, uses Gini impurity and chooses best threshold/value for split.
    def __init__(self, max_depth=10, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None

    def fit(self, X: pd.DataFrame, y: pd.Series):
        data = X.copy()
        data['__y__'] = y.values
        self.tree = self._build(data, depth=0)
        return self

    def _majority(self, y):
        return Counter(y).most_common(1)[0][0]

    def _best_split_continuous(self, data, col):
        vals = sorted(data[col].dropna().unique())
        if len(vals) <= 1:
            return None, -1
        candidates = [(vals[i]+vals[i+1])/2.0 for i in range(len(vals)-1)]
        best_thr, best_gain = None, -1
        parent_gini = gini(data['__y__'].values)
        for thr in candidates:
            left = data[data[col] <= thr]['__y__'].values
            right = data[data[col] > thr]['__y__'].values
            if len(left)==0 or len(right)==0: continue
            gain = parent_gini - (len(left)/len(data))*gini(left) - (len(right)/len(data))*gini(right)
            if gain > best_gain:
                best_gain = gain
                best_thr = thr
        return best_thr, best_gain

    def _best_split_categorical(self, data, col):
        # For categorical, consider binary partition: value == v vs rest
        parent_gini = gini(data['__y__'].values)
        best_v, best_gain = None, -1
        for v in data[col].unique():
            left = data[data[col]==v]['__y__'].values
            right = data[data[col]!=v]['__y__'].values
            gain = parent_gini - (len(left)/len(data))*gini(left) - (len(right)/len(data))*gini(right)
            if gain > best_gain:
                best_gain = gain
                best_v = v
        return best_v, best_gain

    def _build(self, data: pd.DataFrame, depth):
        y = data['__y__'].values
        if len(set(y)) == 1:
            return TreeNode(is_leaf=True, prediction=y[0])
        if depth >= self.max_depth or len(data) < self.min_samples_split:
            return TreeNode(is_leaf=True, prediction=self._majority(y))
        best_attr, best_thr, best_gain, best_is_cont = None, None, -1, False
        for col in data.columns:
            if col == '__y__': continue
            if pd.api.types.is_numeric_dtype(data[col]):
                thr, gain = self._best_split_continuous(data, col)
                if gain > best_gain:
                    best_gain = gain; best_attr = col; best_thr = thr; best_is_cont = True
            else:
                v, gain = self._best_split_categorical(data, col)
                if gain > best_gain:
                    best_gain = gain; best_attr = col; best_thr = v; best_is_cont = False
        if best_attr is None or best_gain <= 0:
            return TreeNode(is_leaf=True, prediction=self._majority(y))
        if best_is_cont:
            node = TreeNode(is_leaf=False, feature=best_attr, threshold=best_thr)
            left = data[data[best_attr] <= best_thr].copy()
            right = data[data[best_attr] > best_thr].copy()
            node.children['le'] = self._build(left, depth+1)
            node.children['gt'] = self._build(right, depth+1)
        else:
            node = TreeNode(is_leaf=False, feature=best_attr, threshold=None, children={})
            # binary split: best_thr is a category value; create two children: ==v and !=v
            v = best_thr
            node.children[f"=={v}"] = self._build(data[data[best_attr]==v].drop(columns=[best_attr]), depth+1)
            node.children[f"≠{v}"] = self._build(data[data[best_attr]!=v].drop(columns=[best_attr]), depth+1)
        return node

    def predict_row(self, row):
        node = self.tree
        while not node.is_leaf:
            if node.threshold is not None:
                if row[node.feature] <= node.threshold:
                    node = node.children['le']
                else:
                    node = node.children['gt']
            else:
                # categorical binary: keys like '==v' and '≠v'
                key_true = f"=={row.get(node.feature, None)}"
                if key_true in node.children:
                    node = node.children[key_true]
                else:
                    node = node.children.get(f"≠{list(node.children.keys())[0][2:]}", list(node.children.values())[0])
        return node.prediction

    def predict(self, X: pd.DataFrame):
        return [self.predict_row(X.iloc[i]) for i in range(len(X))]

# ---------------------------- Datasets ----------------------------

# Play Tennis (classic)
playtennis = pd.DataFrame([
    ['Sunny','Hot','High','Weak','No'],
    ['Sunny','Hot','High','Strong','No'],
    ['Overcast','Hot','High','Weak','Yes'],
    ['Rain','Mild','High','Weak','Yes'],
    ['Rain','Cool','Normal','Weak','Yes'],
    ['Rain','Cool','Normal','Strong','No'],
    ['Overcast','Cool','Normal','Strong','Yes'],
    ['Sunny','Mild','High','Weak','No'],
    ['Sunny','Cool','Normal','Weak','Yes'],
    ['Rain','Mild','Normal','Weak','Yes'],
    ['Sunny','Mild','Normal','Strong','Yes'],
    ['Overcast','Mild','High','Strong','Yes'],
    ['Overcast','Hot','Normal','Weak','Yes'],
    ['Rain','Mild','High','Strong','No']
], columns=['Outlook','Temperature','Humidity','Wind','Play'])

# Titanic: try load titanic.csv in cwd, otherwise create synthetic representative dataset
titanic_path = 'titanic.csv'
if os.path.exists(titanic_path):
    titanic = pd.read_csv(titanic_path)
    # keep relevant columns and do simple cleaning
    cols = ['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked','Survived']
    titanic = titanic[[c for c in cols if c in titanic.columns]]
    titanic = titanic.dropna(subset=['Survived'])
    # simple imputation for Age and Embarked
    if 'Age' in titanic.columns:
        titanic['Age'] = titanic['Age'].fillna(titanic['Age'].median())
    if 'Embarked' in titanic.columns:
        titanic['Embarked'] = titanic['Embarked'].fillna(titanic['Embarked'].mode()[0])
else:
    # synthetic small-ish dataset to allow demonstration
    random.seed(42)
    n = 300
    Pclass = np.random.choice([1,2,3], size=n, p=[0.24,0.2,0.56])
    Sex = np.random.choice(['male','female'], size=n, p=[0.64,0.36])
    Age = np.clip(np.random.normal(30,14,size=n), 0.5, 80)
    SibSp = np.random.poisson(0.6, size=n)
    Parch = np.random.poisson(0.4, size=n)
    Fare = np.round(np.exp(np.random.normal(3.5,1.0,size=n)),2)
    Embarked = np.random.choice(['C','Q','S'], size=n, p=[0.2,0.05,0.75])
    # simple probabilistic label: females and higher class survive more
    logits = (-0.9*(Pclass-1) + 1.4*(Sex=='female') - 0.02*(Age) + 0.02*(Fare>30)).astype(float)
    probs = 1/(1+np.exp(-logits))
    Survived = (np.random.rand(n) < probs).astype(int)
    titanic = pd.DataFrame({'Pclass':Pclass,'Sex':Sex,'Age':Age,'SibSp':SibSp,'Parch':Parch,'Fare':Fare,'Embarked':Embarked,'Survived':Survived})

# For consistent experiments: prepare features/target
titanic_features = ['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']
titanic = titanic[titanic_features + ['Survived']].reset_index(drop=True)

# ---------------------------- Função de avaliação e rotina de treino ----------------------------

def evaluate_model(model, X_train, X_test, y_train, y_test, name="Model", baseline=False):
    preds_train = model.predict(X_train)
    preds_test = model.predict(X_test)
    # convert to numeric if needed
    y_pred = np.array(preds_test).astype(int)
    y_true = np.array(y_test).astype(int)
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    print(f"=== {name} ===")
    print(f"Train size: {len(y_train)}, Test size: {len(y_test)}")
    print(f"Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")
    print("Confusion matrix (test):")
    print(cm)
    print()

    return {'name':name, 'acc':acc, 'prec':prec, 'rec':rec, 'f1':f1, 'cm':cm}


def run_experiment(X, y, dataset_name="Dataset"):
    print(f"*** Experimento em {dataset_name} ***\n")
    # split (stratify if possible)
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
    except:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Train ID3 (note ID3 will discretize continuous features automatically)
    id3 = ID3(max_depth=6, min_samples_split=2, n_bins_for_continuous=5)
    id3.fit(X_train, y_train)
    print("ID3 learned tree:")
    print(id3.tree.pretty())

    id3_results = evaluate_model(id3, X_train, X_test, y_train, y_test, name="ID3 (from-scratch)")

    print("ID3 rules:")
    for r in id3.tree.rules():
        print(" -", r)
    print()

    # Train C4.5
    c45 = C45(max_depth=6, min_samples_split=2)
    c45.fit(X_train, y_train)
    print("C4.5 learned tree:")
    print(c45.tree.pretty())
    c45_results = evaluate_model(c45, X_train, X_test, y_train, y_test, name="C4.5 (from-scratch)")
    print("C4.5 rules:")
    for r in c45.tree.rules():
        print(" -", r)
    print()

    # Train CART
    cart = CART(max_depth=6, min_samples_split=2)
    cart.fit(X_train, y_train)
    print("CART learned tree:")
    print(cart.tree.pretty())
    cart_results = evaluate_model(cart, X_train, X_test, y_train, y_test, name="CART (from-scratch)")
    print("CART rules:")
    for r in cart.tree.rules():
        print(" -", r)
    print()

    # Baseline sklearn DecisionTreeClassifier (Gini, binary behavior depends on settings)
    clf = DecisionTreeClassifier(random_state=42)
    # simple preprocessing: encode categoricals with pandas get_dummies, fillna
    X_all = pd.get_dummies(X).fillna(0)
    Xtr, Xte, ytr, yte = train_test_split(X_all, y, test_size=0.3, stratify=y, random_state=42)
    clf.fit(Xtr, ytr)
    ypred = clf.predict(Xte)
    acc = accuracy_score(yte, ypred)
    prec = precision_score(yte, ypred, zero_division=0)
    rec = recall_score(yte, ypred, zero_division=0)
    f1 = f1_score(yte, ypred, zero_division=0)
    print("=== Baseline sklearn DecisionTreeClassifier ===")
    print(f"Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")
    print("Confusion matrix:")
    print(confusion_matrix(yte, ypred))
    print()

    return {'id3':id3_results, 'c45':c45_results, 'cart':cart_results, 'sklearn':{'acc':acc,'prec':prec,'rec':rec,'f1':f1}}

# ---------------------------- Run experiments ----------------------------

if __name__ == '__main__':
    print("Running on Play Tennis (small classic dataset)\n")
    X_pt = playtennis.drop(columns=['Play'])
    y_pt = playtennis['Play'].map({'Yes':1,'No':0})
    res_play = run_experiment(X_pt, y_pt, dataset_name="Play Tennis")

    print("\n\nRunning on Titanic (Kaggle style columns; using local file if present, otherwise synthetic sample)\n")
    X_tt = titanic[titanic_features]
    y_tt = titanic['Survived']
    res_titanic = run_experiment(X_tt, y_tt, dataset_name="Titanic (sample)")

    print("=== Experimentos concluídos ===")

# FIM
