# spaCy – notatki (kurs „Advanced NLP with spaCy”, Chapter 1)

1. [Introduction to spaCy](#1-introduction-to-spacy)
2. [Trained pipelines](#2-trained-pipelines)
3. [Rule-based matching](#3-rule-based-matching)

---

## 1. Introduction to spaCy

```python
import spacy

nlp = spacy.blank("en")         # obiekt nlp: pusty angielski potok + reguły tokenizacji dla języka
doc = nlp("It costs $5.")       # Doc: tekst najpierw tokenizowany, potem powstaje Doc

for token in doc:               # Doc działa jak sekwencja – iterujemy po tokenach
    print(token.i, token.text, token.is_alpha, token.is_punct, token.like_num)

token = doc[1]                  # Token – indeks jak w liście Pythona (od 0)
span = doc[1:3]                 # Span – wycinek Doc, koniec wyłączony (tokeny 1 i 2)
next_token = doc[token.i + 1]   # następny token w dokumencie
print(token.text, "|", span.text, "|", next_token.text)
```
```
0 It True False False
1 costs True False False
2 $ False False False
3 5 False False True
4 . False True False
costs | costs $ | $
```

| Co | Do czego |
|---|---|
| `nlp` | zawiera potok przetwarzania i reguły języka (np. tokenizacja); wywołuje się go jak funkcję. `spacy.blank("en")`, `"de"`, `"es"`… (60+ języków) |
| `Doc` | wynik `nlp(tekst)`; trzyma cały tekst w uporządkowanej formie, nic się nie gubi |
| `Token` | jedno słowo / znak interpunkcyjny, `doc[i]` |
| `Span` | wycinek z jednego lub kilku tokenów, `doc[start:end]`; to tylko widok na `Doc`, bez własnych danych |
| `.text` | tekst tokena / spanu |
| `.i` | indeks tokena w dokumencie |
| `.is_alpha`, `.is_punct`, `.like_num` | litery? interpunkcja? *przypomina* liczbę („10” albo „ten”)? |
---

## 2. Trained pipelines

```bash
python -m spacy download en_core_web_sm
```

```python
import spacy

nlp = spacy.load("en_core_web_sm")   # wytrenowany potok: mały, angielski, na tekstach z internetu

doc = nlp("She ate the pizza")
for token in doc:
    # tekst, część mowy, etykieta zależności, nadrzędnik (head)
    print(token.text, token.pos_, token.dep_, token.head.text)

doc = nlp("Apple is looking at buying U.K. startup for $1 billion")
for ent in doc.ents:                 # jednostki nazwane (obiekty Span)
    print(ent.text, ent.label_)

spacy.explain("GPE")                 # 'Countries, cities, states'
spacy.explain("dobj")                # 'direct object'
```
```
She PRON nsubj ate
ate VERB ROOT ate
the DET det pizza
pizza NOUN dobj ate

Apple ORG
U.K. GPE
$1 billion MONEY
```

| Co | Do czego |
|---|---|
| `spacy.load("nazwa")` | wczytuje pakiet z potokiem po nazwie i zwraca `nlp`; nazwa zależy od języka i danych treningowych |
| `.pos_` | część mowy (PRON, VERB, DET, NOUN…) |
| `.dep_` | etykieta zależności: `nsubj` – podmiot, `dobj` – dopełnienie bliższe, `det` – określnik (rodzajnik) |
| `.head` | token „rodzic”, do którego słowo jest dołączone |
| `doc.ents` | jednostki nazwane („obiekty ze świata rzeczywistego” z nazwą: osoba, organizacja, kraj…) |
| `.label_` | etykieta jednostki (ORG, GPE, MONEY…) |
| `spacy.explain()` | krótkie wyjaśnienie tagu / etykiety |

Do zapamiętania:
- Trained pipelines przewidują cechy **w kontekście**: części mowy, zależności, jednostki nazwane. Są trenowane na oznaczonych tekstach i można je douczyć (fine-tune) nowymi przykładami.
- **Podkreślnik `_`**: atrybut z `_` zwraca napis, bez `_` zwraca ID (liczbę).

---

## 3. Rule-based matching

```python
import spacy
from spacy.matcher import Matcher

nlp = spacy.load("en_core_web_sm")
matcher = Matcher(nlp.vocab)          # zawsze ze wspólnym słownikiem nlp.vocab

# wzorzec = lista słowników, jeden słownik = jeden token
matcher.add("IPHONE_PATTERN", [[{"TEXT": "iPhone"}, {"TEXT": "X"}]])         # dokładny tekst
matcher.add("FIFA_PATTERN", [[{"IS_DIGIT": True}, {"LOWER": "fifa"},          # atrybuty leksykalne
                              {"LOWER": "world"}, {"LOWER": "cup"}, {"IS_PUNCT": True}]])
matcher.add("LOVE_PATTERN", [[{"LEMMA": "love", "POS": "VERB"}, {"POS": "NOUN"}]])  # przewidywania modelu
matcher.add("BUY_PATTERN", [[{"LEMMA": "buy"}, {"POS": "DET", "OP": "?"}, {"POS": "NOUN"}]])  # operator

texts = [
    "Upcoming iPhone X release date leaked",
    "2018 FIFA World Cup: France won!",
    "I loved dogs but now I love cats more.",
    "I bought a smartphone. Now I'm buying apps.",
]
for text in texts:
    doc = nlp(text)
    matches = matcher(doc)                    # lista krotek (match_id, start, end)
    for match_id, start, end in matches:
        print(doc[start:end].text)            # dopasowany fragment = Span
```
```
iPhone X
2018 FIFA World Cup:
loved dogs
love cats
bought a smartphone
buying apps
```

| Co | Do czego |
|---|---|
| `Matcher(nlp.vocab)` | tworzy matcher ze wspólnym słownikiem |
| `matcher.add("ID", [wzorce])` | 1. argument – unikalne ID wzorca, 2. – **lista** wzorców |
| `matcher(doc)` | zwraca krotki: `match_id` (hash nazwy wzorca), `start`, `end` |
| `doc[start:end]` | dopasowany `Span` |

Klucze we wzorcach:

| Klucz | Znaczenie |
|---|---|
| `TEXT` | dokładny tekst tokena |
| `LOWER` | tekst małymi literami (bez rozróżniania wielkości liter) |
| `IS_DIGIT` / `IS_PUNCT` | tylko cyfry / interpunkcja |
| `LEMMA` | forma podstawowa: `"buy"` → „bought”, „buying”; `"be"` → „is”, „was”, „being” |
| `POS` | część mowy, np. `"NOUN"`, `"VERB"`, `"PROPN"` (nazwa własna), `"ADJ"` |
| `OP` | ile razy token ma wystąpić (tabela niżej) |

| `OP` | Znaczenie |
|---|---|
| `"!"` | 0 razy (negacja) |
| `"?"` | 0 lub 1 raz (opcjonalny) |
| `"+"` | 1 lub więcej razy |
| `"*"` | 0 lub więcej razy |

Dlaczego Matcher, a nie regex:
- działa na `Doc` i tokenach, nie tylko na napisach;
- dopasowuje po atrybutach tokenów, także po przewidywaniach modelu – np. „duck” tylko jako czasownik, nie rzeczownik.
