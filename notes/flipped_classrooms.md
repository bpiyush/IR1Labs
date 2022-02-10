### Class 1: 10/02/2022

* What are potential problems with stemming?
    1. Inverted index becomes too large? False.
    2. With stemming, more relevant results are returned? True.
    3. Too many documents are retrieved with stemming? True. Since we are going to match to more terms.

* What information can be stored in an inverted index without introducing redundancy?
    1. Term frequency: (a) does the term occur? (b) term frequency.. We do need document index and the term freqeuency.
    2. Document length: Might be redudent since we can possibly compute it from the term frequency. And it does not belong to the inverted index and rather to the document properties.
    3. BM25 score? Again, we can possibly compute it. It is similar to TFIDF.

* Indexing stop words

* Indexing and stemming
