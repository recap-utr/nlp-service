import typing as t
from pprint import pprint

import spacy
from gensim.corpora.dictionary import Dictionary
from gensim.models.ldamodel import LdaModel
from textacy.extract import basics as extract

nlp = spacy.load("en_core_web_lg")


def prepocess(texts: t.Iterable[str]) -> t.Tuple[t.List[t.Tuple[int, int]], Dictionary]:
    docs = nlp.pipe(texts)
    processed_docs = [
        [token.lemma_ for token in extract.words(doc)]
        + [ngram.lemma_ for ngram in extract.ngrams(doc, 2)]
        + [ngram.lemma_ for ngram in extract.ngrams(doc, 3)]
        for doc in docs
    ]
    id2word = Dictionary(processed_docs)
    corpus = t.cast(
        t.List[t.Tuple[int, int]], [id2word.doc2bow(doc) for doc in processed_docs]
    )

    return corpus, id2word


# [
#     "Most of the time, the Supreme Court appears to the public like a cautiously deliberative body. Before issuing major rulings, the justices pore over extensive written briefs, grill lawyers in oral arguments and then take months to draft opinions explaining their reasoning, which they release at precisely calibrated moments.",
#     "Then there is the “shadow docket.”",
#     "With increasing frequency, the court is taking up weighty matters in a rushed way, considering emergency petitions that often yield late-night decisions issued with minimal or no written opinions. Such orders have reshaped the legal landscape in recent years on high-profile matters like changes to immigration enforcement, disputes over election rules, and public-health orders barring religious gatherings and evictions during the pandemic.",
#     "The latest and perhaps most powerful example came just before midnight on Wednesday, when the court ruled 5 to 4 to leave in place a novel Texas law that bars most abortions in the state — a momentous development in the decades-long judicial battle over abortion rights.",
#     "The court spent less than three days dealing with the case. There were no oral arguments before the justices. The majority opinion was unsigned and one paragraph long. In a dissent, Justice Elena Kagan said the case illustrated “just how far the court’s ‘shadow-docket’ decisions may depart” from the usual judicial process and said use of the shadow docket “every day becomes more unreasoned, inconsistent and impossible to defend.”\nThere is nothing new about the court having an orders docket where it swiftly disposes of certain matters. But with the notable exception of emergency applications for last-minute stays of execution, this category of court activity has traditionally received little attention. That is because for the most part, the orders docket centers on routine case management requests by lawyers, like asking for permission to submit an unusually long brief.",
#     "The court also uses it to dispose of emergency appeals. Each justice handles requests from a different region, and can reject them or bring them to the full court. And increasingly, the court has been using its orders docket — which was deemed the “shadow docket” in 2015, in an influential law journal article by William Baude, a University of Chicago law professor — to swiftly decide whether to block government actions, turning it into a powerful tool for affecting public policy without fully hearing from the parties or explaining its actions in writing.",
#     "Criticism of the use of the shadow docket has been building for years but rose to a new level with the Texas abortion case. The chairman of the House Judiciary Committee, Representative Jerrold Nadler, Democrat of New York, denounced the ruling, saying it allowed what he portrayed as a “flagrantly unconstitutional law” to take force and calling it “shameful” that the court’s majority did so without hearing arguments or issuing any signed opinion. He announced hearings.",
#     "“Because the court has now shown repressive state legislatures how to game the system, the House Judiciary Committee will hold hearings to shine a light on the Supreme Court’s dangerous and cowardly use of the shadow docket,” he said in a statement. “Decisions like this one chip away at our democracy.”",
#     "Liberals are not the only ones who see problems in the increasing importance of the court’s exercise of power through emergency orders. When the court issued a shadow-docket order last year letting a Trump administration immigration rule take effect — overturning a lower-court judge’s nationwide injunction blocking the rule — Justice Neil M. Gorsuch, a conservative, supported that result but lamented the process that had led up to it.\nEditors’ Picks",
#     "\nMaggie Nelson Wants to Redefine ‘Freedom’",
#     "The 1970s Brought Change to the Beach Boys. A New Boxed Set Celebrates It.",
#     "Adult Swim: How an Animation Experiment Conquered Late-Night TV\nContinue reading the main story\n“Rather than spending their time methodically developing arguments and evidence in cases limited to the parties at hand, both sides have been forced to rush from one preliminary injunction hearing to another, leaping from one emergency stay application to the next, each with potentially nationwide stakes, and all based on expedited briefing and little opportunity for the adversarial testing of evidence,” he wrote.",
#     "But while there is broad consensus that the Supreme Court’s use of the shadow docket for high-profile rulings is growing — a trend playing out within an increasingly polarized judiciary and nation — defining the precise nature of the problem is complicated and subject to dispute.",
#     "“I don’t think anyone thinks it is good to have a lot of last-minute requests for emergency relief that the court has to focus on and decide,” said Samuel Bray, a University of Notre Dame law professor who testified about the shadow docket this summer before President Biden’s commission studying possible Supreme Court changes. “But there are difficult questions about what has caused the high-profile use of the shadow docket — and what to do about it.”",
#     "Over the past decade or so, such rulings have clearly become more common. Typically, they involve emergency appeals of lower-court rulings over the question of whether to block some change — like a new law or government policy — so it cannot be enforced while the slow process of litigating plays out.",
# ]
corpus, id2word = prepocess(
    [
        "Rent prices should be limited by a cap when there's a change of tenant.",
        "Landlords may want to earn as much as possible,",
        "and many, consistent with market principles, are prepared to pay higher rents,",
        "but that people with the same income suddenly must pay more and can't live in the same flat anymore seems implausible.",
        "Gentrification destroys entire districts and their culture.",
    ]
)
query_corpus, query_id2word = prepocess(
    [
        "Should cost on contracts be limited in terms of previous agreements between payers for arbitrary situations?"
    ]
)
lda = LdaModel(corpus=corpus, id2word=id2word, num_topics=1)

pprint(lda.print_topics())
print(lda.get_document_topics(query_corpus[0]))
