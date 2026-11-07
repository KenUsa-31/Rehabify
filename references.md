# Yakap Project References

Access date for web pages: 2026-11-07. Health content is paraphrased for an educational prototype; consult the cited source for the authoritative text.

## Knowledge-base health sources

1. World Health Organization. *Emergency and critical care*. https://www.who.int/health-topics/emergency-care
   - Supports the general description of emergency care as integrated, time-sensitive care. It does not establish a Yakap hospital service, address, or schedule.
2. World Health Organization. *Vaccines and immunization*. https://www.who.int/health-topics/vaccines-and-immunization
   - Supports the general description of how vaccines help build protection. It does not establish a local vaccination schedule or Yakap clinic hours.
3. World Health Organization. *Child health*. https://www.who.int/health-topics/child-health
   - Supports general educational statements about child health and the role of healthcare access, nutrition, safety, and supportive conditions.
4. World Health Organization. *Cardiovascular diseases (CVDs)*. Fact sheet, 31 July 2025. https://www.who.int/news-room/fact-sheets/detail/cardiovascular-diseases-(cvds)
   - Supports the general description of CVDs and recognized behavioral risk factors. It is not individual medical advice.

## NLP and system-design references

5. Lewis, P. et al. (2020). *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*. NeurIPS 2020. https://arxiv.org/abs/2005.11401
   - Research background for combining a language model with retrieved external knowledge. Yakap's implementation is a simpler TF-IDF retrieval pipeline, not the paper's dense neural retriever.
6. scikit-learn developers. *Feature extraction: text feature extraction and TF-IDF term weighting*. https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction
   - Official documentation for text vectorization and TF-IDF.
7. scikit-learn developers. *Pairwise metrics: cosine similarity*. https://scikit-learn.org/stable/modules/metrics.html#cosine-similarity
   - Official documentation for the similarity measure used to rank retrieved chunks.
8. World Health Organization. (2021). *Ethics and governance of artificial intelligence for health*. https://www.who.int/publications/i/item/9789240029200
   - Ethical and governance context for the prototype's limitations, transparency, and human oversight.

## Fictional organization data

Yakap General Hospital is a fictional organization for this course prototype. The hours, location, telephone, services, appointments, visiting rules, and insurance entries in `knowledge_base.json` are explicitly marked as unverified placeholders. They are not factual claims and must be replaced with instructor-approved, official sources before presenting the chatbot as representing a real facility.
