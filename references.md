# Rehabify Health Project References

Access date for web pages: 2026-11-07. Health content is paraphrased for an educational prototype; consult the cited source for the authoritative text.

## Knowledge-base health sources

1. World Health Organization. *Emergency and critical care*. https://www.who.int/health-topics/emergency-care
   - Supports the general description of emergency care as integrated, time-sensitive care; it does not establish facility services or location details.
2. World Health Organization. *Rehabilitation*. https://www.who.int/news-room/fact-sheets/detail/rehabilitation
   - Supports general descriptions of rehabilitation, functioning, participation, and rehabilitation services. It is not an individual assessment or treatment recommendation.
3. World Health Organization. *Assistive technology*. https://www.who.int/news-room/fact-sheets/detail/assistive-technology
   - Supports general information about assistive products. It does not recommend a particular product or establish that Rehabify Health supplies equipment.

## NLP and system-design references

4. Lewis, P. et al. (2020). *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*. NeurIPS 2020. https://arxiv.org/abs/2005.11401
   - Research background for combining a language model with retrieved external knowledge. Rehabify Health's implementation is a simpler TF-IDF retrieval pipeline, not the paper's dense neural retriever.
5. scikit-learn developers. *Feature extraction: text feature extraction and TF-IDF term weighting*. https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction
   - Official documentation for text vectorization and TF-IDF.
6. scikit-learn developers. *Pairwise metrics: cosine similarity*. https://scikit-learn.org/stable/modules/metrics.html#cosine-similarity
   - Official documentation for the similarity measure used to rank retrieved chunks.
7. World Health Organization. (2021). *Ethics and governance of artificial intelligence for health*. https://www.who.int/publications/i/item/9789240029200
   - Ethical and governance context for the prototype's limitations, transparency, and human oversight.

## Fictional organization data

Rehabify Health is a fictional organization for this course prototype. The rehabilitation information is educational and does not establish that the organization has any particular staff, services, equipment, hours, address, or contact details. Verify all facility-specific information through instructor-approved official sources before representing the chatbot as a real provider.
