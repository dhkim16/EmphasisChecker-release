def create_emb_dict():
    nlp = stanza.Pipeline(lang='en', processors='tokenize, mwt, pos, lemma, depparse')

    # Make an embedding dictionary 
    max_extremum = ["maximum", "max", "peak", 'greatest', 'high']
    min_extremum = ["minimum", "min", "least", 'lowest', 'low']
    increase_trend_lemma_word = ["increase", "rise", "jump", "spike", "add", "grow", "high", "inflation", "rebounding", "uptick", 'soar', 'surge', 'raise', 'up', 'go']
    decrease_trend_lemma_word = ["decrease", "drop", "decline", "down", "negative", "low", "sink", "plummet", "fall", 'dip']


    increase_trend_sentences = ["As the global economy shows signs of rebounding, apositive assessments of the economic situation have risen in several major advanced economies since last year.",
                                "Positive views of the economy have sharply increased in countries like Australia and the United Kingdom.",
                                "Inflation surged to a new pandemic-era peak in June, with US consumer prices jumping by 9.1% year-over-year, according to fresh data released Wednesday by the Bureau of Labor Statistics. ",
                                "This indicator spikes around events that increase economic policy uncertainty. ",
                                "In May 390,000 jobs were added, leaving the total number of jobs 822,000 shy of pre-pandemic levels. ",
                                "An opposite trend of growth in the TFP and rebate rate also suggests a negative relationship between firm productivity and export tax rebates. This evidence is consistent with hypothesis H1",
                                "Figure 1 illustrates the evolution of MES from 1980 to 2012. It is observed that values of systemic risk indicator are remarkably high around 1987, 1998 and 2008, which indicate the heavy consequences of Black Monday, the Asian financial crisis and the GFC, respectively.",
                                "As the global economy shows signs of rebounding, apositive assessments of the economic situation have risen in several major advanced economies since last year.",
                                "But it represents a significant uptick over the low point in the late spring of 2020, when only around 1 million Americans or fewer left the United States."]

    decrease_trend_sentences = ["According to the World Bank, in 2013, Mongolia was  in 19th position in the global ranking of CO2 emissions per  capita,  with  13.50  t  CO2  emissions  per  capita  (figure 7.3), which is more than the double the global average  (4.90  t).  In  2014,  Mongolia  ranked  fifty fourth and its CO2 emissions per capita had decreased to 7.12 t.",
                                "Let's stick with wheat prices. There are a few reasons they've dropped almost 45% from their March highs and are down nearly 25% since the beginning of May, per wheat futures traded in Chicago.",
                                "We further illustrate the trend of the growth of firm TFP and industry rebate rate in Fig. 2. The TFP of firms has increased over time, while the rebate rate of the industries has declined during the sample period.",
                                "Snap's stock is down 74% in the past year. The market sell-off has battered social media firms./The company posted a quarterly net loss of $422 million, compared to a $152 million loss in the same quarter last year. As recession concerns grow, Snap (SNAP) is finding it hard to convince digital advertisers to come onboard.",
                                "We further illustrate the trend of the growth of firm TFP and industry rebate rate in Fig. 2. The TFP of firms has increased over time, while the rebate rate of the industries has declined during the sample period. An opposite trend of growth in the TFP and rebate rate also suggests a negative relationship between firm productivity and export tax rebates.",
                                "The unionization rate is low in France: on average, it amounted to 10.8 percent in 2016, which is among the lowest rates among Organisation for Economic Co-operation and Development (OECD) countries. However, despite some uncertainty about estimation, it appears to have been relatively stable since the early 1990s, after two phases of important decline (between 1950 and 1960, and between 1975 and 1990.",
                                "Regarding membership of employers’ organizations, estimates from the Visser database (Visser 2019) suggest that the percentage of employees working in private sector firms belonging to an employers’ organization out of the total number of employees in employment in the private sector stood at 43.7 per cent in 2008 (58.4 per cent if we include all employers and not just those in the private sector) and 25 per cent in 2013.4 The European Social Survey (ESS) data for the years 2002–10 show that membership of people aged between 30 and 64 years of age is low and declining (see Figure 8.9).",
                                "The unionization rate is low in France: on average, it amounted to 10.8 percent in 2016, which is among the lowest rates among Organisation for Economic Co-operation and Development (OECD) countries. However, despite some uncertainty about estimation, it appears to have been relatively stable since the early 1990s, after two phases of important decline (between 1950 and 1960, and between 1975 and 1990; see Figure 6.1).",
                                "Overall, global CO2-FFI emissions are estimated to have declined by 5.8% (5.1%-6.3%) in 2020, or about 2.2 (1.9-2.4) GtCO2 in total. This exceeds any previous global emissions decline since 1970 both in relative and absolute terms (Box TS.1 Figure 1). During  periods  of  economic  lockdown,  daily  emissions,  estimated  based  on  activity  and  power generation data, declined substantially compared to 2019, particularly in April 2020 –as shown in Box TS.1 Figure  1  –  but rebounded  by the  end of 2020."]

    target_word_embeddings = {}
    trend_sents = increase_trend_sentences + decrease_trend_sentences

    for i, text in enumerate(trend_sents):
        tokenized_text, tokens_tensor, segments_tensors = self.bert_text_preparation(text, tokenizer)
        list_token_embeddings = self.get_bert_embeddings(tokens_tensor, segments_tensors, model)
        
        ## Find the position of lemmatized version of 'increase' in list of tokens
        # Lemmatize
        for token in tokenized_text:
            lemma_tt = nlp(token).sentences[0].words[0].lemma
            
            if lemma_tt in increase_trend_lemma_word:
                word_index = tokenized_text.index(token)

                word_embedding = list_token_embeddings[word_index]
                target_word_embeddings[lemma_tt] = word_embedding

            elif lemma_tt in decrease_trend_lemma_word:
                word_index = tokenized_text.index(token)

                word_embedding = list_token_embeddings[word_index]
                target_word_embeddings[lemma_tt] = word_embedding

    self.emb_dict = target_word_embeddings

    ## For Debugging
    with open("emb_dict.pickle", "wb") as f:
        pickle.dump(self.emb_dict, f)

if __name__ == '__main__':
    print("Exercise")