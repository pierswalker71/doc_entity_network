import streamlit as st
def main():
    
    #==============================================================================
    # Imports
    
    import pandas as pd
    
    #import spacy
    #en = spacy.load("en_core_web_sm")
    #en = spacy.load("en_core_web_md")


    #from itertools import combinations

    #import spacy
    #nlp = spacy.load('en_core_web_sm')
    #import spacy.cli
    #spacy.cli.download("en_core_web_lg")
    #nlp = spacy.load("en_core_web_lg")

    #from spacy import displacy

    #!pip install 'networkx<2.7'
    #import networkx as nx

    #==============================================================================
    # Functions

    def ner_to_dataframe(doc,page_title,most_common_num=5):
        # Searches a spacy doc for named entities, returning a df 

        # Start with blank df, so that an empty df may be returned
        df = pd.DataFrame(columns =['page','entity','label', 'count'])

        # Create list of all the found named entities eg ORG or PERSON
        named_entities = []
        for word in doc.ents:
            if word.label_ not in named_entities: named_entities.append(word.label_)

        # Go through each named entities and find the most common results from each
        for i, entity in enumerate(named_entities):
            label_list = []

            for word in doc.ents:
                if word.label_ == entity:
                    label_list.append(word.text)
            label_counts = Counter(label_list).most_common(most_common_num)
            label_counts = [(page_title,entity,) + t for t in label_counts]

            # Build dataframe
            df = pd.concat([df,pd.DataFrame(label_counts, columns =['page','entity','label', 'count'])])

        df.reset_index(drop=True,inplace=True)
    
        return df


    #==============================================================================


    #from sklearn.datasets import fetch_20newsgroups

#categories = ['alt.atheism', 'comp.graphics',
#              'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware','comp.os.ms-windows.misc',
#              'comp.windows.x', 'misc.forsale', 'rec.autos',
#              'rec.motorcycles', 'rec.sport.baseball',
#              'rec.sport.hockey', 'sci.crypt', 'sci.electronics',
#              'sci.med', 'sci.space', 'soc.religion.christian',
#              'talk.politics.guns', 'talk.politics.mideast',
#              'talk.politics.misc', 'talk.religion.misc']

    #newsgroups = fetch_20newsgroups(categories=['comp.windows.x'],remove=('headers', 'footers', 'quotes'))
    #text = [x.replace('\n', ' ') for x in newsgroups.data]
    #data = pd.DataFrame(data={'text':text})
    #display(data.head(5))
    #row = 1
    #data.iloc[row,0]
    ##'sci.electronics',rec.motorcycles

    #doc = nlp(data.iloc[row,0])
    #displacy.render(doc,style="ent",jupyter=True)


#==============================================================================


if __name__ == '__main__':
    main()
