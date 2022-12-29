import streamlit as st
def main():
    
    
    # Settings
    st.set_page_config(page_title = 'Keyword Analysis') 
    
    #==============================================================================
    # Imports
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    
    from collections import Counter
    from itertools import combinations
    
    import spacy
    nlp = spacy.load("en_core_web_md")
    from spacy_streamlit import visualize_ner
    
    from sklearn.datasets import fetch_20newsgroups
    
    import networkx as nx

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
    # Title
    st.title('Textual Analysis - Keyword identification and mapping')    
    st.write('Piers Walker 2022. https://github.com/pierswalker71')
    
    #==============================================================================   

    # Get data
    st.header('Import data')
    
    news_category = st.selectbox('Select category from "20 news groups" dataset',['comp.windows.x','rec.sport.baseball','rec.sport.hockey'])

    #categories = ['alt.atheism', 'comp.graphics',
    #              'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware','comp.os.ms-windows.misc',
    #              'comp.windows.x', 'misc.forsale', 'rec.autos',
    #              'rec.motorcycles', 'rec.sport.baseball',
    #              'rec.sport.hockey', 'sci.crypt', 'sci.electronics',
    #              'sci.med', 'sci.space', 'soc.religion.christian',
    #              'talk.politics.guns', 'talk.politics.mideast',
    #              'talk.politics.misc', 'talk.religion.misc']

    newsgroups = fetch_20newsgroups(categories=[news_category],remove=('headers', 'footers', 'quotes'))
    text = [x.replace('\n', ' ') for x in newsgroups.data]
    data = pd.DataFrame(data={'text':text})

    

    data = data.iloc[:100,:]

    
    
    # Display whole dataset
    with st.expander('Display data table'):
        st.dataframe(data)

    # Display example NER
    st.header('Named entity recognition example')

    with st.expander('Named entity recognition example'):
        st.write('Named entity recognition example')
        eg_text_row = st.slider('Example row ID',0,len(data.index)-1,0)
        eg_text_length = st.slider('Example text length',5,len(data.iloc[eg_text_row,0]),min(300,int(len(data.iloc[eg_text_row,0])*0.8)))
    
    doc_example = nlp(data.iloc[eg_text_row,0][:eg_text_length])
    visualize_ner(doc_example, labels=nlp.get_pipe("ner").labels,title='', show_table=False)
    #https://github.com/explosion/spacy-streamlit
    
    #fig,ax = plt.subplots(figsize=(15,6))
    #st.pyplot(fig)
    
    #==============================================================================
    # Processing data
    
    
    #-----------------------------------------------
    # Build df for all pages with named entities for every row
    label_df = pd.DataFrame(columns =['page','entity','label', 'count'])

    #for row_id in range(data.shape[0]-1):
    for row_id in range(50):
        doc = nlp(data.iloc[:,0][row_id])
        page_title = str(row_id)
        #doc.user_data['title'] = page_title
        label_df = pd.concat([label_df,ner_to_dataframe(doc,page_title)])

    label_df.reset_index(drop=True,inplace=True)

    #-----------------------------------------------
    # filter relevant entities

    # Add up all occurrences of labels across all pages to find most common 
    label_count = label_df.groupby(['label'], as_index=False)['count'].sum()
    label_count.sort_values(by='count', ascending=False, inplace=True)
    top_label_count = label_count['label'].tolist()[:5]

    # Create list of entities which have produced most common labels
    # Use this list to filter 
    top_label_count_entities = []

    for label in top_label_count:
        for x in label_df[label_df['label']==label]['entity'].tolist ():
            top_label_count_entities.append(x )

    top_label_count_entities = list(set(top_label_count_entities))
    top_label_count_entities = [x for x in top_label_count_entities if x not in ['CARDINAL', 'ORDINAL']]

    selected_entities = top_label_count_entities
    filtered_label_df = label_df[label_df['entity'].isin(selected_entities)]

    # Create list of pages with found entities (just in case some are empty)
    pages = list(set(filtered_label_df['page'].tolist()))
    pages = [int(x) for x in pages]
    pages.sort()
    pages = [str(x) for x in pages]

    # Loop around each set of pages to extract the labels from all filtered entity categories
    labels_loop = []
    for page in pages:
        labels = filtered_label_df[filtered_label_df['page'] == page]['label'].tolist()
        labels = [x.lower() for x in labels]
        labels = list(set(labels))
        labels.sort()
        labels_loop.append(labels)
    
    # Create df containing page id and list of unique labels
    unique_labels_per_page = pd.DataFrame(data={'page':pages,'labels':labels_loop})
 
    #-----------------------------------------------
    # Build dataframe with connections of labels within each page required by networkx
    networkx_data = pd.DataFrame(columns=['source','target'])

    # Add pairs of labels to networks
    for row in range(len(unique_labels_per_page)):
        # Get pairs of labels
        pairs_of_labels = list(combinations(unique_labels_per_page.iloc[row,1], 2))

        for pair in pairs_of_labels:
            networkx_data.loc[len(networkx_data.index)] = pair

    # Create graph
    G = nx.from_pandas_edgelist(networkx_data,source='source',target='target')

    # Determine most connected nodes
    node_to_neighbors_mapping = [(node, len(list(G.neighbors(node)))) for node in G.nodes()]
    node_to_neighbors_ser = pd.Series(data=dict(node_to_neighbors_mapping))
    node_to_neighbors_ser.sort_values(ascending=False).head()
    top_nodes = [x for x in node_to_neighbors_ser.sort_values(ascending=False).index[:6]]
    
    #==============================================================================
    # Network
    st.header('Network')
    
    fig, ax = plt.subplots(figsize=(10,10), dpi=200)
    #fig, ax = plt.subplots(figsize=(8, 8), dpi=200)

    color_map = ['red' if node in top_nodes else 'b' for node in G]
    pos = nx.spring_layout(G, k=0.15, iterations=20)
    #pos=pos,, node_size=50, with_labels=True, font_weight='bold',font_size=6
    #nx.draw(G, ax=ax,node_color=color_map)
    #st.pyplot(fig)
    
    
    #dot = nx.nx_pydot.to_pydot(G)
    #st.graphviz_chart(dot.to_string())


    #plt.figure()    
    nx.draw(G,pos=pos,ax=ax,edge_color='black',width=1,linewidths=1, node_size=10,node_color=color_map,with_labels=True, font_weight='normal',font_size=6)
    #plt.axis('on')
    #plt.show()
    st.pyplot(fig)

    st.header('Next section')
    
    
#==============================================================================

if __name__ == '__main__':
    main()
