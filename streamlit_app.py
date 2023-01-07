import streamlit as st
def main():
    
    
    # Settings
    st.set_page_config(page_title = 'Textual analysis') 
    
    #==============================================================================
    # Imports
    #import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    
    from collections import Counter
    from itertools import combinations
    import re
    
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
    st.write('This app analyses a series of news articles, identifying key words within each text and then linking the words across multiple texts.')
    
    #==============================================================================   
    # Get data
    st.header('Import data')
    st.write('Select news category')
    categories = ['sci.space', 'comp.graphics',
                  'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware','comp.os.ms-windows.misc',
                  'comp.windows.x', 'misc.forsale', 'rec.autos',
                  'rec.motorcycles', 'rec.sport.baseball',
                  'rec.sport.hockey', 'sci.crypt', 'sci.electronics',
                  'sci.med',  'soc.religion.christian',
                  'talk.politics.guns', 'talk.politics.mideast',
                  'talk.politics.misc', 'talk.religion.misc', 'alt.atheism']

    #categories = ['comp.windows.x','rec.sport.baseball','rec.sport.hockey']

    with st.expander('Change news category if required'):
        news_category = st.selectbox('Select a different category from the "20 news groups" dataset',categories)

    newsgroups = fetch_20newsgroups(categories=[news_category],remove=('headers', 'footers', 'quotes'))
    
    #-----------------------------------------------------------------
    # Process input data
    text = [x.lower() for x in newsgroups.data] # lower
    
    # Remove special characters
    text_ = []
    for t in text:
        new_t = re.sub(r"[^a-zA-Z0-9 ]", "", t)
        text_.append(new_t)
        
    text_ = [x.replace('\n', ' ') for x in text_]  # remove /n 
    text = text_  
    
    # Build core data frame
    data = pd.DataFrame(data={'text':text})
    data.dropna(how='all', inplace=True) # drop rows that just had special characters or where blank
  
    # Truncate long strings
    
    # Select just first few rows of data
    with st.expander('Change data length if required'):
        num_rows_required = st.slider('Select number of rows of data for processing',100,len(data.index),100)
    data = data.iloc[:num_rows_required,:]
    
    #-----------------------------------------------------------------  
    # Display dataset
    with st.expander('Display table of text inputs'):
        st.dataframe(data)

    # Display example NER
    st.header('Demonstration of conducting "named entity recognition" on the input data')
    st.write('A machine-learning model is used to identify key word entities within the texts')

    with st.expander('Named entity recognition example'):
        st.write('Select text for processing')
        eg_text_row = st.slider('Row ID - example data', 0, len(data.index)-1,0)
        eg_text_length = st.slider('Text length - example data', 5, len(data.iloc[eg_text_row,0]), min(300,int(len(data.iloc[eg_text_row,0])*0.8)))
    
    doc_example = nlp(data.iloc[eg_text_row,0][:eg_text_length])
    visualize_ner(doc_example, labels=nlp.get_pipe("ner").labels,title='', show_table=False)
    #https://github.com/explosion/spacy-streamlit
    
    #==============================================================================
    # Processing data
    
    
    #-----------------------------------------------
    # Build df for all pages with named entities for every row
    label_df = pd.DataFrame(columns =['page','entity','label', 'count'])

    
    #for row_id in range(data.shape[0]-1):
    #for row_id in range(len(data.index)-1):
    #for row_id in range(num_rows_required):
    for row_id in range(100):
        doc = nlp(data.iloc[:,0][row_id])
        page_title = str(row_id)
        #doc.user_data['title'] = page_title
        label_df = pd.concat([label_df,ner_to_dataframe(doc,page_title)])

    label_df.reset_index(drop=True,inplace=True)

    #-----------------------------------------------
    # Filter relevant entities - eg reduce set of ORG, Cardinal, Person etc

    # Add up all occurrences of labels across all pages to find most common 
    label_count = label_df.groupby(['label'], as_index=False)['count'].sum()
    label_count.sort_values(by='count', ascending=False, inplace=True)
    top_label_count = label_count['label'].tolist()[:5] ## TODO hardcoded num
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
    
    #-----------------------------------------------
    # Create list of pages with found entities (just in case some are empty)
    pages = list(set(filtered_label_df['page'].tolist()))
    pages = [int(x) for x in pages]
    pages.sort()
    pages = [str(x) for x in pages]
    
    #-----------------------------------------------
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
    # Determine most common nodes - those which appear on most num of pages
    top_labels, num_pages = [], []

    for label in filtered_label_df['label'].unique().tolist():
        df = filtered_label_df[filtered_label_df['label']==label]
        top_labels.append(label)
        num_pages.append(df['page'].nunique())

    labels_num_pages = pd.DataFrame(data={'label':top_labels, 'num pages':num_pages})
    labels_num_pages.sort_values(by='num pages', ascending=False, inplace=True)

    top_nodes = labels_num_pages.iloc[:5,0].tolist() ## TODO hardcoded num

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

    # Determine most connected nodes - old method
    #node_to_neighbors_mapping = [(node, len(list(G.neighbors(node)))) for node in G.nodes()]
    #node_to_neighbors_ser = pd.Series(data=dict(node_to_neighbors_mapping))
    #node_to_neighbors_ser.sort_values(ascending=False, inplace=True)
    #top_nodes = [x for x in node_to_neighbors_ser.sort_values(ascending=False).index[:6]]
    
    #==============================================================================
    # Network
    st.header('Visualisation of the network of key 🗝️ words')
    
    #fig, ax = plt.subplots(figsize=(10,10), dpi=200) # good for mobiles
    st.write('The whole network')
    fig, ax = plt.subplots(figsize=(12, 12), dpi=300)

    colour_map = ['red' if node in top_nodes else 'b' for node in G]
    pos = nx.spring_layout(G, k=0.15, iterations=20, seed=42)
   
    #nx.draw_networkx_edges(G, pos, alpha=0.3, width=edgewidth, edge_color="m")
    #nx.draw_networkx_nodes(G, pos, node_size=nodesize, node_color="#210070", alpha=0.9)
    #label_options = {"ec": "k", "fc": "white", "alpha": 0.7}
    #nx.draw_networkx_labels(G, pos, font_size=14, bbox=label_options)

    nx.draw(G, pos=pos, ax=ax, edge_color='black' ,width=1, linewidths=1, node_size=10,
            node_color=colour_map, with_labels=True, font_weight='normal', font_size=9)
    st.pyplot(fig)
    
    #==============================================================================
    st.header('Analysis')
    
    #-----------------------------------------------
    st.write('The words with the most connections i.e. found on the most number of pages')
    st.dataframe(labels_num_pages[labels_num_pages['num pages']>1])
    
    #-----------------------------------------------
    st.write('Texts with the most common word ')
    #st.dataframe(label_df[label_df['label'].isin(top_nodes[:1])]['page'])
    top_node_pages = label_df[label_df['label'].isin(top_nodes[:1])]['page'].tolist()
    
    st.dataframe(data.iloc[top_node_pages])

    with st.expander('Display text containing most common word'):
        st.write('Select text containing most common word')
        top_text_row = st.selectbox('Row ID - top word', top_node_pages, key='top_text_row')
        #top_text_length = st.slider('Text length - top word', 5, len(data.iloc[top_text_row,0]), min(300,int(len(data.iloc[top_text_row,0])*0.8)), key='top_text_length')
        top_text_length = 20
        doc_example = nlp(data.iloc[top_text_row,0][:top_text_length])
        visualize_ner(doc_example, labels=nlp.get_pipe("ner").labels, title='', show_table=False, key='top_node-ner')
    
    
    #-----------------------------------------------
    # Generate smaller trimmed network with only the connected items
    top_nodes_and_connections = []
    for node in top_nodes:
        top_nodes_and_connections_temp = nx.node_connected_component(G, node)
        #[top_nodes_and_connections.append(x) for x in nx.node_connected_component(G, node)]
        for x in nx.node_connected_component(G, node):
            top_nodes_and_connections.append(x)
    
    top_nodes_and_connections = list(set(top_nodes_and_connections))
    
    H = G.copy()
    for node in list(G):
        if node not in top_nodes_and_connections:
            H.remove_node(node)

    st.write('The network comprising only the most connected words')
    fig, ax = plt.subplots(figsize=(10, 10), dpi=150)
    colour_map = ['red' if node in top_nodes else 'b' for node in H]
    nx.draw(H, pos=pos, ax=ax, edge_color='black' ,width=1, linewidths=1, node_size=8,
            node_color=colour_map, with_labels=True, font_weight='normal', font_size=9)
    st.pyplot(fig) 



#==============================================================================

if __name__ == '__main__':
    main()
