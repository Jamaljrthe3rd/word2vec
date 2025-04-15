import streamlit as st
import sys
import subprocess

# Function to install required packages
def install_packages():
    packages = ['nltk', 'gensim', 'scikit-learn', 'matplotlib', 'pandas', 'numpy']
    for package in packages:
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
        except Exception as e:
            st.error(f"Failed to install {package}: {str(e)}")
            return False
    return True

# Ensure packages are installed before importing
if not 'packages_installed' in st.session_state:
    st.session_state.packages_installed = install_packages()

if st.session_state.packages_installed:
    import nltk
    from nltk.corpus import reuters, stopwords
    from gensim.models import Word2Vec
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np

    # Download necessary NLTK datasets with explicit error handling
    @st.cache_resource
    def download_nltk_data():
        try:
            # Download NLTK data at the start
            nltk.download('punkt')
            nltk.download('reuters')
            nltk.download('stopwords')
            return True
        except Exception as e:
            st.error(f"Failed to download NLTK data: {str(e)}")
            return False

    @st.cache_resource
    def load_and_process_data():
        if not download_nltk_data():
            return None
            
        try:
            stop_words = set(stopwords.words('english'))
            corpus_sentences = []
            
            # Show loading progress
            with st.spinner("Loading Reuters corpus and training Word2Vec model..."):
                # Safely load Reuters data
                for fileid in reuters.fileids():
                    try:
                        raw_text = reuters.raw(fileid)
                        tokenized_sentence = [
                            word.lower() 
                            for word in nltk.word_tokenize(raw_text) 
                            if word.isalnum() and word.lower() not in stop_words
                        ]
                        corpus_sentences.append(tokenized_sentence)
                    except Exception as e:
                        st.warning(f"Skipping file {fileid} due to error: {str(e)}")
                        continue
                
                # Train Word2Vec model
                model = Word2Vec(sentences=corpus_sentences, vector_size=100, window=5, min_count=5, workers=4)
                return model
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            return None

    # Main app
    st.title("Word2Vec Interactive Visualization")
    st.write("This app trains a Word2Vec model on the Reuters dataset and allows exploration of word embeddings.")

    # Load model
    model = load_and_process_data()

    if model is not None:
        # User input for word similarity
        word = st.text_input("Enter a word to find similar words:", "money")
        if word in model.wv:
            similar_words = model.wv.most_similar(word)
            st.write("Top similar words:")
            st.write(pd.DataFrame(similar_words, columns=['Word', 'Similarity']))
        else:
            st.write("Word not found in vocabulary.")

        # Visualization of embeddings
        if st.button("Visualize Word Embeddings"):
            words = list(model.wv.index_to_key)[:100]
            vectors = np.array([model.wv[word] for word in words])
            tsne = TSNE(n_components=2, random_state=42)
            reduced_vectors = tsne.fit_transform(vectors)
            
            plt.figure(figsize=(10, 6))
            plt.scatter(reduced_vectors[:, 0], reduced_vectors[:, 1])
            for i, word in enumerate(words):
                plt.annotate(word, (reduced_vectors[i, 0], reduced_vectors[i, 1]))
            st.pyplot(plt)
