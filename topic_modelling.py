pip install wordcloud

pip install pyLDAvis scikit-learn

pip install pyLDAvis

import pandas as pd
import jsonlines

def read_json_chunks(file_path, chunk_size):
    def json_generator():
        for chunk in pd.read_json(file_path, lines=True, chunksize=chunk_size):
            yield chunk

    generator = json_generator()
    df = next(generator)
    return df

chunk_size = 100000
dataset_path_review ='.json'

df_review = read_json_chunks(dataset_path_review, chunk_size)

df_review.info()

# Preprocessing steps of the data

import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download('punkt')
nltk.download('stopwords')

reviews = df_review['text'].values

stop_words = set(stopwords.words('english'))
processed_reviews = []

for review in reviews:
    # Tokenize the review text
    tokens = word_tokenize(review.lower())

    # Remove stop words
    filtered_tokens = [token for token in tokens if token not in stop_words]

    # Reconstruct the processed text
    processed_review = ' '.join(filtered_tokens)

    # Append the processed review to the list
    processed_reviews.append(processed_review)

    # Perform Non-Matrix Factorization

    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import NMF

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(processed_reviews)

    # Perform NMF with n_components as the desired number of components
    n_components = 5
    nmf = NMF(n_components=n_components)
    nmf.fit(X)

    # Get the non-negative components and the transformed data
    components = nmf.components_
    transformed_data = nmf.transform(X)

    print("Components:")
    print(components)
    print("Transformed Data:")
    print(transformed_data)

    import pandas as pd
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    from gensim import corpora

    # Prepare data for LDA Analysis
    stop_words = set(stopwords.words('english'))


    def preprocess_text(text):
        tokens = word_tokenize(text.lower())  # Tokenization
        tokens = [token for token in tokens if token.isalpha()]  # Remove non-alphabetic tokens
        tokens = [token for token in tokens if token not in stop_words]  # Remove stopwords
        return tokens


    df_review['processed_text'] = df_review['text'].apply(preprocess_text)
    # Explore prepared data with word cloud

    # Import the wordcloud library
    from wordcloud import WordCloud

    # Join the different processed titles together.
    long_string = ','.join(list(df_review['text'].values))
    # Create a WordCloud object
    wordcloud = WordCloud(background_color="white", max_words=5000, contour_width=3, contour_color='steelblue')
    # Generate a word cloud
    wordcloud.generate(long_string)
    # Visualize the word cloud
    wordcloud.to_image()

    # Create a dictionary from the preprocessed text
    dictionary = corpora.Dictionary(df_review['processed_text'])

    # Create the document-term matrix
    corpus = [dictionary.doc2bow(doc) for doc in df_review['processed_text']]

    # Perform LDA Analysis for Topic Modeling

    from gensim.models import LdaModel

    num_topics = 20  # Number of topics to generate
    passes = 10  # Number of passes through the corpus during training

    # Train the LDA model
    lda_model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics, passes=passes)

    # Print the topics and associated words
    topics = lda_model.print_topics(num_words=10)
    for topic in topics:
        print(topic)

        # Assign topics to documents
        topic_assignments = [lda_model.get_document_topics(doc) for doc in corpus]

        # Get the topic distribution for a specific document
        document_index = 0  # Specify the document index
        document = df_review['processed_text'][document_index]
        document_bow = dictionary.doc2bow(document)
        document_topics = lda_model.get_document_topics(document_bow)
        print(document_topics)

        # check validity of the LDA model

        from gensim.models import CoherenceModel

        # Assuming you have trained the LDA model and have the corpus and dictionary available

        # Calculate coherence score
        coherence_model = CoherenceModel(model=lda_model, texts=df_review['processed_text'], dictionary=dictionary,
                                         coherence='c_v')
        coherence_score = coherence_model.get_coherence()

        # Print the coherence score
        print("Coherence Score:", coherence_score)

        # Check Topic Similarity and Topic Diversity Evaluation Metrics for LDA model

        from sklearn.metrics.pairwise import cosine_similarity

        # Assuming you have trained the LDA model and have the corpus and dictionary available

        # Get the topic-word distributions
        topics = lda_model.get_topics()

        # Compute topic similarity and diversity
        topic_similarities = []
        topic_diversities = []

        num_topics = lda_model.num_topics

        for i in range(num_topics):
            for j in range(i + 1, num_topics):
                similarity = cosine_similarity([topics[i]], [topics[j]])[0][0]
                diversity = 1 - similarity  # Calculate diversity as 1 - similarity
                topic_similarities.append(similarity)
                topic_diversities.append(diversity)

        average_similarity = sum(topic_similarities) / len(topic_similarities)
        average_diversity = sum(topic_diversities) / len(topic_diversities)

        print("Average Topic Similarity:", average_similarity)
        print("Average Topic Diversity:", average_diversity)

        # Check LDA Visualization

        import pyLDAvis
        import pyLDAvis.gensim_models as gensimvis

        pyLDAvis.enable_notebook()

        vis = gensimvis.prepare(lda_model, corpus, dictionary)
        vis
        # pyLDAvis.display(vis)


