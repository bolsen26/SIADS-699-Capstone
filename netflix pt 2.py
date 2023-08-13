from wordcloud import WordCloud

# Generate word cloud
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_texts)

# Display the word cloud using matplotlib
plt.figure(figsize=(10, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud of Top Terms in Netflix Subreddit')
plt.show()

# Extract mentions of shows/movies
show_mentions = [word for word in filtered_tokens if word not in custom_stopwords]

# Count show mentions
show_mention_counts = Counter(show_mentions)

# Visualize top mentioned shows/movies
top_shows = show_mention_counts.most_common(10)
show_names, show_counts = zip(*top_shows)

plt.figure(figsize=(12, 6))
plt.barh(show_names, show_counts, color='lightgreen')
plt.xlabel('Mentions')
plt.title('Top Mentioned Shows/Movies in Netflix Subreddit')
plt.gca().invert_yaxis()
plt.show()

# Define a list of popular shows/movies to analyze
popular_shows = ["stranger things", "the crown", "narcos", "the witcher"]

# Perform sentiment analysis for each show
show_sentiments = {}
for show in popular_shows:
    show_comments = [comment for comment in breaking_bad_comments if show in comment.lower()]
    sentiments = [TextBlob(comment).sentiment.polarity for comment in show_comments]
    show_sentiments[show] = sentiments

# Visualize sentiment comparison
plt.figure(figsize=(10, 6))
for show, sentiments in show_sentiments.items():
    plt.hist(sentiments, bins=20, alpha=0.5, label=show)
plt.title('Sentiment Comparison for Popular Shows/Movies Comments')
plt.xlabel('Polarity')
plt.ylabel('Number of Comments')
plt.legend()
plt.show()

from datetime import datetime

# Analyze when user engagement is the highest using heatmap
import numpy as np
from matplotlib.colors import LogNorm

# Extract hours of the day from comment and post timestamps
comment_hours = [comment_timestamp.hour for comment_timestamp in comment_timestamps]
post_hours = [post_timestamp.hour for post_timestamp in post_timestamps]

# Create a 2D histogram for comment activity by hour
hist, xedges, yedges = np.histogram2d(comment_hours, range(len(comment_hours)), bins=(24, len(comment_hours)))
hist = hist.T

# Plot heatmap
plt.figure(figsize=(12, 6))
plt.imshow(hist, aspect='auto', interpolation='none', norm=LogNorm())
plt.colorbar(label='Log Count')
plt.xlabel('Hour of the Day')
plt.ylabel('Time')
plt.title('Comment Activity Heatmap by Hour of the Day')
plt.xticks(range(24), labels=[str(i) for i in range(24)])
plt.yticks([])
plt.show()


# Extract post timestamps and comment timestamps
post_timestamps = [datetime.fromtimestamp(post.created_utc) for post in top_posts]
comment_timestamps = [datetime.fromtimestamp(comment.created_utc) for post in top_posts for comment in post.comments if not isinstance(comment, praw.models.MoreComments)]

# Plot post and comment activity over time
plt.figure(figsize=(12, 6))
plt.plot(post_timestamps, range(len(post_timestamps)), label='Posts', color='orange')
plt.plot(comment_timestamps, range(len(comment_timestamps)), label='Comments', color='blue')
plt.xlabel('Time')
plt.ylabel('Activity Count')
plt.title('Post and Comment Activity Over Time')
plt.legend()
plt.show()

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# Create a document-term matrix
vectorizer = CountVectorizer(max_df=0.85, stop_words=stop_words)
dtm = vectorizer.fit_transform(all_texts.split())

# Apply LDA
num_topics = 5  # Define the number of topics
lda_model = LatentDirichletAllocation(n_components=num_topics, random_state=42)
lda_model.fit(dtm)

# Display top words for each topic
for index, topic in enumerate(lda_model.components_):
    top_words = [vectorizer.get_feature_names()[i] for i in topic.argsort()[-10:]]
    print(f"Topic {index+1}: {', '.join(top_words)}")

# Calculate average upvotes and comments per post
average_upvotes = sum([post.score for post in top_posts]) / len(top_posts)
average_comments = sum([len(post.comments) for post in top_posts]) / len(top_posts)

print(f"Average Upvotes per Post: {average_upvotes:.2f}")
print(f"Average Comments per Post: {average_comments:.2f}")

# Visualize the distribution of user engagement using scatterplot
upvotes = [post.score for post in top_posts]
comments = [len(post.comments) for post in top_posts]

plt.figure(figsize=(10, 6))
plt.scatter(upvotes, comments, color='green', alpha=0.5)
plt.xlabel('Upvotes')
plt.ylabel('Number of Comments')
plt.title('User Engagement Distribution')
plt.show()


import spacy

nlp = spacy.load('en_core_web_sm')

# Apply Named Entity Recognition
doc = nlp(all_texts)
entities = [(ent.text, ent.label_) for ent in doc.ents]

# Count and visualize named entities
entity_counts = Counter(entities)
top_entities = entity_counts.most_common(10)
entities, counts = zip(*top_entities)

plt.figure(figsize=(10, 6))
plt.barh(entities, counts, color='purple')
plt.xlabel('Counts')
plt.title('Top Named Entities in Netflix Subreddit')
plt.gca().invert_yaxis()
plt.show()