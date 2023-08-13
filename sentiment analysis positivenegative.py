import praw
from textblob import TextBlob

# Initialize Reddit API wrapper
reddit = praw.Reddit(
    client_id="uTp5HipGLhBhPv1S7oxk8A",
    client_secret="AcT-azqNdIT3onTltclkX4Q3ISUDDQ",
)

# Subreddit name for the brand
subreddit_name = 'netflix'

# Get subreddit instance
subreddit = reddit.subreddit(subreddit_name)

# Collect posts and comments from the subreddit
posts = subreddit.new(limit=100)  # Adjust the limit as needed

# Analyze sentiment of each post and its comments
for post in posts:
    post_text = post.title
    post_sentiment = TextBlob(post_text).sentiment.polarity

    print(f"Post Sentiment: {post_sentiment:.2f} - {post_text}")

    # Analyze sentiment of comments on the post
    post.comments.replace_more(limit=None)
    for comment in post.comments.list():
        comment_text = comment.body
        comment_sentiment = TextBlob(comment_text).sentiment.polarity

        print(f"Comment Sentiment: {comment_sentiment:.2f} - {comment_text}")


def get_sentiment_category(sentiment_polarity):
    if sentiment_polarity > 0.1:
        return "positive"
    elif sentiment_polarity < -0.1:
        return "negative"
    else:
        return "neutral"

reddit = praw.Reddit(
    client_id="uTp5HipGLhBhPv1S7oxk8A",
    client_secret="AcT-azqNdIT3onTltclkX4Q3ISUDDQ"
)

# Subreddit name for the brand
subreddit_name = 'netflix'

# Get subreddit instance
subreddit = reddit.subreddit(subreddit_name)

# Collect posts and comments from the subreddit
posts = subreddit.new(limit=100)

# Analyze sentiment of each post and its comments
for post in posts:
    post_text = post.title
    post_sentiment = TextBlob(post_text).sentiment.polarity
    post_sentiment_category = get_sentiment_category(post_sentiment)

    print(f"Post Sentiment: {post_sentiment_category} - {post_text}")

    # Analyze sentiment of comments on the post
    post.comments.replace_more(limit=None)
    for comment in post.comments.list():
        comment_text = comment.body
        comment_sentiment = TextBlob(comment_text).sentiment.polarity
        comment_sentiment_category = get_sentiment_category(comment_sentiment)

        print(f"Comment Sentiment: {comment_sentiment_category} - {comment_text}")
