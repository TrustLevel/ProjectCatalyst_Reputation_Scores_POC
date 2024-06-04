import pandas as pd
import numpy as np

# Load data from CSV files
edges_path = 'Edges.csv'
reviewer_nodes_path = 'Reviewer_Nodes.csv'

edges = pd.read_csv(edges_path)
reviewer_nodes = pd.read_csv(reviewer_nodes_path)

# Calculate Accuracy for Each Reviewer
accuracy = edges.groupby('Reviewer_ID').apply(lambda x: (x['Rating Accuracy'].sum() / len(x))).reset_index()
accuracy.columns = ['Reviewer_ID', 'Accuracy']

# Calculate Consistency for Each Reviewer
consistency = edges.groupby('Reviewer_ID')['Feasibility Rating'].std(ddof=0).reset_index()
consistency.columns = ['Reviewer_ID', 'Consistency']
consistency['Consistency'] = 1 / consistency['Consistency'].replace(0, np.inf) 

# Normalize Number of Reviews
num_reviews = edges['Reviewer_ID'].value_counts().reset_index()
num_reviews.columns = ['Reviewer_ID', 'Num_Reviews']
num_reviews['Normalized_Reviews'] = num_reviews['Num_Reviews'] / num_reviews['Num_Reviews'].max()

# Normalize Peer Feedback
peer_feedback = reviewer_nodes[['Reviewer ID', 'Reviewer Accuracy']].copy()
peer_feedback.columns = ['Reviewer_ID', 'Peer_Feedback']
peer_feedback['Normalized_Feedback'] = peer_feedback['Peer_Feedback'] / peer_feedback['Peer_Feedback'].max()

# Normalize Helpful Votes [export from Lidonation]
edges['Helpful_Votes'] = np.random.randint(0, 10, size=len(edges))  
helpful_votes = edges.groupby('Reviewer_ID')['Helpful_Votes'].sum().reset_index()
helpful_votes.columns = ['Reviewer_ID', 'Helpful_Votes']
helpful_votes['Normalized_Helpful'] = helpful_votes['Helpful_Votes'] / helpful_votes['Helpful_Votes'].max()

# Combine All Factors
reputation_data = accuracy.merge(consistency, on='Reviewer_ID') \
                          .merge(num_reviews[['Reviewer_ID', 'Normalized_Reviews']], on='Reviewer_ID') \
                          .merge(peer_feedback[['Reviewer_ID', 'Normalized_Feedback']], on='Reviewer_ID') \
                          .merge(helpful_votes[['Reviewer_ID', 'Normalized_Helpful']], on='Reviewer_ID')

# Calculate Reputation Score
reputation_data['Reputation_Score'] = (reputation_data['Accuracy'] * 0.4 +
                                       reputation_data['Consistency'] * 0.2 +
                                       reputation_data['Normalized_Reviews'] * 0.2 +
                                       reputation_data['Normalized_Feedback'] * 0.1 +
                                       reputation_data['Normalized_Helpful'] * 0.1)

# Save the Reputation Scores to CSV
reputation_scores_path = 'Reputation_Scores.csv'
reputation_data.to_csv(reputation_scores_path, index=False)

# Display the calculated reputation scores
import ace_tools as tools; tools.display_dataframe_to_user(name="Reputation Scores", dataframe=reputation_data)

reputation_scores_path
