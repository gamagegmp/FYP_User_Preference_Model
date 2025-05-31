from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from collections import Counter

# Load preprocessed dataset
df = pd.read_csv("Preprocessed_Dataset.csv")

# Select only preference columns
preference_columns = [
    'experience_beach', 'experience_adventure', 'experience_nature', 'experience_culture',
    'experience_nightlife', 'experience_history', 'experience_shopping', 'experience_cuisine',
    'scenery_urban', 'scenery_rural', 'scenery_sea', 'scenery_mountain',
    'scenery_lake', 'scenery_desert', 'scenery_plains', 'scenery_jungle',
    'preferred_region_europe', 'preferred_region_n_america', 'preferred_region_caribbean',
    'preferred_region_asia', 'preferred_region_s_america', 'preferred_region_mid_east',
    'preferred_region_africa', 'preferred_region_oceania'
]

user_preference_matrix = df[['user_id'] + preference_columns]

# Sample 500 users for testing
sample_df = user_preference_matrix.sample(n=10000, random_state=42).reset_index(drop=True)

# Important: set user_id as index for easier access by user_id
sample_df.set_index('user_id', inplace=True)

# Drop user_id for similarity calculation
preference_data = sample_df

# Calculate cosine similarity
cosine_sim = cosine_similarity(preference_data)

# Create similarity DataFrame indexed by user_id
similarity_df = pd.DataFrame(cosine_sim, index=sample_df.index, columns=sample_df.index)

# Define how many similar users to find
TOP_N = 500

# Store results in a dictionary
top_similar_users = {}

for user_id in similarity_df.index:
    # Get similarity scores, sort by highest, ignore self
    similar_scores = similarity_df.loc[user_id].drop(user_id)
    top_users = similar_scores.sort_values(ascending=False).head(TOP_N).index.tolist()
    
    # Save in dictionary
    top_similar_users[user_id] = top_users

# Prepare recommendations dictionary
recommendations = {}

num_similar_users = TOP_N

for user_id in top_similar_users.keys():
    user_prefs = sample_df.loc[user_id]
    similar_users = top_similar_users[user_id]
    
    new_prefs = []
    for sim_user in similar_users:
        sim_user_prefs = sample_df.loc[sim_user]
        for pref in preference_columns:
            if sim_user_prefs[pref] == 1 and user_prefs[pref] == 0:
                new_prefs.append(pref)

    pref_counts = Counter(new_prefs)
    
    ranked_prefs = [
        (pref, count, round((count / num_similar_users) * 100, 2))
        for pref, count in pref_counts.most_common()
    ]
    
    recommendations[user_id] = ranked_prefs[:3]

# Print output for first 5 users
for user_id, prefs in list(recommendations.items())[:5]:
    print(f"User {user_id} may also like:")
    for pref, count, percentage in prefs:
        print(f"  - {pref} ({count} users, {percentage}%)")





# Result

# User 21895 may also like:
#   - scenery_mountain (460 users, 46.0%)
#   - scenery_rural (443 users, 44.3%)
#   - experience_adventure (382 users, 38.2%)
# User 41037 may also like:
#   - scenery_lake (655 users, 65.5%)
#   - experience_cuisine (509 users, 50.9%)
#   - experience_culture (487 users, 48.7%)
# User 38357 may also like:
#   - experience_history (548 users, 54.8%)
#   - scenery_jungle (430 users, 43.0%)
#   - experience_beach (381 users, 38.1%)
# User 32097 may also like:
#   - experience_culture (468 users, 46.8%)
#   - scenery_lake (388 users, 38.8%)
#   - experience_nature (334 users, 33.4%)
# User 55731 may also like:
#   - experience_history (428 users, 42.8%)
#   - scenery_lake (202 users, 20.2%)
#   - scenery_sea (186 users, 18.6%)