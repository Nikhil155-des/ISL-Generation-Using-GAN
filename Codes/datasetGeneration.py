import pandas as pd
import random

# Expanded vocabulary pools
subjects = [
    "I", "She", "He", "They", "We", "You", "It", 
    "The teacher", "The scientist", "The politician", 
    "Children", "Dogs", "Cats", "Birds", "Everyone", "No one", 
    "The doctor", "The artist", "The engineer", "The player", 
    "The student", "The farmer", "The driver", "The nurse"
]
verbs = [
    "eat", "read", "play", "watch", "love", "chase", "solve", "prepare", 
    "drink", "drive", "teach", "invent", "write", "design", "build", 
    "sing", "dance", "study", "analyze", "explore", "imagine", "create", 
    "discover", "paint", "repair", "help", "manage", "organize", "lead"
]
objects = [
    "pizza", "books", "football", "television", "cats", "food", "problem", 
    "coffee", "car", "dinner", "truth", "theory", "garden", "mountains", 
    "ocean", "history", "science", "art", "music", "poem", "building", 
    "story", "game", "technology", "research", "project", "robot", "design"
]
adjectives = [
    "happy", "old", "fast", "small", "tired", "young", "hungry", "smart", 
    "busy", "tall", "brave", "curious", "kind", "funny", "serious", 
    "friendly", "quiet", "energetic", "strong", "polite", "lazy", "angry", 
    "clever", "silly", "bold", "shy", "nervous", "confident", "excited"
]
adverbs = [
    "quickly", "carefully", "easily", "slowly", "silently", "gracefully", 
    "clearly", "happily", "sadly", "angrily", "loudly", "calmly", "bravely", 
    "cautiously", "nervously", "confidently", "energetically", "quietly", 
    "smoothly", "surprisingly", "gently", "fiercely", "casually", "awkwardly"
]
locations = [
    "in the park", "in the kitchen", "on the tree", "at school", "in the library", 
    "on the stage", "in the office", "on the roof", "under the bridge", 
    "in the forest", "by the lake", "on the beach", "in the garden", 
    "at the market", "in the classroom", "on the mountain", "in the city", 
    "in the village", "on the road", "at the museum"
]
times = [
    "in the morning", "at night", "yesterday", "tomorrow", "last week", 
    "next year", "this afternoon", "a few minutes ago", "later today", 
    "last night", "next month", "in the evening", "during the holidays", 
    "this weekend", "on Monday", "on New Year's Eve"
]
conjunctions = [
    "and", "but", "or", "because", "although", "while", "so", "if", 
    "even though", "as", "since", "when", "after", "before", "though", 
    "unless", "whereas", "once", "until"
]
questions = [
    "What are you eating?", "Where is she going?", "Can you help me?", 
    "Do you like coffee?", "When will they arrive?", "How did he solve the puzzle?", 
    "Why is it important?", "What time is it?", "Where have they gone?", 
    "Who is responsible for this?", "Which book should I read?", 
    "How much does it cost?", "Why did they leave?", "When will it be ready?"
]
commands = [
    "Please bring water.", "Open the window.", "Call the manager.", 
    "Clean the whiteboard.", "Solve this equation.", "Prepare the presentation.", 
    "Organize the files.", "Fix the issue.", "Complete the task.", "Make the call."
]

# Sentence generators
def create_sentence():
    subject = random.choice(subjects)
    verb = random.choice(verbs)
    obj = random.choice(objects)
    adjective = random.choice(adjectives)
    adverb = random.choice(adverbs)
    location = random.choice(locations)
    time = random.choice(times)
    english = f"The {adjective} {subject} {verb}s the {obj} {adverb} {location} {time}."
    sov = f"The {adjective} {subject} {obj} {adverb} {location} {time} {verb}s."
    return english, sov

# Generate the dataset
data = []
rows = 10000  # Remove the comma after 10,000

for _ in range(rows):
    english, sov = create_sentence()
    data.append({"English Sentence (SVO)": english, "SOV for ISL": sov})

# Save the dataset
df = pd.DataFrame(data)
df.to_csv("enhanced_svo_to_sov_dataset.csv", index=False)
print("Enhanced dataset saved to 'enhanced_svo_to_sov_dataset.csv'")
