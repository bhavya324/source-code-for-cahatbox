# source-code-for-cahatbox
# Importing necessary libraries
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import gradio as gr

# Load dataset
data = pd.read_csv("qa_pairs.csv")
questions = data["Question"].tolist()
answers = data["Answer"].tolist()

# Load sentence transformer model
model = SentenceTransformer("all-MiniLM-L6-v2")
question_embeddings = model.encode(questions)

# Define chatbot function
def chatbot(user_input):
    user_input = user_input.strip().lower()
    input_embedding = model.encode([user_input])
    sims = cosine_similarity(input_embedding, question_embeddings)[0]
    max_sim = max(sims)
    if max_sim > 0.6:
        index = sims.argmax()
        return answers[index]
    else:
        return "Sorry, I don't understand your question."

# Create Gradio Interface
iface = gr.Interface(fn=chatbot, inputs="text", outputs="text", title="Simple Chatbot")
iface.launch()
