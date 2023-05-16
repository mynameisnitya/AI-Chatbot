import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
import aiml
import requests

# Load the AIML kernel
kernel = aiml.Kernel()
kernel.learn("carbot.aiml")

# Load the CSV file into a Pandas dataframe
df = pd.read_csv("Q&A.csv")

# Preprocess the questions by lemmatizing them
nltk.download("wordnet")
lemmatizer = WordNetLemmatizer()
df["lemmatized_question"] = df["question"].apply(lambda x: " ".join([lemmatizer.lemmatize(word) for word in nltk.word_tokenize(x)]))

# Convert the questions into tf/idf vectors
vectorizer = TfidfVectorizer()
questions_vectors = vectorizer.fit_transform(df["lemmatized_question"])

# Function to find the closest match to a user input using the CSV
# Function to find the closest match to a user input using the CSV
def get_answer_csv(input_question):
    # Preprocess the user input by lemmatizing it
    input_question = " ".join([lemmatizer.lemmatize(word) for word in nltk.word_tokenize(input_question)])
    
    # Convert the user input into a tf/idf vector
    input_vector = vectorizer.transform([input_question])
    
    # Calculate the cosine similarity between the user input and the questions
    similarity = cosine_similarity(input_vector, questions_vectors)
    
    # Find the index of the question with the highest similarity score
    most_similar_index = similarity.argmax()
    
    # Return the answer corresponding to the most similar question
    return df.iloc[most_similar_index]["answer"]










import requests

# Function to handle user input
def handle_input(input_question, previous_input, previous_answer):
    # Add the previous context to the current input
    input_question = previous_input + " " + input_question
    
    # Try to get an answer from the AIML kernel
    aiml_response = kernel.respond(input_question, previous_answer)
    
    # If the AIML kernel didn't have a matching pattern, try to get information about a car make from the NHTSA API
    if aiml_response == "":
        # Split the input into words and look for the keyword "car"
        words = input_question.lower().split()
        if "car" in words:
            # Find the index of the word "car"
            car_index = words.index("car")
            
            # If the index is not the last word in the input, get the next word as the car make
            if car_index < len(words) - 1:
                car_make = words[car_index + 1]
                
                # Make a GET request to the NHTSA API to get information about the car make
                url = "https://vpic.nhtsa.dot.gov/api/vehicles/getmodelsformake/{}?format=json".format(car_make)
                response = requests.get(url)
                if response.ok:
                    data = response.json()
                    if data["Count"] > 0:
                        models = ", ".join([model["Model_Name"] for model in data["Results"]])
                        return "Here are some models of {}: {}".format(car_make.capitalize(), models)
                    else:
                        return "Sorry, I couldn't find any models for {}".format(car_make.capitalize())
                else:
                    return "Sorry, I couldn't get information from the NHTSA API"
        
        # If the input doesn't contain the keyword "car", use the CSV to find a response
        else:
            return get_answer_csv(input_question)
            
    else:
        return aiml_response



    

# Chatbot loop
previous_input = ""
previous_answer = ""
print("Car Chatbot: Hello, how can I help you today?")
while True:
    input_question = input("> ")
    if input_question.lower() == "quit" or input_question.lower()=="bye" or input_question.lower()=="exit" :
        print("Car Chatbot: Goodbye!")
        break
    answer = handle_input(input_question, previous_input, previous_answer)
    previous_input = input_question
    previous_answer = answer
    print("Car Chatbot:"+answer)
