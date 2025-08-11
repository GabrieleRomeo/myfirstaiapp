import os
import streamlit as st
from dotenv import load_dotenv
from tempfile import template
from langchain import PromptTemplate
from langchain import LLMChain

# Using Google Models (Gemini Pro)
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

# Set the Google API key from the environment variable
google_api_key = os.getenv("GOOGLE_API_KEY")

# Initialize Google's Gemini model
gemini_model = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest")

template_consigli_alimentari_cot = """
use the chained-of-thought (CoT) reasoning model to solve the following problem:

build a personalized diet plan along with a customized training table for a specific subject.
the subject is {gender} has {age} years old is {height} cm tall and currently weighs {current_weight} kg
the goal they want to achieve is {objective}
the target weight is {target_weight} kg and must be reached in {numero_settimane} weeks.
the subject has a physical activity level of {activity_level} and practices sports {training_sessions} times a week.
create a training plan of {training_sessions} sessions per week that adheres to the following guidelines:

* the training plan must be divided into training sessions
* the training plan must be written in markdown table format
* the table must contain the following fields:

  - day of the week
  - type of training (e.g., strength, endurance, cardio)
  - duration (in minutes)
  - description of the training 
  
create a weekly diet plan that adheres to the following guidelines:

* the diet plan must be balanced and varied
* divide the days into "food intakes" where "food intakes" refers to a consumption moment (breakfast, snack, lunch, or dinner)
* the diet plan must be written in markdown table format    

the table must contain the following fields:
- time
- type of "food intake" (breakfast, snack, lunch, or dinner)
- foods
- quantity
- calories  
explore multiple options for the diet and training plan, choose the best option, and provide a detailed explanation of why it was chosen.
"""


prompt__consigli_alimentari_cot = PromptTemplate(
    template=template_consigli_alimentari_cot,
    input_variables={
        "gender",
        "age",
        "height",
        "current_weight",
        "objective",
        "training_sessions",
        "activity_level",
    },
)

chain_consigli_alimentari_cot = prompt__consigli_alimentari_cot | gemini_model


st.header("App for generating training and nutrition plans")
st.subheader("Generate your personalized plan")

gender = st.selectbox("Select a gender", ["male", "female"], index=0)
age = st.slider("Select your age", 0, 100, 25)
height = st.number_input(
    "Enter your height (cm)", min_value=100, max_value=250, value=180
)
peso = st.number_input("Enter your weight (kg)", min_value=0, max_value=200, value=70)
objective = st.selectbox(
    "Select an objective",
    ["Lose weight", "Maintenance", "Mass gain"],
    index=0,
)

if objective == "Lose weight":
    target_weight = st.number_input(
        "Enter the target weight (kg)", min_value=0, max_value=200, value=peso - 5
    )
elif objective == "Mass gain":
    target_weight = st.number_input(
        "Enter the target weight (kg)", min_value=0, max_value=200, value=peso + 5
    )
else:
    target_weight = peso


if objective == "Lose weight" or objective == "Mass gain":
    numero_settimane = st.number_input(
        "Enter the number of weeks to reach your goal",
        min_value=1,
        max_value=52,
        value=4,
    )
else:
    numero_settimane = 0

training_sessions = st.slider("Number of training sessions per week", 0, 7, 3)
activity_level = st.selectbox(
    "Select your level of physical activity",
    [
        "Sedentary",
        "Slightly active",
        "Moderately active",
        "Very active",
        "Extremely active",
    ],
    index=1,
)

if st.button("GGenerate Plans"):
    with st.spinner("Generation in progress..."):
        result = chain_consigli_alimentari_cot.invoke(
            {
                "gender": gender,
                "age": age,
                "height": height,
                "current_weight": peso,
                "target_weight": target_weight,
                "numero_settimane": numero_settimane,
                "objective": objective,
                "training_sessions": training_sessions,
                "activity_level": activity_level,
            }
        )
        st.markdown(result.content, unsafe_allow_html=True)
        st.success("Successfully generated!")
