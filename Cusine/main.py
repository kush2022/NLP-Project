import streamlit as st 
from langchain.llms import GooglePalm
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from dotenv import load_dotenv
import os 


# load enviroment variables from .env file
load_dotenv()

# title 
st.title('Restuarant Name Generator')


cusine = st.sidebar.selectbox('Cusine', ('Kenyan', 'Italian', 'Indian', 'Mexican'))


# llm 
api_key = os.getenv('API_KEY')
llm = GooglePalm(temperature=0.7, google_api_key=api_key)


# define the function to handle the llms call 

def restaurant(cusine):
    # prompt for the name 
    prompt_name = PromptTemplate(
        input_variables=['cusine'],
        template="I want to open a restuarant for {cusine} food. Suggest a fancy name for this only one name please."
    )

    name_llm = LLMChain(llm=llm, prompt=prompt_name, output_key='restaurant_name')


    # prompt for the menu
    prompt_menu = PromptTemplate(
        input_variables=['restaurant_name'],
        template="Suggest some menu items for {restaurant_name}. Return it as comma seperated list."
    )

    menu_llm = LLMChain(llm=llm, prompt=prompt_menu, output_key='menu')

    # chain
    food_chain = SequentialChain(
        chains=[name_llm, menu_llm], 
        input_variables=['cusine'], 
        output_variables=['restaurant_name', 'menu']
        )

    response = food_chain({
        'cusine': cusine
    })

    return response


if cusine:
    response = restaurant(cusine)

    st.header(response['restaurant_name'].strip())
    st.divider()
    st.write("**Menu Items 🥗😎**")
    menu_items = response['menu'].strip().split(',')
    for menu in menu_items:
        st.write('-', menu)



        