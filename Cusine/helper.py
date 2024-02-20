from langchain_community.llms import GooglePalm
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain

llm = GooglePalm(temperature=0.7, google_api_key="AIzaSyCKrfbN27nHFV_nunfCmusiN9AXiBzISs4")




def generate_restaurant_name_and_items(cuisine):
    prompt_template_name = PromptTemplate(
    input_variables=['cuisine'],
    template="I want to open to a restaurant for {cuisine}. Suggest a fancy name for this."
    )

    name_chain = LLMChain(llm=llm, prompt=prompt_template_name, output_key="name")

    prompt_template_item = PromptTemplate(
        input_variables=['name'],
        template="Suggest some menu items for {name}."
    )
    menu_chain = LLMChain(llm=llm, prompt=prompt_template_item, output_key="menu")

    food_chain = SequentialChain(
        chains=[name_chain, menu_chain],
        input_variables=['cuisine'],
        output_variables=['name', 'menu']
    )

    response = food_chain({
        "cuisine": cuisine
    })

    return response


if __name__ == "__main__":
    print(generate_restaurant_name_and_items('Italian'))