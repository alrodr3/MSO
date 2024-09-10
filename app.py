import streamlit as st
import numpy as np
from MSO import MercurySearchOptimization

# Set page configuration
st.set_page_config(page_title="Mercury Search Optimization", layout="wide")

# Title and input
st.title("Mercury Search Optimization")

form = st.form("my_form")
form.write("Inside the form")
layer1 = form.columns(2, vertical_alignment="top")
advertiser = layer1[0].text_input("Advertiser")
date = layer1[0].date_input("Initial date")
period = layer1[0].number_input("Period (months)")
budget = layer1[0].number_input("Total budget")
target = layer1[0].number_input("Target")
layer1[1].text("")

# Generate budget_allocation_contraints list
budget_allocation_constraints = (np.zeros(8), np.zeros(8))

# Google
layer_google = form.columns(5, vertical_alignment="bottom")
layer_google[1].subheader("Google")
budget_allocation_constraints[0][0] = layer_google[2].text_input("Min",key='google_min')
budget_allocation_constraints[1][0] = layer_google[3].text_input("Max", key='google_max')

# Bing
layer_bing = form.columns(5, vertical_alignment="bottom")
layer_bing[1].subheader("Bing")
budget_allocation_constraints[0][1] = layer_bing[2].text_input("Min",key='bing_min')
budget_allocation_constraints[1][1] = layer_bing[3].text_input("Max", key='bing_max')

# Meta
layer_meta = form.columns(5, vertical_alignment="bottom")
layer_meta[1].subheader("Meta")
budget_allocation_constraints[0][2] = layer_meta[2].text_input("Min",key='meta_min')
budget_allocation_constraints[1][2] = layer_meta[3].text_input("Max", key='meta_max')

# TikTok
layer_tiktok = form.columns(5, vertical_alignment="bottom")
layer_tiktok[1].subheader("TikTok")
budget_allocation_constraints[0][3] = layer_tiktok[2].text_input("Min",key='tiktok_min')
budget_allocation_constraints[1][3] = layer_tiktok[3].text_input("Max", key='tiktok_max')

# Pinterest
layer_pinterest = form.columns(5, vertical_alignment="bottom")
layer_pinterest[1].subheader("Pinterest")
budget_allocation_constraints[0][4] = layer_pinterest[2].text_input("Min",key='pinterest_min')
budget_allocation_constraints[1][4] = layer_pinterest[3].text_input("Max", key='pinterest_max')

# X
layer_x = form.columns(5, vertical_alignment="bottom")
layer_x[1].subheader("X")
budget_allocation_constraints[0][5] = layer_x[2].text_input("Min",key='x_min')
budget_allocation_constraints[1][5] = layer_x[3].text_input("Max", key='x_max')

# Amazon
layer_amazon = form.columns(5, vertical_alignment="bottom")
layer_amazon[1].subheader("Amazon")
budget_allocation_constraints[0][6] = layer_amazon[2].text_input("Min",key='amazon_min')
budget_allocation_constraints[1][6] = layer_amazon[3].text_input("Max", key='amazon_max')

# Youtube
layer_youtube = form.columns(5, vertical_alignment="bottom")
layer_youtube[1].subheader("Youtube")
budget_allocation_constraints[0][7] = layer_youtube[2].text_input("Min",key='youtube_min')
budget_allocation_constraints[1][7] = layer_youtube[3].text_input("Max", key='youtube_max')


submitted= form.form_submit_button("Run optimization")
if submitted:
    with st.spinner('Optimization in progress...'):
        results = MercurySearchOptimization(
            date=date,
            advertiser=advertiser,
            period=period,
            budget=budget,
            target=target,
            budget_allocation_constraints=budget_allocation_constraints,
            development=True)
    st.succes("Ready!")

# Function to generate nudge
# def calculate(input_text):
#     return eval(input_text)

# if st.button("Calculate"):
#     res = calculate(input_exp)
#     st.write(f"Result: {res}")


# number = st.sidebar.slider("Select inversion",0, 100)
# st.sidebar.page_link("app.py", label="another page to navigate")



# Upload a file
# file = st.file_uploader("Pick a file")
