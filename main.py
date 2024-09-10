import streamlit as st

# Set page configuration
st.set_page_config(page_title="Mercury Search Optimization", layout="wide")

# Title and input
st.title("Mercury Search Optimization")

form = st.form("my_form")
form.write("Inside the form")
layer1 = form.columns(2, vertical_alignment="top")
advertiser = layer1[0].text_input("Advertiser")
init_date = layer1[0].date_input("Initial date")
end_date = layer1[1].date_input("End date")
inv = layer1[0].number_input("Total budget")
layer1[1].text("")

# Google
layer_google = form.columns(5, vertical_alignment="bottom")
layer_google[1].subheader("Google")
google_min = layer_google[2].text_input("Min",key='google_min')
google_max = layer_google[3].text_input("Max", key='google_max')

# Bing
layer_bing = form.columns(5, vertical_alignment="bottom")
layer_bing[1].subheader("Bing")
bing_min = layer_bing[2].text_input("Min",key='bing_min')
bing_max = layer_bing[3].text_input("Max", key='bing_max')

# Meta
layer_meta = form.columns(5, vertical_alignment="bottom")
layer_meta[1].subheader("Meta")
meta_min = layer_meta[2].text_input("Min",key='meta_min')
meta_max = layer_meta[3].text_input("Max", key='meta_max')

# TikTok
layer_tiktok = form.columns(5, vertical_alignment="bottom")
layer_tiktok[1].subheader("TikTok")
tiktok_min = layer_tiktok[2].text_input("Min",key='tiktok_min')
tiktok_max = layer_tiktok[3].text_input("Max", key='tiktok_max')

# Pinterest
layer_pinterest = form.columns(5, vertical_alignment="bottom")
layer_pinterest[1].subheader("Pinterest")
pinterest_min = layer_pinterest[2].text_input("Min",key='pinterest_min')
pinterest_max = layer_pinterest[3].text_input("Max", key='pinterest_max')

# X
layer_x = form.columns(5, vertical_alignment="bottom")
layer_x[1].subheader("X")
x_min = layer_x[2].text_input("Min",key='x_min')
x_max = layer_x[3].text_input("Max", key='x_max')

# Amazon
layer_amazon = form.columns(5, vertical_alignment="bottom")
layer_amazon[1].subheader("Amazon")
amazon_min = layer_amazon[2].text_input("Min",key='amazon_min')
amazon_max = layer_amazon[3].text_input("Max", key='amazon_max')

# Youtube
layer_youtube = form.columns(5, vertical_alignment="bottom")
layer_youtube[1].subheader("Youtube")
youtube_min = layer_youtube[2].text_input("Min",key='youtube_min')
youtube_max = layer_youtube[3].text_input("Max", key='youtube_max')

# Every form must have a submit button.

submitted= form.form_submit_button("Run optimization")
if submitted:
    st.write(f"Optimizando!")

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
