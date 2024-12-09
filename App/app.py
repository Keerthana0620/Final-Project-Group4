import streamlit as st
import home
import model1
import model2

# Sidebar dropdown for navigation
options = ["Home", "Markov Chain", "GPT-2"]
selected_option = st.sidebar.selectbox("Navigate to:", options)

# Display the selected page content
if selected_option == "Home":
    home.input()
elif selected_option == "Markov Chain":
    model1.input()
elif selected_option == "GPT-2":
    model2.input()
