import pickle
from PIL import Image
import streamlit as st
import streamlit.components.v1 as stc
import os

# importing the smaller apps
from ml_app import run_ml_app
from ml_app_enhanced import run_ml_app_enhanced
from eda_app import run_eda_app

html_temp = """
			<div style="background-color:#8A9A5B;padding:10px;border-radius:10px">
			<h1 style="color:white;text-align:center;"> Real Estate Price Prediction</h1>
			</div>
			"""

def main():
	stc.html(html_temp)

	menu = ["Home", "Data Analysis", "Basic Prediction", "Enhanced Prediction", "About"]
	choice = st.sidebar.selectbox("Menu", menu)

	if choice=="Home":
		# Use absolute path for cross-platform compatibility
		script_dir = os.path.dirname(os.path.abspath(__file__))
		img_path = os.path.join(script_dir, "IMG", "Realty_Growth.jpg")
		
		# Add error handling for deployment robustness
		try:
			img1 = Image.open(img_path)
			st.image(img1)
		except FileNotFoundError:
			st.warning("Image file not found: Realty_Growth.jpg - using placeholder")
			st.write("Real Estate Price Prediction App")
		st.write("""
				### Thinking Ahead
				Real estate prices are deeply cyclical and much of it is dependent on factors you can't control.
				Whether you plan on buying a new property or want to use the equity in your home for other expenses,
				it is important to analyze both broader market conditions and your specific property
				to determine how the home's value may fare over the course of time.
				
				### Real Estate Price Prediction ML App
				##### 1. This App predict the price of property.
				##### 2. Estimate your budget as per your requirements.
				""")
	elif choice=="Data Analysis":
		run_eda_app()
	elif choice == "Basic Prediction":
		run_ml_app()
	elif choice == "Enhanced Prediction":
		run_ml_app_enhanced()
	else:
		# Use absolute path for cross-platform compatibility
		script_dir = os.path.dirname(os.path.abspath(__file__))
		html_path = os.path.join(script_dir, "IMG", "mumbai_property.html")

		# Add error handling for deployment robustness
		try:
			with open(html_path,'r') as f: 
				html_data = f.read()
			st.components.v1.html(html_data,height=500)
		except FileNotFoundError:
			st.warning("HTML file not found: mumbai_property.html - displaying placeholder")
			st.write("Data visualization would appear here")
		
		
		
		st.write("### Thanks You... ")

main()