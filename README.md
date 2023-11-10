---
title: Find My Book
emoji: 🌍
colorFrom: red
colorTo: purple
sdk: streamlit
sdk_version: 1.28.1
app_file: app.py
pinned: false
license: apache-2.0
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference



# Recommendation system project
# Find my book  
![](https://sun9-46.userapi.com/impg/HvvkmK5Z3-HWHh2eT7Knv-E-ibwK04HI25ASUg/aBMfuCXNiR0.jpg?size=1920x1285&quality=95&sign=48abf3e3dc9344005305f55f18ddee5d&type=album)
### Additional subtask:
# Find a dish for me
![](https://i.imgur.com/LqccfIw.jpg)

# Elbrus Bootcamp | Phase-2 | Team Project

## Team
* [Daniil Lvov](https://huggingface.co/Norgan97)
* [Dmitry Budazhapov](https://huggingface.co/DmitryDorzhievich)
* [Larisa Khlapushina](https://huggingface.co/Larrisa)
___
## Tasks
In this project, our team has developed a book search system based on user requests. 
The service takes a user's description of the book as input and returns a specified number of suitable options. 
In addition to book search, we can offer a system that will find the desired dish and a link to its recipe.
In this case, the user only needs to enter an approximate description of the characteristics and ingredients.
We have implemented a mechanism for analyzing and comparing textual descriptions of dishes, 
which allows for effectively finding alternatives and variations of desired dishes. 
The work represents an innovative approach to searching for culinary recipes and can be useful for both professional chefs and culinary enthusiasts.
___
## Contents
1. Parsing information from websites.
2. Vectorization using a model *rubert-tiny2*
3. Finding similar vectors using *faiss*.
___
## Deployment
The service is implemented on [Streamlit](https://huggingface.co/spaces/ds-meteors/find_my_book)
_
## How to run locally?
## To run the provided applications on your computer, follow these steps:

1. Clone this repository to your local machine.
2. Install the required libraries by running the command *pip install -r requirements.txt* in your terminal or command prompt.
3. Once the libraries are installed, navigate to the repository's directory in your terminal.
4. Run the command *streamlit run main.py* in your terminal to start the application.

This will launch the Streamlit server, and you can access the applications by opening a browser window and navigating to the specified URL.
