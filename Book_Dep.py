#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import pickle
import numpy as np


# In[2]:


import gzip

with gzip.open('books.pkl.gz', 'rb') as f:
    books = pickle.load(f)


# In[3]:


pt = pickle.load(open('pt.pkl', 'rb'))
cosine_similarity = pickle.load(open('cosine_similarity.pkl', 'rb'))


# In[4]:


def recommended_book(book_name):
    index = np.where(pt.index == book_name)[0][0]
    similar = sorted(
        list(enumerate(cosine_similarity[index])),
        reverse=True,
        key=lambda x: x[1]
    )[1:6]

    data = []
    for i in similar:
        item = []
        temp_df = books[books['Book-Title'] == pt.index[i[0]]]

        item.append(temp_df.drop_duplicates('Book-Title')['Book-Title'].values[0])
        item.append(temp_df.drop_duplicates('Book-Title')['Book-Author'].values[0])
        item.append(temp_df.drop_duplicates('Book-Title')['Image-URL-M'].values[0])

        data.append(item)

    return data


# In[5]:


# Streamlit UI
st.set_page_config(page_title="Book Recommender", layout="wide")

st.title("📚 Book Recommendation System")
st.markdown("Find books similar to the one you love ❤️")

selected_book = st.selectbox(
    "🔍 Select a book",
    pt.index.values
)

if st.button("✨ Recommend"):
    recommendations = recommended_book(selected_book)

    st.subheader("📖 Recommended Books")

    # Show books in rows of 5
    for row in range(0, len(recommendations), 5):
        cols = st.columns(5)

        for idx, col in enumerate(cols):
            if row + idx < len(recommendations):
                book = recommendations[row + idx]
                with col:
                    st.image(book[2], width=180)
                    st.markdown(f"**{book[0]}**")
                    st.markdown(f"*by {book[1]}*")


# In[6]:


st.divider()


# In[7]:


with st.spinner("Finding great books for you..."):
    recommendations = recommended_book(selected_book)


# In[ ]:




