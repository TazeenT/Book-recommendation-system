#!/usr/bin/env python
# coding: utf-8

import os
import streamlit as st
import pickle
import numpy as np
import gzip

# MUST be first Streamlit command
st.set_page_config(page_title="Book Recommender", layout="wide")

# Absolute paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

# Debug (remove after confirming)
st.write("MODEL DIR:", MODEL_DIR)
st.write("FILES FOUND:", os.listdir(MODEL_DIR))

# Load models
books_path = os.path.join(MODEL_DIR, "books.pkl.gz")
with gzip.open(books_path, "rb") as f:
    books = pickle.load(f)

pt_path = os.path.join(MODEL_DIR, "pt.pkl")
cosine_path = os.path.join(MODEL_DIR, "cosine_similarity.pkl")

with open(pt_path, "rb") as f:
    pt = pickle.load(f)

with open(cosine_path, "rb") as f:
    cosine_similarity = pickle.load(f)

# Recommendation function
def recommended_book(book_name):
    if book_name not in pt.index:
        return []

    index = np.where(pt.index == book_name)[0][0]
    similar = sorted(
        list(enumerate(cosine_similarity[index])),
        reverse=True,
        key=lambda x: x[1]
    )[1:6]

    data = []
    for i in similar:
        temp_df = books[books['Book-Title'] == pt.index[i[0]]]
        if temp_df.empty:
            continue

        data.append([
            temp_df.drop_duplicates('Book-Title')['Book-Title'].values[0],
            temp_df.drop_duplicates('Book-Title')['Book-Author'].values[0],
            temp_df.drop_duplicates('Book-Title')['Image-URL-M'].values[0]
        ])

    return data

# UI
st.title("📚 Book Recommendation System")
st.markdown("Find books similar to the one you love ❤️")

selected_book = st.selectbox("Select a book", pt.index.values)

if st.button("Recommend"):
    with st.spinner("Finding great books for you..."):
        recommendations = recommended_book(selected_book)

    st.subheader("📖 Recommended Books")

    for row in range(0, len(recommendations), 5):
        cols = st.columns(5)
        for idx, col in enumerate(cols):
            if row + idx < len(recommendations):
                book = recommendations[row + idx]
                with col:
                    st.image(book[2], width=180)
                    st.markdown(f"**{book[0]}**")
                    st.markdown(f"*by {book[1]}*")





