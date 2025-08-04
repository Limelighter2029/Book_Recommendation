import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain.vectorstores import Chroma
import gradio as gr

load_dotenv()
books = pd.read_csv('books_with_emotions.csv')
books["large_thumbnail"] = books["thumbnail"] + "&fife=w800"
books["large_thumbnail"] = np.where(
    books["large_thumbnail"].isna(),
    "covernotfound.jpg",
    books["large_thumbnail"]
)

# CRITICAL FIX: Ensure isbn13 is treated as a string from the start
books["isbn13"] = books["isbn13"].astype(str)

# Caching the Chroma Vector Store
persist_directory = "./chroma_db"
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# Check if the vector store already exists on disk
if os.path.exists(persist_directory):
    print("Loading existing Chroma database from disk...")
    db_books = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings
    )
else:
    print("Creating new Chroma database from documents...")
    # Load and split documents only on the first run
    raw_documents = TextLoader("tagged_description.txt", encoding="utf-8").load()
    text_splitter = CharacterTextSplitter(
        chunk_size=6000,
        chunk_overlap=200,
        separator="\n"
    )
    documents = text_splitter.split_documents(raw_documents)

    # Create the vector store and persist it to disk
    db_books = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    db_books.persist()  # Explicitly persist the data to the folder
    print("Chroma database created and saved.")


def retrieve_semantic_recommendations(
        query: str,
        category: str = None,
        tone: str = None,
        initial_top_k: int = 50,
        final_top_k: int = 16
) -> pd.DataFrame:
    recs = db_books.similarity_search(query, k=initial_top_k)
    # CRITICAL FIX: Handle ISBNs as strings, remove int()
    books_list = [rec.page_content.strip('"').split()[0] for rec in recs]

    # Filter books from the main DataFrame
    book_recs = books[books["isbn13"].isin(books_list)]

    if category != "All":
        book_recs = book_recs[book_recs["simple_categories"] == category]

    # Use explicit reassignment instead of inplace=True
    if tone == "Happy":
        book_recs = book_recs.sort_values(by="joy", ascending=False)
    elif tone == "Surprising":
        book_recs = book_recs.sort_values(by="surprise", ascending=False)
    elif tone == "Angry":
        book_recs = book_recs.sort_values(by="anger", ascending=False)
    elif tone == "Suspenseful":
        book_recs = book_recs.sort_values(by="fear", ascending=False)
    elif tone == "Sad":
        book_recs = book_recs.sort_values(by="sadness", ascending=False)

    # Take the final top_k after all filtering/sorting
    return book_recs.head(final_top_k)


def recommend_books(
        query: str,
        category: str,
        tone: str
):
    recommendations = retrieve_semantic_recommendations(query, category, tone)
    results = []

    for _, row in recommendations.iterrows():
        description = row["description"]
        truncated_desc_split = description.split()
        truncated_description = " ".join(truncated_desc_split[:30]) + "..."

        authors_split = row["authors"].split(";")
        if len(authors_split) == 2:
            authors_str = f"{authors_split[0]} and {authors_split[1]}"
        elif len(authors_split) > 2:
            authors_str = f"{', '.join(authors_split[:-1])}, and {authors_split[-1]}"
        else:
            authors_str = row["authors"]

        caption = f"{row['title']} by {authors_str}: {truncated_description}"
        results.append((row["large_thumbnail"], caption))
    return results


categories = ["All"] + sorted(books["simple_categories"].unique())
tones = ["All"] + ["Happy", "Surprising", "Angry", "Suspenseful", "Sad"]

# Custom CSS for better styling
custom_css = """
.gradio-container {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    min-height: 100vh;
    padding: 20px;
}

.main-header {
    text-align: center;
    color: white;
    font-size: 2.5rem;
    font-weight: 300;
    margin-bottom: 2rem;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
}

.subtitle {
    text-align: center;
    color: rgba(255,255,255,0.9);
    font-size: 1.1rem;
    margin-bottom: 3rem;
    font-weight: 300;
}

.input-container {
    background: rgba(255,255,255,0.95);
    border-radius: 15px;
    padding: 2rem;
    box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    backdrop-filter: blur(10px);
    margin-bottom: 2rem;
}

.input-row {
    display: flex;
    gap: 1rem;
    align-items: end;
    flex-wrap: wrap;
}

.input-group {
    flex: 1;
    min-width: 200px;
}

.submit-btn {
    background: linear-gradient(45deg, #667eea, #764ba2) !important;
    border: none !important;
    color: white !important;
    padding: 12px 30px !important;
    border-radius: 25px !important;
    font-weight: 600 !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4) !important;
}

.submit-btn:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6) !important;
}

.results-container {
    background: rgba(255,255,255,0.95);
    border-radius: 15px;
    padding: 2rem;
    box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    backdrop-filter: blur(10px);
}

.results-header {
    color: #333;
    font-size: 1.5rem;
    font-weight: 600;
    margin-bottom: 1.5rem;
    text-align: center;
}

.gallery-container {
    border-radius: 10px;
    overflow: hidden;
    margin-top: 1rem;
}

.gallery-container .gallery-item {
    border-radius: 8px;
    overflow: hidden;
    transition: transform 0.3s ease;
    margin: 8px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
}

.gallery-container .gallery-item:hover {
    transform: scale(1.05);
    box-shadow: 0 4px 16px rgba(0,0,0,0.2);
}

.gallery-container .gallery-item img {
    border-radius: 6px;
    object-fit: cover;
    width: 100%;
    height: 200px;
}

.gallery-container .gallery-item .caption {
    padding: 8px 12px;
    background: rgba(255,255,255,0.95);
    border-radius: 0 0 6px 6px;
    font-size: 0.85rem;
    line-height: 1.3;
    color: #333;
    font-weight: 500;
}

.loading {
    text-align: center;
    color: #667eea;
    font-size: 1.1rem;
    font-weight: 500;
}
"""

with gr.Blocks(theme=gr.themes.Soft(), css=custom_css) as dashboard:
    # Header Section
    gr.HTML("""
        <div class="main-header">
            📚 Book Recommendation System
        </div>
        <div class="subtitle">
            Discover your next favorite book with AI-powered recommendations
        </div>
    """)

    # Input Section
    with gr.Row(elem_classes=["input-container"]):
        with gr.Column(scale=2):
            user_query = gr.Textbox(
                label="📖 Describe the book you're looking for",
                placeholder="e.g., A thrilling mystery with unexpected plot twists, or A heartwarming story about friendship",
                lines=3,
                max_lines=4
            )

        with gr.Column(scale=1):
            with gr.Row():
                category_dropdown = gr.Dropdown(
                    choices=categories,
                    label="📂 Category",
                    value="All",
                    info="Filter by book category"
                )

            with gr.Row():
                tone_dropdown = gr.Dropdown(
                    choices=tones,
                    label="😊 Emotional Tone",
                    value="All",
                    info="Choose the emotional mood you prefer"
                )

            with gr.Row():
                submit_button = gr.Button(
                    "🔍 Find Recommendations",
                    elem_classes=["submit-btn"],
                    size="lg"
                )

    # Results Section
    with gr.Row(elem_classes=["results-container"]):
        with gr.Column():
            gr.HTML('<div class="results-header">📚 Your Personalized Recommendations</div>')
            output = gr.Gallery(
                label="",
                columns=8,
                rows=2,
                height=400,
                elem_classes=["gallery-container"],
                show_label=False,
                object_fit="contain"
            )

    # Footer
    gr.HTML("""
        <div style="text-align: center; margin-top: 2rem; color: rgba(255,255,255,0.8); font-size: 0.9rem;">
            Powered by AI • Built with Gradio
        </div>
    """)

    # Event handling
    submit_button.click(
        fn=recommend_books,
        inputs=[user_query, category_dropdown, tone_dropdown],
        outputs=output
    )

    # Add loading state
    submit_button.click(
        fn=lambda: gr.update(value="⏳ Finding recommendations..."),
        outputs=submit_button
    ).then(
        fn=lambda: gr.update(value="🔍 Find Recommendations"),
        outputs=submit_button
    )

if __name__ == "__main__":
    dashboard.launch(share=True, server_name="0.0.0.0", server_port=7860)
