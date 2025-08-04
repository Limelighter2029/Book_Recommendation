# 📚 Book Recommendation System

An AI-powered book recommendation system that uses semantic search and emotional analysis to suggest personalized book recommendations based on user preferences.

## 🌟 Features

- **Semantic Search**: Find books based on natural language descriptions
- **Category Filtering**: Filter recommendations by book categories
- **Emotional Tone Analysis**: Get recommendations based on emotional content (Happy, Surprising, Angry, Suspenseful, Sad)
- **Beautiful UI**: Modern, responsive web interface built with Gradio
- **Caching**: Vector store is cached for faster subsequent searches
- **Real-time Recommendations**: Instant book suggestions with cover images and descriptions

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- Google AI API key (for embeddings)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Book_Recommendation
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   Create a `.env` file in the project root:
   ```env
   GOOGLE_API_KEY=your_google_ai_api_key_here
   ```

4. **Run the application**
   ```bash
   python dashboard.py
   ```

5. **Access the application**
   Open your browser and go to `http://localhost:7860`

## 📁 Project Structure

```
Book_Recommendation/
├── dashboard.py              # Main Gradio web application
├── requirements.txt          # Python dependencies
├── README.md               # This file
├── books_with_emotions.csv # Book dataset with emotional analysis
├── tagged_description.txt   # Processed book descriptions for vector search
├── covernotfound.jpg       # Default book cover image
├── chroma_db/              # Cached vector database (created automatically)
├── book_exploration.ipynb  # Data exploration notebook
├── sentiment_analysis.ipynb # Emotional analysis notebook
├── text_classification.ipynb # Text processing notebook
└── vector_search.ipynb     # Vector search implementation notebook
```

## 🛠️ How It Works

### 1. Data Processing
- Book data is loaded from `books_with_emotions.csv`
- Book descriptions are processed and stored in `tagged_description.txt`
- Emotional analysis scores are calculated for each book

### 2. Vector Search
- Uses Google's Generative AI embeddings (`models/embedding-001`)
- Chroma vector database for efficient similarity search
- Documents are chunked and indexed for semantic matching

### 3. Recommendation Engine
- **Semantic Search**: Finds books similar to user's description
- **Category Filtering**: Filters by book categories (Fiction, Non-fiction, etc.)
- **Emotional Sorting**: Ranks books by emotional content scores
- **Caching**: Vector store is persisted to disk for faster subsequent runs

### 4. User Interface
- **Modern Design**: Beautiful gradient background with glass-morphism effects
- **Responsive Layout**: Works on desktop and mobile devices
- **Interactive Elements**: Hover effects and smooth transitions
- **Real-time Results**: Instant recommendations with book covers

## 🎯 Usage

1. **Enter a book description** in the text box
   - Example: "A thrilling mystery with unexpected plot twists"
   - Example: "A heartwarming story about friendship"

2. **Select a category** (optional)
   - Choose from available book categories or select "All"

3. **Choose an emotional tone** (optional)
   - Happy, Surprising, Angry, Suspenseful, Sad, or All

4. **Click "Find Recommendations"**
   - Get 16 personalized book recommendations
   - View book covers with titles and descriptions

## 🔧 Technical Details

### Dependencies
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computations
- **gradio**: Web interface framework
- **langchain**: LLM framework for embeddings and vector search
- **langchain-google-genai**: Google AI embeddings
- **langchain-community**: Document loaders and text splitters
- **chromadb**: Vector database for similarity search
- **python-dotenv**: Environment variable management

### Key Components

#### Vector Database
```python
# Chroma vector store with Google AI embeddings
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
db_books = Chroma.from_documents(documents, embedding=embeddings, persist_directory="./chroma_db")
```

#### Recommendation Function
```python
def retrieve_semantic_recommendations(query, category=None, tone=None, initial_top_k=50, final_top_k=16):
    # Semantic search
    recs = db_books.similarity_search(query, k=initial_top_k)
    
    # Category filtering
    if category != "All":
        book_recs = book_recs[book_recs["simple_categories"] == category]
    
    # Emotional sorting
    if tone == "Happy":
        book_recs = book_recs.sort_values(by="joy", ascending=False)
    # ... other emotional tones
    
    return book_recs.head(final_top_k)
```

## 📊 Data Sources

- **Book Dataset**: Contains book information, descriptions, and emotional analysis https://www.kaggle.com/datasets/dylanjcastillo/7k-books-with-metadata
- **Emotional Analysis**: Books are scored on joy, surprise, anger, fear, and sadness
- **Categories**: Books are categorized for easy filtering
- **Cover Images**: Book covers with fallback to default image

## 🎨 UI Features

- **Gradient Background**: Beautiful purple-blue gradient
- **Glass-morphism Design**: Semi-transparent containers with blur effects
- **Responsive Grid**: 8x2 grid layout for book recommendations
- **Hover Effects**: Interactive book cards with scale animations
- **Loading States**: Visual feedback during recommendation generation

## 🔍 API Integration

The system uses Google's Generative AI API for:
- **Text Embeddings**: Converting book descriptions to vectors
- **Semantic Search**: Finding similar books based on meaning
- **Model**: `models/embedding-001` for high-quality embeddings from HuggingFace

## 🚀 Performance Optimizations

1. **Vector Store Caching**: Chroma database is persisted to disk
2. **Efficient Filtering**: Category and emotional filtering applied after semantic search
3. **Batch Processing**: Multiple books processed simultaneously
4. **Memory Management**: Optimized data structures for large datasets

**Happy Reading! 📚✨** 
