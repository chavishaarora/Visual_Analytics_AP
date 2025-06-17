# Visual_Analytics_AP


# Adventure Planner Dash App

An interactive multi-day itinerary planner that lets users explore hiking trails, nearby UNESCO sites, and Airbnbs on a map. The app allows feedback (likes/dislikes), generates personalized recommendations, clusters adventures, and exports the final itinerary to a downloadable PDF.

---

## 🧩 Features

- **Interactive Map** (Dash Leaflet): View and filter trails, accommodations, and UNESCO sites
- **User Feedback System**: Like/dislike buttons to personalize your experience
- **Clustering & Embeddings**: Uses t-SNE and KMeans to group trails visually
- **ML-Based Ranking**: Ranks Airbnb options using a Gradient Boosting model trained on feedback
- **Dynamic Budget Tracking**
- **PDF Export**: Generates a printable itinerary with cost breakdown

---

## ⚙️ Installation

1. **Clone this repository:**

git clone https://github.com/chavishaarora/Visual_Analytics_AP.git

2. Create a virtual environment (optional but recommended):
 
python -m venv venv
source venv/bin/activate  # On Windows: venv\\Scripts\\activate

3. Install dependencies:

pip install dash dash-bootstrap-components dash-leaflet pandas numpy scikit-learn plotly reportlab

4. Run the App:

python va_enhanced.py
Then open http://localhost:8051 in your browser



Citation and Attribution

Libraries Used: Dash, scikit-learn, Plotly, ReportLab
Icons/Emojis: Unicode characters for visual feedback
Generative AI: ChatGPT was used to help with code commenting, structure, and documentation





