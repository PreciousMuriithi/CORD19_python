import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from transformers import pipeline
from wordcloud import WordCloud

# --- Streamlit page setup ---
st.set_page_config(page_title="CORD-19 Research Dashboard", layout="wide")

st.title("🦠 CORD-19 Research Insights Dashboard")
st.markdown("Explore patterns and trends in COVID-19 research publications from the **CORD-19 dataset**.")

# --- Load and preprocess data ---
@st.cache_data
def load_data():
    data = pd.read_csv('metadata.csv', low_memory=False)
    data = data.dropna(subset=['title', 'abstract'])
    data = data[['title', 'abstract', 'publish_time', 'journal', 'source_x']]
    data['publish_time'] = pd.to_datetime(data['publish_time'], errors='coerce')
    data['year'] = data['publish_time'].dt.year
    data = data.dropna(subset=['year'])
    return data

data_cleaned = load_data()

# --- Load summarization model (cached) ---
@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="facebook/bart-large-cnn")

summarizer = load_summarizer()

# --- Sidebar filters ---
st.sidebar.header("🔧 Filters")

# Year filter
years = sorted(data_cleaned['year'].dropna().unique())
selected_years = st.sidebar.multiselect("Select publication years", years, default=years)

# Journal filter
journals = sorted(data_cleaned['journal'].dropna().unique())
selected_journals = st.sidebar.multiselect("Select journals", journals[:10], default=journals[:10])

# Apply filters
filtered_data = data_cleaned[
    (data_cleaned['year'].isin(selected_years)) &
    (data_cleaned['journal'].isin(selected_journals))
]

# --- Dataset Overview ---
st.markdown("---")
st.subheader("📊 Dataset Overview")
st.write(f"**Total Papers:** {len(filtered_data):,}")
st.write(f"**Filtered Range:** {int(filtered_data['year'].min())} – {int(filtered_data['year'].max())}")

# --- Publications Over Time ---
st.markdown("---")
st.subheader("📈 Publications Over Time")

papers_per_year = filtered_data['year'].value_counts().sort_index()

fig, ax = plt.subplots(figsize=(10, 5))
sns.set_theme(style="darkgrid")
sns.barplot(x=papers_per_year.index, y=papers_per_year.values, ax=ax, color="#00b4d8")
ax.set_title('Number of COVID-19 Research Papers Published per Year', fontsize=14)
ax.set_xlabel('Year')
ax.set_ylabel('Number of Papers')
st.pyplot(fig)

# --- Top Journals ---
st.markdown("---")
st.subheader("🏛️ Top Journals Publishing COVID-19 Research")

top_journals = filtered_data['journal'].value_counts().head(10)

fig2, ax2 = plt.subplots(figsize=(10, 5))
sns.barplot(x=top_journals.values, y=top_journals.index, ax=ax2, palette="crest")
ax2.set_title('Top 10 Journals Publishing COVID-19 Research', fontsize=14)
ax2.set_xlabel('Number of Papers')
ax2.set_ylabel('Journal')
st.pyplot(fig2)

# --- Word Cloud of Abstracts ---
st.markdown("---")
st.subheader("☁️ Common Research Terms in Abstracts")

text = " ".join(filtered_data['abstract'].dropna().tolist())
wc = WordCloud(width=800, height=400, background_color='white').generate(text)
st.image(wc.to_array())

# --- Search Feature ---
st.markdown("---")
st.subheader("🔍 Search Papers by Keyword")

query = st.text_input("Enter keyword (e.g. vaccine, transmission, mutation):").lower()

if query:
    filtered_search = filtered_data[filtered_data['abstract'].str.lower().str.contains(query, na=False)]
    st.write(f"Found **{len(filtered_search)}** matching papers.")
    st.dataframe(filtered_search[['title', 'journal', 'year']].head(20))
    
    # --- Download filtered results ---
    csv = filtered_search.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Results as CSV",
        data=csv,
        file_name=f"cord19_search_results_{query}.csv",
        mime='text/csv'
    )

    # --- Summarization Feature ---
    st.markdown("---")
    st.subheader("🧠 Summarize an Abstract")

    selected_title = st.selectbox(
        "Select a paper to summarize",
        filtered_search['title'].head(20)
    )

    if selected_title:
        abstract = filtered_search.loc[filtered_search['title'] == selected_title, 'abstract'].values[0]
        st.markdown(f"**Original Abstract:**\n\n{abstract}")

        if st.button("✨ Generate Summary"):
            with st.spinner("Summarizing... this might take a few seconds ⏳"):
                summary = summarizer(abstract, max_length=130, min_length=30, do_sample=False)[0]['summary_text']
            st.success("✅ Summary generated successfully!")
            st.markdown(f"**Summary:**\n\n{summary}")

# --- Footer ---
st.markdown("---")
st.caption("Built with ❤️ using Streamlit, Pandas, Seaborn, WordCloud, and Transformers — by Precious")
