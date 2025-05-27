import streamlit as st
st.set_page_config(page_title="YouTube Sentiment Analyzer 🎬", layout="wide")

import pandas as pd
from multilingual_sentiment_model import (
    extract_video_id, get_video_title, get_comments,
    analyze_sentiment, plot_pie_chart, get_overall_sentiment
)

st.markdown("""
    <style>
    .video-row { display: flex; align-items: center; justify-content: space-between; margin-bottom: 5px; }
    .video-row button { margin-left: 10px; }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<h2 style='text-align: center; margin-bottom: 0;'>🎬 YouTube Sentiment Analyzer</h2>
<p style='text-align: center; font-size: 16px; margin-top: 0;'>Analyze viewer sentiments — Positive, Neutral or Negative</p>
<hr style="margin-top: 0; margin-bottom: 1.5rem;">
""", unsafe_allow_html=True)

# demo videos
example_videos = {
    "https://www.youtube.com/watch?v=JGwWNGJdvx8": "Ed Sheeran - Shape of You",
    "https://www.youtube.com/watch?v=dvgZkm1xWPE": "Coldplay - Viva La Vida",
    "https://www.youtube.com/watch?v=YQHsXMglC9A": "Adele - Hello",
    "https://www.youtube.com/watch?v=09R8_2nJtjg": "Maroon 5 - Sugar",
    "https://www.youtube.com/watch?v=7wtfhZwyrcc": "Imagine Dragons - Believer"
}

col1, col2 = st.columns([1.2, 1.8])

with col1:
    st.markdown("### 📥 Input")

    if "url" not in st.session_state:
        st.session_state["url"] = ""

    url = st.text_input("YouTube Video URL", value=st.session_state["url"], key="youtube_url")
    num_comments = st.slider("Number of Comments ", 10, 50, value=10, step=5)
    analyze_btn = st.button("🔍 Analyze")

    st.markdown("##### 🔗 Example YouTube Videos")
    st.text("Copy the links if you don't have any")

    for link, title in example_videos.items():
        # st.markdown(f"➡️ **{title}**: {link}")
        st.markdown(f"{link}")

with col2:
    if analyze_btn:
        video_id = extract_video_id(url)
        if not video_id:
            st.error("❌ Invalid YouTube URL.")
        else:
            with st.spinner("Fetching video title, comments and analyzing sentiment..."):
                video_title = get_video_title(video_id)
                comments, error = get_comments(video_id, max_results=num_comments)

                if error:
                    st.error(f"❌ {error}")
                elif not comments:
                    st.warning("⚠️ No comments found for this video.")
                else:
                    st.markdown(f"### 🎥 {video_title}", unsafe_allow_html=True)

                    results, counts = analyze_sentiment(comments)
                    sentiment_summary = get_overall_sentiment(counts)
                    pie_chart = plot_pie_chart(counts, video_title)

                    # Sentiment badge
                    color = '#66bb6a' if 'Positive' in sentiment_summary else '#ef5350' if 'Negative' in sentiment_summary else "#2cb1f4"
                    st.markdown(
                        f"<h4 style='color: {color}; text-align: center;'>{sentiment_summary}</h4>",
                        unsafe_allow_html=True
                    )

                    left, center, right = st.columns([1, 2, 1])
                    with center:
                        st.plotly_chart(pie_chart, use_container_width=False)

                    st.markdown("##### 💬 Top 5 Comments")
                    df_sample = pd.DataFrame(results).head(5)
                    st.table(df_sample[['Comment', 'Sentiment']])

                    csv = df_sample.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Download Top Comments as CSV",
                        data=csv,
                        file_name='top_comments.csv',
                        mime='text/csv'
                    )

st.markdown("<hr>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align: center; font-size: 12px; color: gray;'>© 2025 YouTube Sentiment Analyzer | Built with ❤️ using Streamlit</p>",
    unsafe_allow_html=True
)