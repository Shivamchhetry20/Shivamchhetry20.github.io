---
title: "WhatsApp Chat Analysis with Deployment in herokuapp"
date: 2021-03-10
lastmod: 2021-05-04
aliases:
    - /courses/course2/slides4.pdf
    - /courses/course2/slides1.pdf
    - /courses/course2/slides3.pdf
    - /courses/course2/slides2.pdf
    - /courses/course2/notes3.pdf
    - /courses/course2/notes4.pdf
    - /courses/course2/ps3.pdf
    - /courses/course2/ps4.pdf
    - /courses/course1/quiz1.pdf
    - /courses/course1/quiz2.pdf
    - /courses/course2/quiz3.pdf
    - /courses/course2/quiz4.pdf
    - /courses/course1/ps1.pdf
tags: ["Romance languages","philology","irregular verbs","Spanish","Portuguese"]
author: "Shivam Chhetry"
description: "This graduate course presents classical results in Romance philology."
summary: "This graduate course presents classical results in Romance philology. it focuses especially on Portugese and Spanish irregular verbs."
cover:
    image: "whatsapp.png"
    alt: "Villa of Reduced Circumstances"
    relative: false
showToc: true
disableAnchoredHeadings: true

---

#### **1. Introduction**
- **Problem Statement**:  
  *"WhatsApp group chats contain a wealth of data about user behavior, but extracting meaningful insights from raw text exports is tedious. Manual analysis of hundreds/thousands of messages is impractical."*
- **Your Solution**:  
  *"I built a Python web app that automates WhatsApp chat analysis and visualizes key metrics like active users, message patterns, and timelines. It’s deployed on Heroku for easy access."*

---

#### **2. Key Features to Highlight**
1. **Chat Parsing**  
   - Handles WhatsApp’s specific text export format (timestamps, users, multi-line messages).
   - Regex-based extraction of metadata (e.g., `\d+/\d+/\d+, \d+:\d+ - (.*?): (.*)`).

2. **Metrics Calculated**  
   - Most active users (message count, media shares).
   - Daily/hourly activity patterns.
   - Timeline of messages (e.g., peaks during weekends).

3. **Visualizations**  
   - Interactive charts (Bar graphs, timelines).
   - Word clouds for frequent terms (if implemented).

4. **Deployment**  
   - Heroku integration for public access.
   - Flask backend for lightweight web handling.

---

#### **3. Technical Deep Dive (Code Breakdown)**
Include snippets with explanations:

```python
# app.py – Core Flask Logic
@app.route('/analyze', methods=['POST'])
def analyze():
    file = request.files['file']
    chat_text = file.read().decode('utf-8')  # Parse uploaded file
    messages = parse_whatsapp_chat(chat_text)  # Custom parsing function
    df = pd.DataFrame(messages)  # Pandas for analysis
    generate_plots(df)  # Matplotlib/Plotly visualization
    return render_template('results.html', data=df)
```

```python
# Parsing Logic Example (Adapt from your code)
def parse_whatsapp_chat(text):
    pattern = r'(\d+/\d+/\d+, \d+:\d+) - (.*?): (.*)'
    messages = re.findall(pattern, text)
    return [{'timestamp': msg[0], 'user': msg[1], 'content': msg[2]} for msg in messages]
```

---

#### **4. Deployment Challenges & Solutions**
- **Heroku Setup**:  
  *"I used a `Procfile` to define the web process and `requirements.txt` to manage dependencies. Heroku’s ephemeral filesystem required temporary storage for uploaded files."*

- **Scalability**:  
  *"The app uses lightweight libraries (Pandas, Matplotlib) to handle large chat exports efficiently."*

---

#### **5. Screenshots/Visuals**
Include 2-3 visuals:  
1. Upload interface (`upload.html`).  
2. Sample dashboard with charts (`results.html`).  
3. Heroku deployment workflow diagram.

---

#### **6. Future Improvements**
- **Advanced Analytics**:  
  *"Add sentiment analysis using NLTK or spaCy to gauge chat tone."*
- **User Authentication**:  
  *"Implement login via OAuth to save user-specific chat histories."*
- **Real-Time Features**:  
  *"WebSocket integration for live chat analysis during active group conversations."*

---

#### **7. Call-to-Action**
- **Demo Link**: [https://your-heroku-app.herokuapp.com](https://your-heroku-app.herokuapp.com)  
- **GitHub Repo**: [github.com/shivamkc01/WhatsAppChat_AnalysisDeployusingHeroku](https://github.com/shivamkc01/WhatsAppChat_AnalysisDeployusingHeroku)  
  *"Feel free to contribute or fork the repository!"*

---

### **Blog Title Suggestions**
1. *"Building a WhatsApp Chat Analyzer: From Raw Exports to Actionable Insights"*  
2. *"Deploying a Python WhatsApp Analytics Tool on Heroku: A Step-by-Step Guide"*  
3. *"Visualizing Group Chat Dynamics with Python and Flask"*

---

### **Additional Tips**
- **Personal Anecdote**: Share why you built this (e.g., *"I wanted to analyze my college group’s chat activity to identify key contributors"*).
- **Metrics**: Include performance stats (e.g., *"Processes 10k messages in <5 seconds"*).
- **SEO Keywords**: "WhatsApp analysis", "Python Flask", "Heroku deployment", "Chat visualization".

Let me know if you’d like me to draft specific sections or refine the technical explanations! 🚀
