import streamlit as st
import pandas as pd
import json
from nltk.stem import PorterStemmer
import numpy as np
from heapq import nsmallest,nlargest
from openai import OpenAI
from st_aggrid import AgGrid, JsCode, GridOptionsBuilder
import pickle
import pandas as pd
import requests
import random
import joblib


st.set_page_config("Advisor Recommendation", page_icon=":book:")
data = pd.read_csv('updated_dataframe.csv')
lda_model = joblib.load('lda_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')
doc_topic_matrix = joblib.load('doc_topic_matrix.pkl')
options = ["software engineering", "software process", "software system", "software quality", "design debt", "case studies", "software development", "software evolution", "online communities", "websites", "web pages", "related websites", "web spam", "web communities", "web mining", "online community analysis", "spammy website networks", "rescue robots", "autonomous mobile robots", "autonomous mode", "tele-operation mode", "multiple robots", "mobile robot", "proposed system", "mobile applications", "mobile devices", "smart phones", "mobile Internet devices", "context information", "resource-constrained mobile devices", "mobile users", "mobile phone", "mobile devices adaptive"]
API_URL= st.secrets["URL"]
MODEL   = st.secrets["MODEL"] 

count_vector={}
with open('my_dict.json', 'r') as f:
        count_vector = json.load(f)

Term_set=[]



with open('Term_set.json', 'r') as f:
        Term_set = json.load(f)
        
def tokenize(txt):
  txt=str(txt)
  txt = txt.replace(';',' ')
  txt = txt.replace(',',' ')
  return txt.split()

def porter_stemmer(words):
  stemmer = PorterStemmer()
  return [stemmer.stem(word) for word in words]

def user_count_vector(doc): 
    lst=[]
    doc=tokenize(doc)
    doc=porter_stemmer(doc)   
    for j in Term_set:
         lst.append(doc.count(j))
    return lst
        
def cosine_similarity(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)  # add epsilon to avoid divide by zero

def top_similar_doc_cosine(count_vec,doc,k=3):
        lst={}
        for i in count_vec:
            lst[i] = cosine_similarity(count_vec[i], user_count_vector(doc))
        top_similar_doc = nlargest(k, lst, key = lst.get)
        return lst,top_similar_doc
def render_spacer():
    st.markdown("""
    <div style='min-height: 250px; overflow-y: auto; padding: 10px;'>
                <h1></h1>
    </div>
    """, unsafe_allow_html=True)

def chat_stream():
    """Send a message to the API and stream back the assistant's reply."""
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": MODEL,
        "messages": [
            {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages
                                                
        ]
    }
    with requests.post(API_URL, headers=headers, data=json.dumps(payload), stream=True) as r:
            for line in r.iter_lines():
                if line:
                    data = json.loads(line.decode("utf-8"))
                    if "message" in data and "content" in data["message"]:
                        yield data["message"]["content"]  # send token to Streamlit
                    if data.get("done"):
                        break

def cosine_recommender(doc):
    # Read data from stdin
    

    user_score, rec_adv=top_similar_doc_cosine(count_vector,doc,3)
    user_score, rec_adv=top_similar_doc_cosine(count_vector,doc,3)
    rank=[]
    user=[]
    score=[]
    kw=[]
    publication=[]
    affiliation=[]
    count=1
    for i in rec_adv:
                   rank.append(count)
                   user.append(i)
                   score.append(user_score[i])
                   a=data[data['n']==i]['t']
                   publication.append(data[data['n']==i]['paper_list'].values[0])
                   affiliation.append(data[data['n']==i]['affiliation'].values[0])
                 
                   for j in a: 
                        k=" ".join(tokenize(j))
                        #k=tokenize(j)
                   kw.append(k)
                   print(kw)
                   count+=1
    df = {
                                        'Ranking': rank,
                                        'Name': user,
                                        'Similarity Score': score,
                                        'Keywords': kw,
                                        'Publication':publication,
                                        'Affiliation':affiliation
                                    }
    data_str = json.dumps(df)
    
    #print("Done")
    with open('rec_result.json', 'w') as f:
        json.dump(df, f)
    return data_str


COOLDOWN_TIME_LONG = 45
COOLDOWN_TIME_SHORT = 30
import time
from datetime import datetime, timedelta

def countdown_component_html(message, duration_sec, reveal_html):
    # Initialize unlock_time only when not already set
    if "unlock_time" not in st.session_state:
        st.session_state.unlock_time = datetime.now() + timedelta(seconds=duration_sec)

    remaining = int((st.session_state.unlock_time - datetime.now()).total_seconds())
    
    html_code = f"""
    <div style="font-weight:bold;font-size:16px;">
        <span id="timer">{message} — {remaining//60:02d}:{remaining%60:02d}</span>
    </div>

    <div id="reveal-section" style="display:none; margin-top:10px;">
        {reveal_html}
    </div>

    <script>
    var seconds = {remaining};
    var timerElement = document.getElementById("timer");
    var revealSection = document.getElementById("reveal-section");
    var countdown = setInterval(function(){{
        if (seconds > 0) {{
            seconds--;
            var mins = Math.floor(seconds/60);
            var secs = seconds % 60;
            timerElement.innerHTML = "{message} — " + 
                (mins<10?"0":"") + mins + ":" + (secs<10?"0":"") + secs;
        }} else {{
            clearInterval(countdown);
            timerElement.innerHTML = "You can now proceed!";
            revealSection.style.display = "block";
        }}
    }}, 1000);
    </script>
    """

    st.components.v1.html(html_code, height=120)
    
def countdown_with_button(message, duration_sec, button_label, button_key):
    # Initialize countdown state
    if f"{button_key}_done" not in st.session_state:
        st.session_state[f"{button_key}_done"] = False

    if not st.session_state[f"{button_key}_done"]:
        placeholder = st.empty()
        for remaining in range(duration_sec, 0, -1):
            mins, secs = divmod(remaining, 60)
            placeholder.markdown(f"**{message} — {mins:02d}:{secs:02d}**")
            time.sleep(1)
        placeholder.empty()
        st.session_state[f"{button_key}_done"] = True

    # Show button only after countdown done
    return st.button(button_label, key=button_key)


    
def countdown_with_form(message, duration_sec, form_key, input_key, submit_label="➤"):
    """
    Shows a countdown before revealing a form with text input + submit.
    Returns the user input if submitted, else None.
    """
    if f"{form_key}_done" not in st.session_state:
        st.session_state[f"{form_key}_done"] = False

    if not st.session_state[f"{form_key}_done"]:
        placeholder = st.empty()
        for remaining in range(duration_sec, 0, -1):
            mins, secs = divmod(remaining, 60)
            placeholder.markdown(f"**{message} — {mins:02d}:{secs:02d}**")
            time.sleep(1)
        placeholder.empty()
        st.session_state[f"{form_key}_done"] = True

    # Show form after countdown done
    if st.session_state[f"{form_key}_done"]:
        with st.form(form_key, clear_on_submit=True):
            cols = st.columns([4, 0.5])
            user_input = cols[0].text_input("", key=input_key, label_visibility="collapsed")
            send = cols[1].form_submit_button(submit_label)
            if send and user_input:
                return user_input
    return None

def load_dict(filename):
    with open(filename, 'r') as file:
        return json.load(file)

def LDA(keywords):
    rank, top, topic_words, topic_prob = [], [], [], []
    names, sim, kw, publication, affiliation = [], [], [], [], []
    from sklearn.metrics.pairwise import cosine_similarity as cosim

    # Preprocess user keywords
    new_doc = porter_stemmer(tokenize(keywords))
    new_doc_text = " ".join(new_doc)
    new_doc_vector = vectorizer.transform([new_doc_text])

    # Topic distribution
    topic_distribution = lda_model.transform(new_doc_vector)[0]
    top3_indices = topic_distribution.argsort()[-3:][::-1]
    # Similarity with existing documents
    new_topic_matrix = lda_model.transform(new_doc_vector)
    similarities = cosim(new_topic_matrix, doc_topic_matrix)[0]
    sorted_sims = similarities.argsort()[-3:][::-1]



    for topic in top3_indices:
        prob = topic_distribution[topic]
        print(f"Topic {topic} with probability {prob:.4f}")
        topic_terms = lda_model.components_[topic]
        top_words_idx = topic_terms.argsort()[-10:][::-1]
        words = [vectorizer.get_feature_names_out()[i] for i in top_words_idx]
        print(f"Top words for topic {topic}: {', '.join(words)}")
        top.append(topic)
        topic_words.append(words)
        topic_prob.append(prob)



    for count, doc_position in enumerate(sorted_sims, 1):
        score = similarities[doc_position]
        print(f"Document id: {doc_position}, name: {data['n'][doc_position]} with similarity score: {score:.4f}")
        rank.append(count)
        names.append(data['n'][doc_position])
        publication.append(data['paper_list'][doc_position])
        affiliation.append(data['affiliation'][doc_position])
        a = data['t'][doc_position].replace(";", " ")
        kw.append(a)
        sim.append(score)

    df1 = {
        'LDA_rank': rank,
        'LDA_Name': names,
        'Score': sim,
        'Keywords_LDA': kw,
        'Publication': publication,
        'Affiliation': affiliation
    }

    df2 = {
        'Topic': top,
        'Words': topic_words,
        'Probability': topic_prob
    }

    return df1, df2
def reset_version_state():
                # Clear any previous version state 
                st.session_state.prediction_ready = False
                st.session_state.initial_prompt_sent = False
                st.session_state.chat_history = []
                st.session_state.chat_html = ""
                st.session_state.explain_clicked = False
                st.session_state.show_explain_option = False
                st.session_state.question_asked = 0

def load_scenarios(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        scenarios = file.read().split("---")  # "---" as a separator
    scenarios = [ x.strip() for x in scenarios]
    return scenarios
# --- Session state inits ---
for key, default in [
    ("page","home"),
    ("user_name",""),
    ("prediction_ready", False),
    ("initial_prompt_sent", False),
    ("chat_history", []),
    ("chat_html", ""),
    ("explain_clicked", False),
    ("selected_symptoms_clean", None),
    ("show_explain_option",False),
    ("question_asked",0),
    ("questions",["How does the system work?","Tell me how the first advisor in text similarity model recommended?"]),
    ("scenarios_loaded",False)
]:
    if key not in st.session_state:
        st.session_state[key] = default


if 'clicked' not in st.session_state:
    st.session_state.clicked = False

questions=["How does the system work?","Tell me how the first advisor in text similarity model recommended?"]
data_dict={}
flag=0
st.session_state["openai_model"] = "llama3.3:70b-instruct-q8_0"    #"gpt-3.5-turbo"

if st.session_state.page == "home":
    st.title("Grad Student Advisor Recommender System")
    if st.session_state.user_name == "":
        st.markdown("### Please enter your name to get started:")

        # 4.1) Name input
        name_col, submit_col = st.columns([3, 1], vertical_alignment="bottom")
        with name_col:
            if st.session_state["user_name"] == "":
                st.session_state["user_name"] = st.text_input("Your name:", value="", placeholder="Type your name here")
                
        
        with submit_col:
            if st.button("Start"):
                if st.session_state["user_name"].strip() == "":
                    st.warning("Please enter at least one character for your name.")
                else:
                    st.success(f"Hello, {st.session_state['user_name'].strip()}!")
                    st.rerun()
    
    if not st.session_state.scenarios_loaded:            
        scenarios = load_scenarios("grad_student_scenario.md")
        random.shuffle(scenarios)
        # Keep 3 scenarios
        st.session_state.selected_scenarios = scenarios[:3] if len(scenarios) >= 3 else scenarios
        st.session_state.scenarios_loaded = True
    # Only show the welcome text & cards if we have a name
    if st.session_state["user_name"].strip() != "":
        st.markdown(f"#### Hi, **{st.session_state['user_name']}**! Please choose a version below:")
        st.markdown("")

        # 4.2) Two card-style buttons
        c1, c2 = st.columns(2, gap="large")

        with c1:
            st.info("""**Version 1**: Advisor Recommendation & AI Follow-up.
                    \nEnter your research keywords to receive advisor recommendations.
                        \nEngage with the AI chatbot by asking pre-selected follow-up questions or any additional questions you may have.""")
            if st.button("Go to Version 1"):
                st.session_state.page = "v1"
                reset_version_state()
                st.rerun()
        with c2:
            st.info("""**Version 2**: Advisor Recommendation with AI-Assisted Pre-Quiz.
                    \nEnter your research keywords to receive advisor recommendations.
                    \nTake a short pre-quiz on the recommendation, AI will explain your choice and answer follow-up questions.""")
            if st.button("Go to Version 2"):
                st.session_state.page = "v2"
                reset_version_state()
                st.rerun()
                
                

elif st.session_state.page == "v1" or st.session_state.page == "v2":
    # --- UI ---
    back_col, _ = st.columns([1, 4])
    with back_col:
        if st.button("← Back to Home"):
            st.session_state.page = "home"
            reset_version_state()
            st.rerun()
    if st.session_state.page == "v1":
        st.title("Grad Student Advisor Recommender System")
        st.subheader("Version 1 - Advisor Recommendation & AI Follow-up")
        st.divider()
        st.markdown("_Grad Stuedent Scenario:_")
        scenario = st.session_state.selected_scenarios[0]
        st.info(scenario)
        st.markdown("Enter keywords of your reseach interest separated by comma and get system's recommendations.")
        keywords = st.multiselect("Select Research Keywords:", options=options)
    if st.session_state.page == "v2":
        st.title("Grad Student Advisor Recommender System")
        st.subheader("**Version 2**: Advisor Recommendation with AI-Assisted Pre-Quiz")
        st.divider()
        st.markdown("_Grad Stuedent Scenario:_")
        scenario = st.session_state.selected_scenarios[1]
        st.info(scenario)
        st.markdown("Enter keywords of your reseach interest separated by comma and get system's recommendations.")
        keywords = st.multiselect("Select Research Keywords:", options=options)
    if st.button("Predict"):
        if len(keywords) < 1:
            st.warning("Please select at least one keyword.")
        else:
            import time
            name=st.session_state.user_name
            keywords = ", ".join(keywords[:-1]) + f", and {keywords[-1]}" if len(keywords) > 1 else keywords[0]
            with st.spinner(text="Hello "+name+"! Please wait while we retrieve some information."):
                output=cosine_recommender(keywords)           
                data_dict = json.loads(output)
                
                lda1,lda2=LDA(keywords)          
                st.session_state["flag"] = data_dict
                with open('rec_result.txt', 'w') as f:
                            msg="User name is "+ name+". User reseach interests are "+keywords+". Top 3 recommended advisor list based on Cosine similarity:\n"
                            for i in range(len(data_dict['Ranking'])):
                                msg+=str(i+1)+'. name: '+ data_dict['Name'][i]
                                #if i==0:
                                #        st.session_state.questions.append(f"Why was Dr. {data_dict['Name'][i]} recommended?")
                                msg+='. Cosine similarity score: '+str(data_dict['Similarity Score'][i])
                                msg+='. Keywords: '+data_dict['Keywords'][i]+'\n'
                                msg+='. Publication: '+data_dict['Publication'][i]+'\n'
                                msg+='. Affiliaiton: '+data_dict['Affiliation'][i]+'\n'
                            f.write(msg)
                st.session_state["lda1"] = lda1
                with open('rec_result.txt', 'a') as f:
                            msg="\nTop 3 recommended advisor list based on LDA Topic modeling:\n"
                            for i in range(len(lda1['LDA_rank'])):
                                msg+=str(i+1)+'. name: '+ lda1['LDA_Name'][i]
                                msg+='. Similarity score: '+str(lda1['Score'][i])
                                msg+='. Keywords: '+lda1['Keywords_LDA'][i]+'\n'
                                msg+='. Publication: '+lda1['Publication'][i]+'\n'
                                msg+='. Affiliation: '+lda1['Affiliation'][i]+'\n'
                            f.write(msg)
                st.session_state["lda2"] = lda2
                with open('rec_result.txt', 'a') as f:
                            msg="\nTop LDA Topic selected:\n"
                            for i in range(len(lda2['Topic'])):
                                msg+=str(i+1)+'. Topic id: '+ str(lda2['Topic'][i])
                                #msg+='. Cosine similarity score: '+str(data_dict['Similarity Score'][i])
                                msg+='. Keywords: '+" ".join(lda2['Words'][i])+'\n'
                            f.write(msg)
                
                st.session_state.prediction_ready=True
                msg="" 
                with open('rec_result.txt', 'r') as f:
                                    for line in f:
                                        msg+=line
                                        
                prompt="""
                You are an AI-powered academic advisor chatbot designed to explain the reasoning behind advisor recommendations generated by a machine learning system. Your goal is to help prospective graduate students understand how their research interests align with those of faculty members based on two recommendation models.

Below is the **system design** as implemented:

1. **Text Similarity Model (Cosine Similarity):**
   - Inputs: Research keywords provided by the user.
   - Each advisor’s research profile is represented as a numerical count vector of publication keywords.
   - Cosine similarity is calculated between the user’s keyword vector and each advisor’s vector.
   - Output: Top 3 advisors with the highest similarity scores (range: 0 to 1), where values closer to 1 indicate stronger alignment.

2. **Topic Similarity Model (LDA Topic Modeling):**
   - Inputs: User’s research keywords mapped to 30 predefined LDA topics.
   - Each advisor has a topic distribution profile learned from their publication data.
   - The similarity between the user’s topic vector and each advisor’s topic profile is computed.
   - Output: Top 3 advisors with the most similar topic distributions.

**System Inputs and Outputs for This Session:**
     
                """
                prompt2="""**Expected Outcome:**
Help users interpret why these advisors were recommended, how closely their research interests align, and how changes in keywords might affect the results. Word counts more than 200.

**Guidelines:**
- Provide explanations in details why the advisors are recommended.
- You can answer both **general** and **scenario-specific** questions.

**For General Questions** (e.g., *"How does the system work?"*):
- Explain both models:
  • How keyword similarity (cosine similarity) works like comparing the direction of two arrows.
  • How LDA groups keywords into research themes and compares distributions.
- Explain why using both models gives a more robust match.
- Provide Feature-Based Explanation: Explains how users individual research keywords contributed to the results. Provide example using users individual research keywords contributes from rank 1 to rank 3 similar advisor.
- Counterfactual-Based Explanation: Shows how changing reseach keywords would alter predictions. Provide example using users individual research keywords change can make rank 3 to rank 1.
- Model Inner Working with Simple Example: Provides a basic calculation with example of how the system makes decisions. Provide example such as [kw1, kw2,..] to vector using users individual research keywords. Then a similarity score example using dot product. Also a LDA group of words.

**For Scenario-Specific Questions** (e.g., *"Why the top advisor recommended?"*):
- Explain how the user’s keywords closely matched the advisor’s keywords or topics.
- Show which terms contributed to high similarity. Show a dot product calculation.
- Mention concrete alignment in research themes.
- Highlight key differences between text vs. topic model rankings.
- Give “what-if” examples—how changing or refining keywords might change recommendations.
- Clarify what the similarity scores mean and that a lower score can still be meaningful in niche areas.

You are now ready to answer the user’s questions about their recommended graduate advisors.
                """
                st.session_state.messages = [{'role':'system', 'content':prompt+msg+prompt2}]
                    #response="Welcome "+name+"! Would you like an explanation of your recommendation for advisors?"
                response = st.write_stream(chat_stream()) 

                print(msg)
                st.session_state.messages.append({"role": "assistant", "content": response})
                    #connection = connect_to_db()
                    #insert_message(connection, "LLM", response)
                    #connection.close()
                
                    
    if not st.session_state.prediction_ready:
                render_spacer()            
    if st.session_state.prediction_ready:
                df1 = pd.DataFrame(st.session_state["flag"])
                df2 = pd.DataFrame(st.session_state["lda1"])
                df3 = pd.DataFrame(st.session_state["lda2"])
                
                #left_column, right_column = st.columns(2)
                #left_column, right_column = st.tabs(["Text Similarity", "Topic Similarity"])
                #with left_column:
                
                df1_new = df1[['Ranking','Name','Publication','Affiliation']]                
                df1_new = df1_new.to_dict(orient='records')
                st.write("Top 3 recommended advisor based on Text Similarity of keywords:")
                st.dataframe(df1_new, hide_index=True,  column_config={
                "Publication": st.column_config.Column(
                width="large",
                required=True,
                ),
                "Affiliation": st.column_config.Column(
                width="medium",
                required=True,
                )
                },)

                #with right_column:
                st.write("Top 3 recommended advisor based on LDA Topic Similarity of 30 topics:")
                df2_new = df2[['LDA_rank','LDA_Name','Publication','Affiliation']] 
                df2_new = df2_new.to_dict(orient='records')
                st.dataframe(df2_new,hide_index=True, column_config={
                "LDA_rank": "Ranking","LDA_Name": "Name", "Publication": st.column_config.Column(
                width="large",
                required=True,
                ),})
                st.write("Double clicking individual cell will provide detail texts.")
                if st.session_state.page == "v3":
                        st.markdown("---")
                        
                        st.markdown("## 📋 How the Advisor Recommender System Works")
                        
                        st.markdown("""
                        Our system is designed to help prospective graduate students find suitable research advisors by matching them based on shared research interests and publications. The system uses two models: a **Text Similarity Model** and a **Topic Similarity Model**, each generating the top three advisor recommendations based on the user’s input keywords.
                        
                        The **Text Similarity Model** converts research keywords from publications into numerical count vectors and uses cosine similarity to measure how closely a user’s research interests align with those of potential advisors. A score closer to 1 indicates a stronger match, and the top three advisors with the highest similarity scores are recommended.
                        
                        The **Topic Similarity Model** employs Latent Dirichlet Allocation (LDA) to categorize publication keywords into 30 thematic clusters. Each advisor is assigned probability scores across these topics, creating a thematic profile. The system matches the user’s input keywords to these topics and recommends the top three advisors whose profiles align most closely with the user’s interests.
                        
                        Results are displayed in two tabs: one for Text Similarity and one for Topic Similarity, each showing advisors’ names, affiliations, and publication details. The recommendations aim to foster meaningful academic collaborations by aligning students with advisors whose research interests are most compatible.
                        """)
                        
                        with st.expander("**Key Terms**", expanded=True):
                            st.markdown("""
                        - **Text Similarity Model:** Converts publication keywords into count vectors and uses cosine similarity to measure alignment with user interests.
                        - **Topic Similarity Model:** Uses Latent Dirichlet Allocation (LDA) to group keywords into 30 topics and matches user interests to advisors’ thematic profiles.
                        - **Cosine Similarity:** A score (0 to 1) indicating how closely two sets of keywords align; higher scores mean greater similarity.
                        - **Latent Dirichlet Allocation (LDA):** A model that groups keywords into thematic clusters to identify research topics.
                        - **Count Vector:** A numerical representation of keywords, where each value indicates the presence or frequency of a keyword.
                        """)
                        # Link to Quiz
                        st.markdown("---")
                        
                        reveal_button_html = """
                                <a href="https://quiz-rec.streamlit.app/" target="_blank">
                                        <button style="
                                            background-color:#4CAF50;
                                            border:none;
                                            color:white;
                                            padding:10px 20px;
                                            text-align:center;
                                            text-decoration:none;
                                            display:inline-block;
                                            font-size:16px;
                                            border-radius:5px;
                                            cursor:pointer;">
                                            Go to Quiz
                                        </button>
                                    </a>
                                """
                        countdown_component_html("Please read the given text carefully", COOLDOWN_TIME_LONG, reveal_button_html)

                        
                if st.session_state.page == "v1":
                        
                        if "messages"  in st.session_state:
                            for message in st.session_state.messages:
                                if message['role']=='system':
                                    continue
                                if message['role']=='user':
                                    with st.chat_message(message["role"],avatar="👦"):
                                        st.markdown(message["content"])
                                else:
                                    with st.chat_message(message["role"]):
                                        st.markdown(message["content"])
                            def ask_and_advance(i):
                                    st.session_state.messages.append({"role": "user", "content": st.session_state.questions[i]})                                 
                                    response = st.write_stream(chat_stream())                               
                                    st.session_state.question_asked+=1                                  
                                    st.session_state.messages.append({"role": "assistant", "content": response})
                            if st.session_state.question_asked<2:
                                                if countdown_with_button(
                                                            message="Please read the generated text carefully",
                                                            duration_sec=COOLDOWN_TIME_SHORT,
                                                            button_label=questions[st.session_state.question_asked],
                                                            button_key=f"followup_btn_{st.session_state.question_asked}"
                                                        ):
                                                        ask_and_advance(st.session_state.question_asked)
                                                        st.rerun()
                                                       
                                                        
                                                
                            if st.session_state.question_asked>=2:
                                    prompt=countdown_with_form(
                                                    message="Please read carefully before interacting with the chatbot",
                                                    duration_sec=COOLDOWN_TIME_LONG,
                                                    form_key="freeform_followup",
                                                    input_key="freeform_input"
                                                )
                                    if prompt : 
                                        #st.chat_input("Example: 1. Tell me the research interests of the recommended advisor based on cosine similarity. \n2. Tell me why 'X' is recommended.\n 3. What is cosine similarity."):
                                        st.session_state.messages.append({"role": "user", "content": prompt})
                                        with st.chat_message("user",avatar="👦"):
                                            st.markdown(prompt)
                
                                        with st.chat_message("assistant"):    
                                            response = st.write_stream(chat_stream())
                                        st.session_state.messages.append({"role": "assistant", "content": response})
                                        st.rerun()
                
                                                
                
