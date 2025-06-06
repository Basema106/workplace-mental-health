# workplace-mental-health

The "Analyzing the Impact of Workplace Factors on Employee Mental Health" project explores how various occupational and personal factors contribute to mental health challenges. The dataset includes demographic (Gender, Country, Occupation), psychological (Growing Stress, Coping Struggles, Mood Swings, Mental Health History), and behavioral factors (Days Indoors, Work Interest, Changes in Habits) that may influence mental well-being. 

Here I built a machine learning model to predict whether an individual is likely to seek mental health treatment based on personal and workplace factors.
It uses an Artificial Neural Network (ANN) trained on a mental health survey dataset.
I also did a Streamlit web application that allows users to input their information and receive a prediction result in real time.

About Dataset:

This dataset appears to contain a variety of features related to text analysis, sentiment analysis, and psychological indicators, likely derived from posts or text data. Some features include readability indices such as Automated Readability Index (ARI), Coleman Liau Index, and Flesch-Kincaid Grade Level, as well as sentiment analysis scores like sentiment compound, negative, neutral, and positive scores. Additionally, there are features related to psychological aspects such as economic stress, isolation, substance use, and domestic stress. The dataset seems to cover a wide range of linguistic, psychological, and behavioural attributes, potentially suitable for analyzing mental health-related topics in online communities or text data.

Data set:
https://www.kaggle.com/datasets/bhavikjikadara/mental-health-dataset?resource=download

Google colab Notebook:
https://colab.research.google.com/drive/108c4uiDmPpRBISg1fhspyVPRlooBZnUk?usp=sharing

List of required packages:

Available under requiremnts.txt


Instructions on how to run the code:

1-How to run the work_mental.py model:

(Optional) Retrain the Model:

If you wish to retrain the model from scratch, you can run:


python work_mental.py

make sure that all the required packages are downloaded.

2-How to run the StreamLit app:

download the StreamLit folder from JarrarBasema folder, then cd JarrarBasema/StreamLit , then streamlit run app.py. 
