import streamlit as st
from PIL import Image
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import LabelEncoder
import joblib
import warnings
warnings.filterwarnings('ignore')

# import data
iris = pd.read_csv('iris.data.csv')
iris.head(3)

# rename dataframe columns for ease of understanding
iris.rename(columns = {iris.columns[0]:'sepal length (cm)', iris.columns[1]:'sepal width (cm)', iris.columns[2]:'petal length (cm)', iris.columns[3]:'petal width (cm)', iris.columns[4]:'names'}, inplace = True)

# define x and y variables
x = iris.drop(['names'], axis = 1)
y = iris['names']

# label encode variable y
lb = LabelEncoder()
y = lb.fit_transform(y)

# Split data into train and test
x_train , x_test , y_train, y_test = train_test_split(x, y, test_size = 0.2, stratify = y)

# create dataframe for train and test data
train_data = pd.concat([x_train, pd.Series(y_train)], axis = 1)
test_data = pd.concat([x_test, pd.Series(y_test)], axis = 1)

# create your model
logReg = LogisticRegression()
logRegFitted = logReg.fit(x_train, y_train)
y_pred = logRegFitted.predict(x_test)

# using R2 score module from sklearn metrics for the goodness to fit information
score = r2_score(y_test,y_pred)
print(score)

# saving the model using joblib
joblib.dump(logReg, 'Logistic_Model.pkl')

# BEGIN IMPLEMENTATION USING STREAMLIT

st.header('IRIS MODEL DEPLOYMENT')
user_name = st.text_input('Please enter your name: ', key='my_user_name', on_change=None)
button = st.button('Submit')

if user_name != '':
    if button:
        st.write(f'<p style="font-size:13px;">You are welcome {user_name}. We hope you find what you\'re looking for!</p>', unsafe_allow_html=True)
        # st.write(iris)
elif user_name == '' and button:
    st.warning('Please enter your name to get started.')

placeholder = Image.open('flower_images/placeholder.jpeg')
st.sidebar.image(placeholder)

if user_name != '':
    st.sidebar.subheader(f'Hey, {user_name}!')
    metric = st.sidebar.radio('How do you want your feature to be presented?', ('No selection','Slider','Direct Input'))

    if metric == 'No selection':
        st.sidebar.warning('There is no information to display. You need to make a selection.')
        
    elif metric == 'Slider':
        st.write(iris)
        sepal_length = st.sidebar.slider('Sepal length', 0.0, 9.0, (0.0))
        sepal_width = st.sidebar.slider('Sepal width', 0.0, 4.5, (0.0))
        petal_length = st.sidebar.slider('Petal length', 0.0, 8.0, (0.0))
        petal_width = st.sidebar.slider('Petal width', 0.0, 3.0, (0.0))

        input_values = [[sepal_length, sepal_width, petal_length, petal_width]]

        # modelling - import the model

        model = joblib.load(open('Logistic_Model.pkl', 'rb'))
        pred = model.predict(input_values)

        if pred == 0:
            st.success('The flower is an Iris-setosa')
            setosa = 'https://live.staticflickr.com/65535/51376589362_b92e27ae7a_b.jpg'
            st.image(setosa, caption = 'Iris-setosa', width = 400)
        elif pred == 1:
            st.success('The flower is an Iris-versicolor ')
            versicolor = 'https://upload.wikimedia.org/wikipedia/commons/thumb/d/db/Iris_versicolor_4.jpg/1200px-Iris_versicolor_4.jpg'
            st.image(versicolor, caption = 'Iris-versicolor', width = 400)
        else:
            st.success('The flower is an Iris-virginica ')
            virginica = 'https://www.fs.usda.gov/wildflowers/beauty/iris/Blue_Flag/images/iris_virginica/iris_virginica_shrevei_tb2_lg.jpg'
            st.image(virginica, caption = 'Iris-virginica', width = 400 )

    else:
        st.write(iris)
        sepal_length = st.sidebar.number_input('Sepal length')
        sepal_width = st.sidebar.number_input('Sepal width')
        petal_length = st.sidebar.number_input('Petal length')
        petal_width = st.sidebar.number_input('Petal width')

        input_values = [[sepal_length, sepal_width, petal_length, petal_width]]

    # modelling - import the model

        model = joblib.load(open('Logistic_Model.pkl', 'rb'))
        pred = model.predict(input_values)

        if pred == 0:
            st.success('The flower is an Iris-setosa')
            setosa = 'https://live.staticflickr.com/65535/51376589362_b92e27ae7a_b.jpg'
            st.image(setosa, caption = 'Iris-setosa', width = 400)
        elif pred == 1:
            st.success('The flower is an Iris-versicolor ')
            versicolor = 'https://upload.wikimedia.org/wikipedia/commons/thumb/d/db/Iris_versicolor_4.jpg/1200px-Iris_versicolor_4.jpg'
            st.image(versicolor, caption = 'Iris-versicolor', width = 400)
        else:
            st.success('The flower is an Iris-virginica ')
            virginica = 'https://www.fs.usda.gov/wildflowers/beauty/iris/Blue_Flag/images/iris_virginica/iris_virginica_shrevei_tb2_lg.jpg'
            st.image(virginica, caption = 'Iris-virginica', width = 400 )

else:
    st.sidebar.subheader('Hey, there!')


