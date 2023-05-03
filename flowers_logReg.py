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
button = st.button('Click Me!')

if user_name != '':
    if button:
        st.write(f'<p style="font-size:13px;">You are welcome {user_name}. We hope you find what you\'re looking for!</p>', unsafe_allow_html=True)

elif user_name == '' and button:
    st.warning('Please enter your name to get started.')

placeholder = Image.open('flower_images/placeholder.jpeg')

if user_name != '':
    st.sidebar.image(placeholder)
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
        predict = st.sidebar.button('Predict my flower')
        
        # if metric == 'Slider' or metric == 'Direct Input' and predict:
        
        if sepal_length > 0 and sepal_width > 0 and petal_length > 0 and petal_width > 0:
            if predict:
                pred = model.predict(input_values)

            # if (sepal_length, sepal_width, petal_length, petal_width) > 0:
                if pred == 0:
                    st.success('The flower is an Iris-setosa')
                    setosa = 'https://live.staticflickr.com/65535/51376589362_b92e27ae7a_b.jpg'
                    st.image(setosa, caption = 'Iris-setosa', width = 400)
                    st.write('''
                    The Iris setosa is a species of flowering plant that belongs to the genus Iris. It is a perennial plant that is native to North America and Eurasia. The Iris setosa is characterized by its narrow, grass-like leaves, and its showy, delicate flowers that come in a variety of colors, including white, pink, and blue-violet. \n
                    The Iris setosa is different from the Iris Virginia and Iris versicolor in several ways. One of the most noticeable differences is in their appearance. The Iris setosa has a smaller stature and a more delicate appearance compared to the other two species. Additionally, the Iris setosa is adapted to colder environments, and it can withstand harsher conditions than the other two species. Finally, the Iris setosa has a different genetic makeup and a distinct set of physical characteristics that distinguish it from the other two species.
                    ''')
                elif pred == 1:
                    st.success('The flower is an Iris-versicolor ')
                    versicolor = 'https://upload.wikimedia.org/wikipedia/commons/thumb/d/db/Iris_versicolor_4.jpg/1200px-Iris_versicolor_4.jpg'
                    st.image(versicolor, caption = 'Iris-versicolor', width = 400)
                    st.write('''
                    Iris versicolor is a species of flowering plant native to North America. It is one of the three iris species commonly referred to as "iris". It is a herbaceous perennial, growing to 50 cm tall, with leaves 30-60 cm long and 1-3 cm wide. The flowers are produced on a slender stem up to 90 cm tall, with a few to 20 blooms, each flower is 5-10 cm in diameter with six tepals (three petals and three sepals) of varying colors, typically blue to violet with white or yellow accents. The differences between Iris versicolor and Iris setosa are subtle, with the latter having narrower leaves and more elongated flowers, while the former has wider leaves and rounder flowers. Iris virginica, on the other hand, has larger flowers than Iris versicolor and a distinctive purple-tinged stem.
                    ''')
                else:
                    st.success('The flower is an Iris-virginica ')
                    virginica = 'https://www.fs.usda.gov/wildflowers/beauty/iris/Blue_Flag/images/iris_virginica/iris_virginica_shrevei_tb2_lg.jpg'
                    st.image(virginica, caption = 'Iris-virginica', width = 400 )
                    st.write('''
                    Iris virginica is a species of the iris flower and is one of the three species commonly known as irises. It is native to the eastern United States and grows in wetland areas. The plant typically grows up to 3 feet tall and produces large, showy flowers that are usually blue-violet in color but can also be white or pink. \n
                    Iris virginica can be distinguished from the other two iris species (Iris setosa and Iris versicolor) by several characteristics. One of the most noticeable differences is the shape of the flower petals: Iris virginica has long, narrow petals that curve upwards and backwards, while Iris setosa and Iris versicolor have shorter, wider petals that are usually straight or curve slightly downwards. Another difference is the size of the plant and flower: Iris virginica is generally taller and larger overall than the other two species. \n
                    In addition to these physical differences, Iris virginica also has different ecological requirements than Iris setosa and Iris versicolor. While Iris setosa is adapted to cold, northern climates and Iris versicolor is found in a wide variety of habitats, Iris virginica is typically found in wetland areas such as marshes and swamps.
                    ''')

        elif sepal_length == 0 and sepal_width == 0 and petal_length == 0 and petal_width == 0:
            if predict:
                st.error('Please select values for each flower feature.')

    else:
        st.write(iris)
        sepal_length = st.sidebar.number_input('Sepal length')
        sepal_width = st.sidebar.number_input('Sepal width')
        petal_length = st.sidebar.number_input('Petal length')
        petal_width = st.sidebar.number_input('Petal width')

        input_values = [[sepal_length, sepal_width, petal_length, petal_width]]

    # modelling - import the model

        model = joblib.load(open('Logistic_Model.pkl', 'rb'))

        predict = st.sidebar.button('Predict my flower')
        
        # if metric == 'Slider' or metric == 'Direct Input' and predict:
        
        if sepal_length > 0 and sepal_width > 0 and petal_length > 0 and petal_width > 0:
            if predict:
                pred = model.predict(input_values)

            # if (sepal_length, sepal_width, petal_length, petal_width) > 0:
                if pred == 0:
                    st.success('The flower is an Iris-setosa')
                    setosa = 'https://live.staticflickr.com/65535/51376589362_b92e27ae7a_b.jpg'
                    st.image(setosa, caption = 'Iris-setosa', width = 400)
                    st.write('''
                    The Iris setosa is a species of flowering plant that belongs to the genus Iris. It is a perennial plant that is native to North America and Eurasia. The Iris setosa is characterized by its narrow, grass-like leaves, and its showy, delicate flowers that come in a variety of colors, including white, pink, and blue-violet. \n
                    The Iris setosa is different from the Iris Virginia and Iris versicolor in several ways. One of the most noticeable differences is in their appearance. The Iris setosa has a smaller stature and a more delicate appearance compared to the other two species. Additionally, the Iris setosa is adapted to colder environments, and it can withstand harsher conditions than the other two species. Finally, the Iris setosa has a different genetic makeup and a distinct set of physical characteristics that distinguish it from the other two species.
                    ''')
                elif pred == 1:
                    st.success('The flower is an Iris-versicolor ')
                    versicolor = 'https://upload.wikimedia.org/wikipedia/commons/thumb/d/db/Iris_versicolor_4.jpg/1200px-Iris_versicolor_4.jpg'
                    st.image(versicolor, caption = 'Iris-versicolor', width = 400)
                    st.write('''
                    Iris versicolor is a species of flowering plant native to North America. It is one of the three iris species commonly referred to as "iris". It is a herbaceous perennial, growing to 50 cm tall, with leaves 30-60 cm long and 1-3 cm wide. The flowers are produced on a slender stem up to 90 cm tall, with a few to 20 blooms, each flower is 5-10 cm in diameter with six tepals (three petals and three sepals) of varying colors, typically blue to violet with white or yellow accents. The differences between Iris versicolor and Iris setosa are subtle, with the latter having narrower leaves and more elongated flowers, while the former has wider leaves and rounder flowers. Iris virginica, on the other hand, has larger flowers than Iris versicolor and a distinctive purple-tinged stem.
                    ''')
                else:
                    st.success('The flower is an Iris-virginica ')
                    virginica = 'https://www.fs.usda.gov/wildflowers/beauty/iris/Blue_Flag/images/iris_virginica/iris_virginica_shrevei_tb2_lg.jpg'
                    st.image(virginica, caption = 'Iris-virginica', width = 400 )
                    st.write('''
                    Iris virginica is a species of the iris flower and is one of the three species commonly known as irises. It is native to the eastern United States and grows in wetland areas. The plant typically grows up to 3 feet tall and produces large, showy flowers that are usually blue-violet in color but can also be white or pink. \n
                    Iris virginica can be distinguished from the other two iris species (Iris setosa and Iris versicolor) by several characteristics. One of the most noticeable differences is the shape of the flower petals: Iris virginica has long, narrow petals that curve upwards and backwards, while Iris setosa and Iris versicolor have shorter, wider petals that are usually straight or curve slightly downwards. Another difference is the size of the plant and flower: Iris virginica is generally taller and larger overall than the other two species. \n
                    In addition to these physical differences, Iris virginica also has different ecological requirements than Iris setosa and Iris versicolor. While Iris setosa is adapted to cold, northern climates and Iris versicolor is found in a wide variety of habitats, Iris virginica is typically found in wetland areas such as marshes and swamps.
                    ''')

        elif sepal_length == 0 and sepal_width == 0 and petal_length == 0 and petal_width == 0:
            if predict:
                st.error('Please select values for each flower feature.')

        else:
            st.warning('Please select the predict button to proceed.')


