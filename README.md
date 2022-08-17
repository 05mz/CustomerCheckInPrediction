# CustomerCheckInPrediction
ML model to predict if the customer is going to checkin or not

Deployment of the project:

-Using the code from python file in https://github.com/05mz/CustomerCheckInPrediction/tree/main/nonDeployed

-I added the code to a single file, and made necessary changes to be compatible with streamlit

-Listed the necessary libraries in requirement file

-streamlit checks the requirements file and installs all the dependencies

-the code is then executed and output is displayed on the app in streamlit

Deployed App link: https://05mz-customercheckinprediction-checkinprediction-sqn6xh.streamlitapp.com/

Screencast Video link: https://drive.google.com/file/d/1VTXXt1-LD26u4qngpwFMD1eJvjKuRTYI/view?usp=sharing

Bonus Questions:

1. Write about any difficult problem that you solved. (According to us difficult - is something which 90% of people would have only 10% probability in getting a similarly good solution). 

-Faced difficulty while testing the model against new data, due to different dimensions
-to solve this, i set new dimensions for the machine before compiling

2. Explain back propagation and tell us how you handle a dataset if 4 out of 30 parameters have null values more than 40 percentage

Back propagation: it is a way to train artificial neural networks, it is used to reduce the errorrate. the input is sent to a model which is based on ANN, the hidden layer processes the data based on the model and the output layer gives the output. this output may have some error in it, hence the output data is fed back to the hidden layer, so that the new output that is generated has less error rate.

to handle the missing values: if the values that are missing are insignificant to our problem statement, then drop the columns using dropna()
if the data values that are missing in the important attributes, then the most easy method is to use pandas imputation method. we fill the missing values using numeric values using fillna(mean()). this method calculates the mean of the present values in a column and uses that mean to fill the missing values.

