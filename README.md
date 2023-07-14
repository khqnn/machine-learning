# machine-learning

### Day 24:
- load cleaned cuisines data and preview data and its info
- separate labels and features
- choose a classification model for our data
- train a logistic regression model on our data
- get predictios over test data
- calculate accuracy of our model on test data
- print classification report
- train linear SVC on our training data and print accuracy and classification report
- train K-Neighbours classifier on training data and print accuracy and classification report
- train SVC classifier on training data
- train ensemble methods i.e. Random Forest and AdaBoost classifiers
- it is noticed that on cuisines data from above 5 algorithms Random forest outperforms other algorithms

### Day 23:
classification
- load cuisines dataset
- display data info
- plot data in barplot
- find out how much data is available per cuisine
- visualize the ingredients
- visualize each country ingredients separately
- drop the most common ingredients that create confusion between distinct cuisines
- balace dataset with SMOTE
- check number of samples after balancing the data
- create new dataset for balanced dataset
- save transformed data
- 

### Day 22:
for creating a flask web app
- app.py has an api that excepts three query params as seconds, lat and lng
- load pickle model that is trained on ufos dataset
- return country name in response of api

### Day 21:
- load ufos dataset
- reduce dataset to keep only four columns
- filter dataset based on seconds from 1 to 60
- encode labels
- prepare features and labels
- split data into training and testing
- train model
- calculate model performance and print classification report
- pickle the trained model
- load pickle model and predict a sample

### Day 20:
logistic regression on pumpkins data
- load and view data
- our output feature is categorical feature i.g. Color
- correlation should not exists while working on logistic regression unlike linear or polynomial regression
- select few features and visualize dataframe
- remove nulls
- visualize color vs variety on category plot
- identify categorical and ordinal features
- transform categorical and ordinal features into numeric features
- transform output feature(categorical) into labels
- visualize color, variety and item sizes on whiskers box plot
- prepare dataset
- split data into training and testing set
- create a linear regression model
- train the model
- evaluate model on test data
- classification report, F1-score and confusion matrix

### Day 19:
- for linear regression choose one variable of categorical type (that is variety in pumpkins data)
- convert categorical feature into numeric features(one hot encoding)
- prepare data
- split data into test and train
- ceate a pipeline for polynomial features of degree 2 and a linear regression
- train the model on training data
- get predictions of model on test data
- calculate the model's performance on test data

### Day 18:
- now instead of linear model(straight line) we tried polynomial model with 2 features
- created a pipeline for polynomial features and linear regression
- train model using pipeline
- make predictions on test data
- calculate model performance in terms of error
- visualize linear regression model with polynomial features

### Day 17:
- chosse one variable from pumpkins data
- prepare feature and value data
- split data into training and testing
- choose linear regression model
- train model on training data
- get predictions on test data
- find out error scores
- visualize model on training data (one variable on 2d space)

### Day 16: 
- visualize price over month
- calculate day of year for each record
- visualize day wise price distribution
- find correlation between price and day/month
- explore pumpkins variety
- visualize variety wise distribution
- calculate and visualize mean price for each variety
- visualize price distribution for individual variety
- find correlation of price with each variety of pumpkins
- keep data only with PIE TYPE pumpkin variety
- see information about data and remove null

Note:
- while dealing with two variables like month and price visualize it horizontally for each month how price is distributed as well as for each month price aggeregation
- for categorical attribute do some data processing with different categories of data (like finding correlation)

### Day 15:
load US Pumpkins data and do some preprocessing
- load data
- know your data by checking out its shape and info
- drop some columns
- convert month string into number
- calculate price by averaging low and high price
- find out total package types
- filter out data and keep data only that contains bushel package type

### Day 14: 
Regression notebook contains
- load diabeties dataset using load_dataset from sklearn dataset
- analyze diabeties dataset
- features and targets
- split dataset
- fit linear regression model on dataset
- visualize model fit for one feature

### Day 13:
- linear regression is a straight line that tried to fit all data with minimum of mean square error
- polynomial regression is a line with multiple independent variables that fit to data with minimum mean square error
- logistic regression is a line that separates data with minimum log loss

### Day 12:
Regression
- Regression models can help determine the relationship between variables. This type of model can predict values such as length, temperature, or age, thus uncovering relationships between variables as it analyzes data points.
- There multiple regression techniques including
1. linear regression
2. polynomial regression
3. logistic regression 

### Day 11:
Prebuilding tasks
- collect data
- clean and prepare data
- feature selection
- features and targets
- visualize your data
- split data

### Day 10:
The typical machine learning process is as follows
- decide on the question
- collect and prepare data
- choose a training method
- train the model
- evaluate the model
- parameter tuning(fine tuning)
- predict
- deploy
- monitor

### Day 9:
- detect unfairness in your model
- understand your model and build in fairness
- identify harms (and benifits)
- identify the affected groups
- define fairness metrics
- to mitigate the unfairness do above steps and track tradeoff between the accuracy and fairness and choose fairness accordingly

### Day 8:
- besides the faireness there are other factors that must be taken into account while building a machine learning model
- Reliability and safety
- Inclusiveness
- Security and privacy
- Transparency
- Accountability

### Day 7:
**Over-representation or under-representation**
- Skewed image search results can be a good example of this harm. When searching images of professions with an equal or higher percentage of men than women, such as engineering, or CEO, watch for results that are more heavily skewed towards a given gender.

### Day 6:
**Quality of Service**
- Researchers found that several commercial gender classifiers had higher error rates around images of women with darker skin tones as opposed to images of men with lighter skin tones. 
- Another infamous example is a hand soap dispenser that could not seem to be able to sense people with dark skin.

### Day 5:
**Denigration**
- An image labeling technology infamously mislabeled images of dark-skinned people as gorillas. Mislabeling is harmful not just because the system made a mistake because it specifically applied a label that has a long history of being purposefully used to denigrate Black people.

### Day 4:
**

**Stereotyping**
- A stereotypical gender view was found in machine translation. When translating “he is a nurse and she is a doctor” into Turkish, problems were encountered. Turkish is a genderless language which has one pronoun, “o” to convey a singular third person, but translating the sentence back from Turkish to English yields the stereotypical and incorrect as “she is a nurse, and he is a doctor.”

### Day 3:
**Allocation**
- a scenario where a group is favoured over another
- one example is to build a loan approval machine learning model which prefered white man over black. this kind of unfairness should be considered critical
- another example is the hr system for screening application and the system mostly select male candidates over female candidates. this unfairness based on gender demographic.
- this kind of fairness maybe occurred because the historical data on which the ML model is trained have seen more examples of male candidates and female candidates were less in number.

### Day 2:  
**Fairness**
- While building a machine learning model it should be considered the model should be fair
- it should not ignore any group based on race or gender
- there are total 5 categories of unfairness
1. Allocation
2. Quality of service
3. Stereotyping
4. Denigration
5. Over- or under- representation

### Day 1:  
**Introduction to Machine learning**  

- Machine learn data in a similar way as a child learn new things from its surroundings(data and observations)
- Machine learning has many applications like predicting the probability of disease in a patient. detecting spam emails, facial recognition, digital assitants and many more
- There are some computing problems that are very hard to solve by treditional programming techniques. e.g. to distinguish between an image of an orange and and apple. we cannot use just a predicate logic or an algorithm to made that decision. We might need hunderds and thousands of checks. And once we could able to do that then what if also need to distinguish between an apple, orange and mango. So it's just very hard to do that in that way. These kind of problems need different methodolgy than the treditional programming. 
- Machine learning is a subset of the field **Artificial Intellegence**, and the **Deep Learning** is the sub-feild of Machine learning, whereas the term **Data Science** is broader and it has some part of three of them and many other things. 
