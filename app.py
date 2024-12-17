from flask import * 
import pandas as pd 
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression

app = Flask(__name__) 

url = "https://raw.githubusercontent.com/sarwansingh/Python/master/ClassExamples/data/diabetes.csv"
df = pd.read_csv(url)
df.gender.unique()
df.gender.replace(['Male','Female','Other'],[1,2,3],inplace=True)
df.smoking_history.unique()
df.smoking_history.replace(['never', 'No Info', 'current', 'former', 'ever', 'not current'],[1,2,3,4,5,6],inplace=True)


X = df[['gender',	'age'	,'hypertension'	,'heart_disease',	'smoking_history'	,'bmi'	,'HbA1c_level',	'blood_glucose_level'	]]
Y = df['diabetes']
 
dmodel = LinearRegression( )
dmodel.fit(X, Y)
res = dmodel.predict([[1,	67.0,	0,	0,	3,	19	,6.6,	100	]])
op = str( round(res[0]*100 ,2)) + "%"


@app.route('/')
def dproject(): 
   return render_template("index.html" )  
@app.route("/dpredict", methods=['POST'])
def dpredict():
  gender         = int(request.form["gender"])
  age            = int(request.form["age"])
  hypertension   = int(request.form["hypertension"])
  heartdisease   = int(request.form["heartdisease"])
  smokinghistory = int(request.form["smokinghistory"])
  bmi            = int(request.form["bmi"])
  HbA1clevel     = int(request.form["HbA1clevel"])
  bloodglucoselevel = int(request.form["bloodglucoselevel"])
  
  
  res = dmodel.predict([[ gender, age,hypertension,heartdisease,smokinghistory,bmi,HbA1clevel,bloodglucoselevel ]])
  op =  "   Diabetes Risk: " + str( round(res[0]*100 ,2)) + "%"
 
  return render_template("index.html", result=op)          
  
if __name__ == '__main__': 
  app.run()
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  