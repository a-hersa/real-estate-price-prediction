from flask import Flask,request,render_template,jsonify
from src.pipeline.prediction_pipeline import CustomData,PredictPipeline

application=Flask(__name__)

app=application

@app.route('/')
def home_page():
    return render_template('index.html')

@app.route('/predict', methods=['GET','POST'])

def predict_datapoint():
    if request.method=='GET':
        return render_template('form.html')
    
    else:
        data=CustomData(
            deal_type=request.form.get('deal_type'),
            parking=int(request.form.get('parking')),
            rooms=int(request.form.get('rooms')),
            sqrm=int(request.form.get('sqrm')),
            floor=int(request.form.get('floor')),
            surface=int(request.form.get('surface')),
            elevator=int(request.form.get('elevator')),
            property_type=request.form.get('property_type'),
            location_encoded=int(request.form.get('location_encoded'))
        )
        final_new_data = data.get_data_as_dataframe()
        predict_pipeline=PredictPipeline()
        pred = predict_pipeline.predict(final_new_data)

        results=round(pred[0],2)

        return render_template('results.html', final_result=results)
    
if __name__=='__main__':
    app.run(host='0.0.0.0',debug=True)