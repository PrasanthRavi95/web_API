import pickle
from flask import Flask, request, Response, jsonify
app = Flask(__name__)


@app.route('/')
def hello():
    return 'Hello, World! This is a basic Flask API.'


@app.route('/ping', methods=['GET'])
def ping():
    return "Pinging Model!!"


@app.route('/predict', methods=['POST'])
def prediction():
    # Pre-processing user input
    loan_req = request.get_json()
    print(loan_req)
    ret = loan_req['input'] * 5
    result = {
        'loan_approval_status': ret
    }

    return jsonify(result)


# Load the pre-trained Logistic Regression model
with open('.\\artifacts\\loan_approval_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)


@app.route('/predict_loan_approval', methods=['POST'])
def predict_loan_approval():
    try:
        # Get input features from the JSON data in the request body
        data = request.get_json()

        # Ensure that the required features are present in the JSON data
        required_features = ['income', 'credit_score', 'loan_amount']
        if all(feature in data for feature in required_features):
            # Prepare the input features for prediction
            input_features = [[data[feature] for feature in required_features]]

            # Make a prediction using the pre-trained model
            prediction = model.predict(input_features)[0]

            # Create a response JSON
            response_data = {
                "success": True,
                "prediction": "Approved" if prediction == 1 else "Not Approved"
            }
        else:
            response_data = {
                "success": False,
                "message": f"Required features {required_features} not found in the JSON data."
            }

    except Exception as e:
        response_data = {
            "success": False,
            "message": f"An error occurred: {str(e)}"
        }

    # Return the response as JSON
    return jsonify(response_data)


if __name__ == '__main__':
    app.run(debug=True)
