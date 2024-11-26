import json
import joblib
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .models import MLInputData
from .serializers import MLInputDataSerializer
from sklearn.preprocessing import LabelEncoder

# Load the trained model once at server startup
model = joblib.load('/Users/aravajayaram/ML_Project_using_Django/research/research/random_forest.joblib')  # Update with the correct path
encoders = {}

class PredictView(APIView):
    def post(self, request, *args, **kwargs):
        # Parse the input data
        serializer = MLInputDataSerializer(data=request.data)
        if serializer.is_valid():
            input_data = serializer.save()

            # Convert input data to the correct format for prediction
            input_data_dict = {
                "age": input_data.age,
                "workclass": input_data.workclass,
                "fnlwgt": input_data.fnlwgt,
                "education": input_data.education,
                "education_num": input_data.education_num,
                "marital_status": input_data.marital_status,
                "occupation": input_data.occupation,
                "relationship": input_data.relationship,
                "race": input_data.race,
                "sex": input_data.sex,
                "capital_gain": input_data.capital_gain,
                "capital_loss": input_data.capital_loss,
                "hours_per_week": input_data.hours_per_week,
                "native_country": input_data.native_country
            }

            # Apply LabelEncoder to categorical columns
            categorical_columns = [
                'workclass', 'education', 'marital_status', 'occupation', 
                'relationship', 'race', 'sex', 'native_country'
            ]

            for column in categorical_columns:
                if column not in encoders:
                    encoders[column] = LabelEncoder()
                    # Fit the encoder on the existing unique values in the training dataset
                    # Here we simulate it by using the current data as an example
                    # You can use a training dataset or sample data to fit the encoder
                    encoders[column].fit([input_data_dict[column]])  # Fit on the input data (or training data)
                input_data_dict[column] = encoders[column].transform([input_data_dict[column]])[0]

            # Create the features list for prediction
            features = [
                input_data_dict["age"],
                input_data_dict["workclass"],input_data_dict["fnlwgt"],
                input_data_dict["education"],
                input_data_dict["education_num"],
                input_data_dict["marital_status"],
                input_data_dict["occupation"],
                input_data_dict["relationship"],
                input_data_dict["race"],
                input_data_dict["sex"],
                input_data_dict["capital_gain"],
                input_data_dict["capital_loss"],
                input_data_dict["hours_per_week"],
                input_data_dict["native_country"]
            ]

            # Ensure features is a 2D array: (1, n_features)
            features_2d = [features]  # Convert the features list into a 2D array (1 sample)

            # Make the prediction
            prediction = model.predict(features_2d)

            return Response({'prediction': prediction[0]}, status=status.HTTP_200_OK)

        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
