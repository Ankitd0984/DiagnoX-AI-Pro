import unittest

import pandas as pd

import app as predictor_app


class StubModel:
    def __init__(self, prediction):
        self.prediction = prediction
        self.inputs = []

    def predict(self, inputs):
        self.inputs = inputs
        return [self.prediction]


class PredictionRouteTests(unittest.TestCase):
    def setUp(self):
        self.original_model = predictor_app.model
        self.original_medications = predictor_app.medications_df
        self.original_symptoms = predictor_app.symptoms

        self.stub_model = StubModel('Dengue')
        predictor_app.model = self.stub_model
        predictor_app.medications_df = pd.DataFrame(
            [{'Disease': 'Dengue', 'Suggestion': 'Consult a healthcare professional.'}]
        )
        predictor_app.symptoms = [
            'high_fever',
            'skin_rash',
            'red_spots_over_body',
            'blister',
        ]
        predictor_app.app.config['TESTING'] = True
        self.client = predictor_app.app.test_client()

    def tearDown(self):
        predictor_app.model = self.original_model
        predictor_app.medications_df = self.original_medications
        predictor_app.symptoms = self.original_symptoms
        predictor_app.app.config['TESTING'] = False

    def test_prediction_does_not_require_rash_status(self):
        response = self.client.post(
            '/predict',
            data={'symptoms': ['high_fever']},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json['status'], 'prediction')
        self.assertEqual(self.stub_model.inputs[0].tolist(), [1, 0, 0, 0])

    def test_rejects_unsupported_symptom(self):
        response = self.client.post(
            '/predict',
            data={'symptoms': ['not_a_symptom']},
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn('error', response.json)
        self.assertEqual(self.stub_model.inputs, [])

    def test_requires_at_least_one_symptom(self):
        response = self.client.post(
            '/predict',
            data={},
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn('error', response.json)
        self.assertEqual(self.stub_model.inputs, [])


if __name__ == '__main__':
    unittest.main()
