# Reference:
# https://aws.amazon.com/blogs/machine-learning/call-an-amazon-sagemaker-model-endpoint-using-amazon-api-gateway-and-aws-lambda/
# https://docs.aws.amazon.com/apigateway/latest/developerguide/set-up-lambda-proxy-integrations.html#api-gateway-simple-proxy-for-lambda-output-format

import boto3
import math
import dateutil
import json
import os

# grab environment variables
ENDPOINT_NAME = os.environ['ENDPOINT_NAME']
client = boto3.client(service_name='sagemaker-runtime')

# Raw Data Structure: 
# datetime,season,holiday,workingday,weather,temp,atemp,humidity,windspeed,casual,registered,count
# Model expects data in this format (it was trained with these features):
# season,holiday,workingday,weather,temp,atemp,humidity,windspeed,year,month,day,dayofweek,hour
def transform_data(data):
    try:
        features = data.copy()
        # Extract year, month, day, dayofweek, hour
        dt = dateutil.parser.parse(features[0])
    
        features.append(dt.year)
        features.append(dt.month)
        features.append(dt.day)
        features.append(dt.weekday())
        features.append(dt.hour)
        
        # Return the transformed data. skip datetime field
        return ','.join([str(feature) for feature in features[1:]])
        
    except Exception as err:
        print('Error when transforming: {0},{1}'.format(data,err))
        raise Exception('Error when transforming: {0},{1}'.format(data,err))
        
    
def lambda_handler(event, context):
    try:    
        print("Received event: " + json.dumps(event, indent=2))
        
        request = json.loads(json.dumps(event))
        
    
        transformed_data = [transform_data(instance['features']) for instance in request["instances"]]
        
    
        # XGBoost accepts data in CSV. It does not support JSON.
        # So, we need to submit the request in CSV format
        # Prediction for multiple observations in the same call
        result = client.invoke_endpoint(EndpointName=ENDPOINT_NAME, 
                               Body=('\n'.join(transformed_data).encode('utf-8')),
                               ContentType='text/csv')
                               
    
        result = result['Body'].read().decode('utf-8')
        
        # Apply inverse transformation to get the rental count
        print(result)
        result = result.split(',')
        predictions = [math.expm1(float(r)) for r in result]
        
        return {
            'statusCode': 200,
            'isBase64Encoded':False,
            'body': json.dumps(predictions)
        }

    except Exception as err:
        return {
            'statusCode': 400,
            'isBase64Encoded':False,
            'body': 'Call Failed {0}'.format(err)
        }
