from __future__ import print_function

import base64
import boto3
import json
import re

print('Loading function')

client = boto3.client('comprehend')

# Replace embedded new lines, tabs and carriage return
pattern = r'[\n\t\r]+'

def lambda_handler(event, context):
    output = []

    for record in event['records']:
        print(record['recordId'])
        payload = base64.b64decode(record['data'])

        # Do custom processing on the payload here
        payload = json.loads(payload)

        #print (payload)
        
        review = payload['review_headline'] + ' - ' + payload['review_body']
        #print(review)
        
        sentiment = client.detect_sentiment(Text=review[:4500], LanguageCode='en')
        
        print(sentiment['Sentiment'])
        
        payload['sentiment'] = sentiment['Sentiment']
        
        payload = json.dumps(payload,separators=(',', ':'))
        
        payload = re.sub(pattern,' ', payload) + "\n"
        
        output_record = {
            'recordId': record['recordId'],
            'result': 'Ok',
            'data': base64.b64encode(payload)
        }
        output.append(output_record)

    print('Successfully processed {} records.'.format(len(event['records'])))

    return {'records': output}
