import glob
import boto3
import json

def detect_labels(photo, client):
    with open(photo, 'rb') as fd:
        response = client.detect_labels(Image={'Bytes': fd.read()})
        return response["Labels"]

client = boto3.client('rekognition')
combined = []

for filename in glob.glob('public/photos/*.jpeg'):
    labels = detect_labels(filename, client)
    entry = {
        "Filename": filename.replace("public/", ""),
        "Labels": labels
    }
    combined.append(entry)

with open('data.json', 'w') as outfile:
    json.dump(combined, outfile, indent=2)

print('JSON data has been written to data.json')