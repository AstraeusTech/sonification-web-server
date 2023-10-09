import io
from sonification_scale import sonificationConfig, runSonification
from point_cloud import create_point_cloud
import os
import boto3
import dotenv
from flask import Flask, jsonify

dotenv.load_dotenv()

config = sonificationConfig(
    amplitude=4096 * 8,
    guassian_sigma=8,
    sonification_duration=30,
    output_path="soundfiles/combined.wav",
)

def handler(event, context):
    s3_bucket = os.getenv("S3_BUCKET")

    file_link = event['id']

    print("file_link", file_link)
    print("s3_bucket", s3_bucket)

    s3_client = boto3.client('s3')

    image = s3_client.get_object(Bucket=s3_bucket, Key=file_link)
    image = image['Body'].read()

    sonification = runSonification(image, config)
    file_link_without_extension = file_link.split(".")[0]
    s3_client.put_object(Body=sonification, Bucket=s3_bucket, Key=file_link_without_extension + ".wav")

    create_point_cloud(image, "test.pcd")
    s3_client.upload_file("test.pcd", s3_bucket, file_link_without_extension + ".pcd")

    os.remove("test.pcd")

    return {
      'statusCode': 200,
    }

app = Flask(__name__)

@app.route('/<id>')
def get_files(id):
    res = handler({
        'id': id
    }, None)

    return jsonify(res['body'])

if __name__ == "__main__":
    port = os.getenv("PORT", 5000)
    app.run(debug=True, host='0.0.0.0', port=port)
    # res = handler({
    #     'id': '02-medium-cut-with-wavy-hair.jpg'
    # }, None)

    # print(res)