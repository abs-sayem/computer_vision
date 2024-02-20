# import base64
# import requests
# import json

# def encode_image(image_path):
#     with open(image_path, "rb") as image_file:
#         return base64.b64encode(image_file.read()).decode('utf-8')

# def detect_emotion(api_key, image_path, model="gpt-4-vision-preview"):
#     base64_image = encode_image(image_path)

#     headers = {
#         "Content-Type": "application/json",
#         "Authorization": f"Bearer {api_key}"
#     }

#     payload = {
#         "model": model,
#         "messages": [
#             {
#                 "role": "user",
#                 "content": [
#                     {
#                         "type": "text",
#                         "text": "Please estimate the emotion values from this picture."
#                     },
#                     {
#                         "type": "image_url",
#                         "image_url": {
#                             "url": f"data:image/jpeg;base64,{base64_image}"
#                         }
#                     }
#                 ]
#             }
#         ],
#         "max_tokens": 300
#     }

#     response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

#     emotions_str = response.json()["choices"][0]["message"]["content"]
#     emotions_json = json.loads(emotions_str)

#     return emotions_json

import base64
import requests
import json
import openai

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def detect_emotion(api_key, image_path, model="gpt-4-vision-preview"):
    base64_image = encode_image(image_path)

    query = '''# Instruction
    Please estimate the emotion values from this picture in the following JSON format:
    {"fear":"0.x",
    "surprise":"0.x",
    "sadness":"0.x",
    "disgust":"0.x",
    "anger":"0.x",
    "anticipation":"0.x",
    "joy":"0.x",
    "trust":"0.x"}
    # Note
    - The values shuold be from 0.0 to 1.0
    - Output should be only the JSON data'''
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": query
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
        "max_tokens": 300
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    emotions_str = response.json()["choices"][0]["message"]["content"]
    emotions_json = json.loads(emotions_str)

    #print(response.json())
    #print(f'Output JSON {response.json()["choices"][0]["message"]["content"]}')
    #print()
    return emotions_json

# call the detection function
import cv2

api_key = "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"
img_path = 'afraid0.jpeg'
emotion = detect_emotion(api_key=api_key, image_path=img_path)
print(emotion)