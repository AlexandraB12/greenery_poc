import requests

url = "http://127.0.0.1:8000/score"
path = r"C:\Users\alexa\greenery_poc\streetview_images\001.jpg"

with open(path, "rb") as f:
    files = {"file": ("001.jpg", f, "image/jpeg")}
    r = requests.post(url, files=files)

print(r.status_code)
print(r.text)
