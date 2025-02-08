from flask import Flask, request
import controller
from time import sleep

app = Flask(__name__)

@app.route('/', methods=['POST'])
def handle_data():
    if request.is_json:
        data = request.get_json()
    else:
        data = request.form
    
    print(data['text'].lower())

    for letter in data['text'].lower():
      controller.show_letter(letter)
      sleep(1)
  

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=3000, debug=False)