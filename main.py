from flask import Flask
import recognizer


app = Flask(__name__)



# geting and sending response to dialogflow
@app.route('/', methods=['POST','GET'])
def webhook():

    name = recognizer.get_name()

    value = 'Hello'+ ' ' + name
    
    return value
    


    







if __name__ == '__main__':
    app.run(debug=True)