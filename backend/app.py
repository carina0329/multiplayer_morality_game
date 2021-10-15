from flask import Flask, render_template, request, jsonify, make_response, json
from pusher import pusher
from flask_cors import CORS

app = Flask(__name__)

# enable CORS
CORS(app, resources={r'/*': {'origins': '*'}})

pusher = pusher_client = pusher.Pusher(
    app_id="1260676",
    key="531311b5216a9187dc03",
    secret="86c3ca6dc3875b0ba3dd",
    cluster="us2",
    ssl=True
)

name = ''

@app.route('/')
def index():
    return "haha:s"
    
@app.route('/play')
def play():
    global name
    name = request.args.get('username')
    return "gfajhsjah"
    
@app.route("/pusher/auth", methods=['POST'])
def pusher_authentication():
    auth = pusher.authenticate(
    channel=request.form['channel_name'],
    socket_id=request.form['socket_id'],
    custom_data={
        u'user_id': name,
        u'user_info': {
        u'role': u'player'
        }
    }
    )
    return json.dumps(auth)
    
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

name = ''