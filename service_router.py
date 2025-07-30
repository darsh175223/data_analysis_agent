from flask import Flask, jsonify

app = Flask(__name__)

@app.route('/')
def home():
    return jsonify({
        "message": "Welcome to the Flask Server",
        "status": "running"
    })

@app.route('/tool', methods=['POST'])
def tool_call(tool: str):
    






    return jsonify({
        "status": "executed"
    })


@app.route('/health')
def health_check():
    return jsonify({
        "status": "healthy"
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
