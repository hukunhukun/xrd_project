from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    user_input = request.form['user_input']
    # 在这里可以将用户输入传递给您的Python程序进行处理
    # 这里简单地将用户输入原样返回
    response = f"您输入了：{user_input}"
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True)
