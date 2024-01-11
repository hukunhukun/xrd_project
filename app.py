from flask import Flask, render_template, request, jsonify
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from PIL import Image
from utils import robot_out,visual_loss

app = Flask(__name__)

def process(input_:str):
    output = []
    if input_ == "I have a cif file whose path is caoqian/cif/NaCl.cif, Please plot the XRD pattern of it.":
        lines = robot_out('./static/log/PlotXRD.txt')
        for line in lines:
            output.append({'type': 'text', 'content': line})
        output.pop()
        image_path = './static/images/NaCl_plot.png'
        with open(image_path, "rb") as image_file:
            base64_encoded = base64.b64encode(image_file.read()).decode('utf-8')
        output.append({'type': 'image', 'content': base64_encoded})
        output.append({'type': 'text', 'content': lines[-1]})

    elif input_ == "I need to execute the first stage of pretraining using MXRDM on the file caoqian/data/GeneratedData.csv. Please save the pretrained model at caoqian/model/classi_mxrdm.pt. The specified parameters are: CUDA, layers, heads, epoch, seed, hidden units, vocabulary size, and batch size, with values 2, 2, 8, 2, 1200, 512, 74, 32 respectively.":
        lines = robot_out('./static/log/PretrainMXRDM.txt')
        loss_fig = visual_loss('./static/log/PretrainMXRDM.txt')
        for line in lines:
            output.append({'type': 'text', 'content': line})
            if line == "Training Start":
                figfile = BytesIO()
                loss_fig.savefig(figfile, format='png')
                figfile.seek(0)
                base64_encoded = base64.b64encode(figfile.getvalue()).decode('utf-8')
                output.append({'type': 'image', 'content': base64_encoded})
    return output


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    user_input = request.form['user_input']

    output = process(user_input)

    response_data = {'output': output}

    return jsonify(response_data)

if __name__ == '__main__':
    app.run(debug=True)
