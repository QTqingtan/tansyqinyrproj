
import base64
from datetime import timedelta
from flask import *
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import logging as rel_log
import cv2
import shutil
import numpy as np
import process as p
import core.main

app = Flask(__name__,template_folder='./firstend/dist',static_folder='./firstend/dist',static_url_path="")
cors = CORS(app,supports_credentials=True)
app.config['MAX_CONTENT_PATH'] = 1024*1024

werkzeug_logger = rel_log.getLogger('werkzeug')
werkzeug_logger.setLevel(rel_log.ERROR)


#跨域
@app.after_request
def after_request(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Credentials'] = 'true'
    response.headers['Access-Control-Allow-Methods'] = 'POST'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type, X-Requested-With'
    return response


ALLOWED_EXTENSIONS = set(['png', 'jpg'])   #支持两种图片
UPLOAD_FOLDER = r'./uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# 解决缓存刷新问题
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = timedelta(seconds=1)

"""处理调用函数"""
def process_func(img_name,img_path,num):
    if num==1:
        print('ok')

"""判断上传有效"""
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

@app.route('/')
def hello_world():
    return redirect(url_for('static', filename='./index.html'))

@app.route('/upload', methods=['POST','GET'])
def upload_image():
    image=request.files['file']
    if image :
        img_path=os.path.join(app.config['UPLOAD_FOLDER'],image.filename)
        image.save(img_path)
        shutil.copy(img_path,'./tmp/ct')
        pid=os.getpid()
        image_path=os.path.join('./tmp/ct',image.filename)
        pid= core.main.c_main(image_path,0,image.filename.rsplit('.', 1)[1])
        return jsonify({'status':1,
                        'image_url':'http://127.0.0.1:5000/tmp/ct/'+pid,
                         'draw_url':'http://127.0.0.1:5000/tmp/draw/'+pid #应该是/tmp/draw/,是处理后的结果
                        })
    return jsonify({'status':0})


@app.route("/download",methods=['GET'])
def download_file():
    return send_from_directory('data')

@app.route('/tmp/<path:file>',methods=['GET'])
def show_photo(file):
    if request.method == 'GET':
        if not file is None:
            image_data=open(f'tmp/{file}','rb').read()
            response=make_response(image_data)
            response.headers['Content-Type']='image/png'
            return response

if __name__ == '__main__':
    files = [
        'uploads', 'tmp/ct', 'tmp/draw',
        'tmp/image', 'tmp/mask', 'tmp/uploads'
    ]
    for ff in files:
        if not os.path.exists(ff):
            os.makedirs(ff)

    app.run(debug=True)