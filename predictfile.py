from keras.models import Sequential, load_model #Kerasで学習したモデルを呼び込む
from flask import Flask, request, redirect, url_for
from werkzeug.utils import secure_filename #ファイルをチェックする関数
from PIL import Image #回転・反転
import os
import keras, sys
import numpy as np
from flask_bootstrap import Bootstrap

classes = ["monkey","boar","crow"]
num_classes = len(classes)
image_size = 50 #縦横50px

#Patterns for Flask - Uploading Files url(http://flask.pocoo.org/docs/1.0/patterns/fileuploads/)
UPLOAD_FOLDER = './uploads'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'gif'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

#ファイルのアップロード可否判定関数
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
#Flask　データを受け取る
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('ファイルがありません')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('ファイルがありません')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

            model = load_model('./animal_cnn_aug.h5')

            image = Image.open(filepath)
            image = image.convert('RGB')
            image = image.resize((image_size, image_size))
            data = np.asarray(image)
            X = []
            X.append(data)
            X = np.array(X)

            #推定結果
            result = model.predict([X])[0]
            predicted = result.argmax()
            percentage = int(result[predicted] * 100)

            return "ケモノ名: " + classes[predicted] + "  確率: " + str(percentage) + " %"

            #return redirect(url_for('uploaded_file',filename=filename))
    return '''
    <!doctype html>
    <html>
    <head>
    <meta charset="utf-8">
    <title>ファイルをアップロードして判定</title>
    <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css" integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" crossorigin="anonymous">
    <style>
    .cover {
      background-image: url();
      background-size: cover;
    }
    </style>
    </head>
    <body>
    <div class="cover text-center py-5">
        <h1 class="display-4 mb-4 text-warning">害獣判定アプリ</h1>
        <form method = post enctype = multipart/form-data>
        <input type=file name=file　class="m-md-3 m-lg-3">
        <p></p>
        <input type=submit value=判定スタート class="m-md-3 m-lg-3">
        <div class="container">
            <div class="row align-items-center">
                <div class="col-lg">
                  <div class="card">
                      <div class="card-header">
                          <h2>害獣サーチ</h2>
                      </div>
                      <div class="card-body">
                          <p class="text-muted m-b-15">このアプリは写真からどのタイプの害獣かを判定して、<code>その対処法を紹介するものです。</code><br>
                          私の祖母は畑で野菜を育てているのですが、猪が出て食い荒らされる事があるのを嘆いていました。<br>
                          今後は害獣の判定精度を高めて、害獣が現れた時のみ撃退音波を発生させる装置に応用しようと思います。</p>
                          <ul class="nav nav-tabs" id="myTab" role="tablist">
                              <li class="nav-item">
                                  <a class="nav-link active" id="home-tab" data-toggle="tab" href="#home" role="tab" aria-controls="home" aria-selected="true">boar</a>
                              </li>
                              <li class="nav-item">
                                  <a class="nav-link" id="profile-tab" data-toggle="tab" href="#profile" role="tab" aria-controls="profile" aria-selected="false">monkey</a>
                              </li>
                              <li class="nav-item">
                                  <a class="nav-link" id="contact-tab" data-toggle="tab" href="#contact" role="tab" aria-controls="contact" aria-selected="false">crow</a>
                              </li>
                          </ul>
                          <div class="tab-content pl-3 p-1" id="myTabContent">
                              <div class="tab-pane fade show active" id="home" role="tabpanel" aria-labelledby="home-tab">
                                  <h3>イノシシ</h3>
                                  <p>祖母を泣かせる大敵です！直接退治は危険なので超音波・電気柵を使いましょう。<br>
                                  ジャンプ力も高く１m程の柵なら飛び越えてしまいます。</p>
                                  <a href="https://www.choujuhigai.com/blog02/archives/631" target="_blank">参考記事</a>
                              </div>
                              <div class="tab-pane fade" id="profile" role="tabpanel" aria-labelledby="profile-tab">
                                  <h3>サル</h3>
                                  <p>有効対処法：電気柵・電気ネット　サルは非常に学習能力が高いので単純な障害物やネットでは防げない。</p>
                                  <a href="https://www.choujuhigai.com/blog02/archives/1161" target="_blank">参考記事</a>
                              </div>
                              <div class="tab-pane fade" id="contact" role="tabpanel" aria-labelledby="contact-tab">
                                  <h3>カラス</h3>
                                  <p>カラスも大変賢いので単調な対策だと破られてしまいます。<br>
                                  「いやがらす」という対策アイテムが設置が容易でお勧めです。</p>
                                  <a href="https://www.choujuhigai.com/blog02/archives/256" target="_blank">参考記事</a>
                              </div>
                          </div>
                      </div>
                   </div>
                </div>
            </div>
        </div>
    </div>
    </form>
    <footer class="text-center text-muted py-5">
      @Copyright animal_ai app
    </footer>
    <!-- jQuery first, then Popper.js, then Bootstrap JS -->
    <script src="https://code.jquery.com/jquery-3.3.1.slim.min.js" integrity="sha384-q8i/X+965DzO0rT7abK41JStQIAqVgRVzpbzo5smXKp4YfRvH+8abtTE1Pi6jizo" crossorigin="anonymous"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/popper.js/1.14.3/umd/popper.min.js" integrity="sha384-ZMP7rVo3mIykV+2+9J3UJ46jBk0WLaUAdn689aCwoqbBJiSnjAK/l8WvCWPIPm49" crossorigin="anonymous"></script>
    <script src="https://stackpath.bootstrapcdn.com/bootstrap/4.1.1/js/bootstrap.min.js" integrity="sha384-smHYKdLADwkXOn1EmN1qk/HfnUcbVRZyYmZ4qpPea6sjB/pTJ0euyQp0Mk8ck+5T" crossorigin="anonymous"></script>
    </body>
    </html>
    '''
#アップロードされたファイルを表示させる
from flask import send_from_directory

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)
