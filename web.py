from flask import Flask, render_template, request, flash, send_from_directory, url_for, redirect
from flask_uploads import UploadSet, configure_uploads, IMAGES
import hotdog

app = Flask(__name__)

photos = UploadSet('photos', IMAGES)
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])
app.config['UPLOADED_PHOTOS_DEST'] = 'img/'
configure_uploads(app, photos)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST' and 'photo' in request.files:
        filename = photos.save(request.files['photo'])
        hot = hotdog.read_hotdog('img/' + filename)
        return render_template('upload.html', img=filename, res=hot)
    else:
        flash("Invalid upload")
        return redirect(url_for('/'))

@app.route('/img/<filename>')
def send_file(filename):
    return send_from_directory(app.config['UPLOADED_PHOTOS_DEST'], filename)

if __name__ == '__main__':
    app.run(debug=True)
