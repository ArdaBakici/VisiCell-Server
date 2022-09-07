# Gerekli kütüphaneleri koda dahil et
from flask import Flask, redirect, render_template, url_for, request, session
from werkzeug.utils import secure_filename
import os
import base64
import io 
import numpy as np
import cv2
from analyzer import analyze

app = Flask(__name__) # Flask objesi oluştur
app.secret_key = "XaBhgkdj&sJs2s!5df!849" # Sessionları kullanmak için gizli kod oluştur
results_dir = os.path.join(app.root_path, 'results') # Sonuçların kaydedileceği dizin
ALLOWED_EXTENSIONS = {'bmp', 'dib', 'jpg', 'jpeg', 'jpe', 'jp2', 'png', 'pbm', 'pgm', 'ppm', 'sr', 'ras', 'tiff', 'tif'} # İzin verilen dosya formatları
os.makedirs(results_dir, exist_ok=True) # Eğer klasör yoksa oluştur


def get_base64_encoded_image(img_path): # Sunucudaki resim dosyasını oku ve base64 kodlamasını dönüştür
    with open(img_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')

def allowed_file(filename): # Fotoğraf formatı izin verilen dosya formatlarından mı kontrol et
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/<lang>/processing', methods=['GET', 'POST']) # Web sitesinin fotoğraf işleme sayfası
def processing(lang):
    if request.method == 'GET': # Eğer istek türü 'GET' ise ana sayfaya geri döndür
        return redirect(url_for('upload', lang=lang))
    image = request.files.get('image', None)
    if image is None: # Eğer yüklenmiş bir fotoğraf yoksa ana sayfaya geri döndür
        return redirect(url_for('upload', lang=lang))
    elif allowed_file(image.filename): # Eğer fotoğraf varsa ve uzantısı izin verilen uzantılar içindeyse
        session['dilutionFactor'] = request.form.get('dilutionFactor', None) # Seyreltme faktörünü session'a kaydet
        if session['dilutionFactor'] == '': # Seyreltme faktörü boş ise
            session['dilutionFactor'] = -1 # Seyreltme faktörünü -1 olarak ayarla
        else:
            session['dilutionFactor'] = float(session['dilutionFactor']) # Seyreltme faktörü girildiyse float türünden kaydet
            if session['dilutionFactor'] < 0: # Seyreltme faktörü negatif ise
                return redirect(url_for('home')) # Ana sayfaya geri döndür
        in_memory_file = io.BytesIO() # Fotoğrafı bellekte tutmak için bir bytesio nesnesi oluştur
        image.save(in_memory_file) # Fotoğrafı belleğe kaydet
        data = np.fromstring(in_memory_file.getvalue(), dtype=np.uint8) # Bellekteki veriyi oku
        color_image_flag = 1
        img = cv2.imdecode(data, color_image_flag) # Fotoğrafa dönüştür
        #try:
        session['resultpath'], session['livecell_number'], session['deadcell_number'] = analyze(img, results_dir) # Fotoğrafı analiz et ve sonuçları session'a kaydet
        #except Exception as e: # Eğer hata oluşursa ana sayfaya geri döndür
        #    print(e)
        #    return redirect(url_for('upload', lang=lang))
        return redirect(url_for('results', lang=lang)) # Fotoğraf düzgün işlenirse sonuçlar sayfasına yönlendir
    else:
        return redirect(url_for('upload', lang=lang)) # Fotoğraf uzantısı izin verilen uzantılardan farklı ise ana sayfaya geri döndür

@app.route('/api', methods=['POST']) # Mobil uygulamanın fotoğraf işleme sayfası
def api():
    image = request.files.get('photo', None) # Yüklenen fotoğrafı al
    in_memory_file = io.BytesIO() # Fotoğrafı bellekte tutmak için bir bytesio nesnesi oluştur
    image.save(in_memory_file) # Fotoğrafı belleğe kaydet
    data = np.fromstring(in_memory_file.getvalue(), dtype=np.uint8) # Bellekteki veriyi oku
    color_image_flag = 1
    img = cv2.imdecode(data, color_image_flag) # Fotoğrafa dönüştür
    height, width = img.shape[:2] # Fotoğrafın yüksekliğini ve genişliğini al
    res_img, alive_cell, dead_cell = analyze(img) # Fotoğrafı analiz et
    img = cv2.imencode('.png', res_img)[1] # Hücrelerin işaretlenmiş olduğu fotoğrafı gönderilmek üzere PNG formatına dönüştür
    # Göndermek üzere fotoğrafın yüksekliğini, genişliğini, hücre sayısını JSON formatına dönüştür
    ret_data = {"img_width":width, "img_height":height, "alive":alive_cell, "dead":dead_cell, "img":base64.encodebytes(img).decode('utf-8')}
    return ret_data # Sonuçları JSON formatında gönder

@app.route('/<lang>/results') # Sonuçlar sayfası
def results(lang):
    if session['resultpath'] is None: # Eğer önceden işlenmiş bir sonuç yoksa
        return redirect(url_for('upload', lang=lang)) # Ana sayfaya geri döndür
    else: # Eğer önceden işlenmiş bir sonuç varsa
        bytesio = io.BytesIO() # Fotoğrafı bellekte tutmak için bir dosya oluştur
        bytesio = get_base64_encoded_image(session['resultpath']) # İşlenmiş fotoğrafı base64 kodlamasına dönüştür
        os.remove(session['resultpath']) # Sunucudan dosyayı sil
        session['resultpath'] = None # Değeri sıfırlayarak olmayan bir fotoğrafın okunmasını engelle
        # Kullanıcıya html dosyasını gereken değerlerle beraber gönder 
        if lang == 'en':
            return render_template('results_en.html', img= bytesio, alivecell_count= session['livecell_number'], deadcell_count=session['deadcell_number'], dilutionFactor=session['dilutionFactor'])
        elif lang == 'tr':
            return render_template('results_tr.html', img= bytesio, alivecell_count= session['livecell_number'], deadcell_count=session['deadcell_number'], dilutionFactor=session['dilutionFactor'])

@app.route('/') # Ana sayfa
def home():
    return redirect(url_for('upload', lang='tr'))

@app.route('/<lang>/upload')
def upload(lang):
	if lang == 'tr':
		return render_template('index_tr.html')
	elif lang == 'en':
		return render_template('index_en.html')
    
if __name__ == '__main__':
	app.run(debug=True) # Uygulamayı çalıştır