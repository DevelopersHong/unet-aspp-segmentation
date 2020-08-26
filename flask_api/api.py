from flask import Flask,request,Response, jsonify
import seg_color
from PIL import Image

app = Flask(__name__)



@app.route('/',methods = ['POST'])
def seg():
    image = request.files['image']
    filename = image.filename
    img = Image.open(image)
    
    seg_dir = seg_color.main_seg(img,filename)

    resp = Response(seg_dir+'\output_seg.png',mimetype="image/png")
    return jsonify({"res": seg_dir[10:]})

    # return send_file(
    #     io.BytesIO(output_img),
    #     mimetype='image/png',
    #     as_attachment=True,
    #     attachment_filename='output_seg.png'
    # )

if __name__=='__main__':
    app.run(port=7017,debug=True,host='0.0.0.0')
