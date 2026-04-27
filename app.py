from flask import Flask, render_template, request, jsonify
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import base64
import io

app = Flask(__name__)

def figura_a_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=120, bbox_inches='tight',
                facecolor='#F0EBE3', edgecolor='none')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return img_base64

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/varianza', methods=['POST'])
def calcular_varianza():
    datos = request.json['datos']
    numeros = list(map(float, datos.split(',')))
    varianza = np.var(numeros, ddof=1)
    media = np.mean(numeros)
    desviacion = np.std(numeros, ddof=1)
    return jsonify({
        'varianza': round(float(varianza), 4),
        'media': round(float(media), 4),
        'desviacion': round(float(desviacion), 4)
    })

@app.route('/esperado', methods=['POST'])
def calcular_esperado():
    datos = request.json
    valores = list(map(float, datos['valores'].split(',')))
    probabilidades = list(map(float, datos['probabilidades'].split(',')))
    esperado = float(np.dot(valores, probabilidades))
    varianza = float(sum(p * (x - esperado)**2 for x, p in zip(valores, probabilidades)))
    desviacion = float(np.sqrt(varianza))
    return jsonify({
        'esperado': round(esperado, 4),
        'varianza': round(varianza, 4),
        'desviacion': round(desviacion, 4)
    })

@app.route('/normal', methods=['POST'])
def grafica_normal():
    media = float(request.json.get('media', 0))
    desviacion = float(request.json.get('desviacion', 1))
    x = np.linspace(media - 4*desviacion, media + 4*desviacion, 300)
    y = stats.norm.pdf(x, media, desviacion)

    fig, ax = plt.subplots(figsize=(9, 4))
    fig.patch.set_facecolor('#F0EBE3')
    ax.set_facecolor('#F0EBE3')
    ax.plot(x, y, color='#A0692A', linewidth=2.5)
    ax.fill_between(x, y, alpha=0.2, color='#A0692A')

    x_sigma = np.linspace(media - desviacion, media + desviacion, 100)
    y_sigma = stats.norm.pdf(x_sigma, media, desviacion)
    ax.fill_between(x_sigma, y_sigma, alpha=0.35, color='#A0692A', label='68% (±1σ)')
    ax.axvline(media, color='#7a4f1e', linestyle='--', linewidth=1.5, alpha=0.7, label=f'μ = {media}')

    ax.set_title(f'Distribución Normal  μ={media}  σ={desviacion}', color='#1a1a1a', fontsize=13, pad=12)
    ax.tick_params(colors='#555')
    ax.spines['bottom'].set_color('#D4B896')
    ax.spines['left'].set_color('#D4B896')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(facecolor='#fff', edgecolor='#D4B896', labelcolor='#333', fontsize=10)
    ax.grid(True, alpha=0.15, color='#A0692A')
    plt.tight_layout()
    return jsonify({'imagen': figura_a_base64(fig)})

@app.route('/problemas')
def problemas():
    return render_template('problemas.html')

if __name__ == '__main__':
    app.run(debug=True)