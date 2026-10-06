"""Corre UNA vez desde la carpeta del proyecto:  python parche.py
Hace copia .bak de cada archivo antes de tocarlo."""
import re, shutil, os

def leer(p):
    with open(p, encoding='utf-8') as f:
        return f.read()

def guardar(p, t):
    shutil.copy(p, p + '.bak')
    with open(p, 'w', encoding='utf-8') as f:
        f.write(t)

# ---------- app.py
p = 'app.py'
t = leer(p)
if 'register_blueprint(pruebas)' not in t:
    t = t.replace("app = Flask(__name__)",
                  "app = Flask(__name__)\nfrom pruebas import pruebas\napp.register_blueprint(pruebas)", 1)
if "'/inferencia'" not in t:
    t = t.replace("if __name__ == '__main__':",
                  "@app.route('/inferencia')\ndef inferencia():\n    return render_template('inferencia.html')\n\nif __name__ == '__main__':", 1)
guardar(p, t)
print('app.py OK')

# ---------- index.html
p = 'templates/index.html'
t = leer(p)
if '/inferencia' not in t:
    t = re.sub(r"(<button class=\"tab\" onclick=\"window\.location='/problemas'\">.*?</button>)",
               r'\1\n  <button class="tab" onclick="window.location='"'"'/inferencia'"'"'">05 &middot; Inferencia</button>',
               t, count=1)
if 'const num =' not in t:
    t = re.sub(r"const a2 = parseFloat\(document\.getElementById\('n-a2'\)\.value\) \|\| -1;",
               "const num = (id, def) => { const v = parseFloat(document.getElementById(id).value); return isNaN(v) ? def : v; };\n    const a2 = num('n-a2', -1);", t)
    t = re.sub(r"const b2 = parseFloat\(document\.getElementById\('n-b2'\)\.value\) \|\| 1;",
               "const b2 = num('n-b2', 1);", t)
guardar(p, t)
print('index.html OK')

# ---------- problemas.html
p = 'templates/problemas.html'
t = leer(p)
nuevo = """// highlight en tabla
  document.querySelectorAll('#ztable td.highlight').forEach(el=>el.classList.remove('highlight'));
  const c=Math.round(Math.abs(z)*100),lab=(z<0?'-':'')+(Math.floor(c/10)/10).toFixed(1);
  const tr=document.querySelector('#ztable tr[data-l="'+lab+'"]');
  if(tr&&tr.cells[c%10+1])tr.cells[c%10+1].classList.add('highlight');
"""
if 'data-l=' in t and 'const c=Math.round(Math.abs(z)*100)' not in t:
    t = re.sub(r"// highlight en tabla.*?\}\);\n", lambda m: nuevo, t, count=1, flags=re.S)
t = t.replace('valores de 0.0 a 3.4', 'valores de -3.4 a 3.4')
guardar(p, t)
print('problemas.html OK')

for f in ('pruebas.py', 'templates/inferencia.html'):
    if not os.path.exists(f):
        print('FALTA copiar:', f)
        