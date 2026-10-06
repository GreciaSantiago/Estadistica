"""
pruebas.py: modulo nuevo para Estadistica (no modifica nada de lo que ya tienes).

Conectalo en app.py con:
    from pruebas import pruebas
    app.register_blueprint(pruebas)

Todas las rutas reciben JSON por POST y regresan JSON con:
    pasos (lista de textos), decision, conclusion, grafica (PNG en base64)
"""
from flask import Blueprint, request, jsonify
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import base64
import io

pruebas = Blueprint('pruebas', __name__)
FONDO = '#F0EBE3'
AZUL = '#2B5C8A'
ROJO = '#C0392B'


# ---------------------------------------------------------------- utilidades
def fig_a_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=120, bbox_inches='tight',
                facecolor=FONDO, edgecolor='none')
    buf.seek(0)
    img = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return img


def lista(texto):
    """'1, 2, 3' -> [1.0, 2.0, 3.0]"""
    return [float(x) for x in str(texto).replace('\n', ',').split(',') if x.strip()]


def detectar_cola(texto):
    """Lee el enunciado y sugiere la cola. Es una ayuda, SIEMPRE confirma leyendo el problema."""
    t = texto.lower()
    izq = ['menor', 'menos de', 'menos que', 'disminu', 'declin', 'reduce', 'reducir',
           'baja', 'inferior', 'decrece', 'tardan menos']
    der = ['mayor', 'más de', 'mas de', 'más que', 'mas que', 'aument', 'excede',
           'supera', 'superior', 'incrementa', 'conviene', 'produce más', 'produce mas']
    bil = ['diferente', 'distint', 'difiere', '≠', 'cambi', 'es igual a', 'afirma que']
    hits = {
        'izquierda': [p for p in izq if p in t],
        'derecha': [p for p in der if p in t],
        'bilateral': [p for p in bil if p in t],
    }
    con_pistas = [k for k, v in hits.items() if v]
    if len(con_pistas) == 1:
        c = con_pistas[0]
        return {'cola': c, 'pistas': hits[c], 'segura': True}
    return {'cola': None, 'pistas': hits, 'segura': False}


def hipotesis(cola, mu0, simbolo='μ'):
    if cola == 'izquierda':
        return f'H0: {simbolo} ≥ {mu0}', f'Ha: {simbolo} < {mu0}'
    if cola == 'derecha':
        return f'H0: {simbolo} ≤ {mu0}', f'Ha: {simbolo} > {mu0}'
    return f'H0: {simbolo} = {mu0}', f'Ha: {simbolo} ≠ {mu0}'


def evaluar(dist, estad, alpha, cola, gl=None):
    """Valor critico, p-value y decision para Z o t."""
    d = stats.norm if dist == 'z' else stats.t(gl)
    if cola == 'izquierda':
        crit = d.ppf(alpha)
        p = d.cdf(estad)
        rechaza = estad < crit
    elif cola == 'derecha':
        crit = d.ppf(1 - alpha)
        p = d.sf(estad)
        rechaza = estad > crit
    else:
        crit = d.ppf(1 - alpha / 2)
        p = 2 * d.sf(abs(estad))
        rechaza = abs(estad) > crit
    return d, float(crit), float(p), bool(rechaza)


def grafica_rechazo(dist, estad, alpha, cola, gl=None, titulo=''):
    d, crit, p, _ = evaluar(dist, estad, alpha, cola, gl)
    lim = max(4.5, abs(estad) + 1)
    x = np.linspace(-lim, lim, 600)
    y = d.pdf(x)
    fig, ax = plt.subplots(figsize=(7, 3.6))
    fig.patch.set_facecolor(FONDO)
    ax.set_facecolor(FONDO)
    ax.plot(x, y, color=AZUL, lw=2)
    if cola == 'izquierda':
        m = x <= crit
        ax.fill_between(x[m], y[m], color=ROJO, alpha=.45, label=f'Rechazo (α={alpha})')
        ax.axvline(crit, color=ROJO, ls='--', lw=1)
    elif cola == 'derecha':
        m = x >= crit
        ax.fill_between(x[m], y[m], color=ROJO, alpha=.45, label=f'Rechazo (α={alpha})')
        ax.axvline(crit, color=ROJO, ls='--', lw=1)
    else:
        m1, m2 = x <= -crit, x >= crit
        ax.fill_between(x[m1], y[m1], color=ROJO, alpha=.45, label=f'Rechazo (α={alpha})')
        ax.fill_between(x[m2], y[m2], color=ROJO, alpha=.45)
        ax.axvline(-crit, color=ROJO, ls='--', lw=1)
        ax.axvline(crit, color=ROJO, ls='--', lw=1)
    ax.axvline(estad, color='black', lw=2, label=f'Estadístico = {estad:.3f}')
    ax.set_title(titulo or f'Región de rechazo ({cola})')
    ax.set_xlabel('z' if dist == 'z' else 't')
    ax.legend(loc='upper right', fontsize=8)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    return fig_a_base64(fig)


def respuesta(pasos, rechaza, concl_rechaza, concl_no, grafica, **extra):
    return jsonify({
        'pasos': pasos,
        'decision': 'Se rechaza H0' if rechaza else 'No se rechaza H0',
        'conclusion': concl_rechaza if rechaza else concl_no,
        'grafica': grafica,
        **extra,
    })


# ------------------------------------------------------------ detector de cola
@pruebas.route('/detectar_cola', methods=['POST'])
def ruta_detectar_cola():
    return jsonify(detectar_cola(request.json['texto']))


# ------------------------------------------------- 2.1 intervalos de confianza
@pruebas.route('/intervalo', methods=['POST'])
def ruta_intervalo():
    j = request.json
    n, xbar, nivel = int(j['n']), float(j['xbar']), float(j['nivel'])  # nivel en %, ej. 95
    alpha = 1 - nivel / 100
    if j.get('sigma') not in (None, ''):
        dist, desv, gl = 'z', float(j['sigma']), None
        crit = stats.norm.ppf(1 - alpha / 2)
        nombre = 'Zα/2'
    else:
        dist, desv, gl = 't', float(j['s']), n - 1
        crit = stats.t.ppf(1 - alpha / 2, gl)
        nombre = f'tα/2 (gl={gl})'
    margen = crit * desv / np.sqrt(n)
    li, ls = xbar - margen, xbar + margen

    d = stats.norm if dist == 'z' else stats.t(gl)
    x = np.linspace(-4.5, 4.5, 600)
    fig, ax = plt.subplots(figsize=(7, 3.4))
    fig.patch.set_facecolor(FONDO)
    ax.set_facecolor(FONDO)
    ax.plot(x, d.pdf(x), color=AZUL, lw=2)
    m = (x >= -crit) & (x <= crit)
    ax.fill_between(x[m], d.pdf(x[m]), color=AZUL, alpha=.3, label=f'{nivel:g}% de confianza')
    ax.axvline(-crit, color=ROJO, ls='--', lw=1)
    ax.axvline(crit, color=ROJO, ls='--', lw=1)
    ax.set_title(f'{nombre} = ±{crit:.3f}')
    ax.legend(fontsize=8)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)

    pasos = [
        f'Datos: n={n}, x̄={xbar}, {"σ" if dist == "z" else "s"}={desv}, NC={nivel:g}%',
        f'α = {alpha:.4f}  →  {nombre} = {crit:.4f}',
        f'Margen de error = {crit:.4f} · {desv}/√{n} = {margen:.4f}',
        f'Intervalo: {li:.4f} < μ < {ls:.4f}',
    ]
    return jsonify({'pasos': pasos, 'li': li, 'ls': ls, 'crit': float(crit),
                    'metodo': 'Z (σ conocida)' if dist == 'z' else 't (σ desconocida)',
                    'grafica': fig_a_base64(fig)})


# ------------------------------------------ 2.2 y 2.3 prueba de una media Z / t
@pruebas.route('/prueba_media', methods=['POST'])
def ruta_prueba_media():
    j = request.json
    n, xbar, mu0 = int(j['n']), float(j['xbar']), float(j['mu0'])
    alpha, cola = float(j['alpha']), j['cola']  # 'izquierda' | 'derecha' | 'bilateral'
    if j.get('sigma') not in (None, ''):
        dist, desv, gl = 'z', float(j['sigma']), None
    else:
        dist, desv, gl = 't', float(j['s']), n - 1
    ee = desv / np.sqrt(n)
    estad = (xbar - mu0) / ee
    d, crit, p, rechaza = evaluar(dist, estad, alpha, cola, gl)
    h0, ha = hipotesis(cola, mu0)
    simb = 'Z' if dist == 'z' else 't'
    region = {'izquierda': f'{simb} < {crit:.4f}', 'derecha': f'{simb} > {crit:.4f}',
              'bilateral': f'{simb} < {-crit:.4f} o {simb} > {crit:.4f}'}[cola]
    pasos = [
        f'1) {h0}   |   {ha}   (cola {cola})',
        f'2) α = {alpha}' + (f',  gl = {gl}' if gl else ''),
        f'3) Error estándar = {desv}/√{n} = {ee:.4f}',
        f'4) {simb} = ({xbar} − {mu0}) / {ee:.4f} = {estad:.4f}',
        f'5) Región de rechazo: {region}',
        f'6) p-value = {p:.4f}',
    ]
    return respuesta(pasos, rechaza,
                     'Hay evidencia suficiente para rechazar H0 (se cumple Ha).',
                     'No hay evidencia suficiente para rechazar H0.',
                     grafica_rechazo(dist, estad, alpha, cola, gl),
                     estadistico=float(estad), critico=crit, p=p, metodo=simb)


# -------------------------------------------------- 2.4 diferencia de dos medias
@pruebas.route('/dos_medias', methods=['POST'])
def ruta_dos_medias():
    j = request.json
    x1, x2 = float(j['xbar1']), float(j['xbar2'])
    s1, s2 = float(j['s1']), float(j['s2'])
    n1, n2 = int(j['n1']), int(j['n2'])
    D0, alpha, cola = float(j.get('D0', 0)), float(j['alpha']), j['cola']
    ee = np.sqrt(s1 ** 2 / n1 + s2 ** 2 / n2)
    estad = ((x1 - x2) - D0) / ee
    d, crit, p, rechaza = evaluar('z', estad, alpha, cola)
    h0, ha = hipotesis(cola, D0, 'μ1 − μ2')
    pasos = [
        f'1) {h0}   |   {ha}   (cola {cola})',
        f'2) α = {alpha}',
        f'3) Error estándar = √({s1}²/{n1} + {s2}²/{n2}) = {ee:.4f}',
        f'4) Z = (({x1} − {x2}) − {D0}) / {ee:.4f} = {estad:.4f}',
        f'5) Z crítico = {"±" if cola == "bilateral" else ""}{crit:.4f}',
        f'6) p-value = {p:.4f}',
    ]
    return respuesta(pasos, rechaza,
                     'Hay evidencia suficiente para rechazar H0.',
                     'No hay evidencia suficiente para rechazar H0.',
                     grafica_rechazo('z', estad, alpha, cola),
                     estadistico=float(estad), critico=crit, p=p)


# ------------------------------------------------------------ 2.5 Wilcoxon
@pruebas.route('/wilcoxon', methods=['POST'])
def ruta_wilcoxon():
    """
    d = datos1 − datos2.
    cola izquierda  -> Ha: datos1 < datos2  (T+ chica)
    cola derecha    -> Ha: datos1 > datos2  (T− chica)
    bilateral       -> T = min(T+, T−)
    divisor: 24 (formula estandar) o 6 (la de tus apuntes)
    """
    j = request.json
    a, b = np.array(lista(j['datos1'])), np.array(lista(j['datos2']))
    alpha, cola = float(j['alpha']), j['cola']
    divisor = int(j.get('divisor', 24))
    d = a - b
    d = d[d != 0]
    n = len(d)
    rangos = stats.rankdata(np.abs(d))  # empates promediados
    t_pos = float(rangos[d > 0].sum())
    t_neg = float(rangos[d < 0].sum())
    T = {'izquierda': t_pos, 'derecha': t_neg, 'bilateral': min(t_pos, t_neg)}[cola]
    mu_t = n * (n + 1) / 4
    sig_t = np.sqrt(n * (n + 1) * (2 * n + 1) / divisor)
    z = (T - mu_t) / sig_t
    cola_z = 'bilateral' if cola == 'bilateral' else 'izquierda'
    _, crit, p, rechaza = evaluar('z', z, alpha, cola_z)
    tabla = [{'d': float(di), 'rango': float(r)} for di, r in
             sorted(zip(d, rangos), key=lambda t: abs(t[0]))]
    pasos = [
        f'1) H0: poblaciones idénticas  |  Ha según cola {cola}',
        f'2) α = {alpha},  n = {n} (sin diferencias en 0)',
        f'3) T+ = {t_pos:g},  T− = {t_neg:g}  →  T = {T:g}',
        f'   μT = n(n+1)/4 = {mu_t:g}',
        f'   σT = √(n(n+1)(2n+1)/{divisor}) = {sig_t:.4f}',
        f'4) Z = ({T:g} − {mu_t:g}) / {sig_t:.4f} = {z:.4f}',
        f'5) Z crítico = {"±" if cola == "bilateral" else "−"}{abs(crit):.4f}',
        f'6) p-value aprox. = {p:.4f}',
    ]
    return respuesta(pasos, rechaza,
                     'Hay evidencia suficiente para decir que las poblaciones difieren.',
                     'No hay evidencia suficiente para decir que las poblaciones difieren.',
                     grafica_rechazo('z', z, alpha, cola_z),
                     tabla=tabla, T_pos=t_pos, T_neg=t_neg, z=float(z), divisor=divisor)


# ------------------------------------------------------ 2.6 Mann-Whitney-Wilcoxon
@pruebas.route('/mann_whitney', methods=['POST'])
def ruta_mann_whitney():
    j = request.json
    g1, g2 = lista(j['datos1']), lista(j['datos2'])
    alpha, cola = float(j['alpha']), j.get('cola', 'bilateral')
    n1, n2 = len(g1), len(g2)
    todos = np.array(g1 + g2)
    rangos = stats.rankdata(todos)
    T1 = float(rangos[:n1].sum())
    U1 = T1 - n1 * (n1 + 1) / 2
    U2 = n1 * n2 - U1
    Umin = min(U1, U2)
    alt = {'izquierda': 'less', 'derecha': 'greater', 'bilateral': 'two-sided'}[cola]
    res = stats.mannwhitneyu(g1, g2, alternative=alt)
    p = float(res.pvalue)
    rechaza = p < alpha

    fig, ax = plt.subplots(figsize=(7, 3.4))
    fig.patch.set_facecolor(FONDO)
    ax.set_facecolor(FONDO)
    ax.scatter(g1, np.full(n1, 1), s=70, color=AZUL, label='Grupo 1', zorder=3)
    ax.scatter(g2, np.full(n2, 2), s=70, color=ROJO, label='Grupo 2', zorder=3)
    ax.hlines([1, 2], todos.min(), todos.max(), color='gray', lw=.5)
    ax.set_yticks([1, 2])
    ax.set_yticklabels(['Grupo 1', 'Grupo 2'])
    ax.set_ylim(.4, 2.6)
    ax.set_title(f'U mín = {Umin:g}   (p = {p:.4f})')
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)

    orden = np.argsort(todos, kind='stable')
    tabla = [{'valor': float(todos[i]), 'grupo': 1 if i < n1 else 2, 'rango': float(rangos[i])}
             for i in orden]
    pasos = [
        '1) H0: poblaciones idénticas  |  Ha: no idénticas (o según cola)',
        f'2) α = {alpha},  n1 = {n1},  n2 = {n2}',
        f'3) T1 = suma de rangos del grupo 1 = {T1:g}',
        f'4) U1 = T1 − n1(n1+1)/2 = {U1:g}',
        f'   U2 = n1·n2 − U1 = {U2:g}',
        f'   U mín = {Umin:g}  (compárala con tu tabla de valores críticos)',
        f'5) p-value (scipy) = {p:.4f}',
    ]
    return respuesta(pasos, rechaza,
                     'Hay evidencia suficiente para decir que las poblaciones difieren.',
                     'No hay evidencia suficiente para decir que las poblaciones difieren.',
                     fig_a_base64(fig), tabla=tabla, T1=T1, U1=U1, U2=U2, Umin=Umin, p=p)


# ------------------------------------------------ buscador de valores criticos
@pruebas.route('/critico', methods=['POST'])
def ruta_critico():
    j = request.json
    dist, alpha, cola = j['dist'], float(j['alpha']), j['cola']
    gl = int(j['gl']) if j.get('gl') not in (None, '') else None
    d = stats.norm if dist == 'z' else stats.t(gl)
    if cola == 'izquierda':
        v = [d.ppf(alpha)]
    elif cola == 'derecha':
        v = [d.ppf(1 - alpha)]
    else:
        v = [-d.ppf(1 - alpha / 2), d.ppf(1 - alpha / 2)]
    return jsonify({'critico': [round(float(x), 4) for x in v]})