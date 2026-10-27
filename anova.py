"""ANOVA de un factor (diseño completamente aleatorizado) + prueba LSD de Fisher.
Blueprint para Flask. Rutas:  GET /anova   y   POST /anova_calcular
"""
import io
import re
import json
import base64
from itertools import combinations

import numpy as np
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from flask import Blueprint, render_template, request, jsonify, Response

anova = Blueprint('anova', __name__)

CAFE, CAFE_OSC, FONDO = '#A0692A', '#7a4f1e', '#F0EBE3'


class DatosInvalidos(Exception):
    pass


def _fig64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=120, bbox_inches='tight',
                facecolor=FONDO, edgecolor='none')
    buf.seek(0)
    img = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return img


def _n(x, d=4):
    """Número con d decimales (sin ceros inútiles al final)."""
    s = f'{x:.{d}f}'
    return s.rstrip('0').rstrip('.') if '.' in s else s


def _p(p):
    return '< 0.0001' if p < 0.0001 else f'{p:.4f}'


def _parsear(texto):
    partes = [p for p in re.split(r'[,;\s]+', str(texto).strip()) if p]
    try:
        return [float(p) for p in partes]
    except ValueError:
        raise DatosInvalidos('Hay un dato que no es un número. Usa punto decimal (5.2) '
                             'y separa los datos con comas, espacios o saltos de línea.')


def _grafica(grupos, nombres, medias, F, fcrit, gl1, gl2, alpha):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4),
                                   gridspec_kw={'width_ratios': [1.3, 1]})
    fig.patch.set_facecolor(FONDO)
    for ax in (ax1, ax2):
        ax.set_facecolor(FONDO)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_color('#D4B896')
        ax.spines['left'].set_color('#D4B896')
        ax.tick_params(colors='#555')
        ax.grid(True, alpha=.15, color=CAFE)

    # --- distribución F con región de rechazo
    xmax = max(fcrit * 2.5, F * 1.15, 6)
    x = np.linspace(0.02, xmax, 600)
    y = stats.f.pdf(x, gl1, gl2)
    ytop = np.nanmax(y[x >= 0.15]) * 1.25
    ax1.plot(x, y, color=CAFE, linewidth=2.5)
    ax1.fill_between(x, y, where=x >= fcrit, color=CAFE, alpha=.45,
                     label=f'Región de rechazo (α = {_n(alpha)})')
    ax1.axvline(fcrit, color=CAFE_OSC, linestyle='--', linewidth=1.5,
                label=f'F crítico = {_n(fcrit, 3)}')
    ax1.axvline(F, color='#b03a2e', linewidth=2, label=f'F calculado = {_n(F, 3)}')
    ax1.set_ylim(0, ytop)
    ax1.set_title(f'Distribución F  (gl = {gl1}, {gl2})', color='#1a1a1a', fontsize=12)
    ax1.legend(facecolor='#fff', edgecolor='#D4B896', fontsize=8)

    # --- datos por grupo
    rng = np.random.default_rng(1)
    for i, (g, m) in enumerate(zip(grupos, medias), start=1):
        ax2.scatter(i + rng.uniform(-.08, .08, len(g)), g, color=CAFE, alpha=.7, s=28, zorder=3)
        ax2.hlines(m, i - .25, i + .25, color='#b03a2e', linewidth=2.5, zorder=4)
    ax2.set_xticks(range(1, len(grupos) + 1))
    ax2.set_xticklabels(nombres, fontsize=8)
    ax2.set_title('Datos y media de cada grupo', color='#1a1a1a', fontsize=12)
    plt.tight_layout()
    return _fig64(fig)


def resolver(nombres, grupos, alpha):
    k = len(grupos)
    if k < 2:
        raise DatosInvalidos('Necesitas al menos 2 grupos.')
    if not (0 < alpha < 1):
        raise DatosInvalidos('α debe estar entre 0 y 1 (por ejemplo 0.05).')
    ns = [len(g) for g in grupos]
    if min(ns) < 1:
        raise DatosInvalidos('Todos los grupos deben tener al menos un dato.')
    nT = sum(ns)
    if nT <= k:
        raise DatosInvalidos('Faltan datos: se necesita n_T > k para estimar el error.')

    todos = np.concatenate([np.array(g) for g in grupos])
    medias = [float(np.mean(g)) for g in grupos]
    gran = float(np.mean(todos))

    sctr = float(sum(n * (m - gran) ** 2 for n, m in zip(ns, medias)))
    sce = float(sum(np.sum((np.array(g) - m) ** 2) for g, m in zip(grupos, medias)))
    stc = float(np.sum((todos - gran) ** 2))
    gl_tr, gl_e, gl_t = k - 1, nT - k, nT - 1
    if sce == 0:
        raise DatosInvalidos('Dentro de cada grupo todos los datos son iguales (SCE = 0): '
                             'F no se puede calcular.')
    cmtr, cme = sctr / gl_tr, sce / gl_e
    F = cmtr / cme
    p = float(stats.f.sf(F, gl_tr, gl_e))
    fcrit = float(stats.f.ppf(1 - alpha, gl_tr, gl_e))
    rechaza = p <= alpha

    # ---------------- pasos ANOVA
    pasos = [
        'HIPÓTESIS\nH₀: ' + ' = '.join(f'μ({n})' for n in nombres) +
        '\nHₐ: No todas las medias poblacionales son iguales',
        f'k = {k} grupos,  n_T = {nT},  α = {_n(alpha)}',
        'MEDIAS\n' + '\n'.join(f'{n}:  n = {ni},  Σ = {_n(sum(g))},  x̄ = {_n(m)}'
                               for n, g, ni, m in zip(nombres, grupos, ns, medias)) +
        f'\nMedia general ȳ = {_n(gran)}',
        'SUMAS DE CUADRADOS\n'
        'SCTR = Σ nᵢ(ȳᵢ − ȳ)² = ' + _n(sctr) + '\n'
        'SCE = ΣΣ(yᵢⱼ − ȳᵢ)² = ' + _n(sce) + '\n'
        'STC = ΣΣ(yᵢⱼ − ȳ)² = ' + _n(stc) + '   (SCTR + SCE = ' + _n(sctr + sce) + ')',
        'GRADOS DE LIBERTAD\n'
        f'Tratamientos: k − 1 = {gl_tr}\nError: n_T − k = {gl_e}\nTotal: n_T − 1 = {gl_t}',
        'CUADRADOS MEDIOS\n'
        f'CMTR = SCTR / (k − 1) = {_n(sctr)} / {gl_tr} = {_n(cmtr)}\n'
        f'CME = SCE / (n_T − k) = {_n(sce)} / {gl_e} = {_n(cme)}',
        f'ESTADÍSTICO DE PRUEBA\nF = CMTR / CME = {_n(cmtr)} / {_n(cme)} = {_n(F)}',
        f'F crítico:  F(α = {_n(alpha)};  {gl_tr},  {gl_e}) = {_n(fcrit, 4)}\n'
        f'Valor-p = {_p(p)}',
        'REGLA DE RECHAZO\n'
        f'Método del valor-p: rechazar H₀ si valor-p ≤ α  →  {_p(p)} {"≤" if rechaza else ">"} {_n(alpha)}\n'
        f'Método del valor crítico: rechazar H₀ si F ≥ F_α  →  {_n(F)} {"≥" if F >= fcrit else "<"} {_n(fcrit, 4)}',
    ]
    tabla = [
        {'Fuente de variación': 'Tratamientos', 'Suma de cuadrados': _n(sctr),
         'Grados de libertad': gl_tr, 'Cuadrado medio': _n(cmtr), 'F': _n(F), 'Valor-p': _p(p)},
        {'Fuente de variación': 'Error', 'Suma de cuadrados': _n(sce),
         'Grados de libertad': gl_e, 'Cuadrado medio': _n(cme), 'F': '', 'Valor-p': ''},
        {'Fuente de variación': 'Total', 'Suma de cuadrados': _n(stc),
         'Grados de libertad': gl_t, 'Cuadrado medio': '', 'F': '', 'Valor-p': ''},
    ]
    if rechaza:
        decision = 'Se rechaza H₀'
        conclusion = (f'Con α = {_n(alpha)}, no todas las medias poblacionales son iguales: '
                      'al menos una media es distinta.')
    else:
        decision = 'No se rechaza H₀'
        conclusion = (f'Con α = {_n(alpha)}, no hay evidencia suficiente para decir que las '
                      'medias poblacionales sean distintas.')

    # ---------------- LSD de Fisher
    tcrit = float(stats.t.ppf(1 - alpha / 2, gl_e))
    iguales = len(set(ns)) == 1
    lsd_pasos = [
        'Se usa la distribución t con n_T − k = ' + str(gl_e) + ' grados de libertad.\n'
        'LSD = t(α/2) · √( CME · (1/nᵢ + 1/nⱼ) )',
        f't({_n(alpha / 2, 4)}; {gl_e}) = {_n(tcrit, 4)}      CME = {_n(cme)}',
    ]
    if iguales:
        n0 = ns[0]
        lsd0 = tcrit * np.sqrt(cme * (2 / n0))
        lsd_pasos.append(f'LSD = {_n(tcrit, 4)} · √( {_n(cme)} · (1/{n0} + 1/{n0}) ) = {_n(lsd0)}')
    else:
        lsd_pasos.append('Los grupos tienen tamaños distintos, así que el LSD se calcula para cada pareja.')
    if not rechaza:
        lsd_pasos.append('Nota: el ANOVA no rechazó H₀. Normalmente el LSD de Fisher se aplica '
                         'solo cuando el ANOVA sí rechaza; aquí se muestra por completo por si lo necesitas.')

    lsd_tabla, difieren = [], []
    for i, j in combinations(range(k), 2):
        dif = abs(medias[i] - medias[j])
        lsd = tcrit * np.sqrt(cme * (1 / ns[i] + 1 / ns[j]))
        rech = dif >= lsd
        if rech:
            difieren.append(f'{nombres[i]} y {nombres[j]}')
        lsd_tabla.append({
            'Pareja': f'{nombres[i]} vs {nombres[j]}',
            'H₀': f'μ({nombres[i]}) = μ({nombres[j]})',
            '|x̄ᵢ − x̄ⱼ|': _n(dif),
            'LSD': _n(lsd),
            'Decisión': ('Se rechaza H₀ → medias distintas' if rech
                         else 'No se rechaza H₀ → medias iguales'),
        })
    lsd_pasos.append('Regla: rechazar H₀ si |x̄ᵢ − x̄ⱼ| ≥ LSD')
    if difieren:
        lsd_concl = 'Difieren significativamente: ' + '; '.join(difieren) + '.'
    else:
        lsd_concl = 'Ninguna pareja de medias difiere significativamente.'

    return {
        'pasos': pasos, 'tabla': tabla, 'decision': decision, 'conclusion': conclusion,
        'lsd_pasos': lsd_pasos, 'lsd_tabla': lsd_tabla, 'lsd_conclusion': lsd_concl,
        'grafica': _grafica(grupos, nombres, medias, F, fcrit, gl_tr, gl_e, alpha),
        # valores crudos, útiles para comprobar
        'num': {'F': F, 'p': p, 'fcrit': fcrit, 'sctr': sctr, 'sce': sce, 'stc': stc,
                'cmtr': cmtr, 'cme': cme, 'tcrit': tcrit},
    }


@anova.route('/anova')
def pagina_anova():
    return render_template('anova.html')


@anova.route('/anova_calcular', methods=['POST'])
def anova_calcular():
    d = request.get_json(silent=True) or {}
    try:
        crudos = d.get('grupos', [])
        nombres, grupos = [], []
        for i, g in enumerate(crudos, start=1):
            datos = _parsear(g.get('datos', ''))
            if not datos:
                continue                      # ignora grupos vacíos
            nombres.append((g.get('nombre') or '').strip() or f'Grupo {i}')
            grupos.append(datos)
        try:
            alpha = float(str(d.get('alpha', '0.05')).replace(',', '.'))
        except ValueError:
            raise DatosInvalidos('α no es un número válido.')
        # json.dumps (sin ordenar claves) para que las columnas de las tablas
        # lleguen en el orden correcto; jsonify de Flask las ordena alfabéticamente.
        cuerpo = json.dumps(resolver(nombres, grupos, alpha), ensure_ascii=False)
        return Response(cuerpo, mimetype='application/json')
    except DatosInvalidos as e:
        return jsonify({'error': str(e)}), 400