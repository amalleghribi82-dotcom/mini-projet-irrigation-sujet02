#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mini-projet 1GII - Sujet 02 : irrigation automatique d'une plante en pot
Phases 3 et 4 : modèle mathématique + simulation de la commande à hystérésis.

Modèle (indice d'humidité x en %, temps en secondes) :
    dx/dt = -a (x - x_sec) + b u(t - tau_d) - d(t),     x dans [0, 100]

Scénarios simulés (24 h, pas de calcul 1 s) :
    none        : sans pompe (séchage seul)
    timer       : boucle ouverte, une impulsion de 10 s toutes les 2300 s
    naive       : hystérésis 35/55 %, pompe en continu, sans temporisation
    hyst        : hystérésis 35/55 % + impulsions <= 10 s + temporisation 30 s (RETENUE)
    hystP       : variante de hyst avec durée d'impulsion proportionnelle (option P)
    single      : seuil unique 45 % avec impulsions + temporisation
    sing_naive  : seuil unique 45 %, pompe en continu, sans temporisation

Utilisation :   python3 simulation_sujet02.py
Sorties     :   figures/*.png  et  resultats.json
"""
import json
import os
from collections import deque

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "figures")
os.makedirs(OUT, exist_ok=True)

# ---------------------------------------------------------------- paramètres
X_SEC = 5.0        # indice du sol sec [%]
A = 4e-5           # taux de séchage [1/s]  (constante de temps 1/a = 25000 s = 6,9 h)
B = 0.5            # efficacité d'arrosage [%/s] : une impulsion de 10 s -> +5 points
TAU_D = 20         # retard de diffusion [s]
E_NOM = 6e-4       # évaporation nominale d [%/s] (~2,2 %/h)
DT = 1             # pas de calcul [s]
H = 3600

LOW, HIGH = 35.0, 55.0   # seuils de l'hystérésis [%]
T_ON = 10                # impulsion maximale de pompe [s]
T_OBS = 30               # temporisation d'observation après arrosage [s]
N_MAX_H = 8              # nombre maximal d'impulsions par heure glissante
Q_POMPE = 1.0            # débit de la pompe [mL/s] (60 mL/min)
V_MIN = 20.0             # niveau du détecteur "réservoir vide" [mL]
T_TIMER = 2300           # période de la minuterie (boucle ouverte) [s]
T_STAT = 6 * H           # début de la mesure du régime établi [s]
SIGMA = 2.0              # bruit de mesure de la sonde [points d'indice]
EVAP_MULT, EVAP_T0, EVAP_T1 = 2.5, 6, 18   # perturbation : évaporation x2,5 de 6 h à 18 h

C_X, C_REF, C_PUMP, C_BAND, C_PERT = "#1f4e9c", "#6b7280", "#e07b00", "#2e9e6b", "#f2c94c"
plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.25,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 100, "savefig.dpi": 200})


# ---------------------------------------------------------------- simulation
def simulate(ctrl, T=24 * H, x0=60.0, band=None, evap=None, bmult=None,
             v0=1000.0, sigma=0.0, seed=1, thr_single=45.0):
    """Simule le système en boucle ouverte ou fermée. Retourne un dictionnaire."""
    N = int(T / DT)
    rng = np.random.default_rng(seed)
    if band is None:
        band = lambda t: (LOW, HIGH)
    x = x0
    xs = np.zeros(N); xms = np.zeros(N); us = np.zeros(N, dtype=int); Vs = np.zeros(N)
    los = np.zeros(N); his = np.zeros(N); lat = np.zeros(N, dtype=bool)
    V = v0
    state, pump_end, wait_end, latch = 0, 0, 0, False
    starts = deque(); log = []; alarm_t = None; n_bloc = 0
    for k in range(N):
        t = k * DT
        lo, hi = band(t)
        xm = x + sigma * rng.standard_normal() if sigma > 0 else x
        xm = min(max(xm, 0.0), 100.0)
        empty = V <= V_MIN
        u = 0
        if ctrl == "none":
            pass
        elif ctrl == "timer":
            u = 1 if (k % T_TIMER) < T_ON else 0
        elif ctrl == "naive":
            if (not latch) and xm < lo:
                latch = True
            elif latch and xm > hi:
                latch = False
            u = 1 if latch else 0
        elif ctrl == "sing_naive":
            u = 1 if xm < thr_single else 0
        else:  # hyst, hystP, single : automate IDLE -> POMPE -> ATTENTE
            if state == 1 and k >= pump_end:
                state, wait_end = 2, k + T_OBS
            if state == 2 and k >= wait_end:
                state = 0
            if state == 0:
                if ctrl == "single":
                    want = xm < thr_single
                else:
                    if (not latch) and xm < lo:
                        latch = True
                    elif latch and xm > hi:
                        latch = False
                    want = latch
                if want:
                    while starts and starts[0] <= k - H:
                        starts.popleft()
                    if empty:
                        pass
                    elif len(starts) >= N_MAX_H:
                        n_bloc += 1
                    else:
                        L = T_ON if ctrl != "hystP" else int(np.clip(round((hi + 1 - xm) / B), 2, T_ON))
                        state, pump_end = 1, k + L
                        starts.append(k); log.append((t, L, xm))
            u = 1 if state == 1 else 0
        if empty:
            if alarm_t is None:
                alarm_t = t
            if u == 1:
                u = 0; state = 0
        us[k] = u
        ud = us[k - TAU_D] if k >= TAU_D else 0
        bm = bmult(t) if bmult else 1.0
        em = evap(t) if evap else 1.0
        xs[k], xms[k], Vs[k], los[k], his[k], lat[k] = x, xm, V, lo, hi, latch
        dx = -A * (x - X_SEC) + B * bm * ud - E_NOM * em
        x = min(max(x + DT * dx, 0.0), 100.0)
        V -= Q_POMPE * u * DT
    return dict(t=np.arange(N) * DT, x=xs, xm=xms, u=us, V=Vs, lo=los, hi=his,
                latch=lat, log=log, alarm_t=alarm_t, ctrl=ctrl, n_bloc=n_bloc)


def metrics(r, t_from=0.0):
    """Indicateurs de performance d'une simulation."""
    t, x, lo, hi, u = r["t"], r["x"], r["lo"], r["hi"], r["u"]
    inb = (x >= lo) & (x <= hi)
    sel = t >= T_STAT
    idx = np.where(inb)[0]
    e0 = idx[0] if len(idx) else 0
    c = (lo + hi) / 2
    st = np.where((u[1:] == 1) & (u[:-1] == 0))[0] + 1
    if u[0] == 1:
        st = np.insert(st, 0, 0)
    m = dict(
        pct_total=float(100 * inb.mean()),
        pct_apres_6h=float(100 * inb[sel].mean()),
        t_entree_bande_h=float(t[e0] / H),
        depassement=float(max(0.0, np.max(x[e0:] - hi[e0:]))),
        sous_depassement=float(max(0.0, np.max(lo[e0:] - x[e0:]))),
        x_min=float(x.min()), x_max=float(x.max()),
        erreur_moy_abs=float(np.mean(np.abs(x[sel] - c[sel]))),
        erreur_rms=float(np.sqrt(np.mean((x[sel] - c[sel]) ** 2))),
        n_demarrages=int(len(st)),
        temps_marche_s=int(u.sum()),
        volume_mL=float(Q_POMPE * u.sum()),
        intervalle_min_s=(int(np.min(np.diff(st))) if len(st) > 1 else None),
        premiere_activation_h=(float(t[st[0]] / H) if len(st) else None),
        n_blocages_limite=int(r["n_bloc"]),
    )
    lat = r["latch"].astype(int)
    up = np.where((lat[1:] == 1) & (lat[:-1] == 0))[0] + 1
    dn = np.where((lat[1:] == 0) & (lat[:-1] == 1))[0] + 1
    m["n_cycles"] = int(len(up)) if r["ctrl"] in ("naive", "hyst", "hystP") else None
    durs = [dn[dn > a][0] - a for a in up if (dn > a).any()]
    m["duree_cycle_moy_s"] = float(np.mean(durs)) if durs else None
    return m


# ---------------------------------------------------------------- scénarios
def run_all():
    res, met = {}, {}
    nominal = ["none", "timer", "naive", "hyst", "hystP"]
    for c in nominal:
        res[c] = simulate(c)
    # perturbation d'évaporation x2,5 entre 6 h et 18 h (journée chaude et ventée)
    evap = lambda t: EVAP_MULT if EVAP_T0 * H <= t < EVAP_T1 * H else 1.0
    for c in ("timer", "hyst", "hystP"):
        res[c + "_evap"] = simulate(c, evap=evap)
    # drainage accru : efficacité d'arrosage x0,7 pendant toute la simulation
    bm = lambda t: 0.7
    for c in ("timer", "hyst", "hystP"):
        res[c + "_drain"] = simulate(c, bmult=bm)
    # arrosage trop efficace : efficacité d'arrosage x1,8
    for c in ("timer", "hyst", "hystP"):
        res[c + "_fort"] = simulate(c, bmult=lambda t: 1.8)
    # changement de consigne (bande 35/55 -> 50/70 à t = 12 h)
    band = lambda t: (LOW, HIGH) if t < 12 * H else (50.0, 70.0)
    res["hyst_consigne"] = simulate("hyst", band=band)
    # réservoir de 150 mL seulement : il se vide en cours de simulation
    res["hyst_vide"] = simulate("hyst", v0=120.0)
    # bruit de capteur
    for c in ("sing_naive", "naive", "single", "hyst"):
        res[c + "_bruit"] = simulate(c, sigma=SIGMA)
    for k, r in res.items():
        met[k] = metrics(r)
    # réponse à l'échelon de consigne
    r = res["hyst_consigne"]; tc = 12 * H
    sel = r["t"] >= tc
    ok = np.where(sel & (r["x"] >= 50.0))[0]
    met["hyst_consigne"]["t_rep_consigne_s"] = float(r["t"][ok[0]] - tc) if len(ok) else None
    met["hyst_consigne"]["x_a_tc"] = float(r["x"][int(tc)])
    met["hyst_consigne"]["depassement_70"] = float(max(0.0, np.max(r["x"][sel] - 70.0)))
    sub = r["t"] >= tc
    met["hyst_consigne"]["pct_nouvelle_bande_apres_rep"] = float(
        100 * np.mean(((r["x"] >= 50) & (r["x"] <= 70))[sub & (r["t"] >= (r["t"][ok[0]] if len(ok) else tc))]))
    # alarme réservoir vide
    r = res["hyst_vide"]
    met["hyst_vide"]["t_alarme_h"] = None if r["alarm_t"] is None else r["alarm_t"] / H
    met["hyst_vide"]["x_final"] = float(r["x"][-1])
    met["hyst_vide"]["impulsions_apres_alarme"] = int(
        sum(1 for (tt, L, xm) in r["log"] if r["alarm_t"] is not None and tt > r["alarm_t"]))
    # vérification analytique
    x_inf = X_SEC - E_NOM / A
    t_th = (1 / A) * np.log((60 - x_inf) / (LOW - x_inf))
    met["analytique"] = dict(
        x_inf_sans_pompe=x_inf, t_35_th_h=float(t_th / H),
        x_eq_minuterie=X_SEC + (B * T_ON / T_TIMER - E_NOM) / A,
        x_eq_minuterie_evap=X_SEC + (B * T_ON / T_TIMER - EVAP_MULT * E_NOM) / A,
        gain_impulsion=B * T_ON, tau_h=1 / A / H,
        volume_impulsion_mL=Q_POMPE * T_ON,
        duree_mini_entre_impulsions_s=T_ON + T_OBS)
    return res, met


# ---------------------------------------------------------------- figures
def hrs(r):
    return r["t"] / H


def starts_of(r):
    u = r["u"]
    return np.where((u[1:] == 1) & (u[:-1] == 0))[0] + 1


def band_patch(ax, lo=LOW, hi=HIGH, label="Plage visée 35–55 %"):
    ax.axhspan(lo, hi, color=C_BAND, alpha=0.13, label=label)
    ax.axhline(lo, color=C_BAND, lw=0.8, ls="--")
    ax.axhline(hi, color=C_BAND, lw=0.8, ls="--")


def fig_schema_blocs():
    fig, ax = plt.subplots(figsize=(11, 4.3)); ax.set_xlim(0, 22); ax.set_ylim(0, 8.6); ax.axis("off")

    def box(x, y, w, h, txt, fc="#ffffff", ec="#374151", lw=1.2, fs=8.5, bold=False):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.18",
                                    fc=fc, ec=ec, lw=lw))
        ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=fs,
                fontweight="bold" if bold else "normal")

    def arr(x1, y1, x2, y2, txt=None, dy=0.28, dx=0.0, col="#374151", ls="-"):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color=col, lw=1.2, ls=ls, shrinkA=0, shrinkB=0))
        if txt:
            ax.text((x1 + x2) / 2 + dx, (y1 + y2) / 2 + dy, txt, ha="center", fontsize=8, color=col)

    box(0.2, 5.3, 3.4, 1.6, "Comparateur à\nhystérésis (ESP32)\n+ impulsion ≤ 10 s\n+ attente ≥ 30 s", fc="#e5edfb", ec=C_X, bold=True)
    box(4.7, 5.3, 2.3, 1.6, "Pompe\n(ON/OFF)")
    box(8.0, 5.3, 2.4, 1.6, "Retard de\ndiffusion\ne^(−τd·p)")
    box(11.4, 5.3, 1.9, 1.6, "Gain\nb")
    ax.add_patch(plt.Circle((15.0, 6.1), 0.5, fc="white", ec="#374151", lw=1.2))
    ax.text(15.0, 6.1, "Σ", ha="center", va="center", fontsize=12)
    box(16.9, 5.3, 1.9, 1.6, "Intégrateur\n1/p")
    box(19.7, 5.3, 2.0, 1.6, "Sonde\nSEN0193\n(+ bruit)")
    box(13.6, 2.5, 2.8, 1.5, "Séchage\n−a·(x − x_sec)")
    box(4.7, 2.5, 2.8, 1.5, "Réservoir\nniveau bas", fc="#fdf3e3", ec=C_PUMP)

    arr(3.6, 6.1, 4.7, 6.1, "u", dy=0.25)
    arr(7.0, 6.1, 8.0, 6.1)
    arr(10.4, 6.1, 11.4, 6.1, "u(t−τd)", dy=0.25)
    arr(13.3, 6.1, 14.5, 6.1, "+", dy=0.2)
    arr(15.5, 6.1, 16.9, 6.1)
    arr(18.8, 6.1, 19.7, 6.1, "x", dy=0.25)
    # perturbation
    arr(15.0, 8.2, 15.0, 6.6, None)
    ax.text(15.15, 7.7, "− d(t)\névaporation,\ndrainage", fontsize=8, color="#b45309", va="center")
    # retour séchage : prise de x en sortie de l'intégrateur -> bloc séchage -> Σ (signe -)
    ax.plot([19.25, 19.25], [6.1, 3.25], color="#374151", lw=1.2)
    ax.plot([19.25], [6.1], marker="o", ms=4, color="#374151")
    ax.annotate("", xy=(16.4, 3.25), xytext=(19.25, 3.25), arrowprops=dict(arrowstyle="-|>", color="#374151", lw=1.2, shrinkA=0, shrinkB=0))
    ax.annotate("", xy=(15.0, 5.6), xytext=(15.0, 4.0), arrowprops=dict(arrowstyle="-|>", color="#374151", lw=1.2, shrinkA=0, shrinkB=0))
    ax.text(15.15, 4.7, "−", fontsize=12)
    # mesure vers comparateur
    ax.plot([20.7, 20.7], [5.3, 1.1], color="#374151", lw=1.2)
    ax.plot([20.7, 1.9], [1.1, 1.1], color="#374151", lw=1.2)
    ax.annotate("", xy=(1.9, 5.3), xytext=(1.9, 1.1), arrowprops=dict(arrowstyle="-|>", color="#374151", lw=1.2))
    ax.text(11.3, 0.65, "x_mes (indice mesuré, étalonné 0–100 %)", ha="center", fontsize=8.5)
    # consigne
    arr(1.9, 8.3, 1.9, 6.9, None)
    ax.text(2.1, 8.0, "Consigne : 35 % / 55 %", fontsize=8.5)
    # réservoir -> comparateur
    ax.annotate("", xy=(3.0, 5.3), xytext=(4.7, 3.25), arrowprops=dict(arrowstyle="-|>", color=C_PUMP, lw=1.2, ls="--"))
    ax.text(3.0, 3.9, "réservoir\nvide : inhibe", fontsize=7.5, color=C_PUMP, ha="right")
    ax.annotate("", xy=(5.85, 5.3), xytext=(5.85, 4.0), arrowprops=dict(arrowstyle="-|>", color=C_PUMP, lw=1.2))
    ax.text(6.0, 4.35, "eau", fontsize=8, color=C_PUMP)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_01_schema_blocs.png"), bbox_inches="tight"); plt.close(fig)


def fig_boucle_ouverte(res):
    fig, ax = plt.subplots(figsize=(8.2, 3.6))
    band_patch(ax)
    ax.plot(hrs(res["none"]), res["none"]["x"], color=C_REF, lw=1.8, label="Sans pompe (séchage seul)")
    ax.plot(hrs(res["timer"]), res["timer"]["x"], color=C_PUMP, lw=1.6, label="Minuterie (1 impulsion de 10 s / 2300 s)")
    ax.set_xlabel("Temps [h]"); ax.set_ylabel("Indice d'humidité x [%]"); ax.set_xlim(0, 24); ax.set_ylim(0, 70)
    ax.set_title("Boucle ouverte, évaporation nominale", fontsize=10, loc="left")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_02_boucle_ouverte.png")); plt.close(fig)


def fig_hysteresis(res):
    r = res["hyst"]
    fig, axs = plt.subplots(3, 1, figsize=(8.2, 6.2), sharex=True, gridspec_kw=dict(height_ratios=[3, 0.9, 1.4]))
    ax = axs[0]; band_patch(ax)
    ax.plot(hrs(r), r["x"], color=C_X, lw=1.6, label="x(t) avec hystérésis")
    ax.set_ylabel("Indice x [%]"); ax.set_ylim(0, 75); ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    ax.set_title("Hystérésis 35 / 55 % : impulsions ≤ 10 s, attente 30 s", fontsize=10, loc="left")
    s = starts_of(r)
    axs[1].vlines(r["t"][s] / H, 0, 1, color=C_PUMP, lw=1.2); axs[1].set_yticks([]); axs[1].set_ylabel("Pompe")
    axs[2].plot(hrs(r), (1000 - r["V"]), color=C_PUMP, lw=1.5)
    axs[2].set_ylabel("Eau utilisée [mL]"); axs[2].set_xlabel("Temps [h]"); axs[2].set_xlim(0, 24)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_03_hysteresis_nominal.png")); plt.close(fig)


def fig_zoom(res):
    rn, rh = res["naive"], res["hyst"]
    t0 = starts_of(rh)[0]
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for ax, r, ttl, T1 in ((axs[0], rn, "Hystérésis simple (pompe en continu)", 900),
                           (axs[1], rh, "Hystérésis + impulsions + attente", 900)):
        sl = slice(int(t0 - 60), int(t0 + T1))
        tt = r["t"][sl] - t0
        band_patch(ax)
        ax.plot(tt, r["x"][sl], color=C_X, lw=1.8, label="x(t)")
        ax.fill_between(tt, 0, r["u"][sl] * 8, step="mid", color=C_PUMP, alpha=0.6, label="pompe ON (échelle ×8)")
        ax.set_title(ttl, fontsize=9.5, loc="left"); ax.set_xlabel("Temps depuis le 1er démarrage [s]")
        ax.set_ylim(0, 80)
    axs[0].set_ylabel("Indice x [%]"); axs[0].legend(loc="upper right", fontsize=8, framealpha=0.9)
    xm = rn["x"][int(t0):int(t0) + 900].max()
    axs[0].annotate(f"pic {xm:.0f} %", xy=(np.argmax(rn["x"][int(t0):int(t0) + 900]), xm), xytext=(300, xm + 3),
                    fontsize=8.5, arrowprops=dict(arrowstyle="-", color="#374151"))
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_04_zoom_episode.png")); plt.close(fig)


def fig_perturbation(res):
    fig, axs = plt.subplots(2, 1, figsize=(8.2, 5.6), sharex=True, gridspec_kw=dict(height_ratios=[3, 1.4]))
    ax = axs[0]; band_patch(ax)
    ax.axvspan(EVAP_T0, EVAP_T1, color=C_PERT, alpha=0.25, label=f"évaporation ×{EVAP_MULT:.1f}".replace(".", ","))
    ax.plot(hrs(res["timer_evap"]), res["timer_evap"]["x"], color=C_PUMP, lw=1.6, label="Boucle ouverte (minuterie)")
    ax.plot(hrs(res["hyst_evap"]), res["hyst_evap"]["x"], color=C_X, lw=1.6, label="Hystérésis")
    ax.set_ylabel("Indice x [%]"); ax.set_ylim(0, 75); ax.legend(loc="lower left", fontsize=8, framealpha=0.9, ncol=2)
    ax.set_title(f"Effet d'une perturbation : évaporation ×{EVAP_MULT:.1f} entre {EVAP_T0} h et {EVAP_T1} h".replace(".", ","), fontsize=10, loc="left")
    for k, c, l in (("timer_evap", C_PUMP, "minuterie"), ("hyst_evap", C_X, "hystérésis")):
        axs[1].plot(hrs(res[k]), (1000 - res[k]["V"]), color=c, lw=1.5, label=l)
    axs[1].axvspan(EVAP_T0, EVAP_T1, color=C_PERT, alpha=0.25)
    axs[1].set_ylabel("Eau utilisée [mL]"); axs[1].set_xlabel("Temps [h]"); axs[1].set_xlim(0, 24)
    axs[1].legend(loc="upper left", fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_05_perturbation.png")); plt.close(fig)


def fig_consigne(res):
    r = res["hyst_consigne"]
    fig, ax = plt.subplots(figsize=(8.2, 3.8))
    ax.fill_between(hrs(r), r["lo"], r["hi"], color=C_BAND, alpha=0.13, label="Plage visée")
    ax.plot(hrs(r), r["lo"], color=C_BAND, lw=0.9, ls="--"); ax.plot(hrs(r), r["hi"], color=C_BAND, lw=0.9, ls="--")
    ax.plot(hrs(r), r["x"], color=C_X, lw=1.6, label="x(t)")
    s = starts_of(r)
    ax.vlines(r["t"][s] / H, 0, 4, color=C_PUMP, lw=1.2, label="impulsions de pompe")
    ax.axvline(12, color="#374151", lw=0.8, ls=":")
    ax.text(12.1, 3, "changement de plage\n35–55 → 50–70 %", fontsize=8)
    ax.set_xlabel("Temps [h]"); ax.set_ylabel("Indice x [%]"); ax.set_xlim(0, 24); ax.set_ylim(0, 80)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_06_changement_consigne.png")); plt.close(fig)


def fig_vide(res):
    r = res["hyst_vide"]
    fig, axs = plt.subplots(2, 1, figsize=(8.2, 5.0), sharex=True, gridspec_kw=dict(height_ratios=[2.2, 1.3]))
    ta = r["alarm_t"] / H if r["alarm_t"] is not None else None
    ax = axs[0]; band_patch(ax)
    ax.plot(hrs(r), r["x"], color=C_X, lw=1.6, label="x(t)")
    if ta is not None:
        ax.axvspan(ta, 24, color="#dc2626", alpha=0.08, label="alarme · pompe inhibée")
    ax.set_ylabel("Indice x [%]"); ax.set_ylim(0, 75); ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    ax.set_title("Réservoir de 120 mL : détection de réservoir vide", fontsize=10, loc="left")
    axs[1].plot(hrs(r), r["V"], color=C_PUMP, lw=1.6)
    axs[1].axhline(V_MIN, color="#dc2626", lw=0.9, ls="--"); axs[1].text(0.3, V_MIN + 6, "seuil « niveau bas » 20 mL", fontsize=8, color="#dc2626")
    axs[1].set_ylabel("Volume restant [mL]"); axs[1].set_xlabel("Temps [h]"); axs[1].set_xlim(0, 24)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_07_reservoir_vide.png")); plt.close(fig)


def fig_bruit(res, met):
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.7), gridspec_kw=dict(width_ratios=[1.5, 1]))
    r = res["hyst_bruit"]; sl = slice(6 * H, 12 * H)
    ax = axs[0]; band_patch(ax)
    ax.plot(r["t"][sl] / H, r["xm"][sl], color="#9ca3af", lw=0.4, label="mesure bruitée (σ = 2 pts)")
    ax.plot(r["t"][sl] / H, r["x"][sl], color=C_X, lw=1.6, label="indice réel x")
    ax.set_xlabel("Temps [h]"); ax.set_ylabel("Indice [%]"); ax.set_ylim(0, 95)
    ax.set_title("Hystérésis retenue avec bruit de capteur", fontsize=9.5, loc="left")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    ax = axs[1]
    names = ["Seuil unique 45 %\n(pompe continue)", "Hystérésis 35/55 %\n(pompe continue)", "Hystérésis 35/55 %\n+ impulsions + attente"]
    keys = ["sing_naive_bruit", "naive_bruit", "hyst_bruit"]
    vals = [met[k]["n_demarrages"] for k in keys]
    bars = ax.bar(range(3), vals, color=[C_REF, "#93a8d4", C_X], width=0.6)
    ax.set_yscale("log"); ax.set_xticks(range(3)); ax.set_xticklabels(names, fontsize=7.5)
    for b_, v in zip(bars, vals):
        ax.text(b_.get_x() + b_.get_width() / 2, v * 1.12, str(v), ha="center", fontsize=9)
    ax.set_ylim(1, max(vals) * 4); ax.set_ylabel("Démarrages de pompe en 24 h")
    ax.set_title("Pompage avec bruit σ = 2 pts", fontsize=9.5, loc="left")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_08_bruit_pompage.png")); plt.close(fig)


def main():
    res, met = run_all()
    fig_schema_blocs(); fig_boucle_ouverte(res); fig_hysteresis(res); fig_zoom(res)
    fig_perturbation(res); fig_consigne(res); fig_vide(res); fig_bruit(res, met)
    # journal des cycles (mémorisation demandée par le cahier des charges)
    log = [dict(t_s=float(a), duree_s=int(b), x_mesure=float(c)) for a, b, c in res["hyst"]["log"]]
    out = dict(parametres=dict(X_SEC=X_SEC, A=A, B=B, TAU_D=TAU_D, E_NOM=E_NOM, LOW=LOW, HIGH=HIGH,
                               T_ON=T_ON, T_OBS=T_OBS, N_MAX_H=N_MAX_H, Q_POMPE=Q_POMPE, V_MIN=V_MIN,
                               T_TIMER=T_TIMER, SIGMA=SIGMA, EVAP_MULT=EVAP_MULT),
               metriques=met, journal_hyst=log)
    with open(os.path.join(HERE, "resultats.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    for k in ("none", "timer", "naive", "hyst", "hystP", "timer_evap", "hyst_evap", "hystP_evap", "timer_drain",
              "hyst_drain", "hystP_drain", "timer_fort", "hyst_fort", "hystP_fort", "hyst_consigne", "hyst_vide", "sing_naive_bruit", "naive_bruit", "single_bruit", "hyst_bruit"):
        m = met[k]
        print(f"{k:18s} bande={m['pct_total']:5.1f}% (>6h {m['pct_apres_6h']:5.1f}%) dep={m['depassement']:5.1f} "
              f"sdep={m['sous_depassement']:4.1f} xmin={m['x_min']:5.1f} xmax={m['x_max']:5.1f} "
              f"start={m['n_demarrages']:4d} cyc={m['n_cycles']} vol={m['volume_mL']:6.1f} "
              f"dtmin={m['intervalle_min_s']} err={m['erreur_moy_abs']:.1f} bloc={m['n_blocages_limite']}")
    print(json.dumps({k: v for k, v in met.items() if k in ("analytique",)}, indent=1))
    print({k: met["hyst_consigne"].get(k) for k in ("t_rep_consigne_s", "x_a_tc", "depassement_70", "pct_nouvelle_bande_apres_rep")})
    print({k: met["hyst_vide"].get(k) for k in ("t_alarme_h", "x_final", "impulsions_apres_alarme")})


if __name__ == "__main__":
    main()
