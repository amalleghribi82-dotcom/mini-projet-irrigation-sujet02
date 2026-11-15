# Mini-projet 1GII — Sujet 02 : irrigation automatique d'une plante en pot

Phases 3 et 4 : modèle mathématique et simulation de la commande à hystérésis.

- ENET'Com — Université de Sfax, année 2026/2027, classe 1GII1
- Groupe : Ghribi Amal, Abdeli Hadil
- Encadrante : Dr.-Ing. Ferdaous Masmoudi
- Rapport : projet Overleaf (lien envoyé à l'encadrante)

## Modèle

Indice d'humidité `x` (en %), temps en secondes :

    dx/dt = -a (x - x_sec) + b u(t - tau_d) - d(t),   x dans [0, 100]

Paramètres (hypothèses de travail, à identifier sur un vrai pot) : `a = 4e-5 1/s`, `x_sec = 5 %`, `b = 0,5 %/s`, `tau_d = 20 s`, évaporation nominale `6e-4 %/s`.

## Commande simulée

Hystérésis 35 % / 55 %, impulsions de pompe de 10 s au maximum, temporisation de 30 s après chaque impulsion, limite de 8 impulsions par heure, inhibition de la pompe si le réservoir est vide.
Scénarios comparés : sans pompe, minuterie (boucle ouverte), hystérésis simple, hystérésis avec impulsions (retenue), variante à durée d'impulsion proportionnelle, perturbations, changement de consigne, réservoir vide, bruit de capteur.

## Utilisation

    pip install -r requirements.txt
    python3 simulation_sujet02.py

Le script écrit les figures dans `figures/` et les indicateurs dans `resultats.json`, puis affiche un résumé dans le terminal (durée de calcul : environ 10 s).

## Résultat principal

Avec l'hystérésis retenue, l'indice reste dans la plage 35–55 % pendant 83,8 % des 24 h simulées (objectif : 80 %), avec 35 impulsions et 350 mL d'eau.
