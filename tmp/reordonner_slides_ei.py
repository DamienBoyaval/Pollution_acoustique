from pathlib import Path
import re, shutil

root=Path(__file__).resolve().parents[1]
path=root/'presentation_optimisation_binarisation.tex'
source=path.read_text(encoding='utf-8')
matches=list(re.finditer(r'\\begin\{frame\}(?:\[[^]]*\])?(?:\{[^}]*\})?',source))
assert len(matches)==26, len(matches)
prefix=source[:matches[0].start()]
frames=[]
for i,m in enumerate(matches):
    end=matches[i+1].start() if i+1<len(matches) else source.rfind(r'\end{document}')
    frames.append(source[m.start():end])

def frame(title,body):
    return '\\begin{frame}{'+title+'}\n'+body+'\n\\end{frame}\n\n'

fig=root/'figures_presentation'
shutil.copy2('C:/Users/damie/Downloads/resultats/chi_binaire_0.5.png',fig/'descente_chi_binaire_05.png')
shutil.copy2('C:/Users/damie/Downloads/resultats/Dernier Programme/5 - binarize 1 max iter 80 - 6 cycles/Figure_11.png',fig/'six_cycles_seuil_80_barres.png')

gradient=frame('La descente réduit l’énergie à la fréquence optimisée',r'''
\centering
\includegraphics[width=.95\textwidth,height=.67\textheight,keepaspectratio]{figures_presentation/descente_chi_binaire_05.png}

\small Sur cette figure, chaque fréquence a son propre $\chi$ optimisé :
la courbe continue (bleu) est sous la courbe initiale (gris) aux points calculés.
C'est un contrôle numérique de la descente à fréquence fixée ; après seuillage
(orange), le gain diminue et la quantité peut changer.
''')

bars=frame('Six cycles avec seuillage : le meilleur est le cycle 3',r'''
\centering
\includegraphics[width=.95\textwidth,height=.67\textheight,keepaspectratio]{figures_presentation/six_cycles_seuil_80_barres.png}

\small Essai distinct : \code{chi_binarize_1}, 80 itérations maximum et 6 cycles.
La somme minimale est $7{,}0664\,10^{-4}$ au cycle 3 ($-40{,}8\,\%$ face à
l'initiale). La fraction de matériau de ce cycle est $13{,}70\,\%$ :
ce gain ne respecte plus le budget initial de $30\,\%$.
''')

kept={i:frames[i-1] for i in range(1,27) if i not in {10,11,12,13,18,19,24,26}}
order=[1,2,3,4,5,7,6,'gradient',8,9,17,14,15,16,20,'bars',21,22,23,25]
assert set(x for x in order if isinstance(x,int))==set(kept)
out=''.join(kept[x] if isinstance(x,int) else {'gradient':gradient,'bars':bars}[x] for x in order)
path.write_text(prefix+out+r'\end{document}'+'\n',encoding='utf-8')
print('Slides:',len(order))
for i,x in enumerate(order,1):
    title=re.search(r'\\begin\{frame\}(?:\{([^}]*)\})?',kept[x] if isinstance(x,int) else {'gradient':gradient,'bars':bars}[x]).group(1)
    print(i,x,title)
