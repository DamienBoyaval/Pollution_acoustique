from PIL import Image, ImageDraw
from pathlib import Path
p = Path('tmp/pdfs/verification_EI')
fs = sorted(p.glob('page-*.png'))
for start in range(0, len(fs), 8):
    canvas = Image.new('RGB', (1100,1320), '#cbd0d5')
    for i,f in enumerate(fs[start:start+8]):
        im = Image.open(f)
        im.thumbnail((540,304))
        x,y = (i%2)*550,(i//2)*330
        canvas.paste(im,(x,y+24))
        ImageDraw.Draw(canvas).text((x+5,y+5),f.stem,fill='black')
    canvas.save(p/f'contact_{start//8}.png')
