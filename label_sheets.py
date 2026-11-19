"""Build numbered-grid sheets for manual labelling: one reCAPTCHA grid per sheet.

Tiles are numbered row-major so a label is just the list of matching indices.
"""
import sys, json, glob, os, re
import cv2, numpy as np

def clean_task(t):
    t=re.sub(r"\s+"," ",(t or "").replace("\n"," "))
    t=re.sub(r"Click verify.*$","",t).strip()
    t=re.sub(r"If there are none.*$","",t).strip()
    return t

def build(jpg, js, out, scale=1.0):
    img=cv2.imread(jpg)
    if img is None: return False
    meta=json.load(open(js))
    rows,cols=meta.get("rows",3),meta.get("cols",3)
    h,w=img.shape[:2]; th,tw=h//rows,w//cols
    canvas=img.copy()
    for i in range(rows*cols):
        r,c=divmod(i,cols)
        y0,x0=r*th,c*tw
        cv2.rectangle(canvas,(x0,y0),(x0+tw,y0+th),(0,0,0),2)
        cv2.putText(canvas,str(i),(x0+6,y0+26),cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,0,0),4)
        cv2.putText(canvas,str(i),(x0+6,y0+26),cv2.FONT_HERSHEY_SIMPLEX,0.9,(255,255,255),2)
    bar=np.full((40,canvas.shape[1],3),255,np.uint8)
    cv2.putText(bar,clean_task(meta.get("task",""))[:70],(8,28),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,0),2)
    sheet=np.vstack([bar,canvas])
    if scale!=1.0: sheet=cv2.resize(sheet,None,fx=scale,fy=scale)
    cv2.imwrite(out,sheet,[cv2.IMWRITE_JPEG_QUALITY,88])
    return True

if __name__=="__main__":
    files=sorted(glob.glob("capture/*.json"))
    want=sys.argv[1] if len(sys.argv)>1 else None
    n=0
    os.makedirs("label_sheets",exist_ok=True)
    for js in files:
        meta=json.load(open(js))
        t=clean_task(meta.get("task","")).lower()
        if want and want not in t: continue
        jpg=js.replace(".json",".jpg")
        if not os.path.exists(jpg): continue
        out=f"label_sheets/{os.path.basename(js).replace('.json','')}.jpg"
        if build(jpg,js,out): n+=1
        if n>=int(sys.argv[2]) if len(sys.argv)>2 else False: break
    print("sheets built:", n)
