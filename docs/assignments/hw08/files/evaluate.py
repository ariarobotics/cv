"""Provided public evaluator: python evaluate.py [--submission hw08.py]."""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
from hw08_support import load_and_run,corner_score,band_summary,SubmissionError


def main():
    home=Path(__file__).resolve().parent
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--submission',type=Path,default=home/'hw08.py')
    p.add_argument('--output-dir',type=Path,default=Path('results'))
    args=p.parse_args();args.output_dir.mkdir(parents=True,exist_ok=True)
    cv2.setNumThreads(1)
    annotations=json.loads((home/'public_corners.json').read_text())
    results={};tiles=[]
    for name,reference in annotations.items():
        image=cv2.imread(str(home/name))
        if image is None:raise FileNotFoundError(home/name)
        canvas=image.copy();prediction=None
        try:
            prediction,counts=load_and_run(args.submission.resolve(),image)
            result=corner_score(prediction,reference)
            result['required_calls']=counts
            result['prediction']=prediction.tolist()
            message=band_summary(result)
        except SubmissionError as e:
            result={'accuracy':0.,'message':str(e)};message=str(e)
        results[name]=result
        print(f"{name}: {100*result['accuracy']:.2f}% accuracy. {message}")
        for x,y in np.rint(reference).astype(int):cv2.circle(canvas,(x,y),8,(0,200,0),2)
        if prediction is not None and prediction.shape==(4,2):
            center=prediction.mean(axis=0)
            order=np.argsort(np.arctan2(prediction[:,1]-center[1],prediction[:,0]-center[0]))
            drawable=np.rint(np.clip(prediction[order],-10000,10000)).astype(np.int32)
            cv2.polylines(canvas,[drawable],True,(0,200,255),3)
            for x,y in drawable:cv2.drawMarker(canvas,(int(x),int(y)),(0,0,255),cv2.MARKER_CROSS,12,2)
        cv2.imwrite(str(args.output_dir/(Path(name).stem+'_overlay.jpg')),canvas)
        tile=cv2.resize(canvas,(300,400));cv2.rectangle(tile,(0,0),(300,28),(255,255,255),-1)
        cv2.putText(tile,f"{name}: {100*result['accuracy']:.1f}%",(5,20),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,0,0),1)
        tiles.append(tile)
    score=sum(r['accuracy'] for r in results.values())/3
    print(f'Public score: {score:.6f}/1.000000. Green circles: reference; red crosses/yellow lines: prediction.')
    cv2.imwrite(str(args.output_dir/'public_overview.jpg'),np.hstack(tiles))
    (args.output_dir/'public_results.json').write_text(json.dumps({'public_score':score,'images':results},indent=2)+'\n')

if __name__=='__main__':main()
