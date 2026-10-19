"""Provided HW08 evaluation support. Leave unchanged; do not submit.

Contains the public scoring rules and execution checks, not a detector.
"""
import ast
import contextlib
import importlib.util
import itertools
import os
from pathlib import Path
import cv2
import numpy as np

HELPER_SOURCE = 'def intersect_lines(line1, line2):\n    """Intersect two Hough (r, theta) lines; return (x,y), or None if parallel."""\n    r1, t1 = line1\n    r2, t2 = line2\n    matrix = np.array([[np.cos(t1), np.sin(t1)],\n                       [np.cos(t2), np.sin(t2)]], dtype=float)\n    if abs(np.linalg.det(matrix)) < 1e-6:\n        return None\n    return np.linalg.solve(matrix, np.array([r1, r2], dtype=float))'
BANDS = np.array([8.,16.,32.,48.])
CREDITS = np.array([1.,.75,.5,.25,0.])

class SubmissionError(Exception):
    pass


def corner_score(prediction, reference):
    """Maximize one-to-one corner credit; distance breaks score ties."""
    if not isinstance(prediction,np.ndarray):
        raise SubmissionError('Return a NumPy array with shape (4, 2).')
    if prediction.shape == (0,2):
        return {'accuracy':0.,'bands':[0,0,0,0,4],'errors':None,'order':None,
                'message':'No document detected.'}
    if prediction.shape != (4,2):
        raise SubmissionError('Return exactly four corners with shape (4, 2), or an empty (0, 2) array.')
    if prediction.dtype.kind not in 'fiu' or not np.isfinite(prediction).all():
        raise SubmissionError('Corner coordinates must be finite real numbers.')
    ref=np.asarray(reference,dtype=float)
    if ref.shape != (4,2) or not np.isfinite(ref).all():
        raise ValueError('Invalid reference annotations.')
    best=None
    for order in itertools.permutations(range(4)):
        errors=np.linalg.norm(prediction[list(order)].astype(float)-ref,axis=1)
        bins=np.searchsorted(BANDS,errors,side='left')
        credit=float(CREDITS[bins].mean())
        key=(credit,-float(errors.sum()))
        if best is None or key>best[0]:best=(key,errors,bins,order)
    key,errors,bins,order=best
    return {'accuracy':key[0],'bands':np.bincount(bins,minlength=5).tolist(),
            'errors':errors.tolist(),'order':list(order),'message':''}


def band_summary(result):
    names=['<=8 px','>8-16 px','>16-32 px','>32-48 px','>48 px / missing']
    return 'Corner counts: '+ '; '.join(f'{name}: {count}' for name,count in zip(names,result['bands']))+'.'


def load_and_run(path,image):
    """Execute trusted local student code; Gradescope calls this in isolation.

    Instrumentation verifies calls, not semantic dependence on their outputs.
    Deliberately unused/dummy calls remain subject to code review.
    """
    path=Path(path)
    if not path.is_file():raise SubmissionError('Submit hw08.py.')
    try:tree=ast.parse(path.read_text(encoding='utf-8-sig'))
    except SyntaxError as e:raise SubmissionError(f'Python syntax error on line {e.lineno}.') from None
    expected=ast.dump(ast.parse(HELPER_SOURCE).body[0],include_attributes=False)
    helpers=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='intersect_lines']
    if len(helpers)!=1 or ast.dump(helpers[0],include_attributes=False)!=expected:
        raise SubmissionError('Keep the provided intersect_lines helper unchanged in hw08.py.')
    counts={'hough':0,'intersections':0}
    original_hough=cv2.HoughLines; original_probabilistic=cv2.HoughLinesP
    def counted(*a,**kw):counts['hough']+=1;return original_hough(*a,**kw)
    def prohibited(*a,**kw):raise SubmissionError('Use cv2.HoughLines; HoughLinesP is not allowed.')
    cv2.HoughLines=counted;cv2.HoughLinesP=prohibited
    try:
        spec=importlib.util.spec_from_file_location('hw08_submission',path)
        module=importlib.util.module_from_spec(spec)
        # Printed content and arbitrary exception messages never become hidden-test feedback.
        with open(os.devnull,'w') as null,contextlib.redirect_stdout(null),contextlib.redirect_stderr(null):
            spec.loader.exec_module(module)
            if not callable(getattr(module,'detect_document',None)):
                raise SubmissionError('Missing detect_document(image).')
            supplied=module.intersect_lines
            def intersection(*a,**kw):counts['intersections']+=1;return supplied(*a,**kw)
            module.intersect_lines=intersection
            counts['hough']=0
            before=image.copy()
            prediction=module.detect_document(image)
        if not np.array_equal(image,before):raise SubmissionError('Do not modify the input image.')
        if counts['hough']==0:raise SubmissionError('detect_document must call cv2.HoughLines.')
        empty=isinstance(prediction,np.ndarray) and prediction.shape==(0,2)
        if not empty and counts['intersections']<4:
            raise SubmissionError('Compute the four returned corners using the provided intersect_lines helper.')
        if not isinstance(prediction,np.ndarray) or prediction.shape not in ((4,2),(0,2)):
            raise SubmissionError('Return a NumPy array of shape (4, 2), or an empty (0, 2) array.')
        if prediction.dtype.kind not in 'fiu' or not np.isfinite(prediction).all():
            raise SubmissionError('Corner coordinates must be finite real numbers.')
        return prediction,counts
    except SubmissionError:raise
    except Exception as e:
        raise SubmissionError('Student code raised '+type(e).__name__+'. Run evaluate.py locally to debug.') from None
    finally:
        cv2.HoughLines=original_hough;cv2.HoughLinesP=original_probabilistic
