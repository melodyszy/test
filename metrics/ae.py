# metrics/ae.py
def compute_ae(scores):
    valid_scores = []
    for s in scores:
        if isinstance(s,(list,tuple)) and len(s)==3:
            valid_scores.append(s)
        else:
            valid_scores.append([0.0,0.0,0.0])
    t_err=-1
    for i,(c,_,_) in enumerate(valid_scores):
        if c<0.5:
            t_err=i
            break
    if t_err==-1: return 1.0
    t_rec=-1
    for i in range(t_err,len(valid_scores)):
        if valid_scores[i][0]>=0.8:
            t_rec=i
            break
    if t_rec==-1: return 0.0
    valid=0
    total=0
    for i in range(t_err,t_rec+1):
        if valid_scores[i][0]>=0.5: valid+=1
        total+=1
    return valid/total if total>0 else 0.0