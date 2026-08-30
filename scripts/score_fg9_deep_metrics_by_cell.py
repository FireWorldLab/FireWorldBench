import json, re
from pathlib import Path
from collections import defaultdict

ROOT=Path('/root/autodl-tmp/FireWorldBench/2/5.项目实现/v1')
BASE=ROOT/'artifacts/validation/fg9_dual_model_n40_final_20260812'
RUNS={
 'main_grok':('模拟数据主集',['main_gate_C/gold.jsonl','main_gate_B/gold.jsonl']),
 'main_claude':('模拟数据主集',['main_gate_C/gold.jsonl','main_gate_B/gold.jsonl']),
 'c06_grok':('真实事件模拟',['c06_candidate/build_A/gold.jsonl','c06_gate_A/gold.jsonl','c06_pool80/build_A/gold.jsonl','c06_pool80/build_B/gold.jsonl']),
 'c06_claude':('真实事件模拟',['c06_candidate/build_A/gold.jsonl','c06_gate_A/gold.jsonl','c06_pool80/build_A/gold.jsonl','c06_pool80/build_B/gold.jsonl'])}
VARS={'temperature','visibility','soot','flow','velocity','risk','fire','smoke','sensor','ventilation'}
def text(v):
 if isinstance(v,str):return v.lower()
 if isinstance(v,list):return ' '.join(text(x) for x in v)
 if isinstance(v,dict):return ' '.join(text(x) for x in v.values())
 return str(v).lower()
def toks(s):return set(re.findall(r'[a-z][a-z0-9_]+',s.lower()))
def ece(rows,bins=10):
 if not rows:return None
 z=0
 for i in range(bins):
  b=[r for r in rows if i/bins<=r[0]<(i+1)/bins or (i==bins-1 and r[0]==1)]
  if b:z+=len(b)/len(rows)*abs(sum(x[0] for x in b)/len(b)-sum(x[1] for x in b)/len(b))
 return z
def main():
 out=[]
 for run,(dataset,grels) in RUNS.items():
  gold={}
  for grel in grels:
   for x in map(json.loads,(BASE/grel).open(encoding='utf-8')):
    if x.get('question_type')=='open':gold[x['qa_id']]=x
  cells=defaultdict(list)
  for x in map(json.loads,(BASE/'runs'/run/'merged_responses.jsonl').open(encoding='utf-8')):
   if x.get('question_type')!='open' or x.get('qa_id') not in gold:continue
   g=gold[x['qa_id']]; raw=x.get('response'); r=raw if isinstance(raw,dict) else {}; pred=r.get('prediction') or {}; free=text([r.get('conclusion',''),r.get('evidence',[]),r.get('mechanism','')]); ft=toks(free)
   anchors=[]
   for c in g.get('evidence_claims') or []:
    obs=(c.get('observation') or '').lower(); var=next((v for v in VARS if v in obs),None); anchors.append((str(c.get('region','')).lower(),var))
   rec=sum(((reg in free)+(var in free if var else 0))/(2 if var else 1) for reg,var in anchors)/max(1,len(anchors))
   claims=set(z.lower() for z in re.findall(r'\bR\d+\b',free,re.I))|(ft&VARS); supported={z for a in anchors for z in a if z}; prec=len(claims&supported)/len(claims) if claims else 0
   gt=toks(' '.join(c.get('statement','') for c in g.get('mechanism_claims') or []))-{'the','and','with','that','this','from','into','first'}; mech=toks(text(r.get('mechanism',''))); align=len(gt&mech)/len(gt) if gt else 0
   cons=sum(str(v).lower() in free for v in pred.values())/max(1,len(pred)); conf=r.get('confidence'); conf=float(conf) if isinstance(conf,(int,float)) else 0; correct=int(pred==g.get('prediction')); valid=int(x.get('status')=='ok')
   cells[(g.get('task'),g.get('track'))].append((rec,prec,align,cons,conf,correct,valid))
  for (task,track),rs in cells.items():
   valid=[r for r in rs if r[6]]; cal=[(r[4],r[5]) for r in valid]
   def avg(i):return sum(r[i] for r in valid)/len(valid) if valid else 0
   out.append({'dataset':dataset,'model':'Grok' if run.endswith('grok') else 'Claude','task':task,'track':track,'cell':f'{task}-{track}','n':len(rs),'valid_n':len(valid),'format_valid_rate':len(valid)/len(rs),'evidence_anchor_recall':avg(0),'evidence_precision_proxy':avg(1),'mechanism_alignment':avg(2),'prediction_explanation_consistency':avg(3),'brier_score':sum((c-y)**2 for c,y in cal)/len(cal) if cal else None,'ece_10bin':ece(cal)})
 out.sort(key=lambda x:(x['dataset'],x['task'],x['track'],x['model']))
 dest=BASE/'final_report_20260813/n40_open_deep_metrics_18cells.json'; dest.write_text(json.dumps({'schema_version':'FWB-FG9-DEEP-CELL-METRICS-v1','open_only':True,'rows':out},ensure_ascii=False,indent=2),encoding='utf-8'); print(dest,len(out))
if __name__=='__main__':main()
