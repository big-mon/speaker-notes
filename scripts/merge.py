"""Time-overlap join. Text is never invented, dropped or copied to multiple speakers."""
import collections, copy, hashlib, json, math, pathlib

WINDOWS=[{'name':'冒頭','start':0,'end':120},{'name':'中盤','start':1290,'end':1410},{'name':'終盤','start':2580,'end':2700}]
def sha256(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def stamp(t):
 ms=round(t*1000);return f'{ms//60000:02}:{ms//1000%60:02}.{ms%1000:03}'
def export(document,path):
 path=pathlib.Path(path);path.mkdir(parents=True,exist_ok=True)
 (path/'transcript.json').write_text(json.dumps(document,ensure_ascii=False,indent=2))
 parts=[]
 for segment in document['segments']:
  label=' / '.join(document['speaker_names'].get(x,x) for x in segment['speakers']) or '話者不明'
  if segment['state']=='mixed':label+='（主話者・一部未分離）'
  entry=f'[{stamp(segment["start"])}–{stamp(segment["end"])}] '+label+'\n'+segment['text']
  if segment.get('annotations'):
   entry+='\n\n区間内の話者情報（本文の発言者は未分離）: '+ ' / '.join(f'{stamp(a["start"])}–{stamp(a["end"])} '+(','.join(a['speakers']) or '不明') for a in segment['annotations'])
  parts.append(entry)
 text='\n\n'.join(parts)
 (path/'transcript.txt').write_text(text)
 (path/'transcript.md').write_text('# 話者ラベル付き文字起こし\n\n未校正。複数話者の行は同時発話候補で、各人の発言に分解したものではありません。\n\n'+text+'\n')

def _lexical_segments(document, asr, lexical, diar):
 """Build reading rows from pre-attribution buckets; keep native audit intact."""
 from lexical_speaker_buckets import build_buckets
 buckets=build_buckets(asr,lexical,diar,collar=False)
 rows=[]
 for bucket in buckets['buckets']:
  a,b=bucket['start'],bucket['end'];fallback=not bucket['timing_safe']
  reasons=list(bucket['reasons'])
  if fallback:
   parent=asr[bucket['asr_segment']];a,b=parent.get('start'),parent.get('end')
   if not all(isinstance(t,(int,float)) and not isinstance(t,bool) and math.isfinite(t) for t in (a,b)) or not 0<=a<=b:
    raise ValueError('Invalid native bucket timing also lacks a valid parent ASR playback extent')
   reasons.append('parent_asr_extent_for_playback_only; bucket_timing_unresolved')
  speakers=[document['speaker_mapping'][s] for s in bucket['speakers']]
  state=bucket['state']
  # Unknown or overlapping buckets never inherit a neighboring single speaker.
  row={'start':a,'end':b,'text':bucket['text'],'speakers':speakers,'state':state,
       'asr_segment':bucket['asr_segment'],
       'unit_start':bucket['native_unit_indices'][0],'unit_end':bucket['native_unit_indices'][-1]+1,
       'bucket_ids':[bucket['id']],'timing_fallback':fallback,
       'time_kind':'parent_asr_playback_envelope' if fallback else 'native_bucket_envelope',
       'reasons':reasons,'attribution_status':'speaker_activity_candidate_not_verified_word_ownership',
       'human_verified':False}
  if (rows and not fallback and not rows[-1]['timing_fallback']
      and rows[-1]['speakers']==speakers and rows[-1]['state']==state
      and rows[-1]['asr_segment']==row['asr_segment']
      and a>=rows[-1]['start'] and b>=rows[-1]['end']
      and a-rows[-1]['end']<1.5 and b-rows[-1]['start']<=18):
   previous=rows[-1];previous['end']=b;previous['text']+=row['text'];previous['unit_end']=row['unit_end']
   previous['bucket_ids']+=row['bucket_ids']
   previous['reasons']=list(dict.fromkeys(previous['reasons']+reasons))
  else:rows.append(dict(row,id=str(len(rows))))
 assert ''.join(r['text'] for r in rows)==buckets['original_text']==''.join(s['text'] for s in asr)
 result=dict(document,segments=rows,lexical_tokens=copy.deepcopy(lexical),lexical_buckets=buckets)
 result['merge']={'algorithm':'lexical buckets before temporal speaker attribution',
                  'lexical_tokenizer':'Apple NaturalLanguage NLTokenizer(.word), Japanese',
                  'settings':buckets['settings'],'native_units_algorithm':document['merge'],
                  'native_units':'unchanged legacy per-run attribution, retained as audit evidence',
                  'speaker_ids':'raw IDs in lexical_buckets; first-appearance A/B in displayed segments',
                  'row_grouping':'adjacent equal attribution only; same ASR result; gap <1.5s and extent <=18s',
                  'collar_enabled':False,'silence_snap_enabled':False,'short_response_absorption':False,
                  'neighbor_speaker_inheritance':False,'coverage_is_confidence':False}
 result['validation']=dict(document['validation'],lexical_text_preserved=True,
                           lexical_bucket_count=len(buckets['buckets']),reading_row_count=len(rows),
                           lexical_unknown_buckets=sum(b['state']=='unknown' for b in buckets['buckets']),
                           lexical_overlap_buckets=sum(b['state']=='overlap' for b in buckets['buckets']),
                           native_unit_statistics_scope='unchanged legacy per-run audit; not displayed bucket counts')
 return result

def merge(asr,diar,input_path,engine,*,lexical=None):
 spans=sorted(diar['segments'],key=lambda x:(x['start'],x['end'],str(x['speaker'])))
 ids={}
 for s in spans:
  assert math.isfinite(s['start']) and math.isfinite(s['end']) and 0<=s['start']<s['end']
  key=str(s['speaker'])
  if key not in ids:ids[key]=chr(65+len(ids))
 spans=[dict(s,speaker=ids[str(s['speaker'])]) for s in spans]
 tokens=[];active=[];cursor=0
 for si,s in enumerate(asr):
  assert ''.join(u['text'] for u in s['units'])==s['text']
  for ui,u in enumerate(s['units']):
   a,b=u['start'],u['end']; labels=[];reason=None
   if a is None or b is None: a,b=s['start'],s['end'];reason='missing_audio_time_range'
   else:
    assert math.isfinite(a) and math.isfinite(b) and 0<=a<=b
    while cursor<len(spans) and spans[cursor]['start']<b:
     active.append(spans[cursor]);cursor+=1
    active=[d for d in active if d['end']>a]
    dur=b-a
    if dur>0:
     amounts=collections.defaultdict(float)
     for d in active:amounts[d['speaker']]+=max(0,min(b,d['end'])-max(a,d['start']))
     labels=sorted(k for k,v in amounts.items() if v/dur>=0.65)
     if not labels:reason='boundary_or_no_speech'
     elif len(labels)>1:reason='overlap_unseparated_text'
    else:reason='zero_duration'
   tokens.append({'start':a,'end':b,'text':u['text'],'speakers':labels,'state':'unknown' if not labels else ('overlap' if len(labels)>1 else 'single'),'reason':reason,'asr_segment':si,'asr_unit':ui})
 rows=[]
 for i,t in enumerate(tokens):
  if rows and rows[-1]['speakers']==t['speakers'] and rows[-1]['state']==t['state'] and t['start']-rows[-1]['end']<1.5 and t['end']-rows[-1]['start']<=18 and rows[-1]['asr_segment']==t['asr_segment']:
   rows[-1]['end']=max(rows[-1]['end'],t['end']);rows[-1]['text']+=t['text'];rows[-1]['unit_end']=i+1
  else:rows.append(dict(t,id=str(len(rows)),unit_start=i,unit_end=i+1))
 assert ''.join(x['text'] for x in rows)==''.join(s['text'] for s in asr)
 document={'schema_version':1,'versions':json.loads((pathlib.Path(__file__).resolve().parents[1]/'config/runtime.json').read_text()),'input':{'path':str(pathlib.Path(input_path).resolve()),'sha256':sha256(input_path),'bytes':pathlib.Path(input_path).stat().st_size},'engine':engine,'asr':{'engine':'Apple SpeechAnalyzer / SpeechTranscriber','locale':'ja-JP','attribute':'audioTimeRange','unit':'attributed string run; not assumed to be a word'},'merge':{'algorithm':'temporal overlap per attributed run','minimum_coverage':0.65,'coverage_is_confidence':False},'speaker_names':{v:'話者'+v for v in ids.values()},'speaker_mapping':ids,'diarization':spans,'units':tokens,'segments':rows,'review':{'windows':WINDOWS,'human_verified':False,'notes':[]},'validation':{'asr_text_preserved':True,'unit_count':len(tokens),'unknown_units':sum(not t['speakers'] for t in tokens),'overlap_units':sum(len(t['speakers'])>1 for t in tokens),'accuracy_verified':False}}
 return _lexical_segments(document,asr,lexical,diar) if lexical is not None else document

if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('asr');p.add_argument('diarization');p.add_argument('input');p.add_argument('output');p.add_argument('--lexical-json');a=p.parse_args()
 diar=json.load(open(a.diarization));lexical=json.load(open(a.lexical_json)) if a.lexical_json else None
 doc=merge(json.load(open(a.asr)),diar,a.input,{k:v for k,v in diar.items() if k!='segments'},lexical=lexical)
 export(doc,a.output)
 print(json.dumps(doc['validation'],ensure_ascii=False))
