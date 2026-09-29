import pathlib,tempfile,unittest
from merge import merge
class MergeTests(unittest.TestCase):
 def test_handover_overlap_unknown_and_text_conservation(self):
  with tempfile.NamedTemporaryFile() as f:
   f.write(b'audio');f.flush()
   a=[{'start':0,'end':4,'text':'あいうえ','units':[{'start':i,'end':i+1,'text':c} for i,c in enumerate('あいうえ')]}]
   d={'segments':[{'start':0,'end':2,'speaker':'x'},{'start':1,'end':3,'speaker':'y'}]}
   r=merge(a,d,f.name,{})
   self.assertEqual([u['speakers'] for u in r['units']],[['A'],['A','B'],['B'],[]])
   self.assertEqual(''.join(s['text'] for s in r['segments']),'あいうえ')
 def test_boundary_is_not_forced(self):
  with tempfile.NamedTemporaryFile() as f:
   a=[{'start':0,'end':1,'text':'字','units':[{'start':0,'end':1,'text':'字'}]}]
   d={'segments':[{'start':0,'end':.5,'speaker':'x'},{'start':.5,'end':1,'speaker':'y'}]}
   self.assertEqual(merge(a,d,f.name,{})['units'][0]['state'],'unknown')
if __name__=='__main__':unittest.main()
