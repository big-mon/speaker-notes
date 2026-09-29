"""Exercise the actual export JavaScript without downloads or a browser.

DOM stubs check state/error handling; actual browser integration is separate.
Uses an existing Node executable only, never installs a test dependency.
"""
import json
import os
from pathlib import Path
import shutil
import subprocess
import unittest

from scene_transcript import REVIEW_HTML


NODE = (os.environ.get('SCENE_REVIEW_NODE') or shutil.which('node') or
        str(Path.home()/'.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node'))
EXPORT_SCRIPT = REVIEW_HTML.split('// Review JSON export:', 1)[1].split('</script>', 1)[0]
# Keep the first comment line a comment after selecting this section.
EXPORT_SCRIPT = '// Review JSON export:' + EXPORT_SCRIPT
HARNESS = r'''
const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict');
const input=JSON.parse(fs.readFileSync(0,'utf8'));
const elements=new Map(),state={clicks:0,clipboard:null,appended:0,removed:0,timers:[],revoked:[]};
function element(id){if(!elements.has(id))elements.set(id,{value:'',hidden:true,textContent:'',
 setAttribute(k,v){this[k]=v},focus(){state.focus=id},select(){state.selected=id}});return elements.get(id)}
const review={schema_version:1,source_artifact_id:'sha256:original',reviewer:'',listened_ranges:[],
 reviewed_at:null,rows:{'0':{text:'原文。',note:'訂正のメモ',content:'review_required',
 corrections:{text:'修正後の本文。'}}}};
const document={getElementById:element,body:{append(a){a.connected=true;state.appended++}},
 createElement(){return {download:'',click(){assert.equal(this.connected,true);state.clicks++},
 remove(){state.removed++}}}};
const URL={createObjectURL(blob){state.blob=blob;return 'blob:review'},revokeObjectURL(u){state.revoked.push(u)}};
const navigator={clipboard:{async writeText(text){state.clipboard=text}}};
const context=vm.createContext({assert,state,review,document,URL,navigator,$:element,
 data:{source_window:{start:0,end:180}},Blob:class{constructor(parts){this.text=parts.join('')}},
 setTimeout(fn,ms){state.timers.push({fn,ms})}});
vm.runInContext(input.script,context);
(async()=>{await vm.runInContext('(async()=>{'+input.test+'})()',context)})().catch(e=>{console.error(e);process.exitCode=1});
'''


@unittest.skipUnless(Path(NODE).is_file(), 'Existing Node executable unavailable')
class ReviewExportTests(unittest.TestCase):
    def js(self, body, valid=True):
        setup = "$('reviewer').value='自分';$('listened').value='0-180';" if valid else ''
        process = subprocess.run([NODE, '-e', HARNESS], input=json.dumps({
            'script': EXPORT_SCRIPT, 'test': setup + body}), text=True, capture_output=True)
        self.assertEqual(process.returncode, 0, process.stderr)

    def test_required_fields_are_visible_errors_not_download_errors(self):
        self.js("""
          $('download').onclick();
          assert.match($('export-status').textContent,/入力不足/);
          assert.equal(state.focus,'reviewer');assert.equal(state.clicks,0);
          assert.equal($('json-fallback').hidden,true);
          $('reviewer').value='自分';$('show-json').onclick();
          assert.match($('export-status').textContent,/再生しただけでは記録されません/);
          assert.equal(state.focus,'listened');assert.equal(state.clicks,0);
        """, valid=False)

    def test_invalid_ranges_do_not_produce_a_stale_export(self):
        self.js("""
          $('show-json').onclick();assert.equal($('json-fallback').hidden,false);
          $('listened').value='180-0';$('download').onclick();
          assert.match($('export-status').textContent,/入力形式のエラー/);
          assert.equal($('review-json').value,'');assert.equal($('json-fallback').hidden,true);
          $('listened').value='0-181';$('show-json').onclick();
          assert.match($('export-status').textContent,/入力範囲のエラー/);
          assert.equal(state.clicks,0);
        """)

    def test_display_preserves_unicode_review_and_corrections_without_download(self):
        self.js("""
          $('listened').value='0-90\\n90〜180';$('show-json').onclick();
          const exported=JSON.parse($('review-json').value);
          assert.equal(exported.source_artifact_id,'sha256:original');
          assert.equal(exported.rows['0'].text,'原文。');
          assert.equal(exported.rows['0'].corrections.text,'修正後の本文。');
          assert.equal(exported.rows['0'].content,'review_required');
          assert.equal(exported.listened_ranges.length,2);assert.ok(exported.reviewed_at);
          assert.equal(state.clicks,0);assert.equal($('json-fallback').hidden,false);
          assert.match($('export-status').textContent,/まだファイルには保存していません/);
        """)

    def test_unsupported_download_still_exposes_json(self):
        self.js("""
          URL.createObjectURL=()=>{throw new Error('not supported')};$('download').onclick();
          assert.match($('export-status').textContent,/ダウンロードを開始できませんでした/);
          assert.match($('export-status').textContent,/not supported/);
          assert.equal(JSON.parse($('review-json').value).reviewer,'自分');
          assert.equal($('json-fallback').hidden,false);assert.equal(state.clicks,0);
        """)

    def test_download_request_never_claims_the_file_was_saved(self):
        self.js("""
          $('download').onclick();assert.equal(state.clicks,1);assert.equal(state.appended,1);
          assert.equal(state.removed,1);assert.equal(state.blob.text,$('review-json').value);
          assert.match($('export-status').textContent,/保存の完了はこの画面から確認できません/);
          assert.equal(state.timers.length,1);assert.ok(state.timers[0].ms>1000);
          state.timers[0].fn();assert.equal(state.revoked[0],'blob:review');
        """)

    def test_clipboard_failure_selects_json_and_explains_manual_copy(self):
        self.js("""
          navigator.clipboard.writeText=async()=>{throw new Error('permission denied')};
          await $('copy-json').onclick();
          assert.equal(state.selected,'review-json');assert.equal(state.focus,'review-json');
          assert.match($('export-status').textContent,/⌘C/);
          assert.match($('export-status').textContent,/permission denied/);
          assert.equal($('json-fallback').hidden,false);
          delete navigator.clipboard;await $('copy-json').onclick();
          assert.match($('export-status').textContent,/自動コピーできませんでした/);
        """)

    def test_copy_and_manual_select_regenerate_latest_input(self):
        self.js("""
          $('show-json').onclick();review.rows['0'].note='変更したメモ';
          await $('copy-json').onclick();
          assert.equal(JSON.parse(state.clipboard).rows['0'].note,'変更したメモ');
          assert.match($('export-status').textContent,/クリップボードにコピーしました/);
          $('select-json').onclick();assert.equal(state.selected,'review-json');
          assert.equal(state.clicks,0);
        """)


if __name__ == '__main__':
    unittest.main()
