/* Coverage and score-preservation checks for whole-curve intervention display. */
'use strict';
const assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path');
const api=require('../experiments/templates/task4c_all_segments_1_4.js');
const geometry=Array.from({length:2},(_,l)=>Array.from({length:32},(_,j)=>[j,l,j*j]));
const patches=[];
for(let l=0;l<2;l++)for(let a=0;a<=24;a+=4)patches.push({indices:[l,a,Math.min(a+8,31)],margin_delta:a%8===0?(a===0?-2:1):1000,probability:.7});
const synthetic={geometry,patches};
assert.equal(api.indices(synthetic).length,8);
assert.equal(api.limit([synthetic]),2,'Unused overlapping windows must not set the new color range');
const missing={geometry,patches:patches.filter(p=>p.indices.join(':')!=='0:8:16')};
assert.throws(()=>api.indices(missing),/Missing measured/);
const duplicate={geometry,patches:[...patches,patches[0]]};
assert.throws(()=>api.indices(duplicate),/Duplicate measured/);

function check(data,id) {
  const original=JSON.stringify(data),built=api.build(data,data.geometry,id),coverage=data.geometry.map(()=>Array(31).fill(0));
  let cursor=0;
  for(const pi of built.indices) {
    const p=data.patches[pi],[l,a,b]=p.indices;
    for(let j=a;j<=b;j++,cursor++) {
      assert.deepEqual(built.points[cursor],data.geometry[l][j]);
      assert.equal(built.values[cursor],p.margin_delta);
      assert.equal(built.customdata[cursor][3],pi);
      assert.equal(built.customdata[cursor][5],p.probability);
      if(j<b)coverage[l][j]++;
    }
    assert.equal(built.points[cursor],null,'No line/color interpolation across distinct windows');cursor++;
  }
  assert.equal(cursor,built.points.length);
  assert.ok(coverage.every(line=>line.every(count=>count===1)),'Every real edge is displayed exactly once');
  assert.equal(JSON.stringify(data),original,'Display must not mutate scientific data');
  return built.indices.length;
}
check(synthetic,'synthetic');
const folder=process.argv[2]||'outputs/Other_FMT_AnalysisWorkbench_1.4';
const source=path.join(folder,'saliency','data');
let bundles=0,lines=0,segments=0;
for(const file of fs.readdirSync(source).filter(p=>p.endsWith('.json'))) {
  const item=JSON.parse(fs.readFileSync(path.join(source,file),'utf8'));
  const d=item.geometry_and_scores;
  segments+=check(d,item.record.id);bundles++;lines+=d.geometry.length;
}
assert.equal(bundles,400);assert.equal(lines,7244);assert.equal(segments,4*lines);
const report={passed:true,bundles,lines,segments,windows:api.windows,all_31_edges_covered_once:true,exact_geometry_scores_and_probabilities_preserved:true,missing_or_duplicate_windows_rejected:true};
fs.writeFileSync(path.join(folder,'segment_audit.json'),JSON.stringify(report,null,2)+'\n');
console.log(JSON.stringify(report));
