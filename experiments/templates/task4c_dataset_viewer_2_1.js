/* Dataset v2 viewer controls: per-instance assignment filter, fold / assignment details, split summary. */
window.installDatasetViewer=function(){
 const KIND=['在标注单元内','GT 包围盒内','候选分量重叠','最近实例'];
 // Region filter by assigned instance (default) instead of the GT bounding box.
 const regionLabel=$('region').closest('label');
 const box=document.createElement('label');box.className='check';box.innerHTML='<input id="byAssignment" type="checkbox" checked>区域按归属实例筛选（含划归该实例的负样本）';
 regionLabel.after(box);
 const oldGtBounds=gtBounds;
 gtBounds=function(instance){
  const b=oldGtBounds(instance);if(instance==='all'||!$('byAssignment').checked)return b;
  const s=current(),target=+instance;
  for(let i=0;i<s.selected;i++){if(s.instance[i]!==target)continue;const c=s.center[i];for(let a=0;a<3;a++){b[2*a]=Math.min(b[2*a],c[a]);b[2*a+1]=Math.max(b[2*a+1],c[a]);}}
  return b;
 };
 const oldMode=modeIndices;
 modeIndices=function(split,indices,headsOnly){
  let ids=oldMode(split,indices,headsOnly);const region=$('region').value;
  if(region!=='all'&&$('byAssignment').checked)ids=ids.filter(i=>split.instance[i]===+region);
  return ids;
 };
 $('byAssignment').onchange=()=>{state.revision++;update(true);};
 const oldRegions=refreshRegions;
 refreshRegions=function(){
  oldRegions();const s=current(),count=new Map();
  for(let i=0;i<s.selected;i++){const k=s.instance[i],c=count.get(k)||{n:0,pos:0};c.n++;if(s.labels[i]===1)c.pos++;count.set(k,c);}
  for(const o of [...$('region').options]){if(o.value==='all')continue;const c=count.get(+o.value);if(!c){o.remove();continue;}o.textContent=`实例 ${o.value} · ${c.n} 样本（hairpin ${c.pos}）`;}
  if(![...$('region').options].some(o=>o.value===$('region').value))$('region').value='all';
 };
 // Summary panel above the plot.
 const panel=document.createElement('section');panel.id='datasetPanel';panel.innerHTML='<strong id="datasetSummary"></strong><p id="datasetDetail"></p>';
 document.querySelector('article').insertBefore(panel,$('plot'));
 function summary(){
  const s=current(),region=$('region').value,byAssign=$('byAssignment').checked,visible=state.visibleIndices||[];
  const scope=region==='all'?Array.from({length:s.selected},(_,i)=>i):(byAssign?Array.from({length:s.selected},(_,i)=>i).filter(i=>s.instance[i]===+region):null);
  const splitName=state.split==='test'?'测试折 0':'训练折 1–4';
  let head=`${splitName}：${s.selected.toLocaleString()} 样本，hairpin ${s.labels.reduce((a,v)=>a+v,0).toLocaleString()}`;
  if(state.split==='train'&&s.validation)head+=`，其中 FMT 2.1 验证子集 ${s.validation.reduce((a,v)=>a+(v?1:0),0).toLocaleString()}`;
  head+=`；当前显示 ${visible.length}`;
  let detail='';
  if(scope&&region!=='all'){
   const kinds=[0,0,0,0],pos=[0,0,0,0];for(const i of scope){kinds[s.assignment_kind[i]]++;if(s.labels[i]===1)pos[s.assignment_kind[i]]++;}
   detail=`实例 ${region} 归属 ${scope.length} 样本：`+kinds.map((n,k)=>n?`${KIND[k]} ${n}${pos[k]?`（hairpin ${pos[k]}）`:''}`:null).filter(Boolean).join('，')+'。';
   if(state.model!=='labels'&&s.fmt_test_prediction_available){const p=s.probabilities[state.model];let tp=0,fn=0,fp=0;for(const i of scope){if(s.labels[i]===1){if(p[i]>=.5)tp++;else fn++;}else if(p[i]>=.5)fp++;}detail+=` FMT 2.1：正确识别 ${tp}，漏检 ${fn}，误报 ${fp}。`;}
  }else if(region!=='all')detail='按 GT 包围盒附近筛选（未勾选归属筛选）。';
  else{const kinds=[0,0,0,0];for(let i=0;i<s.selected;i++)kinds[s.assignment_kind[i]]++;detail='归属方式：'+kinds.map((n,k)=>`${KIND[k]} ${n.toLocaleString()}`).join('，')+'。滑块控制可见样本数；选择一个实例可查看划归它的全部样本。';}
  $('datasetSummary').textContent=head;$('datasetDetail').textContent=detail;
 }
 const oldRender=render;
 render=async function(reset=false){
  refreshRegions();                       // the instance list depends on the split; drop a selection that has no samples here
  await oldRender(reset);
  const model=DATA.models.find(m=>m.id===state.model),s=current();
  if(state.model==='labels'){$('metricLabel').textContent='着色';$('f1').textContent='标签';}
  else if(!s.fmt_test_prediction_available){$('metricLabel').textContent='FMT 2.1';$('f1').textContent='无训练折预测';}
  else{$('metricLabel').textContent='FMT 2.1 测试折 F1（行级）';$('f1').textContent=model.metrics.test.per_flow[state.flow].f1.toFixed(4);}
  summary();
 };
 const oldDetails=selectedDetails;
 selectedDetails=function(id){
  oldDetails(id);const s=current();
  let extra='<br>折 '+s.fold[id]+(s.validation&&s.validation[id]?'（FMT 2.1 验证子集）':'')+'<br>归属实例 '+s.instance[id]+'（'+KIND[s.assignment_kind[id]]+'）'+'<br>候选分量 '+s.head_component[id]+'（'+s.component_size[id]+' 格点）';
  if(s.fmt_test_prediction_available&&s.probabilities.fmt21)extra+='<br>FMT 2.1 P(Hairpin)（三长度均值）= '+s.probabilities.fmt21[id].toFixed(3);
  if(window.candidateDetail)extra+=candidateDetail(s,id);
  $('detail').innerHTML=$('detail').innerHTML.replace('尺度 0','三条长度').replace(/<br>预测：[^<]*/,state.model==='labels'?'':'<br>预测：'+(s.probabilities[state.model][id]>=.5?'Hairpin':'Non-hairpin'))+extra;
 };
 // The FMT 2.1 model has predictions only on the test fold: fall back to labels on the training fold.
 $('split').addEventListener('change',()=>{if(state.split==='train'&&state.model!=='labels'){state.model='labels';$('model').value='labels';}});
};
