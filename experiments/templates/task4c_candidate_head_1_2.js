/* Optional step-1 candidate-head surface: lambda2 < threshold and oyf > 0, loaded on demand. */
window.candidateDetail=function(s,id){
 if(!s.center_lambda2)return '';
 const thr=window.TASK4C_CANDIDATE.flows[state.flow].threshold,ok=s.center_step1[id]===1;
 return '<br>中心 λ₂ = '+s.center_lambda2[id].toPrecision(4)+'（阈值 '+thr+'），oyf = '+s.center_oyf[id].toPrecision(4)+'<br>'+(ok?'中心满足 step1 候选条件':'中心不满足 step1 候选条件');
};
window.installCandidateHead=function(){
 const C=window.TASK4C_CANDIDATE,cache=new Map(),waiters=new Map(),requests=new Map();
 const opacityLabel=$('opacity').closest('label');
 const box=document.createElement('div');
 box.innerHTML='<label class="check"><input id="candidate" type="checkbox">显示 candidate head 区域（λ₂ &lt; 阈值 ∧ oyf &gt; 0）</label>'
 +'<label class="control"><span class="row"><span>候选区域不透明度</span><span id="candidateOpacityText">30%</span></span><input id="candidateOpacity" type="range" min="5" max="80" step="1" value="30"></label>'
 +'<label class="control">隐藏连通格点数少于 <input id="candidateMinSize" type="number" min="1" step="1" value="1" style="width:66px"> 的碎片<span class="muted">（仅区域/聚焦视图的原始分辨率网格；全场用简化网格，不过滤）</span></label>'
 +'<div id="candidateHint" class="muted" style="margin:6px 0 10px"></div>';
 opacityLabel.after(box);
 const gtKey=[...document.querySelectorAll('.key')].find(k=>k.textContent.includes('Ground truth'));
 const key=document.createElement('div');key.className='key';key.innerHTML='<i class="swatch" style="height:13px;background:'+C.color+'66;border:1px solid '+C.color+'"></i>Candidate head 区域：λ₂ &lt; 阈值 ∧ oyf &gt; 0';
 gtKey.after(key);
 window.task4cCandidateChunk=(k,payload)=>{const w=waiters.get(k);if(!w)return;
  try{const v={verts:unpack(payload.verts),faces:unpack(payload.faces),sizes:payload.sizes?unpack(payload.sizes):null,bounds:payload.bounds};
   if(v.faces.length%3||v.verts.length%3||(v.sizes&&v.sizes.length*3!==v.faces.length))throw new Error('候选区域网格长度不匹配');cache.set(k,v);w.resolve(v);}
  catch(e){w.reject(e);}finally{waiters.delete(k);}};
 function load(ref){
  if(cache.has(ref.key))return Promise.resolve(cache.get(ref.key));
  if(requests.has(ref.key))return requests.get(ref.key);
  const p=new Promise((resolve,reject)=>{waiters.set(ref.key,{resolve,reject});const sc=document.createElement('script');sc.src=ref.path;
   sc.onload=()=>{sc.remove();if(waiters.has(ref.key)){waiters.delete(ref.key);reject(new Error('候选区域文件没有返回数据'));}};
   sc.onerror=()=>{sc.remove();waiters.delete(ref.key);reject(new Error('无法加载 '+ref.path));};document.head.appendChild(sc);}).finally(()=>requests.delete(ref.key));
  requests.set(ref.key,p);return p;
 }
 const intersects=(b,r)=>[0,1,2].every(a=>b[2*a]<=r[2*a+1]&&b[2*a+1]>=r[2*a]);
 function assemble(chunks,roi,minSize){
  let nv=0,nf=0;for(const c of chunks){nv+=c.verts.length/3;nf+=c.faces.length/3;}
  const x=new Float32Array(nv),y=new Float32Array(nv),z=new Float32Array(nv),i=[],j=[],k=[];let base=0;
  for(const c of chunks){
   const n=c.verts.length/3;for(let p=0;p<n;p++){x[base+p]=c.verts[3*p];y[base+p]=c.verts[3*p+1];z[base+p]=c.verts[3*p+2];}
   for(let f=0;f<c.faces.length/3;f++){
    if(c.sizes&&c.sizes[f]<minSize)continue;
    const a=c.faces[3*f],b=c.faces[3*f+1],d=c.faces[3*f+2];
    if(roi){const cx=(c.verts[3*a]+c.verts[3*b]+c.verts[3*d])/3,cy=(c.verts[3*a+1]+c.verts[3*b+1]+c.verts[3*d+1])/3,cz=(c.verts[3*a+2]+c.verts[3*b+2]+c.verts[3*d+2])/3;
     if(cx<roi[0]||cx>roi[1]||cy<roi[2]||cy>roi[3]||cz<roi[4]||cz>roi[5])continue;}
    i.push(base+a);j.push(base+b);k.push(base+d);
   }
   base+=n;
  }
  return {x,y,z,i,j,k,total:nf};
 }
 function trace(m){return {type:'mesh3d',x:m.x,y:m.y,z:m.z,i:m.i,j:m.j,k:m.k,color:C.color,opacity:+$('candidateOpacity').value/100,hoverinfo:'skip',showlegend:false,name:'Candidate head region',flatshading:false,lighting:{ambient:.7,diffuse:.6,specular:.05,roughness:.9},lightposition:{x:100,y:-200,z:300}};}
 function centerSummary(){
  const s=current(),vis=state.visibleIndices||[];if(!s.center_step1)return '';
  let heads=0,headsOk=0,neg=0,negOk=0;for(const id of vis){if(s.labels[id]===1&&s.center_is_head?.[id]){heads++;if(s.center_step1[id])headsOk++;}else if(s.labels[id]===0){neg++;if(s.center_step1[id])negOk++;}}
  return '当前显示束的中心满足 step1：GT头部 '+headsOk+' / '+heads+'，Non-hairpin '+negOk+' / '+neg+'。';
 }
 const oldRender=render;
 render=async function(reset=false){
  await oldRender(reset);
  const token=state.renderToken;
  if(!$('candidate').checked){$('candidateHint').textContent=centerSummary();return;}
  const flow=C.flows[state.flow],layout=$('plot').layout,roi=['x','y','z'].flatMap(a=>layout.scene[a+'axis'].range);
  const overview=$('region').value==='all'&&!state.focus;
  const refs=overview?[flow.overview]:flow.chunks.filter(c=>intersects(c.bounds,roi));
  $('status').textContent='加载 candidate head 区域（'+refs.length+' 块）…';
  let chunks;try{chunks=await Promise.all(refs.map(load));}catch(e){$('candidateHint').textContent='候选区域加载失败：'+e.message;return;}
  if(token!==state.renderToken)return;
  const minSize=Math.max(1,Math.floor(+$('candidateMinSize').value)||1);
  const m=assemble(chunks,overview?null:roi,overview?1:minSize);
  if(m.i.length)await Plotly.addTraces('plot',[trace(m)]);
  if(token!==state.renderToken)return;
  const st=flow.stats;
  $('candidateHint').textContent=(overview?'全场：简化网格 '+m.i.length.toLocaleString()+' 个三角形（原始 '+st.surface_faces.toLocaleString()+'，约保留 '+Math.round(100*(1-flow.overview.actual_reduction))+'%）。':'当前区域：原始分辨率 '+m.i.length.toLocaleString()+' 个三角形（'+refs.length+' 块）。')
   +' 候选格点占 '+(100*st.candidate_point_fraction).toFixed(2)+'%，26连通分量 '+st.connected_components_26.toLocaleString()+' 个，中位 '+st.component_size_quantiles_50_90_99_100[0]+' 格点。 '+centerSummary();
  $('status').textContent='可旋转、缩放；点击中心点查看单束预测与中心 λ₂ / oyf。';
 };
 $('candidate').onchange=()=>update();
 $('candidateMinSize').onchange=()=>{if($('candidate').checked)update();};
 $('candidateOpacity').oninput=()=>{$('candidateOpacityText').textContent=$('candidateOpacity').value+'%';const data=$('plot').data;if(!data)return;const idx=data.findIndex(t=>t.name==='Candidate head region');if(idx>=0)Plotly.restyle('plot',{opacity:+$('candidateOpacity').value/100},[idx]);};
};
