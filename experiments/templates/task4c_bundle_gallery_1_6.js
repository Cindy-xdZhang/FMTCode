/* Actual Couette sample integration, with the original two-flow gallery preserved. */
'use strict';
const gallery14={render,filtered,detail,download:$('download').onclick};
const oldGalleryHeader=document.querySelector('header .muted').textContent;
const oldGalleryFooter=document.querySelector('footer:last-of-type').innerHTML;
const oldAllLabelText=$('label').querySelector('[value="all"]').textContent;
const instanceControl=document.createElement('label');instanceControl.id='couetteInstanceControl';
instanceControl.innerHTML='所属 hairpin 实例<select id="couetteInstance"><option value="all">全部实例</option>'+Array.from({length:6},(_,i)=>`<option value="${i}">instance ${i}</option>`).join('')+'</select>';
$('flow').closest('label').after(instanceControl);instanceControl.hidden=true;
filtered=function(role){const ids=gallery14.filtered(role);return value('flow')==='couette'&&value('couetteInstance')!=='all'?ids.filter(i=>current().splits[role].records[i].instance===+value('couetteInstance')):ids;};
render=async function(){
  const couette=value('flow')==='couette';instanceControl.hidden=!couette;
  $('label').querySelector('[value="all"]').textContent=couette?'全部 · 展示抽样':oldAllLabelText;
  document.querySelector('header .muted').textContent=couette?'Task4-c · Couette 未归一化 curl 实际数据集 · 训练／测试按 hairpin 实例隔离':oldGalleryHeader;
  await gallery14.render();
  if(value('flow')!=='couette'){document.querySelector('footer:last-of-type').innerHTML=oldGalleryFooter;return;}
  const d=current();
  $('sourceNote').innerHTML=`Couette · 增量构建<br>λ₂ 阈值 −0.00125<br>未归一化 curl · RK4<br>参数步长 <b>0.0002</b><br>单方向参数区间（非弧长）<b>0.007 / 0.01 / 0.013</b><br>每方向更新次数 <b>35 / 50 / 65</b><br>h = ${fmt(d.h,11)}<br>初始球半径 3h = ${fmt(3*d.h,9)}<br>完整池 ${fmt(d.full_pool_samples,0)}（正 ${fmt(d.full_pool_positive,0)}）<br>载入索引：训练 ${fmt(d.splits.train.population,0)} / 测试 ${fmt(d.splits.test.population,0)}；均正:负=1:2。`;
  document.querySelector('footer:last-of-type').textContent='Couette 标签：最短线至少17/32点在GT内，且最长线全部原始积分点中含GT内head和leg各至少一点。短线已判负时不再检查最长线。画廊从fold0载入索引按实例／类别抽取、随机补足每集合256束；统计来自该集合完整载入索引，实例筛选只作用于画廊卡片。大量负类只在载入索引中屏蔽，完整池全部保留；邻居从Couette全部有效样本查询。Channel／TBL数据未改。';
};
detail=async function(role,index){
  if(value('flow')!=='couette')return gallery14.detail(role,index);
  const rev=++instanceView.revision;$('detailTitle').textContent='读取 Couette 实际线束及所属实例…';
  $('instanceSummary').classList.remove('error');$('instanceSummary').textContent='读取 GT 表面…';$('detailInfo').textContent='';
  if(!$('detailDialog').open)$('detailDialog').showModal();
  try{
    const b=await bundle(role,index),manifest=await instanceManifest(),meta=manifest.flows.couette.samples[`${role}:${b.r.row}`];
    if(!meta||meta.instance!==b.r.instance||meta.label!==b.r.label)throw Error('Couette 样本与实例信息不一致');
    const mesh=await instanceMesh('couette',b.r.instance);
    if(rev!==instanceView.revision||!$('detailDialog').open)return;
    state.detail=b;Object.assign(instanceView,{bundle:b,mesh,flow:'couette',meta,focus:'instance',camera:null});
    $('showInstance').checked=true;$('detailNeighbors').checked=true;
    $('detailTitle').textContent=`Couette · ${role==='train'?'训练':'测试'} #${b.r.row} · ${b.r.label?'正类':'负类'} · 所属 instance ${b.r.instance}`;
    const owner=meta.seed_gt_instance<0?'种子在 GT 外':`种子在 GT instance ${meta.seed_gt_instance} 内`;
    const longest=meta.longest_evaluated?`最长线全部 ${meta.longest_raw_point_count} 个原始积分点：GT head ${meta.longest_head_count}，GT leg ${meta.longest_leg_count}。`:'最短线已不满足多数GT条件，最长线无需检查，计数未计算。';
    $('instanceSummary').textContent=`${owner}；分组归属 instance ${b.r.instance}（${assignmentNames[b.r.assignment_kind]}）。最短线 ${meta.short_gt_count}/32 点在 GT 内，${meta.assigned_instance_short_points} 点在当前实例。${longest} 正类要求短线 ≥17/32 且最长线含 GT head 和 leg。`;
    let html=`<p>真实数据集 mainExp_Task4C_CouetteDataset_2.1 · 折 ${b.r.fold} · 当前单方向参数区间（非弧长）${current().half_lengths[+value('length')]} / 步长 0.0002 · ${b.k} 邻居 · 初始球内 ${b.r.initial_count} · 实际半径 ${fmt(b.radius,8)} = ${fmt(b.radius/current().h,3)}h。</p><table><tr><th>邻居种子</th><th>标签</th><th>折</th><th>所属实例</th><th>距中心</th></tr>`;
    for(let j=0;j<b.k;j++)html+=`<tr><td>#${b.nb.ids[index][j]}</td><td>${b.nb.labels[index][j]?'正':'负'}</td><td>${b.nb.folds[index][j]}</td><td>${b.nb.instances[index][j]}</td><td>${fmt(Math.hypot(...b.nb.seeds[index][j].map((x,d)=>x-b.r.seed[d])),7)}</td></tr>`;
    $('detailInfo').innerHTML=html+'</table>';await drawInstanceContext(true);
    $('detailDialog').dataset.instanceReady=`couette:${role}:${b.r.row}:${b.r.instance}`;
  }catch(e){$('instanceSummary').classList.add('error');$('instanceSummary').textContent=e.message;console.error(e);}
};
$('download').onclick=()=>{
  if(value('flow')!=='couette')return gallery14.download();
  const b=state.detail;if(!b)return;
  const a=document.createElement('a'),data={dataset:current().dataset_version,source_audit_sha256:current().dataset_audit_sha256,
    flow:'couette',role:b.role,center:b.r,neighbors:b.nb.ids[b.index],physical_curves32:b.raw,
    length:current().half_lengths[+value('length')],ds:.0002,metadata:instanceView.meta};
  a.href=URL.createObjectURL(new Blob([JSON.stringify(data)],{type:'application/json'}));a.download=`couette_${b.role}_${b.r.row}.json`;a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000);
};
$('couetteInstance').onchange=()=>{state.pages={train:0,test:0};render();};
const requestedFlow=new URLSearchParams(location.search).get('flow');
if(['channel','tbl','couette'].includes(requestedFlow))$('flow').value=requestedFlow;
if(state.manifest)render();
