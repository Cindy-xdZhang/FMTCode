/* Report actual GT-head coverage separately from prediction and display limits. */
window.installGTCoverage=function(){
 const panel=document.createElement('section');panel.id='coveragePanel';
 panel.innerHTML='<strong id="coverageSummary"></strong><p id="coverageExplanation"></p><details><summary>逐个GT实例：可用、显示及漏检数量</summary><div class="coverage-scroll"><table><thead><tr><th>GT实例</th><th>可用头部束</th><th>其中新增</th><th>实际显示</th><th>正确识别</th><th>漏检</th></tr></thead><tbody id="coverageRows"></tbody></table></div></details>';
 document.querySelector('article').insertBefore(panel,$('plot'));
 const counts=()=>{
   const s=current(),visible=new Set(state.visibleIndices||[]),rows=new Map(DATA.flows[state.flow].gt.instances.map(id=>[id,{id,available:0,added:0,shown:0,tp:0,fn:0}]));
   for(let i=0;i<s.selected;i++){
     if(s.labels[i]!==1||!s.center_is_head[i])continue;
     const r=rows.get(s.center_gt_instance[i]);if(!r)throw new Error('GT head belongs to an unknown instance');
     r.available++;if(s.mandatory_head?.[i])r.added++;if(visible.has(i))r.shown++;
     if(s.probabilities[state.model][i]>=.5)r.tp++;else r.fn++;
   }
   return Array.from(rows.values());
 };
 const oldRegions=refreshRegions;
 refreshRegions=function(){oldRegions();const byId=new Map(counts().map(r=>[String(r.id),r]));for(const o of $('region').options){const r=byId.get(o.value);if(r)o.textContent=`GT ${r.id} · 头部${r.available}束${r.available?'':'（缺样本）'}`;}};
 const oldMode=modeIndices;
 modeIndices=function(split,indices,headsOnly){const ids=oldMode(split,indices,headsOnly),region=$('region').value;
   return headsOnly&&region!=='all'?ids.filter(i=>split.labels[i]===0||split.center_gt_instance[i]===Number(region)):ids;};
 const oldRender=render;
 render=async function(reset=false){await oldRender(reset);refreshRegions();const rows=counts(),covered=rows.filter(r=>r.available>0).length,shown=rows.filter(r=>r.shown>0).length;
   $('coverageSummary').textContent=`${state.split==='test'?'测试':'训练展示'}头部覆盖：${covered} / ${rows.length} 个GT实例；当前视图显示 ${shown} 个`;
   $('coverageExplanation').textContent=covered===rows.length?'每个GT都有真实头部线束。下表的正确识别与漏检按全部可用头部束统计，不受显示数量影响；数量设为0或关闭类别会隐藏线束。':'有GT缺少符合头部条件的样本。这是数据覆盖缺口，不是模型把它判为Non-hairpin；缺样本实例在下表明确列出。';
   $('coverageRows').replaceChildren();
   for(const r of rows.sort((a,b)=>(a.available>0)-(b.available>0)||a.id-b.id)){
     const tr=document.createElement('tr');if(!r.available)tr.className='missing';const first=document.createElement('td'),button=document.createElement('button');button.textContent='GT '+r.id;
     button.onclick=()=>{$('region').value=String(r.id);$('viewMode').value='heads';state.focus=false;state.selected=null;$('focus').disabled=true;state.revision++;state.counts.hairpin=Infinity;$('hairpin').checked=true;update(true);};first.append(button);tr.append(first);
     for(const v of [r.available,r.added,r.shown,r.tp,r.fn]){const td=document.createElement('td');td.textContent=v;tr.append(td);}$('coverageRows').append(tr);
   }
 };
};
