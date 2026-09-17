/* Visual explanation of the frozen 141-dimensional p35 feature layout. */
'use strict';
window.updateFeatureDiagram = function (data, index, sample) {
  const f = data.features[index], svg = document.getElementById('featurePipeline');
  if (!svg) return;
  const ns = 'http://www.w3.org/2000/svg';
  svg.replaceChildren();
  const node = (tag, attrs, text) => { const e = document.createElementNS(ns, tag); for (const [k,v] of Object.entries(attrs)) e.setAttribute(k, v); if (text !== undefined) e.textContent = text; svg.append(e); return e; };
  const active = f.block === 'direction' ? 1 : f.block === 'center' ? 0 : 2;
  const stroke = '#c56331', pale = '#a4b2bd', dark = '#20465b';
  function box(x,y,w,h,lines,on=false) {
    node('rect',{x,y,width:w,height:h,rx:9,fill:on?'#fff0e4':'#f0f4f7',stroke:on?stroke:'#d9e2e9','stroke-width':on?2:1});
    lines.forEach((line,j)=>node('text',{x:x+w/2,y:y+22+j*20,'text-anchor':'middle',fill:on?dark:'#6d808e','font-size':j?12:14,'font-weight':j?400:600},line));
  }
  function arrow(x1,y1,x2,y2,on=false) {
    const color=on?stroke:pale;
    node('path',{d:`M ${x1} ${y1} L ${x2} ${y2}`,stroke:color,'stroke-width':on?2.5:1.2,fill:'none'});
    const a=Math.atan2(y2-y1,x2-x1),l=7;
    node('path',{d:`M ${x2-l*Math.cos(a-.45)} ${y2-l*Math.sin(a-.45)} L ${x2} ${y2} L ${x2-l*Math.cos(a+.45)} ${y2-l*Math.sin(a+.45)}`,stroke:color,'stroke-width':2,fill:'none'});
  }
  box(12,129,153,83,['七条轨线','中心 + 六邻居','每条32个三维点'],true);
  box(188,129,174,83,['几何归一化','减整组质心','除整组最大半径'],true); arrow(165,170,188,170,true);
  const starts=[30,137,244], names=[['中心步进向量','u[n] = q₀[n+1] − q₀[n]','31个三维向量'],['中心坐标 / 单位切向','q₀[n] 与 t[n]','32点 × 6通道'],['邻居相对变化向量','Δ(qᵢ[n] − q₀[n])','6条 × 31个三维向量']];
  for(let b=0;b<3;b++) {
    const y=starts[b],on=active===b;
    arrow(362,170,397,y+39,on); box(397,y,216,80,names[b],on);
    arrow(613,y+39,640,y+39,on);
    box(640,y,167,80,['傅里叶变换',b===1?'32点 · 正交归一化':'31点 · 不除以长度','保留 k = 0…5'],on);
    arrow(807,y+39,838,y+39,on);
    let descriptor;
    if(b!==active) descriptor=b===1?['取实部 / 虚部','72维方向频谱']:['范数 / 余弦 / 三重积',b===0?'23维中心描述':'每个邻居23维'];
    else if(b===1) descriptor=[`${f.channel} · k=${f.frequency}`,f.component==='Re'?'取实部 Re(F[k])':'取虚部 Im(F[k])'];
    else descriptor=[`k = ${f.frequency}${f.component.includes('三重积')?' 与 k+1':''}`,f.component.includes('三重积')?'实/虚/相邻实向量三重积':f.component];
    box(838,y+4,210,72,descriptor,on);
    arrow(1048,y+39,1073,y+39,on);
    const pool=b===2?(!on?['六邻居逐维池化','均值23维 + 最大值23维']:f.block==='neighbor_max'?['六邻居逐维最大值','数值最大，保留符号']:['六邻居逐维均值','先编码，再算术平均']):['直接保留','此分支不做邻居池化'];
    box(1073,y+4,180,72,pool,on);
  }
  node('text',{x:12,y:352,fill:dark,'font-size':14,'font-weight':600},`选中路径 → ${f.label}`);
  document.getElementById('pipelineFormula').textContent=f.formula;
  const rv=data.raw[sample][index],zv=data.z[sample][index];
  document.getElementById('pipelineOutput').textContent=`拼接到 features[${index}]：原特征值 ${rv.toPrecision(6)} → 减全体样本该维均值、除该维标准差 → 标准化值 ${zv.toPrecision(6)}（样本ID ${data.ids[sample]}）。`;
  let note='这里的原特征值已经经过傅里叶运算；不是原始几何坐标。';
  if(f.component==='实虚夹角余弦') note+=' 此余弦比较同一频率的实部与虚部三维向量，不是两条物理邻居线之间的夹角。';
  if(f.component.includes('三重积')) note+=' 三重积使用 aₖ、bₖ、aₖ₊₁，并以三者范数乘积归一化。';
  if(f.theoretical_zero) note+=' k=0的虚部恒为0，因此当前槽位理论上恒为0。';
  document.getElementById('pipelineNote').textContent=note;
};
