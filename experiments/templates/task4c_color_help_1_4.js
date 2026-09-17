/* Plain-language interpretation of frozen shape-intervention and gradient maps. */
'use strict';
const COLOR_HELP = {
  whole: {
    title: '每条线的每一段，都按拉直后得分的变化着色',
    question: '整条线都着色：红色支持Hairpin，蓝色起反作用；越深影响越大。',
    method: '每条32点曲线划为点0–8、8–16、16–24、24–31四段。这四段首尾相接、覆盖整条线；各段直接使用它单独拉直后的实际得分变化，不取重叠平均。',
    color: '原片段红色：拉直后Hairpin得分降低；蓝色：拉直后得分升高。深浅表示变化大小，灰色接近零。当前显示的全部线束共用同一红蓝色标。',
    limit: '每个值属于一整段，不是每个点的独立贡献；分数也不能相加。每条线的四段分别做过实验，没有把整条线同时拉直。',
    hint: '覆盖所有有效线和全部曲线长度。悬停看分数，点击曲线或热图查看该段的前后概率。',
    legend: '深蓝＝较强反作用；灰＝变化接近零；深红＝较强正支持。每段内部颜色相同，段与段之间不平均。'
  },
  support: {
    title: '把一小段弯曲拉直，看看模型的判断怎样变化',
    question: '用于回答：保留哪一段原始形状，会让模型更倾向判为 Hairpin？',
    method: '每次只把一条线中连续的8或9个采样点拉直：两端不动，中间点换成直线；其余线保持原样，再把整束送入同一个模型。图上的粗线仍是原来的弯曲片段。',
    color: '红色：拉直后Hairpin得分下降，原片段提供正支持。蓝色：拉直后得分上升，原片段起反作用。灰色：当前未高亮，不能理解成“没有用”。',
    limit: '每个分数属于一整段，不能拆成每个点的贡献，也不能把重叠片段的分数相加。所有红段颜色相同；强弱看分数变化，不看红色深浅。',
    hint: '你要找“哪段形状支持Hairpin”，可用这一项查看单段，或用“全部线段”查看整体。',
    legend: '红色粗线＝原片段支持Hairpin；蓝色粗线＝原片段起反作用；灰线＝未高亮的原始几何。'
  },
  legacy: {
    title: '旧版画法：把多个拉直实验的结果平均到每个点',
    question: '这是同一批片段实验的汇总画法，没有额外做逐点实验。',
    method: '相邻测试片段有重叠，例如点0–8和点4–12。一个点若处于多个被改动的片段内部，就用这些片段的得分变化的平均值给它着色。',
    color: '红色：平均后偏向支持Hairpin；蓝色：平均后偏向反对；浅灰色：平均接近零，或没有片段改动到该点。',
    limit: '例如同一点收到+2和−2，平均变成0，正负作用被抵消。浅色不等于不重要；这个值也不是该点自己的贡献。保留此模式仅便于查看旧图，判断具体支持片段请用“全部线段”或“选一段”。',
    hint: '旧图会把重叠片段的正负作用抵消，不适合定位最支持Hairpin的一段。',
    legend: '红＝平均值为正；蓝＝平均值为负；浅灰＝平均接近零或未被改动。深浅按每束自己的范围显示，不能跨束比较。'
  },
  gradient: {
    title: '轻微移动一个点，模型得分会有多敏感？',
    question: '用于找敏感位置；不能回答该点支持还是反对Hairpin。',
    method: '不把线段拉直。通过网络求导，计算每个点的三维坐标发生极小变化时，Hairpin得分的变化率，再取这个三维梯度的大小。',
    color: '深红：对坐标微小变化更敏感；浅黄：更不敏感。这里只有非负的大小，没有正支持或负作用的符号。',
    limit: '红点可能让Hairpin分数上升，也可能让它下降，取决于移动方向。高敏感度不等于该点的弯曲支持Hairpin，也不是把这个点删除后的效果。',
    hint: '此处红色只表示“容易影响分数”，不表示“支持Hairpin”。',
    legend: '浅黄→深红＝坐标敏感度由小到大；不区分支持/反对。每束独立缩放颜色，不能跨束比较红色深浅。'
  },
  smooth_gradient: {
    title: '轻微加噪16次，再看平均后的坐标敏感度',
    question: '用于查看噪声平均后的敏感位置；仍然不是正类支持图。',
    method: '给点坐标加入16组独立的小扰动，每组都计算一次梯度；先把三维梯度向量平均，再取大小。这种方法叫SmoothGrad（对加噪后的梯度求平均）。',
    color: '深红：平均梯度的大小更大；浅黄：更小。它和上一项一样没有正负贡献方向，也没有做局部拉直。',
    limit: '不同扰动下的梯度方向可能抵消。这幅图没有显示16次结果的波动大小，因此不能单凭颜色宣称稳定性或抗噪能力更强。',
    hint: '与上一项的区别是加噪后重复16次再平均，不是另一种正支持分数。',
    legend: '浅黄→深红＝平均梯度大小由小到大；不是支持程度或稳定性评分。每束独立缩放颜色。'
  },
  plain: {
    title: '只看真实线束形状，不显示解释分数',
    question: '用于先辨认整体弯曲、头部或腿部的几何形状。',
    method: '所有线使用同一种颜色，保留原始几何形状。画廊只为排版平移并缩放线束，不重新积分。',
    color: '统一蓝灰色没有数值含义，不代表支持、反对或敏感程度。',
    limit: '这一项没有把模型贡献画出来。若要找支持Hairpin的片段，切回“全部线段”或“选一段”。',
    hint: '只显示几何，不计算或显示局部贡献颜色。',
    legend: '统一颜色仅用于显示线条，没有模型分数含义。'
  }
};

window.updateColorHelp = function () {
  const get=id=>document.getElementById(id),mode=get('map').value,help=COLOR_HELP[mode],isWhole=mode==='whole',isSupport=mode==='support'||isWhole;
  for(const [id,key] of [['modeTitle','title'],['modeQuestion','question'],['modeMethod','method'],['modeColor','color'],['modeLimit','limit'],['modeHint','hint']])get(id).textContent=help[key];
  get('legend').textContent=help.legend;
  get('legend').classList.remove('hidden');
  get('interventionControls').classList.toggle('hidden',!isSupport);
  get('scopeControl').classList.toggle('hidden',isWhole);
  get('topCountControl').classList.toggle('hidden',isWhole);
  get('colorRange').classList.toggle('hidden',!isWhole);
  get('windowCard').classList.toggle('hidden',!isSupport);
  get('patchTableCard').classList.toggle('hidden',!isSupport);
  get('interventionBackground').classList.toggle('hidden',!isSupport);
  get('reason').classList.toggle('hidden',!isSupport);
  get('changedProbability').parentElement.classList.toggle('hidden',!isSupport);
  get('delta').parentElement.classList.toggle('hidden',!isSupport);
  get('viewTitle').textContent=isWhole?'全部线段 · 颜色深浅表示拉直影响':isSupport?'原始线束与所选形状片段':help.title;
  get('viewNote').textContent=get('layout').value==='gallery'
    ? '画廊为每束采用质心/半径归一化并分格摆放；上方概率只属于侧栏当前选中的线束。'
    : get('layout').value==='physical'
      ? '线束位于原始物理位置；上方概率只属于侧栏当前选中的线束。'
      : isWhole?'每条线均完整着色。上方前后概率属于侧栏当前选择的一个片段；在曲线上悬停或点击可查看别的片段。':isSupport?'整束保留为灰色背景，粗线标出所选原片段；完整线束的判断来自整束输入，不是只输入这一段。':'当前只显示侧栏选中的一束；着色含义见上方说明。';
};
