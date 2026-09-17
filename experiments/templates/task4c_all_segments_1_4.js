/* Use four measured, non-overlapping interventions to cover every real edge. */
(function (root) {
  'use strict';
  const windows = Object.freeze([[0,8],[8,16],[16,24],[24,31]].map(Object.freeze));
  const colorscale = [[0,'#2166ac'],[0.5,'#dce1e6'],[1,'#b2182b']];
  const cached = new WeakMap();

  function indices(data) {
    if (cached.has(data)) return cached.get(data);
    const lookup = new Map();
    data.patches.forEach((p,i) => {
      const key=p.indices.join(':');
      if (lookup.has(key)) throw new Error('Duplicate measured window: '+key);
      lookup.set(key,i);
    });
    const result=[];
    data.geometry.forEach((line,l) => {
      if (line.length!==32) throw new Error('Whole-curve display requires the frozen 32-point geometry');
      for (const [a,b] of windows) {
        const key=[l,a,b].join(':'),i=lookup.get(key);
        if (i===undefined) throw new Error('Missing measured straightening window: '+key);
        if (!Number.isFinite(data.patches[i].margin_delta)) throw new Error('Nonfinite intervention score');
        result.push(i);
      }
    });
    cached.set(data,result);
    return result;
  }

  function build(data,lines,id) {
    if(lines.length!==data.geometry.length) throw new Error('Display/source line counts differ');
    const points=[],values=[],customdata=[],selected=indices(data);
    for(const pi of selected) {
      const p=data.patches[pi],[l,a,b]=p.indices;
      for(let j=a;j<=b;j++) {
        points.push(lines[l][j]);values.push(p.margin_delta);
        customdata.push([id,l,j,pi,p.margin_delta,p.probability,a,b]);
      }
      // A gap prevents interpolation of scores between independently tested segments.
      points.push(null);values.push(0);customdata.push(null);
    }
    return {points,values,customdata,indices:selected};
  }

  function limit(packs) {
    let value=0;
    for(const data of packs)for(const i of indices(data))value=Math.max(value,Math.abs(data.patches[i].margin_delta));
    return value||1;
  }

  const api={windows,colorscale,indices,build,limit};
  if(typeof module==='object'&&module.exports)module.exports=api;
  else root.WholeCurveSegments=api;
})(typeof window!=='undefined'?window:this);
