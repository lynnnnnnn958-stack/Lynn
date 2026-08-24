#!/usr/bin/env python3
"""
Canyon v9 — Research Website Generator
Reads live CSVs and daily report, generates canyon_v24_research.html.
Run daily:  .venv/bin/python update_research_html.py
"""
from __future__ import annotations
import json, re, os
from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np

ROOT = Path(__file__).parent
OUT  = ROOT / "canyon_v24_research.html"

# Chart.js 4.4.0 bundled inline (offline support)
_CHARTJS_JS: str = '/**\n * Skipped minification because the original files appears to be already minified.\n * Original file: /npm/chart.js@4.4.0/dist/chart.umd.js\n *\n * Do NOT use SRI with dynamically generated files! More information: https://www.jsdelivr.com/using-sri-with-dynamic-files\n */\n/*!\n * Chart.js v4.4.0\n * https://www.chartjs.org\n * (c) 2023 Chart.js Contributors\n * Released under the MIT License\n */\n!function(t,e){"object"==typeof exports&&"undefined"!=typeof module?module.exports=e():"function"==typeof define&&define.amd?define(e):(t="undefined"!=typeof globalThis?globalThis:t||self).Chart=e()}(this,(function(){"use strict";var t=Object.freeze({__proto__:null,get Colors(){return Go},get Decimation(){return Qo},get Filler(){return ma},get Legend(){return ya},get SubTitle(){return ka},get Title(){return Ma},get Tooltip(){return Ba}});function e(){}const i=(()=>{let t=0;return()=>t++})();function s(t){return null==t}function n(t){if(Array.isArray&&Array.isArray(t))return!0;const e=Object.prototype.toString.call(t);return"[object"===e.slice(0,7)&&"Array]"===e.slice(-6)}function o(t){return null!==t&&"[object Object]"===Object.prototype.toString.call(t)}function a(t){return("number"==typeof t||t instanceof Number)&&isFinite(+t)}function r(t,e){return a(t)?t:e}function l(t,e){return void 0===t?e:t}const h=(t,e)=>"string"==typeof t&&t.endsWith("%")?parseFloat(t)/100:+t/e,c=(t,e)=>"string"==typeof t&&t.endsWith("%")?parseFloat(t)/100*e:+t;function d(t,e,i){if(t&&"function"==typeof t.call)return t.apply(i,e)}function u(t,e,i,s){let a,r,l;if(n(t))if(r=t.length,s)for(a=r-1;a>=0;a--)e.call(i,t[a],a);else for(a=0;a<r;a++)e.call(i,t[a],a);else if(o(t))for(l=Object.keys(t),r=l.length,a=0;a<r;a++)e.call(i,t[l[a]],l[a])}function f(t,e){let i,s,n,o;if(!t||!e||t.length!==e.length)return!1;for(i=0,s=t.length;i<s;++i)if(n=t[i],o=e[i],n.datasetIndex!==o.datasetIndex||n.index!==o.index)return!1;return!0}function g(t){if(n(t))return t.map(g);if(o(t)){const e=Object.create(null),i=Object.keys(t),s=i.length;let n=0;for(;n<s;++n)e[i[n]]=g(t[i[n]]);return e}return t}function p(t){return-1===["__proto__","prototype","constructor"].indexOf(t)}function m(t,e,i,s){if(!p(t))return;const n=e[t],a=i[t];o(n)&&o(a)?b(n,a,s):e[t]=g(a)}function b(t,e,i){const s=n(e)?e:[e],a=s.length;if(!o(t))return t;const r=(i=i||{}).merger||m;let l;for(let e=0;e<a;++e){if(l=s[e],!o(l))continue;const n=Object.keys(l);for(let e=0,s=n.length;e<s;++e)r(n[e],t,l,i)}return t}function x(t,e){return b(t,e,{merger:_})}function _(t,e,i){if(!p(t))return;const s=e[t],n=i[t];o(s)&&o(n)?x(s,n):Object.prototype.hasOwnProperty.call(e,t)||(e[t]=g(n))}const y={"":t=>t,x:t=>t.x,y:t=>t.y};function v(t){const e=t.split("."),i=[];let s="";for(const t of e)s+=t,s.endsWith("\\\\")?s=s.slice(0,-1)+".":(i.push(s),s="");return i}function M(t,e){const i=y[e]||(y[e]=function(t){const e=v(t);return t=>{for(const i of e){if(""===i)break;t=t&&t[i]}return t}}(e));return i(t)}function w(t){return t.charAt(0).toUpperCase()+t.slice(1)}const k=t=>void 0!==t,S=t=>"function"==typeof t,P=(t,e)=>{if(t.size!==e.size)return!1;for(const i of t)if(!e.has(i))return!1;return!0};function D(t){return"mouseup"===t.type||"click"===t.type||"contextmenu"===t.type}const C=Math.PI,O=2*C,A=O+C,T=Number.POSITIVE_INFINITY,L=C/180,E=C/2,R=C/4,I=2*C/3,z=Math.log10,F=Math.sign;function V(t,e,i){return Math.abs(t-e)<i}function B(t){const e=Math.round(t);t=V(t,e,t/1e3)?e:t;const i=Math.pow(10,Math.floor(z(t))),s=t/i;return(s<=1?1:s<=2?2:s<=5?5:10)*i}function W(t){const e=[],i=Math.sqrt(t);let s;for(s=1;s<i;s++)t%s==0&&(e.push(s),e.push(t/s));return i===(0|i)&&e.push(i),e.sort(((t,e)=>t-e)).pop(),e}function N(t){return!isNaN(parseFloat(t))&&isFinite(t)}function H(t,e){const i=Math.round(t);return i-e<=t&&i+e>=t}function j(t,e,i){let s,n,o;for(s=0,n=t.length;s<n;s++)o=t[s][i],isNaN(o)||(e.min=Math.min(e.min,o),e.max=Math.max(e.max,o))}function $(t){return t*(C/180)}function Y(t){return t*(180/C)}function U(t){if(!a(t))return;let e=1,i=0;for(;Math.round(t*e)/e!==t;)e*=10,i++;return i}function X(t,e){const i=e.x-t.x,s=e.y-t.y,n=Math.sqrt(i*i+s*s);let o=Math.atan2(s,i);return o<-.5*C&&(o+=O),{angle:o,distance:n}}function q(t,e){return Math.sqrt(Math.pow(e.x-t.x,2)+Math.pow(e.y-t.y,2))}function K(t,e){return(t-e+A)%O-C}function G(t){return(t%O+O)%O}function Z(t,e,i,s){const n=G(t),o=G(e),a=G(i),r=G(o-n),l=G(a-n),h=G(n-o),c=G(n-a);return n===o||n===a||s&&o===a||r>l&&h<c}function J(t,e,i){return Math.max(e,Math.min(i,t))}function Q(t){return J(t,-32768,32767)}function tt(t,e,i,s=1e-6){return t>=Math.min(e,i)-s&&t<=Math.max(e,i)+s}function et(t,e,i){i=i||(i=>t[i]<e);let s,n=t.length-1,o=0;for(;n-o>1;)s=o+n>>1,i(s)?o=s:n=s;return{lo:o,hi:n}}const it=(t,e,i,s)=>et(t,i,s?s=>{const n=t[s][e];return n<i||n===i&&t[s+1][e]===i}:s=>t[s][e]<i),st=(t,e,i)=>et(t,i,(s=>t[s][e]>=i));function nt(t,e,i){let s=0,n=t.length;for(;s<n&&t[s]<e;)s++;for(;n>s&&t[n-1]>i;)n--;return s>0||n<t.length?t.slice(s,n):t}const ot=["push","pop","shift","splice","unshift"];function at(t,e){t._chartjs?t._chartjs.listeners.push(e):(Object.defineProperty(t,"_chartjs",{configurable:!0,enumerable:!1,value:{listeners:[e]}}),ot.forEach((e=>{const i="_onData"+w(e),s=t[e];Object.defineProperty(t,e,{configurable:!0,enumerable:!1,value(...e){const n=s.apply(this,e);return t._chartjs.listeners.forEach((t=>{"function"==typeof t[i]&&t[i](...e)})),n}})})))}function rt(t,e){const i=t._chartjs;if(!i)return;const s=i.listeners,n=s.indexOf(e);-1!==n&&s.splice(n,1),s.length>0||(ot.forEach((e=>{delete t[e]})),delete t._chartjs)}function lt(t){const e=new Set(t);return e.size===t.length?t:Array.from(e)}const ht="undefined"==typeof window?function(t){return t()}:window.requestAnimationFrame;function ct(t,e){let i=[],s=!1;return function(...n){i=n,s||(s=!0,ht.call(window,(()=>{s=!1,t.apply(e,i)})))}}function dt(t,e){let i;return function(...s){return e?(clearTimeout(i),i=setTimeout(t,e,s)):t.apply(this,s),e}}const ut=t=>"start"===t?"left":"end"===t?"right":"center",ft=(t,e,i)=>"start"===t?e:"end"===t?i:(e+i)/2,gt=(t,e,i,s)=>t===(s?"left":"right")?i:"center"===t?(e+i)/2:e;function pt(t,e,i){const s=e.length;let n=0,o=s;if(t._sorted){const{iScale:a,_parsed:r}=t,l=a.axis,{min:h,max:c,minDefined:d,maxDefined:u}=a.getUserBounds();d&&(n=J(Math.min(it(r,l,h).lo,i?s:it(e,l,a.getPixelForValue(h)).lo),0,s-1)),o=u?J(Math.max(it(r,a.axis,c,!0).hi+1,i?0:it(e,l,a.getPixelForValue(c),!0).hi+1),n,s)-n:s-n}return{start:n,count:o}}function mt(t){const{xScale:e,yScale:i,_scaleRanges:s}=t,n={xmin:e.min,xmax:e.max,ymin:i.min,ymax:i.max};if(!s)return t._scaleRanges=n,!0;const o=s.xmin!==e.min||s.xmax!==e.max||s.ymin!==i.min||s.ymax!==i.max;return Object.assign(s,n),o}class bt{constructor(){this._request=null,this._charts=new Map,this._running=!1,this._lastDate=void 0}_notify(t,e,i,s){const n=e.listeners[s],o=e.duration;n.forEach((s=>s({chart:t,initial:e.initial,numSteps:o,currentStep:Math.min(i-e.start,o)})))}_refresh(){this._request||(this._running=!0,this._request=ht.call(window,(()=>{this._update(),this._request=null,this._running&&this._refresh()})))}_update(t=Date.now()){let e=0;this._charts.forEach(((i,s)=>{if(!i.running||!i.items.length)return;const n=i.items;let o,a=n.length-1,r=!1;for(;a>=0;--a)o=n[a],o._active?(o._total>i.duration&&(i.duration=o._total),o.tick(t),r=!0):(n[a]=n[n.length-1],n.pop());r&&(s.draw(),this._notify(s,i,t,"progress")),n.length||(i.running=!1,this._notify(s,i,t,"complete"),i.initial=!1),e+=n.length})),this._lastDate=t,0===e&&(this._running=!1)}_getAnims(t){const e=this._charts;let i=e.get(t);return i||(i={running:!1,initial:!0,items:[],listeners:{complete:[],progress:[]}},e.set(t,i)),i}listen(t,e,i){this._getAnims(t).listeners[e].push(i)}add(t,e){e&&e.length&&this._getAnims(t).items.push(...e)}has(t){return this._getAnims(t).items.length>0}start(t){const e=this._charts.get(t);e&&(e.running=!0,e.start=Date.now(),e.duration=e.items.reduce(((t,e)=>Math.max(t,e._duration)),0),this._refresh())}running(t){if(!this._running)return!1;const e=this._charts.get(t);return!!(e&&e.running&&e.items.length)}stop(t){const e=this._charts.get(t);if(!e||!e.items.length)return;const i=e.items;let s=i.length-1;for(;s>=0;--s)i[s].cancel();e.items=[],this._notify(t,e,Date.now(),"complete")}remove(t){return this._charts.delete(t)}}var xt=new bt;\n/*!\n * @kurkle/color v0.3.2\n * https://github.com/kurkle/color#readme\n * (c) 2023 Jukka Kurkela\n * Released under the MIT License\n */function _t(t){return t+.5|0}const yt=(t,e,i)=>Math.max(Math.min(t,i),e);function vt(t){return yt(_t(2.55*t),0,255)}function Mt(t){return yt(_t(255*t),0,255)}function wt(t){return yt(_t(t/2.55)/100,0,1)}function kt(t){return yt(_t(100*t),0,100)}const St={0:0,1:1,2:2,3:3,4:4,5:5,6:6,7:7,8:8,9:9,A:10,B:11,C:12,D:13,E:14,F:15,a:10,b:11,c:12,d:13,e:14,f:15},Pt=[..."0123456789ABCDEF"],Dt=t=>Pt[15&t],Ct=t=>Pt[(240&t)>>4]+Pt[15&t],Ot=t=>(240&t)>>4==(15&t);function At(t){var e=(t=>Ot(t.r)&&Ot(t.g)&&Ot(t.b)&&Ot(t.a))(t)?Dt:Ct;return t?"#"+e(t.r)+e(t.g)+e(t.b)+((t,e)=>t<255?e(t):"")(t.a,e):void 0}const Tt=/^(hsla?|hwb|hsv)\\(\\s*([-+.e\\d]+)(?:deg)?[\\s,]+([-+.e\\d]+)%[\\s,]+([-+.e\\d]+)%(?:[\\s,]+([-+.e\\d]+)(%)?)?\\s*\\)$/;function Lt(t,e,i){const s=e*Math.min(i,1-i),n=(e,n=(e+t/30)%12)=>i-s*Math.max(Math.min(n-3,9-n,1),-1);return[n(0),n(8),n(4)]}function Et(t,e,i){const s=(s,n=(s+t/60)%6)=>i-i*e*Math.max(Math.min(n,4-n,1),0);return[s(5),s(3),s(1)]}function Rt(t,e,i){const s=Lt(t,1,.5);let n;for(e+i>1&&(n=1/(e+i),e*=n,i*=n),n=0;n<3;n++)s[n]*=1-e-i,s[n]+=e;return s}function It(t){const e=t.r/255,i=t.g/255,s=t.b/255,n=Math.max(e,i,s),o=Math.min(e,i,s),a=(n+o)/2;let r,l,h;return n!==o&&(h=n-o,l=a>.5?h/(2-n-o):h/(n+o),r=function(t,e,i,s,n){return t===n?(e-i)/s+(e<i?6:0):e===n?(i-t)/s+2:(t-e)/s+4}(e,i,s,h,n),r=60*r+.5),[0|r,l||0,a]}function zt(t,e,i,s){return(Array.isArray(e)?t(e[0],e[1],e[2]):t(e,i,s)).map(Mt)}function Ft(t,e,i){return zt(Lt,t,e,i)}function Vt(t){return(t%360+360)%360}function Bt(t){const e=Tt.exec(t);let i,s=255;if(!e)return;e[5]!==i&&(s=e[6]?vt(+e[5]):Mt(+e[5]));const n=Vt(+e[2]),o=+e[3]/100,a=+e[4]/100;return i="hwb"===e[1]?function(t,e,i){return zt(Rt,t,e,i)}(n,o,a):"hsv"===e[1]?function(t,e,i){return zt(Et,t,e,i)}(n,o,a):Ft(n,o,a),{r:i[0],g:i[1],b:i[2],a:s}}const Wt={x:"dark",Z:"light",Y:"re",X:"blu",W:"gr",V:"medium",U:"slate",A:"ee",T:"ol",S:"or",B:"ra",C:"lateg",D:"ights",R:"in",Q:"turquois",E:"hi",P:"ro",O:"al",N:"le",M:"de",L:"yello",F:"en",K:"ch",G:"arks",H:"ea",I:"ightg",J:"wh"},Nt={OiceXe:"f0f8ff",antiquewEte:"faebd7",aqua:"ffff",aquamarRe:"7fffd4",azuY:"f0ffff",beige:"f5f5dc",bisque:"ffe4c4",black:"0",blanKedOmond:"ffebcd",Xe:"ff",XeviTet:"8a2be2",bPwn:"a52a2a",burlywood:"deb887",caMtXe:"5f9ea0",KartYuse:"7fff00",KocTate:"d2691e",cSO:"ff7f50",cSnflowerXe:"6495ed",cSnsilk:"fff8dc",crimson:"dc143c",cyan:"ffff",xXe:"8b",xcyan:"8b8b",xgTMnPd:"b8860b",xWay:"a9a9a9",xgYF:"6400",xgYy:"a9a9a9",xkhaki:"bdb76b",xmagFta:"8b008b",xTivegYF:"556b2f",xSange:"ff8c00",xScEd:"9932cc",xYd:"8b0000",xsOmon:"e9967a",xsHgYF:"8fbc8f",xUXe:"483d8b",xUWay:"2f4f4f",xUgYy:"2f4f4f",xQe:"ced1",xviTet:"9400d3",dAppRk:"ff1493",dApskyXe:"bfff",dimWay:"696969",dimgYy:"696969",dodgerXe:"1e90ff",fiYbrick:"b22222",flSOwEte:"fffaf0",foYstWAn:"228b22",fuKsia:"ff00ff",gaRsbSo:"dcdcdc",ghostwEte:"f8f8ff",gTd:"ffd700",gTMnPd:"daa520",Way:"808080",gYF:"8000",gYFLw:"adff2f",gYy:"808080",honeyMw:"f0fff0",hotpRk:"ff69b4",RdianYd:"cd5c5c",Rdigo:"4b0082",ivSy:"fffff0",khaki:"f0e68c",lavFMr:"e6e6fa",lavFMrXsh:"fff0f5",lawngYF:"7cfc00",NmoncEffon:"fffacd",ZXe:"add8e6",ZcSO:"f08080",Zcyan:"e0ffff",ZgTMnPdLw:"fafad2",ZWay:"d3d3d3",ZgYF:"90ee90",ZgYy:"d3d3d3",ZpRk:"ffb6c1",ZsOmon:"ffa07a",ZsHgYF:"20b2aa",ZskyXe:"87cefa",ZUWay:"778899",ZUgYy:"778899",ZstAlXe:"b0c4de",ZLw:"ffffe0",lime:"ff00",limegYF:"32cd32",lRF:"faf0e6",magFta:"ff00ff",maPon:"800000",VaquamarRe:"66cdaa",VXe:"cd",VScEd:"ba55d3",VpurpN:"9370db",VsHgYF:"3cb371",VUXe:"7b68ee",VsprRggYF:"fa9a",VQe:"48d1cc",VviTetYd:"c71585",midnightXe:"191970",mRtcYam:"f5fffa",mistyPse:"ffe4e1",moccasR:"ffe4b5",navajowEte:"ffdead",navy:"80",Tdlace:"fdf5e6",Tive:"808000",TivedBb:"6b8e23",Sange:"ffa500",SangeYd:"ff4500",ScEd:"da70d6",pOegTMnPd:"eee8aa",pOegYF:"98fb98",pOeQe:"afeeee",pOeviTetYd:"db7093",papayawEp:"ffefd5",pHKpuff:"ffdab9",peru:"cd853f",pRk:"ffc0cb",plum:"dda0dd",powMrXe:"b0e0e6",purpN:"800080",YbeccapurpN:"663399",Yd:"ff0000",Psybrown:"bc8f8f",PyOXe:"4169e1",saddNbPwn:"8b4513",sOmon:"fa8072",sandybPwn:"f4a460",sHgYF:"2e8b57",sHshell:"fff5ee",siFna:"a0522d",silver:"c0c0c0",skyXe:"87ceeb",UXe:"6a5acd",UWay:"708090",UgYy:"708090",snow:"fffafa",sprRggYF:"ff7f",stAlXe:"4682b4",tan:"d2b48c",teO:"8080",tEstN:"d8bfd8",tomato:"ff6347",Qe:"40e0d0",viTet:"ee82ee",JHt:"f5deb3",wEte:"ffffff",wEtesmoke:"f5f5f5",Lw:"ffff00",LwgYF:"9acd32"};let Ht;function jt(t){Ht||(Ht=function(){const t={},e=Object.keys(Nt),i=Object.keys(Wt);let s,n,o,a,r;for(s=0;s<e.length;s++){for(a=r=e[s],n=0;n<i.length;n++)o=i[n],r=r.replace(o,Wt[o]);o=parseInt(Nt[a],16),t[r]=[o>>16&255,o>>8&255,255&o]}return t}(),Ht.transparent=[0,0,0,0]);const e=Ht[t.toLowerCase()];return e&&{r:e[0],g:e[1],b:e[2],a:4===e.length?e[3]:255}}const $t=/^rgba?\\(\\s*([-+.\\d]+)(%)?[\\s,]+([-+.e\\d]+)(%)?[\\s,]+([-+.e\\d]+)(%)?(?:[\\s,/]+([-+.e\\d]+)(%)?)?\\s*\\)$/;const Yt=t=>t<=.0031308?12.92*t:1.055*Math.pow(t,1/2.4)-.055,Ut=t=>t<=.04045?t/12.92:Math.pow((t+.055)/1.055,2.4);function Xt(t,e,i){if(t){let s=It(t);s[e]=Math.max(0,Math.min(s[e]+s[e]*i,0===e?360:1)),s=Ft(s),t.r=s[0],t.g=s[1],t.b=s[2]}}function qt(t,e){return t?Object.assign(e||{},t):t}function Kt(t){var e={r:0,g:0,b:0,a:255};return Array.isArray(t)?t.length>=3&&(e={r:t[0],g:t[1],b:t[2],a:255},t.length>3&&(e.a=Mt(t[3]))):(e=qt(t,{r:0,g:0,b:0,a:1})).a=Mt(e.a),e}function Gt(t){return"r"===t.charAt(0)?function(t){const e=$t.exec(t);let i,s,n,o=255;if(e){if(e[7]!==i){const t=+e[7];o=e[8]?vt(t):yt(255*t,0,255)}return i=+e[1],s=+e[3],n=+e[5],i=255&(e[2]?vt(i):yt(i,0,255)),s=255&(e[4]?vt(s):yt(s,0,255)),n=255&(e[6]?vt(n):yt(n,0,255)),{r:i,g:s,b:n,a:o}}}(t):Bt(t)}class Zt{constructor(t){if(t instanceof Zt)return t;const e=typeof t;let i;var s,n,o;"object"===e?i=Kt(t):"string"===e&&(o=(s=t).length,"#"===s[0]&&(4===o||5===o?n={r:255&17*St[s[1]],g:255&17*St[s[2]],b:255&17*St[s[3]],a:5===o?17*St[s[4]]:255}:7!==o&&9!==o||(n={r:St[s[1]]<<4|St[s[2]],g:St[s[3]]<<4|St[s[4]],b:St[s[5]]<<4|St[s[6]],a:9===o?St[s[7]]<<4|St[s[8]]:255})),i=n||jt(t)||Gt(t)),this._rgb=i,this._valid=!!i}get valid(){return this._valid}get rgb(){var t=qt(this._rgb);return t&&(t.a=wt(t.a)),t}set rgb(t){this._rgb=Kt(t)}rgbString(){return this._valid?(t=this._rgb)&&(t.a<255?`rgba(${t.r}, ${t.g}, ${t.b}, ${wt(t.a)})`:`rgb(${t.r}, ${t.g}, ${t.b})`):void 0;var t}hexString(){return this._valid?At(this._rgb):void 0}hslString(){return this._valid?function(t){if(!t)return;const e=It(t),i=e[0],s=kt(e[1]),n=kt(e[2]);return t.a<255?`hsla(${i}, ${s}%, ${n}%, ${wt(t.a)})`:`hsl(${i}, ${s}%, ${n}%)`}(this._rgb):void 0}mix(t,e){if(t){const i=this.rgb,s=t.rgb;let n;const o=e===n?.5:e,a=2*o-1,r=i.a-s.a,l=((a*r==-1?a:(a+r)/(1+a*r))+1)/2;n=1-l,i.r=255&l*i.r+n*s.r+.5,i.g=255&l*i.g+n*s.g+.5,i.b=255&l*i.b+n*s.b+.5,i.a=o*i.a+(1-o)*s.a,this.rgb=i}return this}interpolate(t,e){return t&&(this._rgb=function(t,e,i){const s=Ut(wt(t.r)),n=Ut(wt(t.g)),o=Ut(wt(t.b));return{r:Mt(Yt(s+i*(Ut(wt(e.r))-s))),g:Mt(Yt(n+i*(Ut(wt(e.g))-n))),b:Mt(Yt(o+i*(Ut(wt(e.b))-o))),a:t.a+i*(e.a-t.a)}}(this._rgb,t._rgb,e)),this}clone(){return new Zt(this.rgb)}alpha(t){return this._rgb.a=Mt(t),this}clearer(t){return this._rgb.a*=1-t,this}greyscale(){const t=this._rgb,e=_t(.3*t.r+.59*t.g+.11*t.b);return t.r=t.g=t.b=e,this}opaquer(t){return this._rgb.a*=1+t,this}negate(){const t=this._rgb;return t.r=255-t.r,t.g=255-t.g,t.b=255-t.b,this}lighten(t){return Xt(this._rgb,2,t),this}darken(t){return Xt(this._rgb,2,-t),this}saturate(t){return Xt(this._rgb,1,t),this}desaturate(t){return Xt(this._rgb,1,-t),this}rotate(t){return function(t,e){var i=It(t);i[0]=Vt(i[0]+e),i=Ft(i),t.r=i[0],t.g=i[1],t.b=i[2]}(this._rgb,t),this}}function Jt(t){if(t&&"object"==typeof t){const e=t.toString();return"[object CanvasPattern]"===e||"[object CanvasGradient]"===e}return!1}function Qt(t){return Jt(t)?t:new Zt(t)}function te(t){return Jt(t)?t:new Zt(t).saturate(.5).darken(.1).hexString()}const ee=["x","y","borderWidth","radius","tension"],ie=["color","borderColor","backgroundColor"];const se=new Map;function ne(t,e,i){return function(t,e){e=e||{};const i=t+JSON.stringify(e);let s=se.get(i);return s||(s=new Intl.NumberFormat(t,e),se.set(i,s)),s}(e,i).format(t)}const oe={values:t=>n(t)?t:""+t,numeric(t,e,i){if(0===t)return"0";const s=this.chart.options.locale;let n,o=t;if(i.length>1){const e=Math.max(Math.abs(i[0].value),Math.abs(i[i.length-1].value));(e<1e-4||e>1e15)&&(n="scientific"),o=function(t,e){let i=e.length>3?e[2].value-e[1].value:e[1].value-e[0].value;Math.abs(i)>=1&&t!==Math.floor(t)&&(i=t-Math.floor(t));return i}(t,i)}const a=z(Math.abs(o)),r=isNaN(a)?1:Math.max(Math.min(-1*Math.floor(a),20),0),l={notation:n,minimumFractionDigits:r,maximumFractionDigits:r};return Object.assign(l,this.options.ticks.format),ne(t,s,l)},logarithmic(t,e,i){if(0===t)return"0";const s=i[e].significand||t/Math.pow(10,Math.floor(z(t)));return[1,2,3,5,10,15].includes(s)||e>.8*i.length?oe.numeric.call(this,t,e,i):""}};var ae={formatters:oe};const re=Object.create(null),le=Object.create(null);function he(t,e){if(!e)return t;const i=e.split(".");for(let e=0,s=i.length;e<s;++e){const s=i[e];t=t[s]||(t[s]=Object.create(null))}return t}function ce(t,e,i){return"string"==typeof e?b(he(t,e),i):b(he(t,""),e)}class de{constructor(t,e){this.animation=void 0,this.backgroundColor="rgba(0,0,0,0.1)",this.borderColor="rgba(0,0,0,0.1)",this.color="#666",this.datasets={},this.devicePixelRatio=t=>t.chart.platform.getDevicePixelRatio(),this.elements={},this.events=["mousemove","mouseout","click","touchstart","touchmove"],this.font={family:"\'Helvetica Neue\', \'Helvetica\', \'Arial\', sans-serif",size:12,style:"normal",lineHeight:1.2,weight:null},this.hover={},this.hoverBackgroundColor=(t,e)=>te(e.backgroundColor),this.hoverBorderColor=(t,e)=>te(e.borderColor),this.hoverColor=(t,e)=>te(e.color),this.indexAxis="x",this.interaction={mode:"nearest",intersect:!0,includeInvisible:!1},this.maintainAspectRatio=!0,this.onHover=null,this.onClick=null,this.parsing=!0,this.plugins={},this.responsive=!0,this.scale=void 0,this.scales={},this.showLine=!0,this.drawActiveElementsOnTop=!0,this.describe(t),this.apply(e)}set(t,e){return ce(this,t,e)}get(t){return he(this,t)}describe(t,e){return ce(le,t,e)}override(t,e){return ce(re,t,e)}route(t,e,i,s){const n=he(this,t),a=he(this,i),r="_"+e;Object.defineProperties(n,{[r]:{value:n[e],writable:!0},[e]:{enumerable:!0,get(){const t=this[r],e=a[s];return o(t)?Object.assign({},e,t):l(t,e)},set(t){this[r]=t}}})}apply(t){t.forEach((t=>t(this)))}}var ue=new de({_scriptable:t=>!t.startsWith("on"),_indexable:t=>"events"!==t,hover:{_fallback:"interaction"},interaction:{_scriptable:!1,_indexable:!1}},[function(t){t.set("animation",{delay:void 0,duration:1e3,easing:"easeOutQuart",fn:void 0,from:void 0,loop:void 0,to:void 0,type:void 0}),t.describe("animation",{_fallback:!1,_indexable:!1,_scriptable:t=>"onProgress"!==t&&"onComplete"!==t&&"fn"!==t}),t.set("animations",{colors:{type:"color",properties:ie},numbers:{type:"number",properties:ee}}),t.describe("animations",{_fallback:"animation"}),t.set("transitions",{active:{animation:{duration:400}},resize:{animation:{duration:0}},show:{animations:{colors:{from:"transparent"},visible:{type:"boolean",duration:0}}},hide:{animations:{colors:{to:"transparent"},visible:{type:"boolean",easing:"linear",fn:t=>0|t}}}})},function(t){t.set("layout",{autoPadding:!0,padding:{top:0,right:0,bottom:0,left:0}})},function(t){t.set("scale",{display:!0,offset:!1,reverse:!1,beginAtZero:!1,bounds:"ticks",clip:!0,grace:0,grid:{display:!0,lineWidth:1,drawOnChartArea:!0,drawTicks:!0,tickLength:8,tickWidth:(t,e)=>e.lineWidth,tickColor:(t,e)=>e.color,offset:!1},border:{display:!0,dash:[],dashOffset:0,width:1},title:{display:!1,text:"",padding:{top:4,bottom:4}},ticks:{minRotation:0,maxRotation:50,mirror:!1,textStrokeWidth:0,textStrokeColor:"",padding:3,display:!0,autoSkip:!0,autoSkipPadding:3,labelOffset:0,callback:ae.formatters.values,minor:{},major:{},align:"center",crossAlign:"near",showLabelBackdrop:!1,backdropColor:"rgba(255, 255, 255, 0.75)",backdropPadding:2}}),t.route("scale.ticks","color","","color"),t.route("scale.grid","color","","borderColor"),t.route("scale.border","color","","borderColor"),t.route("scale.title","color","","color"),t.describe("scale",{_fallback:!1,_scriptable:t=>!t.startsWith("before")&&!t.startsWith("after")&&"callback"!==t&&"parser"!==t,_indexable:t=>"borderDash"!==t&&"tickBorderDash"!==t&&"dash"!==t}),t.describe("scales",{_fallback:"scale"}),t.describe("scale.ticks",{_scriptable:t=>"backdropPadding"!==t&&"callback"!==t,_indexable:t=>"backdropPadding"!==t})}]);function fe(){return"undefined"!=typeof window&&"undefined"!=typeof document}function ge(t){let e=t.parentNode;return e&&"[object ShadowRoot]"===e.toString()&&(e=e.host),e}function pe(t,e,i){let s;return"string"==typeof t?(s=parseInt(t,10),-1!==t.indexOf("%")&&(s=s/100*e.parentNode[i])):s=t,s}const me=t=>t.ownerDocument.defaultView.getComputedStyle(t,null);function be(t,e){return me(t).getPropertyValue(e)}const xe=["top","right","bottom","left"];function _e(t,e,i){const s={};i=i?"-"+i:"";for(let n=0;n<4;n++){const o=xe[n];s[o]=parseFloat(t[e+"-"+o+i])||0}return s.width=s.left+s.right,s.height=s.top+s.bottom,s}const ye=(t,e,i)=>(t>0||e>0)&&(!i||!i.shadowRoot);function ve(t,e){if("native"in t)return t;const{canvas:i,currentDevicePixelRatio:s}=e,n=me(i),o="border-box"===n.boxSizing,a=_e(n,"padding"),r=_e(n,"border","width"),{x:l,y:h,box:c}=function(t,e){const i=t.touches,s=i&&i.length?i[0]:t,{offsetX:n,offsetY:o}=s;let a,r,l=!1;if(ye(n,o,t.target))a=n,r=o;else{const t=e.getBoundingClientRect();a=s.clientX-t.left,r=s.clientY-t.top,l=!0}return{x:a,y:r,box:l}}(t,i),d=a.left+(c&&r.left),u=a.top+(c&&r.top);let{width:f,height:g}=e;return o&&(f-=a.width+r.width,g-=a.height+r.height),{x:Math.round((l-d)/f*i.width/s),y:Math.round((h-u)/g*i.height/s)}}const Me=t=>Math.round(10*t)/10;function we(t,e,i,s){const n=me(t),o=_e(n,"margin"),a=pe(n.maxWidth,t,"clientWidth")||T,r=pe(n.maxHeight,t,"clientHeight")||T,l=function(t,e,i){let s,n;if(void 0===e||void 0===i){const o=ge(t);if(o){const t=o.getBoundingClientRect(),a=me(o),r=_e(a,"border","width"),l=_e(a,"padding");e=t.width-l.width-r.width,i=t.height-l.height-r.height,s=pe(a.maxWidth,o,"clientWidth"),n=pe(a.maxHeight,o,"clientHeight")}else e=t.clientWidth,i=t.clientHeight}return{width:e,height:i,maxWidth:s||T,maxHeight:n||T}}(t,e,i);let{width:h,height:c}=l;if("content-box"===n.boxSizing){const t=_e(n,"border","width"),e=_e(n,"padding");h-=e.width+t.width,c-=e.height+t.height}h=Math.max(0,h-o.width),c=Math.max(0,s?h/s:c-o.height),h=Me(Math.min(h,a,l.maxWidth)),c=Me(Math.min(c,r,l.maxHeight)),h&&!c&&(c=Me(h/2));return(void 0!==e||void 0!==i)&&s&&l.height&&c>l.height&&(c=l.height,h=Me(Math.floor(c*s))),{width:h,height:c}}function ke(t,e,i){const s=e||1,n=Math.floor(t.height*s),o=Math.floor(t.width*s);t.height=Math.floor(t.height),t.width=Math.floor(t.width);const a=t.canvas;return a.style&&(i||!a.style.height&&!a.style.width)&&(a.style.height=`${t.height}px`,a.style.width=`${t.width}px`),(t.currentDevicePixelRatio!==s||a.height!==n||a.width!==o)&&(t.currentDevicePixelRatio=s,a.height=n,a.width=o,t.ctx.setTransform(s,0,0,s,0,0),!0)}const Se=function(){let t=!1;try{const e={get passive(){return t=!0,!1}};window.addEventListener("test",null,e),window.removeEventListener("test",null,e)}catch(t){}return t}();function Pe(t,e){const i=be(t,e),s=i&&i.match(/^(\\d+)(\\.\\d+)?px$/);return s?+s[1]:void 0}function De(t){return!t||s(t.size)||s(t.family)?null:(t.style?t.style+" ":"")+(t.weight?t.weight+" ":"")+t.size+"px "+t.family}function Ce(t,e,i,s,n){let o=e[n];return o||(o=e[n]=t.measureText(n).width,i.push(n)),o>s&&(s=o),s}function Oe(t,e,i,s){let o=(s=s||{}).data=s.data||{},a=s.garbageCollect=s.garbageCollect||[];s.font!==e&&(o=s.data={},a=s.garbageCollect=[],s.font=e),t.save(),t.font=e;let r=0;const l=i.length;let h,c,d,u,f;for(h=0;h<l;h++)if(u=i[h],null==u||n(u)){if(n(u))for(c=0,d=u.length;c<d;c++)f=u[c],null==f||n(f)||(r=Ce(t,o,a,r,f))}else r=Ce(t,o,a,r,u);t.restore();const g=a.length/2;if(g>i.length){for(h=0;h<g;h++)delete o[a[h]];a.splice(0,g)}return r}function Ae(t,e,i){const s=t.currentDevicePixelRatio,n=0!==i?Math.max(i/2,.5):0;return Math.round((e-n)*s)/s+n}function Te(t,e){(e=e||t.getContext("2d")).save(),e.resetTransform(),e.clearRect(0,0,t.width,t.height),e.restore()}function Le(t,e,i,s){Ee(t,e,i,s,null)}function Ee(t,e,i,s,n){let o,a,r,l,h,c,d,u;const f=e.pointStyle,g=e.rotation,p=e.radius;let m=(g||0)*L;if(f&&"object"==typeof f&&(o=f.toString(),"[object HTMLImageElement]"===o||"[object HTMLCanvasElement]"===o))return t.save(),t.translate(i,s),t.rotate(m),t.drawImage(f,-f.width/2,-f.height/2,f.width,f.height),void t.restore();if(!(isNaN(p)||p<=0)){switch(t.beginPath(),f){default:n?t.ellipse(i,s,n/2,p,0,0,O):t.arc(i,s,p,0,O),t.closePath();break;case"triangle":c=n?n/2:p,t.moveTo(i+Math.sin(m)*c,s-Math.cos(m)*p),m+=I,t.lineTo(i+Math.sin(m)*c,s-Math.cos(m)*p),m+=I,t.lineTo(i+Math.sin(m)*c,s-Math.cos(m)*p),t.closePath();break;case"rectRounded":h=.516*p,l=p-h,a=Math.cos(m+R)*l,d=Math.cos(m+R)*(n?n/2-h:l),r=Math.sin(m+R)*l,u=Math.sin(m+R)*(n?n/2-h:l),t.arc(i-d,s-r,h,m-C,m-E),t.arc(i+u,s-a,h,m-E,m),t.arc(i+d,s+r,h,m,m+E),t.arc(i-u,s+a,h,m+E,m+C),t.closePath();break;case"rect":if(!g){l=Math.SQRT1_2*p,c=n?n/2:l,t.rect(i-c,s-l,2*c,2*l);break}m+=R;case"rectRot":d=Math.cos(m)*(n?n/2:p),a=Math.cos(m)*p,r=Math.sin(m)*p,u=Math.sin(m)*(n?n/2:p),t.moveTo(i-d,s-r),t.lineTo(i+u,s-a),t.lineTo(i+d,s+r),t.lineTo(i-u,s+a),t.closePath();break;case"crossRot":m+=R;case"cross":d=Math.cos(m)*(n?n/2:p),a=Math.cos(m)*p,r=Math.sin(m)*p,u=Math.sin(m)*(n?n/2:p),t.moveTo(i-d,s-r),t.lineTo(i+d,s+r),t.moveTo(i+u,s-a),t.lineTo(i-u,s+a);break;case"star":d=Math.cos(m)*(n?n/2:p),a=Math.cos(m)*p,r=Math.sin(m)*p,u=Math.sin(m)*(n?n/2:p),t.moveTo(i-d,s-r),t.lineTo(i+d,s+r),t.moveTo(i+u,s-a),t.lineTo(i-u,s+a),m+=R,d=Math.cos(m)*(n?n/2:p),a=Math.cos(m)*p,r=Math.sin(m)*p,u=Math.sin(m)*(n?n/2:p),t.moveTo(i-d,s-r),t.lineTo(i+d,s+r),t.moveTo(i+u,s-a),t.lineTo(i-u,s+a);break;case"line":a=n?n/2:Math.cos(m)*p,r=Math.sin(m)*p,t.moveTo(i-a,s-r),t.lineTo(i+a,s+r);break;case"dash":t.moveTo(i,s),t.lineTo(i+Math.cos(m)*(n?n/2:p),s+Math.sin(m)*p);break;case!1:t.closePath()}t.fill(),e.borderWidth>0&&t.stroke()}}function Re(t,e,i){return i=i||.5,!e||t&&t.x>e.left-i&&t.x<e.right+i&&t.y>e.top-i&&t.y<e.bottom+i}function Ie(t,e){t.save(),t.beginPath(),t.rect(e.left,e.top,e.right-e.left,e.bottom-e.top),t.clip()}function ze(t){t.restore()}function Fe(t,e,i,s,n){if(!e)return t.lineTo(i.x,i.y);if("middle"===n){const s=(e.x+i.x)/2;t.lineTo(s,e.y),t.lineTo(s,i.y)}else"after"===n!=!!s?t.lineTo(e.x,i.y):t.lineTo(i.x,e.y);t.lineTo(i.x,i.y)}function Ve(t,e,i,s){if(!e)return t.lineTo(i.x,i.y);t.bezierCurveTo(s?e.cp1x:e.cp2x,s?e.cp1y:e.cp2y,s?i.cp2x:i.cp1x,s?i.cp2y:i.cp1y,i.x,i.y)}function Be(t,e,i,s,n){if(n.strikethrough||n.underline){const o=t.measureText(s),a=e-o.actualBoundingBoxLeft,r=e+o.actualBoundingBoxRight,l=i-o.actualBoundingBoxAscent,h=i+o.actualBoundingBoxDescent,c=n.strikethrough?(l+h)/2:h;t.strokeStyle=t.fillStyle,t.beginPath(),t.lineWidth=n.decorationWidth||2,t.moveTo(a,c),t.lineTo(r,c),t.stroke()}}function We(t,e){const i=t.fillStyle;t.fillStyle=e.color,t.fillRect(e.left,e.top,e.width,e.height),t.fillStyle=i}function Ne(t,e,i,o,a,r={}){const l=n(e)?e:[e],h=r.strokeWidth>0&&""!==r.strokeColor;let c,d;for(t.save(),t.font=a.string,function(t,e){e.translation&&t.translate(e.translation[0],e.translation[1]),s(e.rotation)||t.rotate(e.rotation),e.color&&(t.fillStyle=e.color),e.textAlign&&(t.textAlign=e.textAlign),e.textBaseline&&(t.textBaseline=e.textBaseline)}(t,r),c=0;c<l.length;++c)d=l[c],r.backdrop&&We(t,r.backdrop),h&&(r.strokeColor&&(t.strokeStyle=r.strokeColor),s(r.strokeWidth)||(t.lineWidth=r.strokeWidth),t.strokeText(d,i,o,r.maxWidth)),t.fillText(d,i,o,r.maxWidth),Be(t,i,o,d,r),o+=Number(a.lineHeight);t.restore()}function He(t,e){const{x:i,y:s,w:n,h:o,radius:a}=e;t.arc(i+a.topLeft,s+a.topLeft,a.topLeft,1.5*C,C,!0),t.lineTo(i,s+o-a.bottomLeft),t.arc(i+a.bottomLeft,s+o-a.bottomLeft,a.bottomLeft,C,E,!0),t.lineTo(i+n-a.bottomRight,s+o),t.arc(i+n-a.bottomRight,s+o-a.bottomRight,a.bottomRight,E,0,!0),t.lineTo(i+n,s+a.topRight),t.arc(i+n-a.topRight,s+a.topRight,a.topRight,0,-E,!0),t.lineTo(i+a.topLeft,s)}function je(t,e=[""],i,s,n=(()=>t[0])){const o=i||t;void 0===s&&(s=ti("_fallback",t));const a={[Symbol.toStringTag]:"Object",_cacheable:!0,_scopes:t,_rootScopes:o,_fallback:s,_getTarget:n,override:i=>je([i,...t],e,o,s)};return new Proxy(a,{deleteProperty:(e,i)=>(delete e[i],delete e._keys,delete t[0][i],!0),get:(i,s)=>qe(i,s,(()=>function(t,e,i,s){let n;for(const o of e)if(n=ti(Ue(o,t),i),void 0!==n)return Xe(t,n)?Je(i,s,t,n):n}(s,e,t,i))),getOwnPropertyDescriptor:(t,e)=>Reflect.getOwnPropertyDescriptor(t._scopes[0],e),getPrototypeOf:()=>Reflect.getPrototypeOf(t[0]),has:(t,e)=>ei(t).includes(e),ownKeys:t=>ei(t),set(t,e,i){const s=t._storage||(t._storage=n());return t[e]=s[e]=i,delete t._keys,!0}})}function $e(t,e,i,s){const a={_cacheable:!1,_proxy:t,_context:e,_subProxy:i,_stack:new Set,_descriptors:Ye(t,s),setContext:e=>$e(t,e,i,s),override:n=>$e(t.override(n),e,i,s)};return new Proxy(a,{deleteProperty:(e,i)=>(delete e[i],delete t[i],!0),get:(t,e,i)=>qe(t,e,(()=>function(t,e,i){const{_proxy:s,_context:a,_subProxy:r,_descriptors:l}=t;let h=s[e];S(h)&&l.isScriptable(e)&&(h=function(t,e,i,s){const{_proxy:n,_context:o,_subProxy:a,_stack:r}=i;if(r.has(t))throw new Error("Recursion detected: "+Array.from(r).join("->")+"->"+t);r.add(t);let l=e(o,a||s);r.delete(t),Xe(t,l)&&(l=Je(n._scopes,n,t,l));return l}(e,h,t,i));n(h)&&h.length&&(h=function(t,e,i,s){const{_proxy:n,_context:a,_subProxy:r,_descriptors:l}=i;if(void 0!==a.index&&s(t))return e[a.index%e.length];if(o(e[0])){const i=e,s=n._scopes.filter((t=>t!==i));e=[];for(const o of i){const i=Je(s,n,t,o);e.push($e(i,a,r&&r[t],l))}}return e}(e,h,t,l.isIndexable));Xe(e,h)&&(h=$e(h,a,r&&r[e],l));return h}(t,e,i))),getOwnPropertyDescriptor:(e,i)=>e._descriptors.allKeys?Reflect.has(t,i)?{enumerable:!0,configurable:!0}:void 0:Reflect.getOwnPropertyDescriptor(t,i),getPrototypeOf:()=>Reflect.getPrototypeOf(t),has:(e,i)=>Reflect.has(t,i),ownKeys:()=>Reflect.ownKeys(t),set:(e,i,s)=>(t[i]=s,delete e[i],!0)})}function Ye(t,e={scriptable:!0,indexable:!0}){const{_scriptable:i=e.scriptable,_indexable:s=e.indexable,_allKeys:n=e.allKeys}=t;return{allKeys:n,scriptable:i,indexable:s,isScriptable:S(i)?i:()=>i,isIndexable:S(s)?s:()=>s}}const Ue=(t,e)=>t?t+w(e):e,Xe=(t,e)=>o(e)&&"adapters"!==t&&(null===Object.getPrototypeOf(e)||e.constructor===Object);function qe(t,e,i){if(Object.prototype.hasOwnProperty.call(t,e))return t[e];const s=i();return t[e]=s,s}function Ke(t,e,i){return S(t)?t(e,i):t}const Ge=(t,e)=>!0===t?e:"string"==typeof t?M(e,t):void 0;function Ze(t,e,i,s,n){for(const o of e){const e=Ge(i,o);if(e){t.add(e);const o=Ke(e._fallback,i,n);if(void 0!==o&&o!==i&&o!==s)return o}else if(!1===e&&void 0!==s&&i!==s)return null}return!1}function Je(t,e,i,s){const a=e._rootScopes,r=Ke(e._fallback,i,s),l=[...t,...a],h=new Set;h.add(s);let c=Qe(h,l,i,r||i,s);return null!==c&&((void 0===r||r===i||(c=Qe(h,l,r,c,s),null!==c))&&je(Array.from(h),[""],a,r,(()=>function(t,e,i){const s=t._getTarget();e in s||(s[e]={});const a=s[e];if(n(a)&&o(i))return i;return a||{}}(e,i,s))))}function Qe(t,e,i,s,n){for(;i;)i=Ze(t,e,i,s,n);return i}function ti(t,e){for(const i of e){if(!i)continue;const e=i[t];if(void 0!==e)return e}}function ei(t){let e=t._keys;return e||(e=t._keys=function(t){const e=new Set;for(const i of t)for(const t of Object.keys(i).filter((t=>!t.startsWith("_"))))e.add(t);return Array.from(e)}(t._scopes)),e}function ii(t,e,i,s){const{iScale:n}=t,{key:o="r"}=this._parsing,a=new Array(s);let r,l,h,c;for(r=0,l=s;r<l;++r)h=r+i,c=e[h],a[r]={r:n.parse(M(c,o),h)};return a}const si=Number.EPSILON||1e-14,ni=(t,e)=>e<t.length&&!t[e].skip&&t[e],oi=t=>"x"===t?"y":"x";function ai(t,e,i,s){const n=t.skip?e:t,o=e,a=i.skip?e:i,r=q(o,n),l=q(a,o);let h=r/(r+l),c=l/(r+l);h=isNaN(h)?0:h,c=isNaN(c)?0:c;const d=s*h,u=s*c;return{previous:{x:o.x-d*(a.x-n.x),y:o.y-d*(a.y-n.y)},next:{x:o.x+u*(a.x-n.x),y:o.y+u*(a.y-n.y)}}}function ri(t,e="x"){const i=oi(e),s=t.length,n=Array(s).fill(0),o=Array(s);let a,r,l,h=ni(t,0);for(a=0;a<s;++a)if(r=l,l=h,h=ni(t,a+1),l){if(h){const t=h[e]-l[e];n[a]=0!==t?(h[i]-l[i])/t:0}o[a]=r?h?F(n[a-1])!==F(n[a])?0:(n[a-1]+n[a])/2:n[a-1]:n[a]}!function(t,e,i){const s=t.length;let n,o,a,r,l,h=ni(t,0);for(let c=0;c<s-1;++c)l=h,h=ni(t,c+1),l&&h&&(V(e[c],0,si)?i[c]=i[c+1]=0:(n=i[c]/e[c],o=i[c+1]/e[c],r=Math.pow(n,2)+Math.pow(o,2),r<=9||(a=3/Math.sqrt(r),i[c]=n*a*e[c],i[c+1]=o*a*e[c])))}(t,n,o),function(t,e,i="x"){const s=oi(i),n=t.length;let o,a,r,l=ni(t,0);for(let h=0;h<n;++h){if(a=r,r=l,l=ni(t,h+1),!r)continue;const n=r[i],c=r[s];a&&(o=(n-a[i])/3,r[`cp1${i}`]=n-o,r[`cp1${s}`]=c-o*e[h]),l&&(o=(l[i]-n)/3,r[`cp2${i}`]=n+o,r[`cp2${s}`]=c+o*e[h])}}(t,o,e)}function li(t,e,i){return Math.max(Math.min(t,i),e)}function hi(t,e,i,s,n){let o,a,r,l;if(e.spanGaps&&(t=t.filter((t=>!t.skip))),"monotone"===e.cubicInterpolationMode)ri(t,n);else{let i=s?t[t.length-1]:t[0];for(o=0,a=t.length;o<a;++o)r=t[o],l=ai(i,r,t[Math.min(o+1,a-(s?0:1))%a],e.tension),r.cp1x=l.previous.x,r.cp1y=l.previous.y,r.cp2x=l.next.x,r.cp2y=l.next.y,i=r}e.capBezierPoints&&function(t,e){let i,s,n,o,a,r=Re(t[0],e);for(i=0,s=t.length;i<s;++i)a=o,o=r,r=i<s-1&&Re(t[i+1],e),o&&(n=t[i],a&&(n.cp1x=li(n.cp1x,e.left,e.right),n.cp1y=li(n.cp1y,e.top,e.bottom)),r&&(n.cp2x=li(n.cp2x,e.left,e.right),n.cp2y=li(n.cp2y,e.top,e.bottom)))}(t,i)}const ci=t=>0===t||1===t,di=(t,e,i)=>-Math.pow(2,10*(t-=1))*Math.sin((t-e)*O/i),ui=(t,e,i)=>Math.pow(2,-10*t)*Math.sin((t-e)*O/i)+1,fi={linear:t=>t,easeInQuad:t=>t*t,easeOutQuad:t=>-t*(t-2),easeInOutQuad:t=>(t/=.5)<1?.5*t*t:-.5*(--t*(t-2)-1),easeInCubic:t=>t*t*t,easeOutCubic:t=>(t-=1)*t*t+1,easeInOutCubic:t=>(t/=.5)<1?.5*t*t*t:.5*((t-=2)*t*t+2),easeInQuart:t=>t*t*t*t,easeOutQuart:t=>-((t-=1)*t*t*t-1),easeInOutQuart:t=>(t/=.5)<1?.5*t*t*t*t:-.5*((t-=2)*t*t*t-2),easeInQuint:t=>t*t*t*t*t,easeOutQuint:t=>(t-=1)*t*t*t*t+1,easeInOutQuint:t=>(t/=.5)<1?.5*t*t*t*t*t:.5*((t-=2)*t*t*t*t+2),easeInSine:t=>1-Math.cos(t*E),easeOutSine:t=>Math.sin(t*E),easeInOutSine:t=>-.5*(Math.cos(C*t)-1),easeInExpo:t=>0===t?0:Math.pow(2,10*(t-1)),easeOutExpo:t=>1===t?1:1-Math.pow(2,-10*t),easeInOutExpo:t=>ci(t)?t:t<.5?.5*Math.pow(2,10*(2*t-1)):.5*(2-Math.pow(2,-10*(2*t-1))),easeInCirc:t=>t>=1?t:-(Math.sqrt(1-t*t)-1),easeOutCirc:t=>Math.sqrt(1-(t-=1)*t),easeInOutCirc:t=>(t/=.5)<1?-.5*(Math.sqrt(1-t*t)-1):.5*(Math.sqrt(1-(t-=2)*t)+1),easeInElastic:t=>ci(t)?t:di(t,.075,.3),easeOutElastic:t=>ci(t)?t:ui(t,.075,.3),easeInOutElastic(t){const e=.1125;return ci(t)?t:t<.5?.5*di(2*t,e,.45):.5+.5*ui(2*t-1,e,.45)},easeInBack(t){const e=1.70158;return t*t*((e+1)*t-e)},easeOutBack(t){const e=1.70158;return(t-=1)*t*((e+1)*t+e)+1},easeInOutBack(t){let e=1.70158;return(t/=.5)<1?t*t*((1+(e*=1.525))*t-e)*.5:.5*((t-=2)*t*((1+(e*=1.525))*t+e)+2)},easeInBounce:t=>1-fi.easeOutBounce(1-t),easeOutBounce(t){const e=7.5625,i=2.75;return t<1/i?e*t*t:t<2/i?e*(t-=1.5/i)*t+.75:t<2.5/i?e*(t-=2.25/i)*t+.9375:e*(t-=2.625/i)*t+.984375},easeInOutBounce:t=>t<.5?.5*fi.easeInBounce(2*t):.5*fi.easeOutBounce(2*t-1)+.5};function gi(t,e,i,s){return{x:t.x+i*(e.x-t.x),y:t.y+i*(e.y-t.y)}}function pi(t,e,i,s){return{x:t.x+i*(e.x-t.x),y:"middle"===s?i<.5?t.y:e.y:"after"===s?i<1?t.y:e.y:i>0?e.y:t.y}}function mi(t,e,i,s){const n={x:t.cp2x,y:t.cp2y},o={x:e.cp1x,y:e.cp1y},a=gi(t,n,i),r=gi(n,o,i),l=gi(o,e,i),h=gi(a,r,i),c=gi(r,l,i);return gi(h,c,i)}const bi=/^(normal|(\\d+(?:\\.\\d+)?)(px|em|%)?)$/,xi=/^(normal|italic|initial|inherit|unset|(oblique( -?[0-9]?[0-9]deg)?))$/;function _i(t,e){const i=(""+t).match(bi);if(!i||"normal"===i[1])return 1.2*e;switch(t=+i[2],i[3]){case"px":return t;case"%":t/=100}return e*t}const yi=t=>+t||0;function vi(t,e){const i={},s=o(e),n=s?Object.keys(e):e,a=o(t)?s?i=>l(t[i],t[e[i]]):e=>t[e]:()=>t;for(const t of n)i[t]=yi(a(t));return i}function Mi(t){return vi(t,{top:"y",right:"x",bottom:"y",left:"x"})}function wi(t){return vi(t,["topLeft","topRight","bottomLeft","bottomRight"])}function ki(t){const e=Mi(t);return e.width=e.left+e.right,e.height=e.top+e.bottom,e}function Si(t,e){t=t||{},e=e||ue.font;let i=l(t.size,e.size);"string"==typeof i&&(i=parseInt(i,10));let s=l(t.style,e.style);s&&!(""+s).match(xi)&&(console.warn(\'Invalid font style specified: "\'+s+\'"\'),s=void 0);const n={family:l(t.family,e.family),lineHeight:_i(l(t.lineHeight,e.lineHeight),i),size:i,style:s,weight:l(t.weight,e.weight),string:""};return n.string=De(n),n}function Pi(t,e,i,s){let o,a,r,l=!0;for(o=0,a=t.length;o<a;++o)if(r=t[o],void 0!==r&&(void 0!==e&&"function"==typeof r&&(r=r(e),l=!1),void 0!==i&&n(r)&&(r=r[i%r.length],l=!1),void 0!==r))return s&&!l&&(s.cacheable=!1),r}function Di(t,e,i){const{min:s,max:n}=t,o=c(e,(n-s)/2),a=(t,e)=>i&&0===t?0:t+e;return{min:a(s,-Math.abs(o)),max:a(n,o)}}function Ci(t,e){return Object.assign(Object.create(t),e)}function Oi(t,e,i){return t?function(t,e){return{x:i=>t+t+e-i,setWidth(t){e=t},textAlign:t=>"center"===t?t:"right"===t?"left":"right",xPlus:(t,e)=>t-e,leftForLtr:(t,e)=>t-e}}(e,i):{x:t=>t,setWidth(t){},textAlign:t=>t,xPlus:(t,e)=>t+e,leftForLtr:(t,e)=>t}}function Ai(t,e){let i,s;"ltr"!==e&&"rtl"!==e||(i=t.canvas.style,s=[i.getPropertyValue("direction"),i.getPropertyPriority("direction")],i.setProperty("direction",e,"important"),t.prevTextDirection=s)}function Ti(t,e){void 0!==e&&(delete t.prevTextDirection,t.canvas.style.setProperty("direction",e[0],e[1]))}function Li(t){return"angle"===t?{between:Z,compare:K,normalize:G}:{between:tt,compare:(t,e)=>t-e,normalize:t=>t}}function Ei({start:t,end:e,count:i,loop:s,style:n}){return{start:t%i,end:e%i,loop:s&&(e-t+1)%i==0,style:n}}function Ri(t,e,i){if(!i)return[t];const{property:s,start:n,end:o}=i,a=e.length,{compare:r,between:l,normalize:h}=Li(s),{start:c,end:d,loop:u,style:f}=function(t,e,i){const{property:s,start:n,end:o}=i,{between:a,normalize:r}=Li(s),l=e.length;let h,c,{start:d,end:u,loop:f}=t;if(f){for(d+=l,u+=l,h=0,c=l;h<c&&a(r(e[d%l][s]),n,o);++h)d--,u--;d%=l,u%=l}return u<d&&(u+=l),{start:d,end:u,loop:f,style:t.style}}(t,e,i),g=[];let p,m,b,x=!1,_=null;const y=()=>x||l(n,b,p)&&0!==r(n,b),v=()=>!x||0===r(o,p)||l(o,b,p);for(let t=c,i=c;t<=d;++t)m=e[t%a],m.skip||(p=h(m[s]),p!==b&&(x=l(p,n,o),null===_&&y()&&(_=0===r(p,n)?t:i),null!==_&&v()&&(g.push(Ei({start:_,end:t,loop:u,count:a,style:f})),_=null),i=t,b=p));return null!==_&&g.push(Ei({start:_,end:d,loop:u,count:a,style:f})),g}function Ii(t,e){const i=[],s=t.segments;for(let n=0;n<s.length;n++){const o=Ri(s[n],t.points,e);o.length&&i.push(...o)}return i}function zi(t,e){const i=t.points,s=t.options.spanGaps,n=i.length;if(!n)return[];const o=!!t._loop,{start:a,end:r}=function(t,e,i,s){let n=0,o=e-1;if(i&&!s)for(;n<e&&!t[n].skip;)n++;for(;n<e&&t[n].skip;)n++;for(n%=e,i&&(o+=n);o>n&&t[o%e].skip;)o--;return o%=e,{start:n,end:o}}(i,n,o,s);if(!0===s)return Fi(t,[{start:a,end:r,loop:o}],i,e);return Fi(t,function(t,e,i,s){const n=t.length,o=[];let a,r=e,l=t[e];for(a=e+1;a<=i;++a){const i=t[a%n];i.skip||i.stop?l.skip||(s=!1,o.push({start:e%n,end:(a-1)%n,loop:s}),e=r=i.stop?a:null):(r=a,l.skip&&(e=a)),l=i}return null!==r&&o.push({start:e%n,end:r%n,loop:s}),o}(i,a,r<a?r+n:r,!!t._fullLoop&&0===a&&r===n-1),i,e)}function Fi(t,e,i,s){return s&&s.setContext&&i?function(t,e,i,s){const n=t._chart.getContext(),o=Vi(t.options),{_datasetIndex:a,options:{spanGaps:r}}=t,l=i.length,h=[];let c=o,d=e[0].start,u=d;function f(t,e,s,n){const o=r?-1:1;if(t!==e){for(t+=l;i[t%l].skip;)t-=o;for(;i[e%l].skip;)e+=o;t%l!=e%l&&(h.push({start:t%l,end:e%l,loop:s,style:n}),c=n,d=e%l)}}for(const t of e){d=r?d:t.start;let e,o=i[d%l];for(u=d+1;u<=t.end;u++){const r=i[u%l];e=Vi(s.setContext(Ci(n,{type:"segment",p0:o,p1:r,p0DataIndex:(u-1)%l,p1DataIndex:u%l,datasetIndex:a}))),Bi(e,c)&&f(d,u-1,t.loop,c),o=r,c=e}d<u-1&&f(d,u-1,t.loop,c)}return h}(t,e,i,s):e}function Vi(t){return{backgroundColor:t.backgroundColor,borderCapStyle:t.borderCapStyle,borderDash:t.borderDash,borderDashOffset:t.borderDashOffset,borderJoinStyle:t.borderJoinStyle,borderWidth:t.borderWidth,borderColor:t.borderColor}}function Bi(t,e){if(!e)return!1;const i=[],s=function(t,e){return Jt(e)?(i.includes(e)||i.push(e),i.indexOf(e)):e};return JSON.stringify(t,s)!==JSON.stringify(e,s)}var Wi=Object.freeze({__proto__:null,HALF_PI:E,INFINITY:T,PI:C,PITAU:A,QUARTER_PI:R,RAD_PER_DEG:L,TAU:O,TWO_THIRDS_PI:I,_addGrace:Di,_alignPixel:Ae,_alignStartEnd:ft,_angleBetween:Z,_angleDiff:K,_arrayUnique:lt,_attachContext:$e,_bezierCurveTo:Ve,_bezierInterpolation:mi,_boundSegment:Ri,_boundSegments:Ii,_capitalize:w,_computeSegments:zi,_createResolver:je,_decimalPlaces:U,_deprecated:function(t,e,i,s){void 0!==e&&console.warn(t+\': "\'+i+\'" is deprecated. Please use "\'+s+\'" instead\')},_descriptors:Ye,_elementsEqual:f,_factorize:W,_filterBetween:nt,_getParentNode:ge,_getStartAndCountOfVisiblePoints:pt,_int16Range:Q,_isBetween:tt,_isClickEvent:D,_isDomSupported:fe,_isPointInArea:Re,_limitValue:J,_longestText:Oe,_lookup:et,_lookupByKey:it,_measureText:Ce,_merger:m,_mergerIf:_,_normalizeAngle:G,_parseObjectDataRadialScale:ii,_pointInLine:gi,_readValueToProps:vi,_rlookupByKey:st,_scaleRangesChanged:mt,_setMinAndMaxByKey:j,_splitKey:v,_steppedInterpolation:pi,_steppedLineTo:Fe,_textX:gt,_toLeftRightCenter:ut,_updateBezierControlPoints:hi,addRoundedRectPath:He,almostEquals:V,almostWhole:H,callback:d,clearCanvas:Te,clipArea:Ie,clone:g,color:Qt,createContext:Ci,debounce:dt,defined:k,distanceBetweenPoints:q,drawPoint:Le,drawPointLegend:Ee,each:u,easingEffects:fi,finiteOrDefault:r,fontString:function(t,e,i){return e+" "+t+"px "+i},formatNumber:ne,getAngleFromPoint:X,getHoverColor:te,getMaximumSize:we,getRelativePosition:ve,getRtlAdapter:Oi,getStyle:be,isArray:n,isFinite:a,isFunction:S,isNullOrUndef:s,isNumber:N,isObject:o,isPatternOrGradient:Jt,listenArrayEvents:at,log10:z,merge:b,mergeIf:x,niceNum:B,noop:e,overrideTextDirection:Ai,readUsedSize:Pe,renderText:Ne,requestAnimFrame:ht,resolve:Pi,resolveObjectKey:M,restoreTextDirection:Ti,retinaScale:ke,setsEqual:P,sign:F,splineCurve:ai,splineCurveMonotone:ri,supportsEventListenerOptions:Se,throttled:ct,toDegrees:Y,toDimension:c,toFont:Si,toFontString:De,toLineHeight:_i,toPadding:ki,toPercentage:h,toRadians:$,toTRBL:Mi,toTRBLCorners:wi,uid:i,unclipArea:ze,unlistenArrayEvents:rt,valueOrDefault:l});function Ni(t,e,i,s){const{controller:n,data:o,_sorted:a}=t,r=n._cachedMeta.iScale;if(r&&e===r.axis&&"r"!==e&&a&&o.length){const t=r._reversePixels?st:it;if(!s)return t(o,e,i);if(n._sharedOptions){const s=o[0],n="function"==typeof s.getRange&&s.getRange(e);if(n){const s=t(o,e,i-n),a=t(o,e,i+n);return{lo:s.lo,hi:a.hi}}}}return{lo:0,hi:o.length-1}}function Hi(t,e,i,s,n){const o=t.getSortedVisibleDatasetMetas(),a=i[e];for(let t=0,i=o.length;t<i;++t){const{index:i,data:r}=o[t],{lo:l,hi:h}=Ni(o[t],e,a,n);for(let t=l;t<=h;++t){const e=r[t];e.skip||s(e,i,t)}}}function ji(t,e,i,s,n){const o=[];if(!n&&!t.isPointInArea(e))return o;return Hi(t,i,e,(function(i,a,r){(n||Re(i,t.chartArea,0))&&i.inRange(e.x,e.y,s)&&o.push({element:i,datasetIndex:a,index:r})}),!0),o}function $i(t,e,i,s,n,o){let a=[];const r=function(t){const e=-1!==t.indexOf("x"),i=-1!==t.indexOf("y");return function(t,s){const n=e?Math.abs(t.x-s.x):0,o=i?Math.abs(t.y-s.y):0;return Math.sqrt(Math.pow(n,2)+Math.pow(o,2))}}(i);let l=Number.POSITIVE_INFINITY;return Hi(t,i,e,(function(i,h,c){const d=i.inRange(e.x,e.y,n);if(s&&!d)return;const u=i.getCenterPoint(n);if(!(!!o||t.isPointInArea(u))&&!d)return;const f=r(e,u);f<l?(a=[{element:i,datasetIndex:h,index:c}],l=f):f===l&&a.push({element:i,datasetIndex:h,index:c})})),a}function Yi(t,e,i,s,n,o){return o||t.isPointInArea(e)?"r"!==i||s?$i(t,e,i,s,n,o):function(t,e,i,s){let n=[];return Hi(t,i,e,(function(t,i,o){const{startAngle:a,endAngle:r}=t.getProps(["startAngle","endAngle"],s),{angle:l}=X(t,{x:e.x,y:e.y});Z(l,a,r)&&n.push({element:t,datasetIndex:i,index:o})})),n}(t,e,i,n):[]}function Ui(t,e,i,s,n){const o=[],a="x"===i?"inXRange":"inYRange";let r=!1;return Hi(t,i,e,((t,s,l)=>{t[a](e[i],n)&&(o.push({element:t,datasetIndex:s,index:l}),r=r||t.inRange(e.x,e.y,n))})),s&&!r?[]:o}var Xi={evaluateInteractionItems:Hi,modes:{index(t,e,i,s){const n=ve(e,t),o=i.axis||"x",a=i.includeInvisible||!1,r=i.intersect?ji(t,n,o,s,a):Yi(t,n,o,!1,s,a),l=[];return r.length?(t.getSortedVisibleDatasetMetas().forEach((t=>{const e=r[0].index,i=t.data[e];i&&!i.skip&&l.push({element:i,datasetIndex:t.index,index:e})})),l):[]},dataset(t,e,i,s){const n=ve(e,t),o=i.axis||"xy",a=i.includeInvisible||!1;let r=i.intersect?ji(t,n,o,s,a):Yi(t,n,o,!1,s,a);if(r.length>0){const e=r[0].datasetIndex,i=t.getDatasetMeta(e).data;r=[];for(let t=0;t<i.length;++t)r.push({element:i[t],datasetIndex:e,index:t})}return r},point:(t,e,i,s)=>ji(t,ve(e,t),i.axis||"xy",s,i.includeInvisible||!1),nearest(t,e,i,s){const n=ve(e,t),o=i.axis||"xy",a=i.includeInvisible||!1;return Yi(t,n,o,i.intersect,s,a)},x:(t,e,i,s)=>Ui(t,ve(e,t),"x",i.intersect,s),y:(t,e,i,s)=>Ui(t,ve(e,t),"y",i.intersect,s)}};const qi=["left","top","right","bottom"];function Ki(t,e){return t.filter((t=>t.pos===e))}function Gi(t,e){return t.filter((t=>-1===qi.indexOf(t.pos)&&t.box.axis===e))}function Zi(t,e){return t.sort(((t,i)=>{const s=e?i:t,n=e?t:i;return s.weight===n.weight?s.index-n.index:s.weight-n.weight}))}function Ji(t,e){const i=function(t){const e={};for(const i of t){const{stack:t,pos:s,stackWeight:n}=i;if(!t||!qi.includes(s))continue;const o=e[t]||(e[t]={count:0,placed:0,weight:0,size:0});o.count++,o.weight+=n}return e}(t),{vBoxMaxWidth:s,hBoxMaxHeight:n}=e;let o,a,r;for(o=0,a=t.length;o<a;++o){r=t[o];const{fullSize:a}=r.box,l=i[r.stack],h=l&&r.stackWeight/l.weight;r.horizontal?(r.width=h?h*s:a&&e.availableWidth,r.height=n):(r.width=s,r.height=h?h*n:a&&e.availableHeight)}return i}function Qi(t,e,i,s){return Math.max(t[i],e[i])+Math.max(t[s],e[s])}function ts(t,e){t.top=Math.max(t.top,e.top),t.left=Math.max(t.left,e.left),t.bottom=Math.max(t.bottom,e.bottom),t.right=Math.max(t.right,e.right)}function es(t,e,i,s){const{pos:n,box:a}=i,r=t.maxPadding;if(!o(n)){i.size&&(t[n]-=i.size);const e=s[i.stack]||{size:0,count:1};e.size=Math.max(e.size,i.horizontal?a.height:a.width),i.size=e.size/e.count,t[n]+=i.size}a.getPadding&&ts(r,a.getPadding());const l=Math.max(0,e.outerWidth-Qi(r,t,"left","right")),h=Math.max(0,e.outerHeight-Qi(r,t,"top","bottom")),c=l!==t.w,d=h!==t.h;return t.w=l,t.h=h,i.horizontal?{same:c,other:d}:{same:d,other:c}}function is(t,e){const i=e.maxPadding;function s(t){const s={left:0,top:0,right:0,bottom:0};return t.forEach((t=>{s[t]=Math.max(e[t],i[t])})),s}return s(t?["left","right"]:["top","bottom"])}function ss(t,e,i,s){const n=[];let o,a,r,l,h,c;for(o=0,a=t.length,h=0;o<a;++o){r=t[o],l=r.box,l.update(r.width||e.w,r.height||e.h,is(r.horizontal,e));const{same:a,other:d}=es(e,i,r,s);h|=a&&n.length,c=c||d,l.fullSize||n.push(r)}return h&&ss(n,e,i,s)||c}function ns(t,e,i,s,n){t.top=i,t.left=e,t.right=e+s,t.bottom=i+n,t.width=s,t.height=n}function os(t,e,i,s){const n=i.padding;let{x:o,y:a}=e;for(const r of t){const t=r.box,l=s[r.stack]||{count:1,placed:0,weight:1},h=r.stackWeight/l.weight||1;if(r.horizontal){const s=e.w*h,o=l.size||t.height;k(l.start)&&(a=l.start),t.fullSize?ns(t,n.left,a,i.outerWidth-n.right-n.left,o):ns(t,e.left+l.placed,a,s,o),l.start=a,l.placed+=s,a=t.bottom}else{const s=e.h*h,a=l.size||t.width;k(l.start)&&(o=l.start),t.fullSize?ns(t,o,n.top,a,i.outerHeight-n.bottom-n.top):ns(t,o,e.top+l.placed,a,s),l.start=o,l.placed+=s,o=t.right}}e.x=o,e.y=a}var as={addBox(t,e){t.boxes||(t.boxes=[]),e.fullSize=e.fullSize||!1,e.position=e.position||"top",e.weight=e.weight||0,e._layers=e._layers||function(){return[{z:0,draw(t){e.draw(t)}}]},t.boxes.push(e)},removeBox(t,e){const i=t.boxes?t.boxes.indexOf(e):-1;-1!==i&&t.boxes.splice(i,1)},configure(t,e,i){e.fullSize=i.fullSize,e.position=i.position,e.weight=i.weight},update(t,e,i,s){if(!t)return;const n=ki(t.options.layout.padding),o=Math.max(e-n.width,0),a=Math.max(i-n.height,0),r=function(t){const e=function(t){const e=[];let i,s,n,o,a,r;for(i=0,s=(t||[]).length;i<s;++i)n=t[i],({position:o,options:{stack:a,stackWeight:r=1}}=n),e.push({index:i,box:n,pos:o,horizontal:n.isHorizontal(),weight:n.weight,stack:a&&o+a,stackWeight:r});return e}(t),i=Zi(e.filter((t=>t.box.fullSize)),!0),s=Zi(Ki(e,"left"),!0),n=Zi(Ki(e,"right")),o=Zi(Ki(e,"top"),!0),a=Zi(Ki(e,"bottom")),r=Gi(e,"x"),l=Gi(e,"y");return{fullSize:i,leftAndTop:s.concat(o),rightAndBottom:n.concat(l).concat(a).concat(r),chartArea:Ki(e,"chartArea"),vertical:s.concat(n).concat(l),horizontal:o.concat(a).concat(r)}}(t.boxes),l=r.vertical,h=r.horizontal;u(t.boxes,(t=>{"function"==typeof t.beforeLayout&&t.beforeLayout()}));const c=l.reduce(((t,e)=>e.box.options&&!1===e.box.options.display?t:t+1),0)||1,d=Object.freeze({outerWidth:e,outerHeight:i,padding:n,availableWidth:o,availableHeight:a,vBoxMaxWidth:o/2/c,hBoxMaxHeight:a/2}),f=Object.assign({},n);ts(f,ki(s));const g=Object.assign({maxPadding:f,w:o,h:a,x:n.left,y:n.top},n),p=Ji(l.concat(h),d);ss(r.fullSize,g,d,p),ss(l,g,d,p),ss(h,g,d,p)&&ss(l,g,d,p),function(t){const e=t.maxPadding;function i(i){const s=Math.max(e[i]-t[i],0);return t[i]+=s,s}t.y+=i("top"),t.x+=i("left"),i("right"),i("bottom")}(g),os(r.leftAndTop,g,d,p),g.x+=g.w,g.y+=g.h,os(r.rightAndBottom,g,d,p),t.chartArea={left:g.left,top:g.top,right:g.left+g.w,bottom:g.top+g.h,height:g.h,width:g.w},u(r.chartArea,(e=>{const i=e.box;Object.assign(i,t.chartArea),i.update(g.w,g.h,{left:0,top:0,right:0,bottom:0})}))}};class rs{acquireContext(t,e){}releaseContext(t){return!1}addEventListener(t,e,i){}removeEventListener(t,e,i){}getDevicePixelRatio(){return 1}getMaximumSize(t,e,i,s){return e=Math.max(0,e||t.width),i=i||t.height,{width:e,height:Math.max(0,s?Math.floor(e/s):i)}}isAttached(t){return!0}updateConfig(t){}}class ls extends rs{acquireContext(t){return t&&t.getContext&&t.getContext("2d")||null}updateConfig(t){t.options.animation=!1}}const hs="$chartjs",cs={touchstart:"mousedown",touchmove:"mousemove",touchend:"mouseup",pointerenter:"mouseenter",pointerdown:"mousedown",pointermove:"mousemove",pointerup:"mouseup",pointerleave:"mouseout",pointerout:"mouseout"},ds=t=>null===t||""===t;const us=!!Se&&{passive:!0};function fs(t,e,i){t.canvas.removeEventListener(e,i,us)}function gs(t,e){for(const i of t)if(i===e||i.contains(e))return!0}function ps(t,e,i){const s=t.canvas,n=new MutationObserver((t=>{let e=!1;for(const i of t)e=e||gs(i.addedNodes,s),e=e&&!gs(i.removedNodes,s);e&&i()}));return n.observe(document,{childList:!0,subtree:!0}),n}function ms(t,e,i){const s=t.canvas,n=new MutationObserver((t=>{let e=!1;for(const i of t)e=e||gs(i.removedNodes,s),e=e&&!gs(i.addedNodes,s);e&&i()}));return n.observe(document,{childList:!0,subtree:!0}),n}const bs=new Map;let xs=0;function _s(){const t=window.devicePixelRatio;t!==xs&&(xs=t,bs.forEach(((e,i)=>{i.currentDevicePixelRatio!==t&&e()})))}function ys(t,e,i){const s=t.canvas,n=s&&ge(s);if(!n)return;const o=ct(((t,e)=>{const s=n.clientWidth;i(t,e),s<n.clientWidth&&i()}),window),a=new ResizeObserver((t=>{const e=t[0],i=e.contentRect.width,s=e.contentRect.height;0===i&&0===s||o(i,s)}));return a.observe(n),function(t,e){bs.size||window.addEventListener("resize",_s),bs.set(t,e)}(t,o),a}function vs(t,e,i){i&&i.disconnect(),"resize"===e&&function(t){bs.delete(t),bs.size||window.removeEventListener("resize",_s)}(t)}function Ms(t,e,i){const s=t.canvas,n=ct((e=>{null!==t.ctx&&i(function(t,e){const i=cs[t.type]||t.type,{x:s,y:n}=ve(t,e);return{type:i,chart:e,native:t,x:void 0!==s?s:null,y:void 0!==n?n:null}}(e,t))}),t);return function(t,e,i){t.addEventListener(e,i,us)}(s,e,n),n}class ws extends rs{acquireContext(t,e){const i=t&&t.getContext&&t.getContext("2d");return i&&i.canvas===t?(function(t,e){const i=t.style,s=t.getAttribute("height"),n=t.getAttribute("width");if(t[hs]={initial:{height:s,width:n,style:{display:i.display,height:i.height,width:i.width}}},i.display=i.display||"block",i.boxSizing=i.boxSizing||"border-box",ds(n)){const e=Pe(t,"width");void 0!==e&&(t.width=e)}if(ds(s))if(""===t.style.height)t.height=t.width/(e||2);else{const e=Pe(t,"height");void 0!==e&&(t.height=e)}}(t,e),i):null}releaseContext(t){const e=t.canvas;if(!e[hs])return!1;const i=e[hs].initial;["height","width"].forEach((t=>{const n=i[t];s(n)?e.removeAttribute(t):e.setAttribute(t,n)}));const n=i.style||{};return Object.keys(n).forEach((t=>{e.style[t]=n[t]})),e.width=e.width,delete e[hs],!0}addEventListener(t,e,i){this.removeEventListener(t,e);const s=t.$proxies||(t.$proxies={}),n={attach:ps,detach:ms,resize:ys}[e]||Ms;s[e]=n(t,e,i)}removeEventListener(t,e){const i=t.$proxies||(t.$proxies={}),s=i[e];if(!s)return;({attach:vs,detach:vs,resize:vs}[e]||fs)(t,e,s),i[e]=void 0}getDevicePixelRatio(){return window.devicePixelRatio}getMaximumSize(t,e,i,s){return we(t,e,i,s)}isAttached(t){const e=ge(t);return!(!e||!e.isConnected)}}function ks(t){return!fe()||"undefined"!=typeof OffscreenCanvas&&t instanceof OffscreenCanvas?ls:ws}var Ss=Object.freeze({__proto__:null,BasePlatform:rs,BasicPlatform:ls,DomPlatform:ws,_detectPlatform:ks});const Ps="transparent",Ds={boolean:(t,e,i)=>i>.5?e:t,color(t,e,i){const s=Qt(t||Ps),n=s.valid&&Qt(e||Ps);return n&&n.valid?n.mix(s,i).hexString():e},number:(t,e,i)=>t+(e-t)*i};class Cs{constructor(t,e,i,s){const n=e[i];s=Pi([t.to,s,n,t.from]);const o=Pi([t.from,n,s]);this._active=!0,this._fn=t.fn||Ds[t.type||typeof o],this._easing=fi[t.easing]||fi.linear,this._start=Math.floor(Date.now()+(t.delay||0)),this._duration=this._total=Math.floor(t.duration),this._loop=!!t.loop,this._target=e,this._prop=i,this._from=o,this._to=s,this._promises=void 0}active(){return this._active}update(t,e,i){if(this._active){this._notify(!1);const s=this._target[this._prop],n=i-this._start,o=this._duration-n;this._start=i,this._duration=Math.floor(Math.max(o,t.duration)),this._total+=n,this._loop=!!t.loop,this._to=Pi([t.to,e,s,t.from]),this._from=Pi([t.from,s,e])}}cancel(){this._active&&(this.tick(Date.now()),this._active=!1,this._notify(!1))}tick(t){const e=t-this._start,i=this._duration,s=this._prop,n=this._from,o=this._loop,a=this._to;let r;if(this._active=n!==a&&(o||e<i),!this._active)return this._target[s]=a,void this._notify(!0);e<0?this._target[s]=n:(r=e/i%2,r=o&&r>1?2-r:r,r=this._easing(Math.min(1,Math.max(0,r))),this._target[s]=this._fn(n,a,r))}wait(){const t=this._promises||(this._promises=[]);return new Promise(((e,i)=>{t.push({res:e,rej:i})}))}_notify(t){const e=t?"res":"rej",i=this._promises||[];for(let t=0;t<i.length;t++)i[t][e]()}}class Os{constructor(t,e){this._chart=t,this._properties=new Map,this.configure(e)}configure(t){if(!o(t))return;const e=Object.keys(ue.animation),i=this._properties;Object.getOwnPropertyNames(t).forEach((s=>{const a=t[s];if(!o(a))return;const r={};for(const t of e)r[t]=a[t];(n(a.properties)&&a.properties||[s]).forEach((t=>{t!==s&&i.has(t)||i.set(t,r)}))}))}_animateOptions(t,e){const i=e.options,s=function(t,e){if(!e)return;let i=t.options;if(!i)return void(t.options=e);i.$shared&&(t.options=i=Object.assign({},i,{$shared:!1,$animations:{}}));return i}(t,i);if(!s)return[];const n=this._createAnimations(s,i);return i.$shared&&function(t,e){const i=[],s=Object.keys(e);for(let e=0;e<s.length;e++){const n=t[s[e]];n&&n.active()&&i.push(n.wait())}return Promise.all(i)}(t.options.$animations,i).then((()=>{t.options=i}),(()=>{})),n}_createAnimations(t,e){const i=this._properties,s=[],n=t.$animations||(t.$animations={}),o=Object.keys(e),a=Date.now();let r;for(r=o.length-1;r>=0;--r){const l=o[r];if("$"===l.charAt(0))continue;if("options"===l){s.push(...this._animateOptions(t,e));continue}const h=e[l];let c=n[l];const d=i.get(l);if(c){if(d&&c.active()){c.update(d,h,a);continue}c.cancel()}d&&d.duration?(n[l]=c=new Cs(d,t,l,h),s.push(c)):t[l]=h}return s}update(t,e){if(0===this._properties.size)return void Object.assign(t,e);const i=this._createAnimations(t,e);return i.length?(xt.add(this._chart,i),!0):void 0}}function As(t,e){const i=t&&t.options||{},s=i.reverse,n=void 0===i.min?e:0,o=void 0===i.max?e:0;return{start:s?o:n,end:s?n:o}}function Ts(t,e){const i=[],s=t._getSortedDatasetMetas(e);let n,o;for(n=0,o=s.length;n<o;++n)i.push(s[n].index);return i}function Ls(t,e,i,s={}){const n=t.keys,o="single"===s.mode;let r,l,h,c;if(null!==e){for(r=0,l=n.length;r<l;++r){if(h=+n[r],h===i){if(s.all)continue;break}c=t.values[h],a(c)&&(o||0===e||F(e)===F(c))&&(e+=c)}return e}}function Es(t,e){const i=t&&t.options.stacked;return i||void 0===i&&void 0!==e.stack}function Rs(t,e,i){const s=t[e]||(t[e]={});return s[i]||(s[i]={})}function Is(t,e,i,s){for(const n of e.getMatchingVisibleMetas(s).reverse()){const e=t[n.index];if(i&&e>0||!i&&e<0)return n.index}return null}function zs(t,e){const{chart:i,_cachedMeta:s}=t,n=i._stacks||(i._stacks={}),{iScale:o,vScale:a,index:r}=s,l=o.axis,h=a.axis,c=function(t,e,i){return`${t.id}.${e.id}.${i.stack||i.type}`}(o,a,s),d=e.length;let u;for(let t=0;t<d;++t){const i=e[t],{[l]:o,[h]:d}=i;u=(i._stacks||(i._stacks={}))[h]=Rs(n,c,o),u[r]=d,u._top=Is(u,a,!0,s.type),u._bottom=Is(u,a,!1,s.type);(u._visualValues||(u._visualValues={}))[r]=d}}function Fs(t,e){const i=t.scales;return Object.keys(i).filter((t=>i[t].axis===e)).shift()}function Vs(t,e){const i=t.controller.index,s=t.vScale&&t.vScale.axis;if(s){e=e||t._parsed;for(const t of e){const e=t._stacks;if(!e||void 0===e[s]||void 0===e[s][i])return;delete e[s][i],void 0!==e[s]._visualValues&&void 0!==e[s]._visualValues[i]&&delete e[s]._visualValues[i]}}}const Bs=t=>"reset"===t||"none"===t,Ws=(t,e)=>e?t:Object.assign({},t);class Ns{static defaults={};static datasetElementType=null;static dataElementType=null;constructor(t,e){this.chart=t,this._ctx=t.ctx,this.index=e,this._cachedDataOpts={},this._cachedMeta=this.getMeta(),this._type=this._cachedMeta.type,this.options=void 0,this._parsing=!1,this._data=void 0,this._objectData=void 0,this._sharedOptions=void 0,this._drawStart=void 0,this._drawCount=void 0,this.enableOptionSharing=!1,this.supportsDecimation=!1,this.$context=void 0,this._syncList=[],this.datasetElementType=new.target.datasetElementType,this.dataElementType=new.target.dataElementType,this.initialize()}initialize(){const t=this._cachedMeta;this.configure(),this.linkScales(),t._stacked=Es(t.vScale,t),this.addElements(),this.options.fill&&!this.chart.isPluginEnabled("filler")&&console.warn("Tried to use the \'fill\' option without the \'Filler\' plugin enabled. Please import and register the \'Filler\' plugin and make sure it is not disabled in the options")}updateIndex(t){this.index!==t&&Vs(this._cachedMeta),this.index=t}linkScales(){const t=this.chart,e=this._cachedMeta,i=this.getDataset(),s=(t,e,i,s)=>"x"===t?e:"r"===t?s:i,n=e.xAxisID=l(i.xAxisID,Fs(t,"x")),o=e.yAxisID=l(i.yAxisID,Fs(t,"y")),a=e.rAxisID=l(i.rAxisID,Fs(t,"r")),r=e.indexAxis,h=e.iAxisID=s(r,n,o,a),c=e.vAxisID=s(r,o,n,a);e.xScale=this.getScaleForId(n),e.yScale=this.getScaleForId(o),e.rScale=this.getScaleForId(a),e.iScale=this.getScaleForId(h),e.vScale=this.getScaleForId(c)}getDataset(){return this.chart.data.datasets[this.index]}getMeta(){return this.chart.getDatasetMeta(this.index)}getScaleForId(t){return this.chart.scales[t]}_getOtherScale(t){const e=this._cachedMeta;return t===e.iScale?e.vScale:e.iScale}reset(){this._update("reset")}_destroy(){const t=this._cachedMeta;this._data&&rt(this._data,this),t._stacked&&Vs(t)}_dataCheck(){const t=this.getDataset(),e=t.data||(t.data=[]),i=this._data;if(o(e))this._data=function(t){const e=Object.keys(t),i=new Array(e.length);let s,n,o;for(s=0,n=e.length;s<n;++s)o=e[s],i[s]={x:o,y:t[o]};return i}(e);else if(i!==e){if(i){rt(i,this);const t=this._cachedMeta;Vs(t),t._parsed=[]}e&&Object.isExtensible(e)&&at(e,this),this._syncList=[],this._data=e}}addElements(){const t=this._cachedMeta;this._dataCheck(),this.datasetElementType&&(t.dataset=new this.datasetElementType)}buildOrUpdateElements(t){const e=this._cachedMeta,i=this.getDataset();let s=!1;this._dataCheck();const n=e._stacked;e._stacked=Es(e.vScale,e),e.stack!==i.stack&&(s=!0,Vs(e),e.stack=i.stack),this._resyncElements(t),(s||n!==e._stacked)&&zs(this,e._parsed)}configure(){const t=this.chart.config,e=t.datasetScopeKeys(this._type),i=t.getOptionScopes(this.getDataset(),e,!0);this.options=t.createResolver(i,this.getContext()),this._parsing=this.options.parsing,this._cachedDataOpts={}}parse(t,e){const{_cachedMeta:i,_data:s}=this,{iScale:a,_stacked:r}=i,l=a.axis;let h,c,d,u=0===t&&e===s.length||i._sorted,f=t>0&&i._parsed[t-1];if(!1===this._parsing)i._parsed=s,i._sorted=!0,d=s;else{d=n(s[t])?this.parseArrayData(i,s,t,e):o(s[t])?this.parseObjectData(i,s,t,e):this.parsePrimitiveData(i,s,t,e);const a=()=>null===c[l]||f&&c[l]<f[l];for(h=0;h<e;++h)i._parsed[h+t]=c=d[h],u&&(a()&&(u=!1),f=c);i._sorted=u}r&&zs(this,d)}parsePrimitiveData(t,e,i,s){const{iScale:n,vScale:o}=t,a=n.axis,r=o.axis,l=n.getLabels(),h=n===o,c=new Array(s);let d,u,f;for(d=0,u=s;d<u;++d)f=d+i,c[d]={[a]:h||n.parse(l[f],f),[r]:o.parse(e[f],f)};return c}parseArrayData(t,e,i,s){const{xScale:n,yScale:o}=t,a=new Array(s);let r,l,h,c;for(r=0,l=s;r<l;++r)h=r+i,c=e[h],a[r]={x:n.parse(c[0],h),y:o.parse(c[1],h)};return a}parseObjectData(t,e,i,s){const{xScale:n,yScale:o}=t,{xAxisKey:a="x",yAxisKey:r="y"}=this._parsing,l=new Array(s);let h,c,d,u;for(h=0,c=s;h<c;++h)d=h+i,u=e[d],l[h]={x:n.parse(M(u,a),d),y:o.parse(M(u,r),d)};return l}getParsed(t){return this._cachedMeta._parsed[t]}getDataElement(t){return this._cachedMeta.data[t]}applyStack(t,e,i){const s=this.chart,n=this._cachedMeta,o=e[t.axis];return Ls({keys:Ts(s,!0),values:e._stacks[t.axis]._visualValues},o,n.index,{mode:i})}updateRangeFromParsed(t,e,i,s){const n=i[e.axis];let o=null===n?NaN:n;const a=s&&i._stacks[e.axis];s&&a&&(s.values=a,o=Ls(s,n,this._cachedMeta.index)),t.min=Math.min(t.min,o),t.max=Math.max(t.max,o)}getMinMax(t,e){const i=this._cachedMeta,s=i._parsed,n=i._sorted&&t===i.iScale,o=s.length,r=this._getOtherScale(t),l=((t,e,i)=>t&&!e.hidden&&e._stacked&&{keys:Ts(i,!0),values:null})(e,i,this.chart),h={min:Number.POSITIVE_INFINITY,max:Number.NEGATIVE_INFINITY},{min:c,max:d}=function(t){const{min:e,max:i,minDefined:s,maxDefined:n}=t.getUserBounds();return{min:s?e:Number.NEGATIVE_INFINITY,max:n?i:Number.POSITIVE_INFINITY}}(r);let u,f;function g(){f=s[u];const e=f[r.axis];return!a(f[t.axis])||c>e||d<e}for(u=0;u<o&&(g()||(this.updateRangeFromParsed(h,t,f,l),!n));++u);if(n)for(u=o-1;u>=0;--u)if(!g()){this.updateRangeFromParsed(h,t,f,l);break}return h}getAllParsedValues(t){const e=this._cachedMeta._parsed,i=[];let s,n,o;for(s=0,n=e.length;s<n;++s)o=e[s][t.axis],a(o)&&i.push(o);return i}getMaxOverflow(){return!1}getLabelAndValue(t){const e=this._cachedMeta,i=e.iScale,s=e.vScale,n=this.getParsed(t);return{label:i?""+i.getLabelForValue(n[i.axis]):"",value:s?""+s.getLabelForValue(n[s.axis]):""}}_update(t){const e=this._cachedMeta;this.update(t||"default"),e._clip=function(t){let e,i,s,n;return o(t)?(e=t.top,i=t.right,s=t.bottom,n=t.left):e=i=s=n=t,{top:e,right:i,bottom:s,left:n,disabled:!1===t}}(l(this.options.clip,function(t,e,i){if(!1===i)return!1;const s=As(t,i),n=As(e,i);return{top:n.end,right:s.end,bottom:n.start,left:s.start}}(e.xScale,e.yScale,this.getMaxOverflow())))}update(t){}draw(){const t=this._ctx,e=this.chart,i=this._cachedMeta,s=i.data||[],n=e.chartArea,o=[],a=this._drawStart||0,r=this._drawCount||s.length-a,l=this.options.drawActiveElementsOnTop;let h;for(i.dataset&&i.dataset.draw(t,n,a,r),h=a;h<a+r;++h){const e=s[h];e.hidden||(e.active&&l?o.push(e):e.draw(t,n))}for(h=0;h<o.length;++h)o[h].draw(t,n)}getStyle(t,e){const i=e?"active":"default";return void 0===t&&this._cachedMeta.dataset?this.resolveDatasetElementOptions(i):this.resolveDataElementOptions(t||0,i)}getContext(t,e,i){const s=this.getDataset();let n;if(t>=0&&t<this._cachedMeta.data.length){const e=this._cachedMeta.data[t];n=e.$context||(e.$context=function(t,e,i){return Ci(t,{active:!1,dataIndex:e,parsed:void 0,raw:void 0,element:i,index:e,mode:"default",type:"data"})}(this.getContext(),t,e)),n.parsed=this.getParsed(t),n.raw=s.data[t],n.index=n.dataIndex=t}else n=this.$context||(this.$context=function(t,e){return Ci(t,{active:!1,dataset:void 0,datasetIndex:e,index:e,mode:"default",type:"dataset"})}(this.chart.getContext(),this.index)),n.dataset=s,n.index=n.datasetIndex=this.index;return n.active=!!e,n.mode=i,n}resolveDatasetElementOptions(t){return this._resolveElementOptions(this.datasetElementType.id,t)}resolveDataElementOptions(t,e){return this._resolveElementOptions(this.dataElementType.id,e,t)}_resolveElementOptions(t,e="default",i){const s="active"===e,n=this._cachedDataOpts,o=t+"-"+e,a=n[o],r=this.enableOptionSharing&&k(i);if(a)return Ws(a,r);const l=this.chart.config,h=l.datasetElementScopeKeys(this._type,t),c=s?[`${t}Hover`,"hover",t,""]:[t,""],d=l.getOptionScopes(this.getDataset(),h),u=Object.keys(ue.elements[t]),f=l.resolveNamedOptions(d,u,(()=>this.getContext(i,s,e)),c);return f.$shared&&(f.$shared=r,n[o]=Object.freeze(Ws(f,r))),f}_resolveAnimations(t,e,i){const s=this.chart,n=this._cachedDataOpts,o=`animation-${e}`,a=n[o];if(a)return a;let r;if(!1!==s.options.animation){const s=this.chart.config,n=s.datasetAnimationScopeKeys(this._type,e),o=s.getOptionScopes(this.getDataset(),n);r=s.createResolver(o,this.getContext(t,i,e))}const l=new Os(s,r&&r.animations);return r&&r._cacheable&&(n[o]=Object.freeze(l)),l}getSharedOptions(t){if(t.$shared)return this._sharedOptions||(this._sharedOptions=Object.assign({},t))}includeOptions(t,e){return!e||Bs(t)||this.chart._animationsDisabled}_getSharedOptions(t,e){const i=this.resolveDataElementOptions(t,e),s=this._sharedOptions,n=this.getSharedOptions(i),o=this.includeOptions(e,n)||n!==s;return this.updateSharedOptions(n,e,i),{sharedOptions:n,includeOptions:o}}updateElement(t,e,i,s){Bs(s)?Object.assign(t,i):this._resolveAnimations(e,s).update(t,i)}updateSharedOptions(t,e,i){t&&!Bs(e)&&this._resolveAnimations(void 0,e).update(t,i)}_setStyle(t,e,i,s){t.active=s;const n=this.getStyle(e,s);this._resolveAnimations(e,i,s).update(t,{options:!s&&this.getSharedOptions(n)||n})}removeHoverStyle(t,e,i){this._setStyle(t,i,"active",!1)}setHoverStyle(t,e,i){this._setStyle(t,i,"active",!0)}_removeDatasetHoverStyle(){const t=this._cachedMeta.dataset;t&&this._setStyle(t,void 0,"active",!1)}_setDatasetHoverStyle(){const t=this._cachedMeta.dataset;t&&this._setStyle(t,void 0,"active",!0)}_resyncElements(t){const e=this._data,i=this._cachedMeta.data;for(const[t,e,i]of this._syncList)this[t](e,i);this._syncList=[];const s=i.length,n=e.length,o=Math.min(n,s);o&&this.parse(0,o),n>s?this._insertElements(s,n-s,t):n<s&&this._removeElements(n,s-n)}_insertElements(t,e,i=!0){const s=this._cachedMeta,n=s.data,o=t+e;let a;const r=t=>{for(t.length+=e,a=t.length-1;a>=o;a--)t[a]=t[a-e]};for(r(n),a=t;a<o;++a)n[a]=new this.dataElementType;this._parsing&&r(s._parsed),this.parse(t,e),i&&this.updateElements(n,t,e,"reset")}updateElements(t,e,i,s){}_removeElements(t,e){const i=this._cachedMeta;if(this._parsing){const s=i._parsed.splice(t,e);i._stacked&&Vs(i,s)}i.data.splice(t,e)}_sync(t){if(this._parsing)this._syncList.push(t);else{const[e,i,s]=t;this[e](i,s)}this.chart._dataChanges.push([this.index,...t])}_onDataPush(){const t=arguments.length;this._sync(["_insertElements",this.getDataset().data.length-t,t])}_onDataPop(){this._sync(["_removeElements",this._cachedMeta.data.length-1,1])}_onDataShift(){this._sync(["_removeElements",0,1])}_onDataSplice(t,e){e&&this._sync(["_removeElements",t,e]);const i=arguments.length-2;i&&this._sync(["_insertElements",t,i])}_onDataUnshift(){this._sync(["_insertElements",0,arguments.length])}}class Hs{static defaults={};static defaultRoutes=void 0;x;y;active=!1;options;$animations;tooltipPosition(t){const{x:e,y:i}=this.getProps(["x","y"],t);return{x:e,y:i}}hasValue(){return N(this.x)&&N(this.y)}getProps(t,e){const i=this.$animations;if(!e||!i)return this;const s={};return t.forEach((t=>{s[t]=i[t]&&i[t].active()?i[t]._to:this[t]})),s}}function js(t,e){const i=t.options.ticks,n=function(t){const e=t.options.offset,i=t._tickSize(),s=t._length/i+(e?0:1),n=t._maxLength/i;return Math.floor(Math.min(s,n))}(t),o=Math.min(i.maxTicksLimit||n,n),a=i.major.enabled?function(t){const e=[];let i,s;for(i=0,s=t.length;i<s;i++)t[i].major&&e.push(i);return e}(e):[],r=a.length,l=a[0],h=a[r-1],c=[];if(r>o)return function(t,e,i,s){let n,o=0,a=i[0];for(s=Math.ceil(s),n=0;n<t.length;n++)n===a&&(e.push(t[n]),o++,a=i[o*s])}(e,c,a,r/o),c;const d=function(t,e,i){const s=function(t){const e=t.length;let i,s;if(e<2)return!1;for(s=t[0],i=1;i<e;++i)if(t[i]-t[i-1]!==s)return!1;return s}(t),n=e.length/i;if(!s)return Math.max(n,1);const o=W(s);for(let t=0,e=o.length-1;t<e;t++){const e=o[t];if(e>n)return e}return Math.max(n,1)}(a,e,o);if(r>0){let t,i;const n=r>1?Math.round((h-l)/(r-1)):null;for($s(e,c,d,s(n)?0:l-n,l),t=0,i=r-1;t<i;t++)$s(e,c,d,a[t],a[t+1]);return $s(e,c,d,h,s(n)?e.length:h+n),c}return $s(e,c,d),c}function $s(t,e,i,s,n){const o=l(s,0),a=Math.min(l(n,t.length),t.length);let r,h,c,d=0;for(i=Math.ceil(i),n&&(r=n-s,i=r/Math.floor(r/i)),c=o;c<0;)d++,c=Math.round(o+d*i);for(h=Math.max(o,0);h<a;h++)h===c&&(e.push(t[h]),d++,c=Math.round(o+d*i))}const Ys=(t,e,i)=>"top"===e||"left"===e?t[e]+i:t[e]-i,Us=(t,e)=>Math.min(e||t,t);function Xs(t,e){const i=[],s=t.length/e,n=t.length;let o=0;for(;o<n;o+=s)i.push(t[Math.floor(o)]);return i}function qs(t,e,i){const s=t.ticks.length,n=Math.min(e,s-1),o=t._startPixel,a=t._endPixel,r=1e-6;let l,h=t.getPixelForTick(n);if(!(i&&(l=1===s?Math.max(h-o,a-h):0===e?(t.getPixelForTick(1)-h)/2:(h-t.getPixelForTick(n-1))/2,h+=n<e?l:-l,h<o-r||h>a+r)))return h}function Ks(t){return t.drawTicks?t.tickLength:0}function Gs(t,e){if(!t.display)return 0;const i=Si(t.font,e),s=ki(t.padding);return(n(t.text)?t.text.length:1)*i.lineHeight+s.height}function Zs(t,e,i){let s=ut(t);return(i&&"right"!==e||!i&&"right"===e)&&(s=(t=>"left"===t?"right":"right"===t?"left":t)(s)),s}class Js extends Hs{constructor(t){super(),this.id=t.id,this.type=t.type,this.options=void 0,this.ctx=t.ctx,this.chart=t.chart,this.top=void 0,this.bottom=void 0,this.left=void 0,this.right=void 0,this.width=void 0,this.height=void 0,this._margins={left:0,right:0,top:0,bottom:0},this.maxWidth=void 0,this.maxHeight=void 0,this.paddingTop=void 0,this.paddingBottom=void 0,this.paddingLeft=void 0,this.paddingRight=void 0,this.axis=void 0,this.labelRotation=void 0,this.min=void 0,this.max=void 0,this._range=void 0,this.ticks=[],this._gridLineItems=null,this._labelItems=null,this._labelSizes=null,this._length=0,this._maxLength=0,this._longestTextCache={},this._startPixel=void 0,this._endPixel=void 0,this._reversePixels=!1,this._userMax=void 0,this._userMin=void 0,this._suggestedMax=void 0,this._suggestedMin=void 0,this._ticksLength=0,this._borderValue=0,this._cache={},this._dataLimitsCached=!1,this.$context=void 0}init(t){this.options=t.setContext(this.getContext()),this.axis=t.axis,this._userMin=this.parse(t.min),this._userMax=this.parse(t.max),this._suggestedMin=this.parse(t.suggestedMin),this._suggestedMax=this.parse(t.suggestedMax)}parse(t,e){return t}getUserBounds(){let{_userMin:t,_userMax:e,_suggestedMin:i,_suggestedMax:s}=this;return t=r(t,Number.POSITIVE_INFINITY),e=r(e,Number.NEGATIVE_INFINITY),i=r(i,Number.POSITIVE_INFINITY),s=r(s,Number.NEGATIVE_INFINITY),{min:r(t,i),max:r(e,s),minDefined:a(t),maxDefined:a(e)}}getMinMax(t){let e,{min:i,max:s,minDefined:n,maxDefined:o}=this.getUserBounds();if(n&&o)return{min:i,max:s};const a=this.getMatchingVisibleMetas();for(let r=0,l=a.length;r<l;++r)e=a[r].controller.getMinMax(this,t),n||(i=Math.min(i,e.min)),o||(s=Math.max(s,e.max));return i=o&&i>s?s:i,s=n&&i>s?i:s,{min:r(i,r(s,i)),max:r(s,r(i,s))}}getPadding(){return{left:this.paddingLeft||0,top:this.paddingTop||0,right:this.paddingRight||0,bottom:this.paddingBottom||0}}getTicks(){return this.ticks}getLabels(){const t=this.chart.data;return this.options.labels||(this.isHorizontal()?t.xLabels:t.yLabels)||t.labels||[]}getLabelItems(t=this.chart.chartArea){return this._labelItems||(this._labelItems=this._computeLabelItems(t))}beforeLayout(){this._cache={},this._dataLimitsCached=!1}beforeUpdate(){d(this.options.beforeUpdate,[this])}update(t,e,i){const{beginAtZero:s,grace:n,ticks:o}=this.options,a=o.sampleSize;this.beforeUpdate(),this.maxWidth=t,this.maxHeight=e,this._margins=i=Object.assign({left:0,right:0,top:0,bottom:0},i),this.ticks=null,this._labelSizes=null,this._gridLineItems=null,this._labelItems=null,this.beforeSetDimensions(),this.setDimensions(),this.afterSetDimensions(),this._maxLength=this.isHorizontal()?this.width+i.left+i.right:this.height+i.top+i.bottom,this._dataLimitsCached||(this.beforeDataLimits(),this.determineDataLimits(),this.afterDataLimits(),this._range=Di(this,n,s),this._dataLimitsCached=!0),this.beforeBuildTicks(),this.ticks=this.buildTicks()||[],this.afterBuildTicks();const r=a<this.ticks.length;this._convertTicksToLabels(r?Xs(this.ticks,a):this.ticks),this.configure(),this.beforeCalculateLabelRotation(),this.calculateLabelRotation(),this.afterCalculateLabelRotation(),o.display&&(o.autoSkip||"auto"===o.source)&&(this.ticks=js(this,this.ticks),this._labelSizes=null,this.afterAutoSkip()),r&&this._convertTicksToLabels(this.ticks),this.beforeFit(),this.fit(),this.afterFit(),this.afterUpdate()}configure(){let t,e,i=this.options.reverse;this.isHorizontal()?(t=this.left,e=this.right):(t=this.top,e=this.bottom,i=!i),this._startPixel=t,this._endPixel=e,this._reversePixels=i,this._length=e-t,this._alignToPixels=this.options.alignToPixels}afterUpdate(){d(this.options.afterUpdate,[this])}beforeSetDimensions(){d(this.options.beforeSetDimensions,[this])}setDimensions(){this.isHorizontal()?(this.width=this.maxWidth,this.left=0,this.right=this.width):(this.height=this.maxHeight,this.top=0,this.bottom=this.height),this.paddingLeft=0,this.paddingTop=0,this.paddingRight=0,this.paddingBottom=0}afterSetDimensions(){d(this.options.afterSetDimensions,[this])}_callHooks(t){this.chart.notifyPlugins(t,this.getContext()),d(this.options[t],[this])}beforeDataLimits(){this._callHooks("beforeDataLimits")}determineDataLimits(){}afterDataLimits(){this._callHooks("afterDataLimits")}beforeBuildTicks(){this._callHooks("beforeBuildTicks")}buildTicks(){return[]}afterBuildTicks(){this._callHooks("afterBuildTicks")}beforeTickToLabelConversion(){d(this.options.beforeTickToLabelConversion,[this])}generateTickLabels(t){const e=this.options.ticks;let i,s,n;for(i=0,s=t.length;i<s;i++)n=t[i],n.label=d(e.callback,[n.value,i,t],this)}afterTickToLabelConversion(){d(this.options.afterTickToLabelConversion,[this])}beforeCalculateLabelRotation(){d(this.options.beforeCalculateLabelRotation,[this])}calculateLabelRotation(){const t=this.options,e=t.ticks,i=Us(this.ticks.length,t.ticks.maxTicksLimit),s=e.minRotation||0,n=e.maxRotation;let o,a,r,l=s;if(!this._isVisible()||!e.display||s>=n||i<=1||!this.isHorizontal())return void(this.labelRotation=s);const h=this._getLabelSizes(),c=h.widest.width,d=h.highest.height,u=J(this.chart.width-c,0,this.maxWidth);o=t.offset?this.maxWidth/i:u/(i-1),c+6>o&&(o=u/(i-(t.offset?.5:1)),a=this.maxHeight-Ks(t.grid)-e.padding-Gs(t.title,this.chart.options.font),r=Math.sqrt(c*c+d*d),l=Y(Math.min(Math.asin(J((h.highest.height+6)/o,-1,1)),Math.asin(J(a/r,-1,1))-Math.asin(J(d/r,-1,1)))),l=Math.max(s,Math.min(n,l))),this.labelRotation=l}afterCalculateLabelRotation(){d(this.options.afterCalculateLabelRotation,[this])}afterAutoSkip(){}beforeFit(){d(this.options.beforeFit,[this])}fit(){const t={width:0,height:0},{chart:e,options:{ticks:i,title:s,grid:n}}=this,o=this._isVisible(),a=this.isHorizontal();if(o){const o=Gs(s,e.options.font);if(a?(t.width=this.maxWidth,t.height=Ks(n)+o):(t.height=this.maxHeight,t.width=Ks(n)+o),i.display&&this.ticks.length){const{first:e,last:s,widest:n,highest:o}=this._getLabelSizes(),r=2*i.padding,l=$(this.labelRotation),h=Math.cos(l),c=Math.sin(l);if(a){const e=i.mirror?0:c*n.width+h*o.height;t.height=Math.min(this.maxHeight,t.height+e+r)}else{const e=i.mirror?0:h*n.width+c*o.height;t.width=Math.min(this.maxWidth,t.width+e+r)}this._calculatePadding(e,s,c,h)}}this._handleMargins(),a?(this.width=this._length=e.width-this._margins.left-this._margins.right,this.height=t.height):(this.width=t.width,this.height=this._length=e.height-this._margins.top-this._margins.bottom)}_calculatePadding(t,e,i,s){const{ticks:{align:n,padding:o},position:a}=this.options,r=0!==this.labelRotation,l="top"!==a&&"x"===this.axis;if(this.isHorizontal()){const a=this.getPixelForTick(0)-this.left,h=this.right-this.getPixelForTick(this.ticks.length-1);let c=0,d=0;r?l?(c=s*t.width,d=i*e.height):(c=i*t.height,d=s*e.width):"start"===n?d=e.width:"end"===n?c=t.width:"inner"!==n&&(c=t.width/2,d=e.width/2),this.paddingLeft=Math.max((c-a+o)*this.width/(this.width-a),0),this.paddingRight=Math.max((d-h+o)*this.width/(this.width-h),0)}else{let i=e.height/2,s=t.height/2;"start"===n?(i=0,s=t.height):"end"===n&&(i=e.height,s=0),this.paddingTop=i+o,this.paddingBottom=s+o}}_handleMargins(){this._margins&&(this._margins.left=Math.max(this.paddingLeft,this._margins.left),this._margins.top=Math.max(this.paddingTop,this._margins.top),this._margins.right=Math.max(this.paddingRight,this._margins.right),this._margins.bottom=Math.max(this.paddingBottom,this._margins.bottom))}afterFit(){d(this.options.afterFit,[this])}isHorizontal(){const{axis:t,position:e}=this.options;return"top"===e||"bottom"===e||"x"===t}isFullSize(){return this.options.fullSize}_convertTicksToLabels(t){let e,i;for(this.beforeTickToLabelConversion(),this.generateTickLabels(t),e=0,i=t.length;e<i;e++)s(t[e].label)&&(t.splice(e,1),i--,e--);this.afterTickToLabelConversion()}_getLabelSizes(){let t=this._labelSizes;if(!t){const e=this.options.ticks.sampleSize;let i=this.ticks;e<i.length&&(i=Xs(i,e)),this._labelSizes=t=this._computeLabelSizes(i,i.length,this.options.ticks.maxTicksLimit)}return t}_computeLabelSizes(t,e,i){const{ctx:o,_longestTextCache:a}=this,r=[],l=[],h=Math.floor(e/Us(e,i));let c,d,f,g,p,m,b,x,_,y,v,M=0,w=0;for(c=0;c<e;c+=h){if(g=t[c].label,p=this._resolveTickFontOptions(c),o.font=m=p.string,b=a[m]=a[m]||{data:{},gc:[]},x=p.lineHeight,_=y=0,s(g)||n(g)){if(n(g))for(d=0,f=g.length;d<f;++d)v=g[d],s(v)||n(v)||(_=Ce(o,b.data,b.gc,_,v),y+=x)}else _=Ce(o,b.data,b.gc,_,g),y=x;r.push(_),l.push(y),M=Math.max(_,M),w=Math.max(y,w)}!function(t,e){u(t,(t=>{const i=t.gc,s=i.length/2;let n;if(s>e){for(n=0;n<s;++n)delete t.data[i[n]];i.splice(0,s)}}))}(a,e);const k=r.indexOf(M),S=l.indexOf(w),P=t=>({width:r[t]||0,height:l[t]||0});return{first:P(0),last:P(e-1),widest:P(k),highest:P(S),widths:r,heights:l}}getLabelForValue(t){return t}getPixelForValue(t,e){return NaN}getValueForPixel(t){}getPixelForTick(t){const e=this.ticks;return t<0||t>e.length-1?null:this.getPixelForValue(e[t].value)}getPixelForDecimal(t){this._reversePixels&&(t=1-t);const e=this._startPixel+t*this._length;return Q(this._alignToPixels?Ae(this.chart,e,0):e)}getDecimalForPixel(t){const e=(t-this._startPixel)/this._length;return this._reversePixels?1-e:e}getBasePixel(){return this.getPixelForValue(this.getBaseValue())}getBaseValue(){const{min:t,max:e}=this;return t<0&&e<0?e:t>0&&e>0?t:0}getContext(t){const e=this.ticks||[];if(t>=0&&t<e.length){const i=e[t];return i.$context||(i.$context=function(t,e,i){return Ci(t,{tick:i,index:e,type:"tick"})}(this.getContext(),t,i))}return this.$context||(this.$context=Ci(this.chart.getContext(),{scale:this,type:"scale"}))}_tickSize(){const t=this.options.ticks,e=$(this.labelRotation),i=Math.abs(Math.cos(e)),s=Math.abs(Math.sin(e)),n=this._getLabelSizes(),o=t.autoSkipPadding||0,a=n?n.widest.width+o:0,r=n?n.highest.height+o:0;return this.isHorizontal()?r*i>a*s?a/i:r/s:r*s<a*i?r/i:a/s}_isVisible(){const t=this.options.display;return"auto"!==t?!!t:this.getMatchingVisibleMetas().length>0}_computeGridLineItems(t){const e=this.axis,i=this.chart,s=this.options,{grid:n,position:a,border:r}=s,h=n.offset,c=this.isHorizontal(),d=this.ticks.length+(h?1:0),u=Ks(n),f=[],g=r.setContext(this.getContext()),p=g.display?g.width:0,m=p/2,b=function(t){return Ae(i,t,p)};let x,_,y,v,M,w,k,S,P,D,C,O;if("top"===a)x=b(this.bottom),w=this.bottom-u,S=x-m,D=b(t.top)+m,O=t.bottom;else if("bottom"===a)x=b(this.top),D=t.top,O=b(t.bottom)-m,w=x+m,S=this.top+u;else if("left"===a)x=b(this.right),M=this.right-u,k=x-m,P=b(t.left)+m,C=t.right;else if("right"===a)x=b(this.left),P=t.left,C=b(t.right)-m,M=x+m,k=this.left+u;else if("x"===e){if("center"===a)x=b((t.top+t.bottom)/2+.5);else if(o(a)){const t=Object.keys(a)[0],e=a[t];x=b(this.chart.scales[t].getPixelForValue(e))}D=t.top,O=t.bottom,w=x+m,S=w+u}else if("y"===e){if("center"===a)x=b((t.left+t.right)/2);else if(o(a)){const t=Object.keys(a)[0],e=a[t];x=b(this.chart.scales[t].getPixelForValue(e))}M=x-m,k=M-u,P=t.left,C=t.right}const A=l(s.ticks.maxTicksLimit,d),T=Math.max(1,Math.ceil(d/A));for(_=0;_<d;_+=T){const t=this.getContext(_),e=n.setContext(t),s=r.setContext(t),o=e.lineWidth,a=e.color,l=s.dash||[],d=s.dashOffset,u=e.tickWidth,g=e.tickColor,p=e.tickBorderDash||[],m=e.tickBorderDashOffset;y=qs(this,_,h),void 0!==y&&(v=Ae(i,y,o),c?M=k=P=C=v:w=S=D=O=v,f.push({tx1:M,ty1:w,tx2:k,ty2:S,x1:P,y1:D,x2:C,y2:O,width:o,color:a,borderDash:l,borderDashOffset:d,tickWidth:u,tickColor:g,tickBorderDash:p,tickBorderDashOffset:m}))}return this._ticksLength=d,this._borderValue=x,f}_computeLabelItems(t){const e=this.axis,i=this.options,{position:s,ticks:a}=i,r=this.isHorizontal(),l=this.ticks,{align:h,crossAlign:c,padding:d,mirror:u}=a,f=Ks(i.grid),g=f+d,p=u?-d:g,m=-$(this.labelRotation),b=[];let x,_,y,v,M,w,k,S,P,D,C,O,A="middle";if("top"===s)w=this.bottom-p,k=this._getXAxisLabelAlignment();else if("bottom"===s)w=this.top+p,k=this._getXAxisLabelAlignment();else if("left"===s){const t=this._getYAxisLabelAlignment(f);k=t.textAlign,M=t.x}else if("right"===s){const t=this._getYAxisLabelAlignment(f);k=t.textAlign,M=t.x}else if("x"===e){if("center"===s)w=(t.top+t.bottom)/2+g;else if(o(s)){const t=Object.keys(s)[0],e=s[t];w=this.chart.scales[t].getPixelForValue(e)+g}k=this._getXAxisLabelAlignment()}else if("y"===e){if("center"===s)M=(t.left+t.right)/2-g;else if(o(s)){const t=Object.keys(s)[0],e=s[t];M=this.chart.scales[t].getPixelForValue(e)}k=this._getYAxisLabelAlignment(f).textAlign}"y"===e&&("start"===h?A="top":"end"===h&&(A="bottom"));const T=this._getLabelSizes();for(x=0,_=l.length;x<_;++x){y=l[x],v=y.label;const t=a.setContext(this.getContext(x));S=this.getPixelForTick(x)+a.labelOffset,P=this._resolveTickFontOptions(x),D=P.lineHeight,C=n(v)?v.length:1;const e=C/2,i=t.color,o=t.textStrokeColor,h=t.textStrokeWidth;let d,f=k;if(r?(M=S,"inner"===k&&(f=x===_-1?this.options.reverse?"left":"right":0===x?this.options.reverse?"right":"left":"center"),O="top"===s?"near"===c||0!==m?-C*D+D/2:"center"===c?-T.highest.height/2-e*D+D:-T.highest.height+D/2:"near"===c||0!==m?D/2:"center"===c?T.highest.height/2-e*D:T.highest.height-C*D,u&&(O*=-1),0===m||t.showLabelBackdrop||(M+=D/2*Math.sin(m))):(w=S,O=(1-C)*D/2),t.showLabelBackdrop){const e=ki(t.backdropPadding),i=T.heights[x],s=T.widths[x];let n=O-e.top,o=0-e.left;switch(A){case"middle":n-=i/2;break;case"bottom":n-=i}switch(k){case"center":o-=s/2;break;case"right":o-=s}d={left:o,top:n,width:s+e.width,height:i+e.height,color:t.backdropColor}}b.push({label:v,font:P,textOffset:O,options:{rotation:m,color:i,strokeColor:o,strokeWidth:h,textAlign:f,textBaseline:A,translation:[M,w],backdrop:d}})}return b}_getXAxisLabelAlignment(){const{position:t,ticks:e}=this.options;if(-$(this.labelRotation))return"top"===t?"left":"right";let i="center";return"start"===e.align?i="left":"end"===e.align?i="right":"inner"===e.align&&(i="inner"),i}_getYAxisLabelAlignment(t){const{position:e,ticks:{crossAlign:i,mirror:s,padding:n}}=this.options,o=t+n,a=this._getLabelSizes().widest.width;let r,l;return"left"===e?s?(l=this.right+n,"near"===i?r="left":"center"===i?(r="center",l+=a/2):(r="right",l+=a)):(l=this.right-o,"near"===i?r="right":"center"===i?(r="center",l-=a/2):(r="left",l=this.left)):"right"===e?s?(l=this.left+n,"near"===i?r="right":"center"===i?(r="center",l-=a/2):(r="left",l-=a)):(l=this.left+o,"near"===i?r="left":"center"===i?(r="center",l+=a/2):(r="right",l=this.right)):r="right",{textAlign:r,x:l}}_computeLabelArea(){if(this.options.ticks.mirror)return;const t=this.chart,e=this.options.position;return"left"===e||"right"===e?{top:0,left:this.left,bottom:t.height,right:this.right}:"top"===e||"bottom"===e?{top:this.top,left:0,bottom:this.bottom,right:t.width}:void 0}drawBackground(){const{ctx:t,options:{backgroundColor:e},left:i,top:s,width:n,height:o}=this;e&&(t.save(),t.fillStyle=e,t.fillRect(i,s,n,o),t.restore())}getLineWidthForValue(t){const e=this.options.grid;if(!this._isVisible()||!e.display)return 0;const i=this.ticks.findIndex((e=>e.value===t));if(i>=0){return e.setContext(this.getContext(i)).lineWidth}return 0}drawGrid(t){const e=this.options.grid,i=this.ctx,s=this._gridLineItems||(this._gridLineItems=this._computeGridLineItems(t));let n,o;const a=(t,e,s)=>{s.width&&s.color&&(i.save(),i.lineWidth=s.width,i.strokeStyle=s.color,i.setLineDash(s.borderDash||[]),i.lineDashOffset=s.borderDashOffset,i.beginPath(),i.moveTo(t.x,t.y),i.lineTo(e.x,e.y),i.stroke(),i.restore())};if(e.display)for(n=0,o=s.length;n<o;++n){const t=s[n];e.drawOnChartArea&&a({x:t.x1,y:t.y1},{x:t.x2,y:t.y2},t),e.drawTicks&&a({x:t.tx1,y:t.ty1},{x:t.tx2,y:t.ty2},{color:t.tickColor,width:t.tickWidth,borderDash:t.tickBorderDash,borderDashOffset:t.tickBorderDashOffset})}}drawBorder(){const{chart:t,ctx:e,options:{border:i,grid:s}}=this,n=i.setContext(this.getContext()),o=i.display?n.width:0;if(!o)return;const a=s.setContext(this.getContext(0)).lineWidth,r=this._borderValue;let l,h,c,d;this.isHorizontal()?(l=Ae(t,this.left,o)-o/2,h=Ae(t,this.right,a)+a/2,c=d=r):(c=Ae(t,this.top,o)-o/2,d=Ae(t,this.bottom,a)+a/2,l=h=r),e.save(),e.lineWidth=n.width,e.strokeStyle=n.color,e.beginPath(),e.moveTo(l,c),e.lineTo(h,d),e.stroke(),e.restore()}drawLabels(t){if(!this.options.ticks.display)return;const e=this.ctx,i=this._computeLabelArea();i&&Ie(e,i);const s=this.getLabelItems(t);for(const t of s){const i=t.options,s=t.font;Ne(e,t.label,0,t.textOffset,s,i)}i&&ze(e)}drawTitle(){const{ctx:t,options:{position:e,title:i,reverse:s}}=this;if(!i.display)return;const a=Si(i.font),r=ki(i.padding),l=i.align;let h=a.lineHeight/2;"bottom"===e||"center"===e||o(e)?(h+=r.bottom,n(i.text)&&(h+=a.lineHeight*(i.text.length-1))):h+=r.top;const{titleX:c,titleY:d,maxWidth:u,rotation:f}=function(t,e,i,s){const{top:n,left:a,bottom:r,right:l,chart:h}=t,{chartArea:c,scales:d}=h;let u,f,g,p=0;const m=r-n,b=l-a;if(t.isHorizontal()){if(f=ft(s,a,l),o(i)){const t=Object.keys(i)[0],s=i[t];g=d[t].getPixelForValue(s)+m-e}else g="center"===i?(c.bottom+c.top)/2+m-e:Ys(t,i,e);u=l-a}else{if(o(i)){const t=Object.keys(i)[0],s=i[t];f=d[t].getPixelForValue(s)-b+e}else f="center"===i?(c.left+c.right)/2-b+e:Ys(t,i,e);g=ft(s,r,n),p="left"===i?-E:E}return{titleX:f,titleY:g,maxWidth:u,rotation:p}}(this,h,e,l);Ne(t,i.text,0,0,a,{color:i.color,maxWidth:u,rotation:f,textAlign:Zs(l,e,s),textBaseline:"middle",translation:[c,d]})}draw(t){this._isVisible()&&(this.drawBackground(),this.drawGrid(t),this.drawBorder(),this.drawTitle(),this.drawLabels(t))}_layers(){const t=this.options,e=t.ticks&&t.ticks.z||0,i=l(t.grid&&t.grid.z,-1),s=l(t.border&&t.border.z,0);return this._isVisible()&&this.draw===Js.prototype.draw?[{z:i,draw:t=>{this.drawBackground(),this.drawGrid(t),this.drawTitle()}},{z:s,draw:()=>{this.drawBorder()}},{z:e,draw:t=>{this.drawLabels(t)}}]:[{z:e,draw:t=>{this.draw(t)}}]}getMatchingVisibleMetas(t){const e=this.chart.getSortedVisibleDatasetMetas(),i=this.axis+"AxisID",s=[];let n,o;for(n=0,o=e.length;n<o;++n){const o=e[n];o[i]!==this.id||t&&o.type!==t||s.push(o)}return s}_resolveTickFontOptions(t){return Si(this.options.ticks.setContext(this.getContext(t)).font)}_maxDigits(){const t=this._resolveTickFontOptions(0).lineHeight;return(this.isHorizontal()?this.width:this.height)/t}}class Qs{constructor(t,e,i){this.type=t,this.scope=e,this.override=i,this.items=Object.create(null)}isForType(t){return Object.prototype.isPrototypeOf.call(this.type.prototype,t.prototype)}register(t){const e=Object.getPrototypeOf(t);let i;(function(t){return"id"in t&&"defaults"in t})(e)&&(i=this.register(e));const s=this.items,n=t.id,o=this.scope+"."+n;if(!n)throw new Error("class does not have id: "+t);return n in s||(s[n]=t,function(t,e,i){const s=b(Object.create(null),[i?ue.get(i):{},ue.get(e),t.defaults]);ue.set(e,s),t.defaultRoutes&&function(t,e){Object.keys(e).forEach((i=>{const s=i.split("."),n=s.pop(),o=[t].concat(s).join("."),a=e[i].split("."),r=a.pop(),l=a.join(".");ue.route(o,n,l,r)}))}(e,t.defaultRoutes);t.descriptors&&ue.describe(e,t.descriptors)}(t,o,i),this.override&&ue.override(t.id,t.overrides)),o}get(t){return this.items[t]}unregister(t){const e=this.items,i=t.id,s=this.scope;i in e&&delete e[i],s&&i in ue[s]&&(delete ue[s][i],this.override&&delete re[i])}}class tn{constructor(){this.controllers=new Qs(Ns,"datasets",!0),this.elements=new Qs(Hs,"elements"),this.plugins=new Qs(Object,"plugins"),this.scales=new Qs(Js,"scales"),this._typedRegistries=[this.controllers,this.scales,this.elements]}add(...t){this._each("register",t)}remove(...t){this._each("unregister",t)}addControllers(...t){this._each("register",t,this.controllers)}addElements(...t){this._each("register",t,this.elements)}addPlugins(...t){this._each("register",t,this.plugins)}addScales(...t){this._each("register",t,this.scales)}getController(t){return this._get(t,this.controllers,"controller")}getElement(t){return this._get(t,this.elements,"element")}getPlugin(t){return this._get(t,this.plugins,"plugin")}getScale(t){return this._get(t,this.scales,"scale")}removeControllers(...t){this._each("unregister",t,this.controllers)}removeElements(...t){this._each("unregister",t,this.elements)}removePlugins(...t){this._each("unregister",t,this.plugins)}removeScales(...t){this._each("unregister",t,this.scales)}_each(t,e,i){[...e].forEach((e=>{const s=i||this._getRegistryForType(e);i||s.isForType(e)||s===this.plugins&&e.id?this._exec(t,s,e):u(e,(e=>{const s=i||this._getRegistryForType(e);this._exec(t,s,e)}))}))}_exec(t,e,i){const s=w(t);d(i["before"+s],[],i),e[t](i),d(i["after"+s],[],i)}_getRegistryForType(t){for(let e=0;e<this._typedRegistries.length;e++){const i=this._typedRegistries[e];if(i.isForType(t))return i}return this.plugins}_get(t,e,i){const s=e.get(t);if(void 0===s)throw new Error(\'"\'+t+\'" is not a registered \'+i+".");return s}}var en=new tn;class sn{constructor(){this._init=[]}notify(t,e,i,s){"beforeInit"===e&&(this._init=this._createDescriptors(t,!0),this._notify(this._init,t,"install"));const n=s?this._descriptors(t).filter(s):this._descriptors(t),o=this._notify(n,t,e,i);return"afterDestroy"===e&&(this._notify(n,t,"stop"),this._notify(this._init,t,"uninstall")),o}_notify(t,e,i,s){s=s||{};for(const n of t){const t=n.plugin;if(!1===d(t[i],[e,s,n.options],t)&&s.cancelable)return!1}return!0}invalidate(){s(this._cache)||(this._oldCache=this._cache,this._cache=void 0)}_descriptors(t){if(this._cache)return this._cache;const e=this._cache=this._createDescriptors(t);return this._notifyStateChanges(t),e}_createDescriptors(t,e){const i=t&&t.config,s=l(i.options&&i.options.plugins,{}),n=function(t){const e={},i=[],s=Object.keys(en.plugins.items);for(let t=0;t<s.length;t++)i.push(en.getPlugin(s[t]));const n=t.plugins||[];for(let t=0;t<n.length;t++){const s=n[t];-1===i.indexOf(s)&&(i.push(s),e[s.id]=!0)}return{plugins:i,localIds:e}}(i);return!1!==s||e?function(t,{plugins:e,localIds:i},s,n){const o=[],a=t.getContext();for(const r of e){const e=r.id,l=nn(s[e],n);null!==l&&o.push({plugin:r,options:on(t.config,{plugin:r,local:i[e]},l,a)})}return o}(t,n,s,e):[]}_notifyStateChanges(t){const e=this._oldCache||[],i=this._cache,s=(t,e)=>t.filter((t=>!e.some((e=>t.plugin.id===e.plugin.id))));this._notify(s(e,i),t,"stop"),this._notify(s(i,e),t,"start")}}function nn(t,e){return e||!1!==t?!0===t?{}:t:null}function on(t,{plugin:e,local:i},s,n){const o=t.pluginScopeKeys(e),a=t.getOptionScopes(s,o);return i&&e.defaults&&a.push(e.defaults),t.createResolver(a,n,[""],{scriptable:!1,indexable:!1,allKeys:!0})}function an(t,e){const i=ue.datasets[t]||{};return((e.datasets||{})[t]||{}).indexAxis||e.indexAxis||i.indexAxis||"x"}function rn(t){if("x"===t||"y"===t||"r"===t)return t}function ln(t,...e){if(rn(t))return t;for(const s of e){const e=s.axis||("top"===(i=s.position)||"bottom"===i?"x":"left"===i||"right"===i?"y":void 0)||t.length>1&&rn(t[0].toLowerCase());if(e)return e}var i;throw new Error(`Cannot determine type of \'${t}\' axis. Please provide \'axis\' or \'position\' option.`)}function hn(t,e,i){if(i[e+"AxisID"]===t)return{axis:e}}function cn(t,e){const i=re[t.type]||{scales:{}},s=e.scales||{},n=an(t.type,e),a=Object.create(null);return Object.keys(s).forEach((e=>{const r=s[e];if(!o(r))return console.error(`Invalid scale configuration for scale: ${e}`);if(r._proxy)return console.warn(`Ignoring resolver passed as options for scale: ${e}`);const l=ln(e,r,function(t,e){if(e.data&&e.data.datasets){const i=e.data.datasets.filter((e=>e.xAxisID===t||e.yAxisID===t));if(i.length)return hn(t,"x",i[0])||hn(t,"y",i[0])}return{}}(e,t),ue.scales[r.type]),h=function(t,e){return t===e?"_index_":"_value_"}(l,n),c=i.scales||{};a[e]=x(Object.create(null),[{axis:l},r,c[l],c[h]])})),t.data.datasets.forEach((i=>{const n=i.type||t.type,o=i.indexAxis||an(n,e),r=(re[n]||{}).scales||{};Object.keys(r).forEach((t=>{const e=function(t,e){let i=t;return"_index_"===t?i=e:"_value_"===t&&(i="x"===e?"y":"x"),i}(t,o),n=i[e+"AxisID"]||e;a[n]=a[n]||Object.create(null),x(a[n],[{axis:e},s[n],r[t]])}))})),Object.keys(a).forEach((t=>{const e=a[t];x(e,[ue.scales[e.type],ue.scale])})),a}function dn(t){const e=t.options||(t.options={});e.plugins=l(e.plugins,{}),e.scales=cn(t,e)}function un(t){return(t=t||{}).datasets=t.datasets||[],t.labels=t.labels||[],t}const fn=new Map,gn=new Set;function pn(t,e){let i=fn.get(t);return i||(i=e(),fn.set(t,i),gn.add(i)),i}const mn=(t,e,i)=>{const s=M(e,i);void 0!==s&&t.add(s)};class bn{constructor(t){this._config=function(t){return(t=t||{}).data=un(t.data),dn(t),t}(t),this._scopeCache=new Map,this._resolverCache=new Map}get platform(){return this._config.platform}get type(){return this._config.type}set type(t){this._config.type=t}get data(){return this._config.data}set data(t){this._config.data=un(t)}get options(){return this._config.options}set options(t){this._config.options=t}get plugins(){return this._config.plugins}update(){const t=this._config;this.clearCache(),dn(t)}clearCache(){this._scopeCache.clear(),this._resolverCache.clear()}datasetScopeKeys(t){return pn(t,(()=>[[`datasets.${t}`,""]]))}datasetAnimationScopeKeys(t,e){return pn(`${t}.transition.${e}`,(()=>[[`datasets.${t}.transitions.${e}`,`transitions.${e}`],[`datasets.${t}`,""]]))}datasetElementScopeKeys(t,e){return pn(`${t}-${e}`,(()=>[[`datasets.${t}.elements.${e}`,`datasets.${t}`,`elements.${e}`,""]]))}pluginScopeKeys(t){const e=t.id;return pn(`${this.type}-plugin-${e}`,(()=>[[`plugins.${e}`,...t.additionalOptionScopes||[]]]))}_cachedScopes(t,e){const i=this._scopeCache;let s=i.get(t);return s&&!e||(s=new Map,i.set(t,s)),s}getOptionScopes(t,e,i){const{options:s,type:n}=this,o=this._cachedScopes(t,i),a=o.get(e);if(a)return a;const r=new Set;e.forEach((e=>{t&&(r.add(t),e.forEach((e=>mn(r,t,e)))),e.forEach((t=>mn(r,s,t))),e.forEach((t=>mn(r,re[n]||{},t))),e.forEach((t=>mn(r,ue,t))),e.forEach((t=>mn(r,le,t)))}));const l=Array.from(r);return 0===l.length&&l.push(Object.create(null)),gn.has(e)&&o.set(e,l),l}chartOptionScopes(){const{options:t,type:e}=this;return[t,re[e]||{},ue.datasets[e]||{},{type:e},ue,le]}resolveNamedOptions(t,e,i,s=[""]){const o={$shared:!0},{resolver:a,subPrefixes:r}=xn(this._resolverCache,t,s);let l=a;if(function(t,e){const{isScriptable:i,isIndexable:s}=Ye(t);for(const o of e){const e=i(o),a=s(o),r=(a||e)&&t[o];if(e&&(S(r)||_n(r))||a&&n(r))return!0}return!1}(a,e)){o.$shared=!1;l=$e(a,i=S(i)?i():i,this.createResolver(t,i,r))}for(const t of e)o[t]=l[t];return o}createResolver(t,e,i=[""],s){const{resolver:n}=xn(this._resolverCache,t,i);return o(e)?$e(n,e,void 0,s):n}}function xn(t,e,i){let s=t.get(e);s||(s=new Map,t.set(e,s));const n=i.join();let o=s.get(n);if(!o){o={resolver:je(e,i),subPrefixes:i.filter((t=>!t.toLowerCase().includes("hover")))},s.set(n,o)}return o}const _n=t=>o(t)&&Object.getOwnPropertyNames(t).reduce(((e,i)=>e||S(t[i])),!1);const yn=["top","bottom","left","right","chartArea"];function vn(t,e){return"top"===t||"bottom"===t||-1===yn.indexOf(t)&&"x"===e}function Mn(t,e){return function(i,s){return i[t]===s[t]?i[e]-s[e]:i[t]-s[t]}}function wn(t){const e=t.chart,i=e.options.animation;e.notifyPlugins("afterRender"),d(i&&i.onComplete,[t],e)}function kn(t){const e=t.chart,i=e.options.animation;d(i&&i.onProgress,[t],e)}function Sn(t){return fe()&&"string"==typeof t?t=document.getElementById(t):t&&t.length&&(t=t[0]),t&&t.canvas&&(t=t.canvas),t}const Pn={},Dn=t=>{const e=Sn(t);return Object.values(Pn).filter((t=>t.canvas===e)).pop()};function Cn(t,e,i){const s=Object.keys(t);for(const n of s){const s=+n;if(s>=e){const o=t[n];delete t[n],(i>0||s>e)&&(t[s+i]=o)}}}function On(t,e,i){return t.options.clip?t[i]:e[i]}class An{static defaults=ue;static instances=Pn;static overrides=re;static registry=en;static version="4.4.0";static getChart=Dn;static register(...t){en.add(...t),Tn()}static unregister(...t){en.remove(...t),Tn()}constructor(t,e){const s=this.config=new bn(e),n=Sn(t),o=Dn(n);if(o)throw new Error("Canvas is already in use. Chart with ID \'"+o.id+"\' must be destroyed before the canvas with ID \'"+o.canvas.id+"\' can be reused.");const a=s.createResolver(s.chartOptionScopes(),this.getContext());this.platform=new(s.platform||ks(n)),this.platform.updateConfig(s);const r=this.platform.acquireContext(n,a.aspectRatio),l=r&&r.canvas,h=l&&l.height,c=l&&l.width;this.id=i(),this.ctx=r,this.canvas=l,this.width=c,this.height=h,this._options=a,this._aspectRatio=this.aspectRatio,this._layers=[],this._metasets=[],this._stacks=void 0,this.boxes=[],this.currentDevicePixelRatio=void 0,this.chartArea=void 0,this._active=[],this._lastEvent=void 0,this._listeners={},this._responsiveListeners=void 0,this._sortedMetasets=[],this.scales={},this._plugins=new sn,this.$proxies={},this._hiddenIndices={},this.attached=!1,this._animationsDisabled=void 0,this.$context=void 0,this._doResize=dt((t=>this.update(t)),a.resizeDelay||0),this._dataChanges=[],Pn[this.id]=this,r&&l?(xt.listen(this,"complete",wn),xt.listen(this,"progress",kn),this._initialize(),this.attached&&this.update()):console.error("Failed to create chart: can\'t acquire context from the given item")}get aspectRatio(){const{options:{aspectRatio:t,maintainAspectRatio:e},width:i,height:n,_aspectRatio:o}=this;return s(t)?e&&o?o:n?i/n:null:t}get data(){return this.config.data}set data(t){this.config.data=t}get options(){return this._options}set options(t){this.config.options=t}get registry(){return en}_initialize(){return this.notifyPlugins("beforeInit"),this.options.responsive?this.resize():ke(this,this.options.devicePixelRatio),this.bindEvents(),this.notifyPlugins("afterInit"),this}clear(){return Te(this.canvas,this.ctx),this}stop(){return xt.stop(this),this}resize(t,e){xt.running(this)?this._resizeBeforeDraw={width:t,height:e}:this._resize(t,e)}_resize(t,e){const i=this.options,s=this.canvas,n=i.maintainAspectRatio&&this.aspectRatio,o=this.platform.getMaximumSize(s,t,e,n),a=i.devicePixelRatio||this.platform.getDevicePixelRatio(),r=this.width?"resize":"attach";this.width=o.width,this.height=o.height,this._aspectRatio=this.aspectRatio,ke(this,a,!0)&&(this.notifyPlugins("resize",{size:o}),d(i.onResize,[this,o],this),this.attached&&this._doResize(r)&&this.render())}ensureScalesHaveIDs(){u(this.options.scales||{},((t,e)=>{t.id=e}))}buildOrUpdateScales(){const t=this.options,e=t.scales,i=this.scales,s=Object.keys(i).reduce(((t,e)=>(t[e]=!1,t)),{});let n=[];e&&(n=n.concat(Object.keys(e).map((t=>{const i=e[t],s=ln(t,i),n="r"===s,o="x"===s;return{options:i,dposition:n?"chartArea":o?"bottom":"left",dtype:n?"radialLinear":o?"category":"linear"}})))),u(n,(e=>{const n=e.options,o=n.id,a=ln(o,n),r=l(n.type,e.dtype);void 0!==n.position&&vn(n.position,a)===vn(e.dposition)||(n.position=e.dposition),s[o]=!0;let h=null;if(o in i&&i[o].type===r)h=i[o];else{h=new(en.getScale(r))({id:o,type:r,ctx:this.ctx,chart:this}),i[h.id]=h}h.init(n,t)})),u(s,((t,e)=>{t||delete i[e]})),u(i,(t=>{as.configure(this,t,t.options),as.addBox(this,t)}))}_updateMetasets(){const t=this._metasets,e=this.data.datasets.length,i=t.length;if(t.sort(((t,e)=>t.index-e.index)),i>e){for(let t=e;t<i;++t)this._destroyDatasetMeta(t);t.splice(e,i-e)}this._sortedMetasets=t.slice(0).sort(Mn("order","index"))}_removeUnreferencedMetasets(){const{_metasets:t,data:{datasets:e}}=this;t.length>e.length&&delete this._stacks,t.forEach(((t,i)=>{0===e.filter((e=>e===t._dataset)).length&&this._destroyDatasetMeta(i)}))}buildOrUpdateControllers(){const t=[],e=this.data.datasets;let i,s;for(this._removeUnreferencedMetasets(),i=0,s=e.length;i<s;i++){const s=e[i];let n=this.getDatasetMeta(i);const o=s.type||this.config.type;if(n.type&&n.type!==o&&(this._destroyDatasetMeta(i),n=this.getDatasetMeta(i)),n.type=o,n.indexAxis=s.indexAxis||an(o,this.options),n.order=s.order||0,n.index=i,n.label=""+s.label,n.visible=this.isDatasetVisible(i),n.controller)n.controller.updateIndex(i),n.controller.linkScales();else{const e=en.getController(o),{datasetElementType:s,dataElementType:a}=ue.datasets[o];Object.assign(e,{dataElementType:en.getElement(a),datasetElementType:s&&en.getElement(s)}),n.controller=new e(this,i),t.push(n.controller)}}return this._updateMetasets(),t}_resetElements(){u(this.data.datasets,((t,e)=>{this.getDatasetMeta(e).controller.reset()}),this)}reset(){this._resetElements(),this.notifyPlugins("reset")}update(t){const e=this.config;e.update();const i=this._options=e.createResolver(e.chartOptionScopes(),this.getContext()),s=this._animationsDisabled=!i.animation;if(this._updateScales(),this._checkEventBindings(),this._updateHiddenIndices(),this._plugins.invalidate(),!1===this.notifyPlugins("beforeUpdate",{mode:t,cancelable:!0}))return;const n=this.buildOrUpdateControllers();this.notifyPlugins("beforeElementsUpdate");let o=0;for(let t=0,e=this.data.datasets.length;t<e;t++){const{controller:e}=this.getDatasetMeta(t),i=!s&&-1===n.indexOf(e);e.buildOrUpdateElements(i),o=Math.max(+e.getMaxOverflow(),o)}o=this._minPadding=i.layout.autoPadding?o:0,this._updateLayout(o),s||u(n,(t=>{t.reset()})),this._updateDatasets(t),this.notifyPlugins("afterUpdate",{mode:t}),this._layers.sort(Mn("z","_idx"));const{_active:a,_lastEvent:r}=this;r?this._eventHandler(r,!0):a.length&&this._updateHoverStyles(a,a,!0),this.render()}_updateScales(){u(this.scales,(t=>{as.removeBox(this,t)})),this.ensureScalesHaveIDs(),this.buildOrUpdateScales()}_checkEventBindings(){const t=this.options,e=new Set(Object.keys(this._listeners)),i=new Set(t.events);P(e,i)&&!!this._responsiveListeners===t.responsive||(this.unbindEvents(),this.bindEvents())}_updateHiddenIndices(){const{_hiddenIndices:t}=this,e=this._getUniformDataChanges()||[];for(const{method:i,start:s,count:n}of e){Cn(t,s,"_removeElements"===i?-n:n)}}_getUniformDataChanges(){const t=this._dataChanges;if(!t||!t.length)return;this._dataChanges=[];const e=this.data.datasets.length,i=e=>new Set(t.filter((t=>t[0]===e)).map(((t,e)=>e+","+t.splice(1).join(",")))),s=i(0);for(let t=1;t<e;t++)if(!P(s,i(t)))return;return Array.from(s).map((t=>t.split(","))).map((t=>({method:t[1],start:+t[2],count:+t[3]})))}_updateLayout(t){if(!1===this.notifyPlugins("beforeLayout",{cancelable:!0}))return;as.update(this,this.width,this.height,t);const e=this.chartArea,i=e.width<=0||e.height<=0;this._layers=[],u(this.boxes,(t=>{i&&"chartArea"===t.position||(t.configure&&t.configure(),this._layers.push(...t._layers()))}),this),this._layers.forEach(((t,e)=>{t._idx=e})),this.notifyPlugins("afterLayout")}_updateDatasets(t){if(!1!==this.notifyPlugins("beforeDatasetsUpdate",{mode:t,cancelable:!0})){for(let t=0,e=this.data.datasets.length;t<e;++t)this.getDatasetMeta(t).controller.configure();for(let e=0,i=this.data.datasets.length;e<i;++e)this._updateDataset(e,S(t)?t({datasetIndex:e}):t);this.notifyPlugins("afterDatasetsUpdate",{mode:t})}}_updateDataset(t,e){const i=this.getDatasetMeta(t),s={meta:i,index:t,mode:e,cancelable:!0};!1!==this.notifyPlugins("beforeDatasetUpdate",s)&&(i.controller._update(e),s.cancelable=!1,this.notifyPlugins("afterDatasetUpdate",s))}render(){!1!==this.notifyPlugins("beforeRender",{cancelable:!0})&&(xt.has(this)?this.attached&&!xt.running(this)&&xt.start(this):(this.draw(),wn({chart:this})))}draw(){let t;if(this._resizeBeforeDraw){const{width:t,height:e}=this._resizeBeforeDraw;this._resize(t,e),this._resizeBeforeDraw=null}if(this.clear(),this.width<=0||this.height<=0)return;if(!1===this.notifyPlugins("beforeDraw",{cancelable:!0}))return;const e=this._layers;for(t=0;t<e.length&&e[t].z<=0;++t)e[t].draw(this.chartArea);for(this._drawDatasets();t<e.length;++t)e[t].draw(this.chartArea);this.notifyPlugins("afterDraw")}_getSortedDatasetMetas(t){const e=this._sortedMetasets,i=[];let s,n;for(s=0,n=e.length;s<n;++s){const n=e[s];t&&!n.visible||i.push(n)}return i}getSortedVisibleDatasetMetas(){return this._getSortedDatasetMetas(!0)}_drawDatasets(){if(!1===this.notifyPlugins("beforeDatasetsDraw",{cancelable:!0}))return;const t=this.getSortedVisibleDatasetMetas();for(let e=t.length-1;e>=0;--e)this._drawDataset(t[e]);this.notifyPlugins("afterDatasetsDraw")}_drawDataset(t){const e=this.ctx,i=t._clip,s=!i.disabled,n=function(t,e){const{xScale:i,yScale:s}=t;return i&&s?{left:On(i,e,"left"),right:On(i,e,"right"),top:On(s,e,"top"),bottom:On(s,e,"bottom")}:e}(t,this.chartArea),o={meta:t,index:t.index,cancelable:!0};!1!==this.notifyPlugins("beforeDatasetDraw",o)&&(s&&Ie(e,{left:!1===i.left?0:n.left-i.left,right:!1===i.right?this.width:n.right+i.right,top:!1===i.top?0:n.top-i.top,bottom:!1===i.bottom?this.height:n.bottom+i.bottom}),t.controller.draw(),s&&ze(e),o.cancelable=!1,this.notifyPlugins("afterDatasetDraw",o))}isPointInArea(t){return Re(t,this.chartArea,this._minPadding)}getElementsAtEventForMode(t,e,i,s){const n=Xi.modes[e];return"function"==typeof n?n(this,t,i,s):[]}getDatasetMeta(t){const e=this.data.datasets[t],i=this._metasets;let s=i.filter((t=>t&&t._dataset===e)).pop();return s||(s={type:null,data:[],dataset:null,controller:null,hidden:null,xAxisID:null,yAxisID:null,order:e&&e.order||0,index:t,_dataset:e,_parsed:[],_sorted:!1},i.push(s)),s}getContext(){return this.$context||(this.$context=Ci(null,{chart:this,type:"chart"}))}getVisibleDatasetCount(){return this.getSortedVisibleDatasetMetas().length}isDatasetVisible(t){const e=this.data.datasets[t];if(!e)return!1;const i=this.getDatasetMeta(t);return"boolean"==typeof i.hidden?!i.hidden:!e.hidden}setDatasetVisibility(t,e){this.getDatasetMeta(t).hidden=!e}toggleDataVisibility(t){this._hiddenIndices[t]=!this._hiddenIndices[t]}getDataVisibility(t){return!this._hiddenIndices[t]}_updateVisibility(t,e,i){const s=i?"show":"hide",n=this.getDatasetMeta(t),o=n.controller._resolveAnimations(void 0,s);k(e)?(n.data[e].hidden=!i,this.update()):(this.setDatasetVisibility(t,i),o.update(n,{visible:i}),this.update((e=>e.datasetIndex===t?s:void 0)))}hide(t,e){this._updateVisibility(t,e,!1)}show(t,e){this._updateVisibility(t,e,!0)}_destroyDatasetMeta(t){const e=this._metasets[t];e&&e.controller&&e.controller._destroy(),delete this._metasets[t]}_stop(){let t,e;for(this.stop(),xt.remove(this),t=0,e=this.data.datasets.length;t<e;++t)this._destroyDatasetMeta(t)}destroy(){this.notifyPlugins("beforeDestroy");const{canvas:t,ctx:e}=this;this._stop(),this.config.clearCache(),t&&(this.unbindEvents(),Te(t,e),this.platform.releaseContext(e),this.canvas=null,this.ctx=null),delete Pn[this.id],this.notifyPlugins("afterDestroy")}toBase64Image(...t){return this.canvas.toDataURL(...t)}bindEvents(){this.bindUserEvents(),this.options.responsive?this.bindResponsiveEvents():this.attached=!0}bindUserEvents(){const t=this._listeners,e=this.platform,i=(i,s)=>{e.addEventListener(this,i,s),t[i]=s},s=(t,e,i)=>{t.offsetX=e,t.offsetY=i,this._eventHandler(t)};u(this.options.events,(t=>i(t,s)))}bindResponsiveEvents(){this._responsiveListeners||(this._responsiveListeners={});const t=this._responsiveListeners,e=this.platform,i=(i,s)=>{e.addEventListener(this,i,s),t[i]=s},s=(i,s)=>{t[i]&&(e.removeEventListener(this,i,s),delete t[i])},n=(t,e)=>{this.canvas&&this.resize(t,e)};let o;const a=()=>{s("attach",a),this.attached=!0,this.resize(),i("resize",n),i("detach",o)};o=()=>{this.attached=!1,s("resize",n),this._stop(),this._resize(0,0),i("attach",a)},e.isAttached(this.canvas)?a():o()}unbindEvents(){u(this._listeners,((t,e)=>{this.platform.removeEventListener(this,e,t)})),this._listeners={},u(this._responsiveListeners,((t,e)=>{this.platform.removeEventListener(this,e,t)})),this._responsiveListeners=void 0}updateHoverStyle(t,e,i){const s=i?"set":"remove";let n,o,a,r;for("dataset"===e&&(n=this.getDatasetMeta(t[0].datasetIndex),n.controller["_"+s+"DatasetHoverStyle"]()),a=0,r=t.length;a<r;++a){o=t[a];const e=o&&this.getDatasetMeta(o.datasetIndex).controller;e&&e[s+"HoverStyle"](o.element,o.datasetIndex,o.index)}}getActiveElements(){return this._active||[]}setActiveElements(t){const e=this._active||[],i=t.map((({datasetIndex:t,index:e})=>{const i=this.getDatasetMeta(t);if(!i)throw new Error("No dataset found at index "+t);return{datasetIndex:t,element:i.data[e],index:e}}));!f(i,e)&&(this._active=i,this._lastEvent=null,this._updateHoverStyles(i,e))}notifyPlugins(t,e,i){return this._plugins.notify(this,t,e,i)}isPluginEnabled(t){return 1===this._plugins._cache.filter((e=>e.plugin.id===t)).length}_updateHoverStyles(t,e,i){const s=this.options.hover,n=(t,e)=>t.filter((t=>!e.some((e=>t.datasetIndex===e.datasetIndex&&t.index===e.index)))),o=n(e,t),a=i?t:n(t,e);o.length&&this.updateHoverStyle(o,s.mode,!1),a.length&&s.mode&&this.updateHoverStyle(a,s.mode,!0)}_eventHandler(t,e){const i={event:t,replay:e,cancelable:!0,inChartArea:this.isPointInArea(t)},s=e=>(e.options.events||this.options.events).includes(t.native.type);if(!1===this.notifyPlugins("beforeEvent",i,s))return;const n=this._handleEvent(t,e,i.inChartArea);return i.cancelable=!1,this.notifyPlugins("afterEvent",i,s),(n||i.changed)&&this.render(),this}_handleEvent(t,e,i){const{_active:s=[],options:n}=this,o=e,a=this._getActiveElements(t,s,i,o),r=D(t),l=function(t,e,i,s){return i&&"mouseout"!==t.type?s?e:t:null}(t,this._lastEvent,i,r);i&&(this._lastEvent=null,d(n.onHover,[t,a,this],this),r&&d(n.onClick,[t,a,this],this));const h=!f(a,s);return(h||e)&&(this._active=a,this._updateHoverStyles(a,s,e)),this._lastEvent=l,h}_getActiveElements(t,e,i,s){if("mouseout"===t.type)return[];if(!i)return e;const n=this.options.hover;return this.getElementsAtEventForMode(t,n.mode,n,s)}}function Tn(){return u(An.instances,(t=>t._plugins.invalidate()))}function Ln(){throw new Error("This method is not implemented: Check that a complete date adapter is provided.")}class En{static override(t){Object.assign(En.prototype,t)}options;constructor(t){this.options=t||{}}init(){}formats(){return Ln()}parse(){return Ln()}format(){return Ln()}add(){return Ln()}diff(){return Ln()}startOf(){return Ln()}endOf(){return Ln()}}var Rn={_date:En};function In(t){const e=t.iScale,i=function(t,e){if(!t._cache.$bar){const i=t.getMatchingVisibleMetas(e);let s=[];for(let e=0,n=i.length;e<n;e++)s=s.concat(i[e].controller.getAllParsedValues(t));t._cache.$bar=lt(s.sort(((t,e)=>t-e)))}return t._cache.$bar}(e,t.type);let s,n,o,a,r=e._length;const l=()=>{32767!==o&&-32768!==o&&(k(a)&&(r=Math.min(r,Math.abs(o-a)||r)),a=o)};for(s=0,n=i.length;s<n;++s)o=e.getPixelForValue(i[s]),l();for(a=void 0,s=0,n=e.ticks.length;s<n;++s)o=e.getPixelForTick(s),l();return r}function zn(t,e,i,s){return n(t)?function(t,e,i,s){const n=i.parse(t[0],s),o=i.parse(t[1],s),a=Math.min(n,o),r=Math.max(n,o);let l=a,h=r;Math.abs(a)>Math.abs(r)&&(l=r,h=a),e[i.axis]=h,e._custom={barStart:l,barEnd:h,start:n,end:o,min:a,max:r}}(t,e,i,s):e[i.axis]=i.parse(t,s),e}function Fn(t,e,i,s){const n=t.iScale,o=t.vScale,a=n.getLabels(),r=n===o,l=[];let h,c,d,u;for(h=i,c=i+s;h<c;++h)u=e[h],d={},d[n.axis]=r||n.parse(a[h],h),l.push(zn(u,d,o,h));return l}function Vn(t){return t&&void 0!==t.barStart&&void 0!==t.barEnd}function Bn(t,e,i,s){let n=e.borderSkipped;const o={};if(!n)return void(t.borderSkipped=o);if(!0===n)return void(t.borderSkipped={top:!0,right:!0,bottom:!0,left:!0});const{start:a,end:r,reverse:l,top:h,bottom:c}=function(t){let e,i,s,n,o;return t.horizontal?(e=t.base>t.x,i="left",s="right"):(e=t.base<t.y,i="bottom",s="top"),e?(n="end",o="start"):(n="start",o="end"),{start:i,end:s,reverse:e,top:n,bottom:o}}(t);"middle"===n&&i&&(t.enableBorderRadius=!0,(i._top||0)===s?n=h:(i._bottom||0)===s?n=c:(o[Wn(c,a,r,l)]=!0,n=h)),o[Wn(n,a,r,l)]=!0,t.borderSkipped=o}function Wn(t,e,i,s){var n,o,a;return s?(a=i,t=Nn(t=(n=t)===(o=e)?a:n===a?o:n,i,e)):t=Nn(t,e,i),t}function Nn(t,e,i){return"start"===t?e:"end"===t?i:t}function Hn(t,{inflateAmount:e},i){t.inflateAmount="auto"===e?1===i?.33:0:e}class jn extends Ns{static id="doughnut";static defaults={datasetElementType:!1,dataElementType:"arc",animation:{animateRotate:!0,animateScale:!1},animations:{numbers:{type:"number",properties:["circumference","endAngle","innerRadius","outerRadius","startAngle","x","y","offset","borderWidth","spacing"]}},cutout:"50%",rotation:0,circumference:360,radius:"100%",spacing:0,indexAxis:"r"};static descriptors={_scriptable:t=>"spacing"!==t,_indexable:t=>"spacing"!==t&&!t.startsWith("borderDash")&&!t.startsWith("hoverBorderDash")};static overrides={aspectRatio:1,plugins:{legend:{labels:{generateLabels(t){const e=t.data;if(e.labels.length&&e.datasets.length){const{labels:{pointStyle:i,color:s}}=t.legend.options;return e.labels.map(((e,n)=>{const o=t.getDatasetMeta(0).controller.getStyle(n);return{text:e,fillStyle:o.backgroundColor,strokeStyle:o.borderColor,fontColor:s,lineWidth:o.borderWidth,pointStyle:i,hidden:!t.getDataVisibility(n),index:n}}))}return[]}},onClick(t,e,i){i.chart.toggleDataVisibility(e.index),i.chart.update()}}}};constructor(t,e){super(t,e),this.enableOptionSharing=!0,this.innerRadius=void 0,this.outerRadius=void 0,this.offsetX=void 0,this.offsetY=void 0}linkScales(){}parse(t,e){const i=this.getDataset().data,s=this._cachedMeta;if(!1===this._parsing)s._parsed=i;else{let n,a,r=t=>+i[t];if(o(i[t])){const{key:t="value"}=this._parsing;r=e=>+M(i[e],t)}for(n=t,a=t+e;n<a;++n)s._parsed[n]=r(n)}}_getRotation(){return $(this.options.rotation-90)}_getCircumference(){return $(this.options.circumference)}_getRotationExtents(){let t=O,e=-O;for(let i=0;i<this.chart.data.datasets.length;++i)if(this.chart.isDatasetVisible(i)&&this.chart.getDatasetMeta(i).type===this._type){const s=this.chart.getDatasetMeta(i).controller,n=s._getRotation(),o=s._getCircumference();t=Math.min(t,n),e=Math.max(e,n+o)}return{rotation:t,circumference:e-t}}update(t){const e=this.chart,{chartArea:i}=e,s=this._cachedMeta,n=s.data,o=this.getMaxBorderWidth()+this.getMaxOffset(n)+this.options.spacing,a=Math.max((Math.min(i.width,i.height)-o)/2,0),r=Math.min(h(this.options.cutout,a),1),l=this._getRingWeight(this.index),{circumference:d,rotation:u}=this._getRotationExtents(),{ratioX:f,ratioY:g,offsetX:p,offsetY:m}=function(t,e,i){let s=1,n=1,o=0,a=0;if(e<O){const r=t,l=r+e,h=Math.cos(r),c=Math.sin(r),d=Math.cos(l),u=Math.sin(l),f=(t,e,s)=>Z(t,r,l,!0)?1:Math.max(e,e*i,s,s*i),g=(t,e,s)=>Z(t,r,l,!0)?-1:Math.min(e,e*i,s,s*i),p=f(0,h,d),m=f(E,c,u),b=g(C,h,d),x=g(C+E,c,u);s=(p-b)/2,n=(m-x)/2,o=-(p+b)/2,a=-(m+x)/2}return{ratioX:s,ratioY:n,offsetX:o,offsetY:a}}(u,d,r),b=(i.width-o)/f,x=(i.height-o)/g,_=Math.max(Math.min(b,x)/2,0),y=c(this.options.radius,_),v=(y-Math.max(y*r,0))/this._getVisibleDatasetWeightTotal();this.offsetX=p*y,this.offsetY=m*y,s.total=this.calculateTotal(),this.outerRadius=y-v*this._getRingWeightOffset(this.index),this.innerRadius=Math.max(this.outerRadius-v*l,0),this.updateElements(n,0,n.length,t)}_circumference(t,e){const i=this.options,s=this._cachedMeta,n=this._getCircumference();return e&&i.animation.animateRotate||!this.chart.getDataVisibility(t)||null===s._parsed[t]||s.data[t].hidden?0:this.calculateCircumference(s._parsed[t]*n/O)}updateElements(t,e,i,s){const n="reset"===s,o=this.chart,a=o.chartArea,r=o.options.animation,l=(a.left+a.right)/2,h=(a.top+a.bottom)/2,c=n&&r.animateScale,d=c?0:this.innerRadius,u=c?0:this.outerRadius,{sharedOptions:f,includeOptions:g}=this._getSharedOptions(e,s);let p,m=this._getRotation();for(p=0;p<e;++p)m+=this._circumference(p,n);for(p=e;p<e+i;++p){const e=this._circumference(p,n),i=t[p],o={x:l+this.offsetX,y:h+this.offsetY,startAngle:m,endAngle:m+e,circumference:e,outerRadius:u,innerRadius:d};g&&(o.options=f||this.resolveDataElementOptions(p,i.active?"active":s)),m+=e,this.updateElement(i,p,o,s)}}calculateTotal(){const t=this._cachedMeta,e=t.data;let i,s=0;for(i=0;i<e.length;i++){const n=t._parsed[i];null===n||isNaN(n)||!this.chart.getDataVisibility(i)||e[i].hidden||(s+=Math.abs(n))}return s}calculateCircumference(t){const e=this._cachedMeta.total;return e>0&&!isNaN(t)?O*(Math.abs(t)/e):0}getLabelAndValue(t){const e=this._cachedMeta,i=this.chart,s=i.data.labels||[],n=ne(e._parsed[t],i.options.locale);return{label:s[t]||"",value:n}}getMaxBorderWidth(t){let e=0;const i=this.chart;let s,n,o,a,r;if(!t)for(s=0,n=i.data.datasets.length;s<n;++s)if(i.isDatasetVisible(s)){o=i.getDatasetMeta(s),t=o.data,a=o.controller;break}if(!t)return 0;for(s=0,n=t.length;s<n;++s)r=a.resolveDataElementOptions(s),"inner"!==r.borderAlign&&(e=Math.max(e,r.borderWidth||0,r.hoverBorderWidth||0));return e}getMaxOffset(t){let e=0;for(let i=0,s=t.length;i<s;++i){const t=this.resolveDataElementOptions(i);e=Math.max(e,t.offset||0,t.hoverOffset||0)}return e}_getRingWeightOffset(t){let e=0;for(let i=0;i<t;++i)this.chart.isDatasetVisible(i)&&(e+=this._getRingWeight(i));return e}_getRingWeight(t){return Math.max(l(this.chart.data.datasets[t].weight,1),0)}_getVisibleDatasetWeightTotal(){return this._getRingWeightOffset(this.chart.data.datasets.length)||1}}class $n extends Ns{static id="polarArea";static defaults={dataElementType:"arc",animation:{animateRotate:!0,animateScale:!0},animations:{numbers:{type:"number",properties:["x","y","startAngle","endAngle","innerRadius","outerRadius"]}},indexAxis:"r",startAngle:0};static overrides={aspectRatio:1,plugins:{legend:{labels:{generateLabels(t){const e=t.data;if(e.labels.length&&e.datasets.length){const{labels:{pointStyle:i,color:s}}=t.legend.options;return e.labels.map(((e,n)=>{const o=t.getDatasetMeta(0).controller.getStyle(n);return{text:e,fillStyle:o.backgroundColor,strokeStyle:o.borderColor,fontColor:s,lineWidth:o.borderWidth,pointStyle:i,hidden:!t.getDataVisibility(n),index:n}}))}return[]}},onClick(t,e,i){i.chart.toggleDataVisibility(e.index),i.chart.update()}}},scales:{r:{type:"radialLinear",angleLines:{display:!1},beginAtZero:!0,grid:{circular:!0},pointLabels:{display:!1},startAngle:0}}};constructor(t,e){super(t,e),this.innerRadius=void 0,this.outerRadius=void 0}getLabelAndValue(t){const e=this._cachedMeta,i=this.chart,s=i.data.labels||[],n=ne(e._parsed[t].r,i.options.locale);return{label:s[t]||"",value:n}}parseObjectData(t,e,i,s){return ii.bind(this)(t,e,i,s)}update(t){const e=this._cachedMeta.data;this._updateRadius(),this.updateElements(e,0,e.length,t)}getMinMax(){const t=this._cachedMeta,e={min:Number.POSITIVE_INFINITY,max:Number.NEGATIVE_INFINITY};return t.data.forEach(((t,i)=>{const s=this.getParsed(i).r;!isNaN(s)&&this.chart.getDataVisibility(i)&&(s<e.min&&(e.min=s),s>e.max&&(e.max=s))})),e}_updateRadius(){const t=this.chart,e=t.chartArea,i=t.options,s=Math.min(e.right-e.left,e.bottom-e.top),n=Math.max(s/2,0),o=(n-Math.max(i.cutoutPercentage?n/100*i.cutoutPercentage:1,0))/t.getVisibleDatasetCount();this.outerRadius=n-o*this.index,this.innerRadius=this.outerRadius-o}updateElements(t,e,i,s){const n="reset"===s,o=this.chart,a=o.options.animation,r=this._cachedMeta.rScale,l=r.xCenter,h=r.yCenter,c=r.getIndexAngle(0)-.5*C;let d,u=c;const f=360/this.countVisibleElements();for(d=0;d<e;++d)u+=this._computeAngle(d,s,f);for(d=e;d<e+i;d++){const e=t[d];let i=u,g=u+this._computeAngle(d,s,f),p=o.getDataVisibility(d)?r.getDistanceFromCenterForValue(this.getParsed(d).r):0;u=g,n&&(a.animateScale&&(p=0),a.animateRotate&&(i=g=c));const m={x:l,y:h,innerRadius:0,outerRadius:p,startAngle:i,endAngle:g,options:this.resolveDataElementOptions(d,e.active?"active":s)};this.updateElement(e,d,m,s)}}countVisibleElements(){const t=this._cachedMeta;let e=0;return t.data.forEach(((t,i)=>{!isNaN(this.getParsed(i).r)&&this.chart.getDataVisibility(i)&&e++})),e}_computeAngle(t,e,i){return this.chart.getDataVisibility(t)?$(this.resolveDataElementOptions(t,e).angle||i):0}}var Yn=Object.freeze({__proto__:null,BarController:class extends Ns{static id="bar";static defaults={datasetElementType:!1,dataElementType:"bar",categoryPercentage:.8,barPercentage:.9,grouped:!0,animations:{numbers:{type:"number",properties:["x","y","base","width","height"]}}};static overrides={scales:{_index_:{type:"category",offset:!0,grid:{offset:!0}},_value_:{type:"linear",beginAtZero:!0}}};parsePrimitiveData(t,e,i,s){return Fn(t,e,i,s)}parseArrayData(t,e,i,s){return Fn(t,e,i,s)}parseObjectData(t,e,i,s){const{iScale:n,vScale:o}=t,{xAxisKey:a="x",yAxisKey:r="y"}=this._parsing,l="x"===n.axis?a:r,h="x"===o.axis?a:r,c=[];let d,u,f,g;for(d=i,u=i+s;d<u;++d)g=e[d],f={},f[n.axis]=n.parse(M(g,l),d),c.push(zn(M(g,h),f,o,d));return c}updateRangeFromParsed(t,e,i,s){super.updateRangeFromParsed(t,e,i,s);const n=i._custom;n&&e===this._cachedMeta.vScale&&(t.min=Math.min(t.min,n.min),t.max=Math.max(t.max,n.max))}getMaxOverflow(){return 0}getLabelAndValue(t){const e=this._cachedMeta,{iScale:i,vScale:s}=e,n=this.getParsed(t),o=n._custom,a=Vn(o)?"["+o.start+", "+o.end+"]":""+s.getLabelForValue(n[s.axis]);return{label:""+i.getLabelForValue(n[i.axis]),value:a}}initialize(){this.enableOptionSharing=!0,super.initialize();this._cachedMeta.stack=this.getDataset().stack}update(t){const e=this._cachedMeta;this.updateElements(e.data,0,e.data.length,t)}updateElements(t,e,i,n){const o="reset"===n,{index:a,_cachedMeta:{vScale:r}}=this,l=r.getBasePixel(),h=r.isHorizontal(),c=this._getRuler(),{sharedOptions:d,includeOptions:u}=this._getSharedOptions(e,n);for(let f=e;f<e+i;f++){const e=this.getParsed(f),i=o||s(e[r.axis])?{base:l,head:l}:this._calculateBarValuePixels(f),g=this._calculateBarIndexPixels(f,c),p=(e._stacks||{})[r.axis],m={horizontal:h,base:i.base,enableBorderRadius:!p||Vn(e._custom)||a===p._top||a===p._bottom,x:h?i.head:g.center,y:h?g.center:i.head,height:h?g.size:Math.abs(i.size),width:h?Math.abs(i.size):g.size};u&&(m.options=d||this.resolveDataElementOptions(f,t[f].active?"active":n));const b=m.options||t[f].options;Bn(m,b,p,a),Hn(m,b,c.ratio),this.updateElement(t[f],f,m,n)}}_getStacks(t,e){const{iScale:i}=this._cachedMeta,n=i.getMatchingVisibleMetas(this._type).filter((t=>t.controller.options.grouped)),o=i.options.stacked,a=[],r=t=>{const i=t.controller.getParsed(e),n=i&&i[t.vScale.axis];if(s(n)||isNaN(n))return!0};for(const i of n)if((void 0===e||!r(i))&&((!1===o||-1===a.indexOf(i.stack)||void 0===o&&void 0===i.stack)&&a.push(i.stack),i.index===t))break;return a.length||a.push(void 0),a}_getStackCount(t){return this._getStacks(void 0,t).length}_getStackIndex(t,e,i){const s=this._getStacks(t,i),n=void 0!==e?s.indexOf(e):-1;return-1===n?s.length-1:n}_getRuler(){const t=this.options,e=this._cachedMeta,i=e.iScale,s=[];let n,o;for(n=0,o=e.data.length;n<o;++n)s.push(i.getPixelForValue(this.getParsed(n)[i.axis],n));const a=t.barThickness;return{min:a||In(e),pixels:s,start:i._startPixel,end:i._endPixel,stackCount:this._getStackCount(),scale:i,grouped:t.grouped,ratio:a?1:t.categoryPercentage*t.barPercentage}}_calculateBarValuePixels(t){const{_cachedMeta:{vScale:e,_stacked:i,index:n},options:{base:o,minBarLength:a}}=this,r=o||0,l=this.getParsed(t),h=l._custom,c=Vn(h);let d,u,f=l[e.axis],g=0,p=i?this.applyStack(e,l,i):f;p!==f&&(g=p-f,p=f),c&&(f=h.barStart,p=h.barEnd-h.barStart,0!==f&&F(f)!==F(h.barEnd)&&(g=0),g+=f);const m=s(o)||c?g:o;let b=e.getPixelForValue(m);if(d=this.chart.getDataVisibility(t)?e.getPixelForValue(g+p):b,u=d-b,Math.abs(u)<a){u=function(t,e,i){return 0!==t?F(t):(e.isHorizontal()?1:-1)*(e.min>=i?1:-1)}(u,e,r)*a,f===r&&(b-=u/2);const t=e.getPixelForDecimal(0),s=e.getPixelForDecimal(1),o=Math.min(t,s),h=Math.max(t,s);b=Math.max(Math.min(b,h),o),d=b+u,i&&!c&&(l._stacks[e.axis]._visualValues[n]=e.getValueForPixel(d)-e.getValueForPixel(b))}if(b===e.getPixelForValue(r)){const t=F(u)*e.getLineWidthForValue(r)/2;b+=t,u-=t}return{size:u,base:b,head:d,center:d+u/2}}_calculateBarIndexPixels(t,e){const i=e.scale,n=this.options,o=n.skipNull,a=l(n.maxBarThickness,1/0);let r,h;if(e.grouped){const i=o?this._getStackCount(t):e.stackCount,l="flex"===n.barThickness?function(t,e,i,s){const n=e.pixels,o=n[t];let a=t>0?n[t-1]:null,r=t<n.length-1?n[t+1]:null;const l=i.categoryPercentage;null===a&&(a=o-(null===r?e.end-e.start:r-o)),null===r&&(r=o+o-a);const h=o-(o-Math.min(a,r))/2*l;return{chunk:Math.abs(r-a)/2*l/s,ratio:i.barPercentage,start:h}}(t,e,n,i):function(t,e,i,n){const o=i.barThickness;let a,r;return s(o)?(a=e.min*i.categoryPercentage,r=i.barPercentage):(a=o*n,r=1),{chunk:a/n,ratio:r,start:e.pixels[t]-a/2}}(t,e,n,i),c=this._getStackIndex(this.index,this._cachedMeta.stack,o?t:void 0);r=l.start+l.chunk*c+l.chunk/2,h=Math.min(a,l.chunk*l.ratio)}else r=i.getPixelForValue(this.getParsed(t)[i.axis],t),h=Math.min(a,e.min*e.ratio);return{base:r-h/2,head:r+h/2,center:r,size:h}}draw(){const t=this._cachedMeta,e=t.vScale,i=t.data,s=i.length;let n=0;for(;n<s;++n)null!==this.getParsed(n)[e.axis]&&i[n].draw(this._ctx)}},BubbleController:class extends Ns{static id="bubble";static defaults={datasetElementType:!1,dataElementType:"point",animations:{numbers:{type:"number",properties:["x","y","borderWidth","radius"]}}};static overrides={scales:{x:{type:"linear"},y:{type:"linear"}}};initialize(){this.enableOptionSharing=!0,super.initialize()}parsePrimitiveData(t,e,i,s){const n=super.parsePrimitiveData(t,e,i,s);for(let t=0;t<n.length;t++)n[t]._custom=this.resolveDataElementOptions(t+i).radius;return n}parseArrayData(t,e,i,s){const n=super.parseArrayData(t,e,i,s);for(let t=0;t<n.length;t++){const s=e[i+t];n[t]._custom=l(s[2],this.resolveDataElementOptions(t+i).radius)}return n}parseObjectData(t,e,i,s){const n=super.parseObjectData(t,e,i,s);for(let t=0;t<n.length;t++){const s=e[i+t];n[t]._custom=l(s&&s.r&&+s.r,this.resolveDataElementOptions(t+i).radius)}return n}getMaxOverflow(){const t=this._cachedMeta.data;let e=0;for(let i=t.length-1;i>=0;--i)e=Math.max(e,t[i].size(this.resolveDataElementOptions(i))/2);return e>0&&e}getLabelAndValue(t){const e=this._cachedMeta,i=this.chart.data.labels||[],{xScale:s,yScale:n}=e,o=this.getParsed(t),a=s.getLabelForValue(o.x),r=n.getLabelForValue(o.y),l=o._custom;return{label:i[t]||"",value:"("+a+", "+r+(l?", "+l:"")+")"}}update(t){const e=this._cachedMeta.data;this.updateElements(e,0,e.length,t)}updateElements(t,e,i,s){const n="reset"===s,{iScale:o,vScale:a}=this._cachedMeta,{sharedOptions:r,includeOptions:l}=this._getSharedOptions(e,s),h=o.axis,c=a.axis;for(let d=e;d<e+i;d++){const e=t[d],i=!n&&this.getParsed(d),u={},f=u[h]=n?o.getPixelForDecimal(.5):o.getPixelForValue(i[h]),g=u[c]=n?a.getBasePixel():a.getPixelForValue(i[c]);u.skip=isNaN(f)||isNaN(g),l&&(u.options=r||this.resolveDataElementOptions(d,e.active?"active":s),n&&(u.options.radius=0)),this.updateElement(e,d,u,s)}}resolveDataElementOptions(t,e){const i=this.getParsed(t);let s=super.resolveDataElementOptions(t,e);s.$shared&&(s=Object.assign({},s,{$shared:!1}));const n=s.radius;return"active"!==e&&(s.radius=0),s.radius+=l(i&&i._custom,n),s}},DoughnutController:jn,LineController:class extends Ns{static id="line";static defaults={datasetElementType:"line",dataElementType:"point",showLine:!0,spanGaps:!1};static overrides={scales:{_index_:{type:"category"},_value_:{type:"linear"}}};initialize(){this.enableOptionSharing=!0,this.supportsDecimation=!0,super.initialize()}update(t){const e=this._cachedMeta,{dataset:i,data:s=[],_dataset:n}=e,o=this.chart._animationsDisabled;let{start:a,count:r}=pt(e,s,o);this._drawStart=a,this._drawCount=r,mt(e)&&(a=0,r=s.length),i._chart=this.chart,i._datasetIndex=this.index,i._decimated=!!n._decimated,i.points=s;const l=this.resolveDatasetElementOptions(t);this.options.showLine||(l.borderWidth=0),l.segment=this.options.segment,this.updateElement(i,void 0,{animated:!o,options:l},t),this.updateElements(s,a,r,t)}updateElements(t,e,i,n){const o="reset"===n,{iScale:a,vScale:r,_stacked:l,_dataset:h}=this._cachedMeta,{sharedOptions:c,includeOptions:d}=this._getSharedOptions(e,n),u=a.axis,f=r.axis,{spanGaps:g,segment:p}=this.options,m=N(g)?g:Number.POSITIVE_INFINITY,b=this.chart._animationsDisabled||o||"none"===n,x=e+i,_=t.length;let y=e>0&&this.getParsed(e-1);for(let i=0;i<_;++i){const g=t[i],_=b?g:{};if(i<e||i>=x){_.skip=!0;continue}const v=this.getParsed(i),M=s(v[f]),w=_[u]=a.getPixelForValue(v[u],i),k=_[f]=o||M?r.getBasePixel():r.getPixelForValue(l?this.applyStack(r,v,l):v[f],i);_.skip=isNaN(w)||isNaN(k)||M,_.stop=i>0&&Math.abs(v[u]-y[u])>m,p&&(_.parsed=v,_.raw=h.data[i]),d&&(_.options=c||this.resolveDataElementOptions(i,g.active?"active":n)),b||this.updateElement(g,i,_,n),y=v}}getMaxOverflow(){const t=this._cachedMeta,e=t.dataset,i=e.options&&e.options.borderWidth||0,s=t.data||[];if(!s.length)return i;const n=s[0].size(this.resolveDataElementOptions(0)),o=s[s.length-1].size(this.resolveDataElementOptions(s.length-1));return Math.max(i,n,o)/2}draw(){const t=this._cachedMeta;t.dataset.updateControlPoints(this.chart.chartArea,t.iScale.axis),super.draw()}},PieController:class extends jn{static id="pie";static defaults={cutout:0,rotation:0,circumference:360,radius:"100%"}},PolarAreaController:$n,RadarController:class extends Ns{static id="radar";static defaults={datasetElementType:"line",dataElementType:"point",indexAxis:"r",showLine:!0,elements:{line:{fill:"start"}}};static overrides={aspectRatio:1,scales:{r:{type:"radialLinear"}}};getLabelAndValue(t){const e=this._cachedMeta.vScale,i=this.getParsed(t);return{label:e.getLabels()[t],value:""+e.getLabelForValue(i[e.axis])}}parseObjectData(t,e,i,s){return ii.bind(this)(t,e,i,s)}update(t){const e=this._cachedMeta,i=e.dataset,s=e.data||[],n=e.iScale.getLabels();if(i.points=s,"resize"!==t){const e=this.resolveDatasetElementOptions(t);this.options.showLine||(e.borderWidth=0);const o={_loop:!0,_fullLoop:n.length===s.length,options:e};this.updateElement(i,void 0,o,t)}this.updateElements(s,0,s.length,t)}updateElements(t,e,i,s){const n=this._cachedMeta.rScale,o="reset"===s;for(let a=e;a<e+i;a++){const e=t[a],i=this.resolveDataElementOptions(a,e.active?"active":s),r=n.getPointPositionForValue(a,this.getParsed(a).r),l=o?n.xCenter:r.x,h=o?n.yCenter:r.y,c={x:l,y:h,angle:r.angle,skip:isNaN(l)||isNaN(h),options:i};this.updateElement(e,a,c,s)}}},ScatterController:class extends Ns{static id="scatter";static defaults={datasetElementType:!1,dataElementType:"point",showLine:!1,fill:!1};static overrides={interaction:{mode:"point"},scales:{x:{type:"linear"},y:{type:"linear"}}};getLabelAndValue(t){const e=this._cachedMeta,i=this.chart.data.labels||[],{xScale:s,yScale:n}=e,o=this.getParsed(t),a=s.getLabelForValue(o.x),r=n.getLabelForValue(o.y);return{label:i[t]||"",value:"("+a+", "+r+")"}}update(t){const e=this._cachedMeta,{data:i=[]}=e,s=this.chart._animationsDisabled;let{start:n,count:o}=pt(e,i,s);if(this._drawStart=n,this._drawCount=o,mt(e)&&(n=0,o=i.length),this.options.showLine){this.datasetElementType||this.addElements();const{dataset:n,_dataset:o}=e;n._chart=this.chart,n._datasetIndex=this.index,n._decimated=!!o._decimated,n.points=i;const a=this.resolveDatasetElementOptions(t);a.segment=this.options.segment,this.updateElement(n,void 0,{animated:!s,options:a},t)}else this.datasetElementType&&(delete e.dataset,this.datasetElementType=!1);this.updateElements(i,n,o,t)}addElements(){const{showLine:t}=this.options;!this.datasetElementType&&t&&(this.datasetElementType=this.chart.registry.getElement("line")),super.addElements()}updateElements(t,e,i,n){const o="reset"===n,{iScale:a,vScale:r,_stacked:l,_dataset:h}=this._cachedMeta,c=this.resolveDataElementOptions(e,n),d=this.getSharedOptions(c),u=this.includeOptions(n,d),f=a.axis,g=r.axis,{spanGaps:p,segment:m}=this.options,b=N(p)?p:Number.POSITIVE_INFINITY,x=this.chart._animationsDisabled||o||"none"===n;let _=e>0&&this.getParsed(e-1);for(let c=e;c<e+i;++c){const e=t[c],i=this.getParsed(c),p=x?e:{},y=s(i[g]),v=p[f]=a.getPixelForValue(i[f],c),M=p[g]=o||y?r.getBasePixel():r.getPixelForValue(l?this.applyStack(r,i,l):i[g],c);p.skip=isNaN(v)||isNaN(M)||y,p.stop=c>0&&Math.abs(i[f]-_[f])>b,m&&(p.parsed=i,p.raw=h.data[c]),u&&(p.options=d||this.resolveDataElementOptions(c,e.active?"active":n)),x||this.updateElement(e,c,p,n),_=i}this.updateSharedOptions(d,n,c)}getMaxOverflow(){const t=this._cachedMeta,e=t.data||[];if(!this.options.showLine){let t=0;for(let i=e.length-1;i>=0;--i)t=Math.max(t,e[i].size(this.resolveDataElementOptions(i))/2);return t>0&&t}const i=t.dataset,s=i.options&&i.options.borderWidth||0;if(!e.length)return s;const n=e[0].size(this.resolveDataElementOptions(0)),o=e[e.length-1].size(this.resolveDataElementOptions(e.length-1));return Math.max(s,n,o)/2}}});function Un(t,e,i,s){const n=vi(t.options.borderRadius,["outerStart","outerEnd","innerStart","innerEnd"]);const o=(i-e)/2,a=Math.min(o,s*e/2),r=t=>{const e=(i-Math.min(o,t))*s/2;return J(t,0,Math.min(o,e))};return{outerStart:r(n.outerStart),outerEnd:r(n.outerEnd),innerStart:J(n.innerStart,0,a),innerEnd:J(n.innerEnd,0,a)}}function Xn(t,e,i,s){return{x:i+t*Math.cos(e),y:s+t*Math.sin(e)}}function qn(t,e,i,s,n,o){const{x:a,y:r,startAngle:l,pixelMargin:h,innerRadius:c}=e,d=Math.max(e.outerRadius+s+i-h,0),u=c>0?c+s+i+h:0;let f=0;const g=n-l;if(s){const t=((c>0?c-s:0)+(d>0?d-s:0))/2;f=(g-(0!==t?g*t/(t+s):g))/2}const p=(g-Math.max(.001,g*d-i/C)/d)/2,m=l+p+f,b=n-p-f,{outerStart:x,outerEnd:_,innerStart:y,innerEnd:v}=Un(e,u,d,b-m),M=d-x,w=d-_,k=m+x/M,S=b-_/w,P=u+y,D=u+v,O=m+y/P,A=b-v/D;if(t.beginPath(),o){const e=(k+S)/2;if(t.arc(a,r,d,k,e),t.arc(a,r,d,e,S),_>0){const e=Xn(w,S,a,r);t.arc(e.x,e.y,_,S,b+E)}const i=Xn(D,b,a,r);if(t.lineTo(i.x,i.y),v>0){const e=Xn(D,A,a,r);t.arc(e.x,e.y,v,b+E,A+Math.PI)}const s=(b-v/u+(m+y/u))/2;if(t.arc(a,r,u,b-v/u,s,!0),t.arc(a,r,u,s,m+y/u,!0),y>0){const e=Xn(P,O,a,r);t.arc(e.x,e.y,y,O+Math.PI,m-E)}const n=Xn(M,m,a,r);if(t.lineTo(n.x,n.y),x>0){const e=Xn(M,k,a,r);t.arc(e.x,e.y,x,m-E,k)}}else{t.moveTo(a,r);const e=Math.cos(k)*d+a,i=Math.sin(k)*d+r;t.lineTo(e,i);const s=Math.cos(S)*d+a,n=Math.sin(S)*d+r;t.lineTo(s,n)}t.closePath()}function Kn(t,e,i,s,n){const{fullCircles:o,startAngle:a,circumference:r,options:l}=e,{borderWidth:h,borderJoinStyle:c,borderDash:d,borderDashOffset:u}=l,f="inner"===l.borderAlign;if(!h)return;t.setLineDash(d||[]),t.lineDashOffset=u,f?(t.lineWidth=2*h,t.lineJoin=c||"round"):(t.lineWidth=h,t.lineJoin=c||"bevel");let g=e.endAngle;if(o){qn(t,e,i,s,g,n);for(let e=0;e<o;++e)t.stroke();isNaN(r)||(g=a+(r%O||O))}f&&function(t,e,i){const{startAngle:s,pixelMargin:n,x:o,y:a,outerRadius:r,innerRadius:l}=e;let h=n/r;t.beginPath(),t.arc(o,a,r,s-h,i+h),l>n?(h=n/l,t.arc(o,a,l,i+h,s-h,!0)):t.arc(o,a,n,i+E,s-E),t.closePath(),t.clip()}(t,e,g),o||(qn(t,e,i,s,g,n),t.stroke())}function Gn(t,e,i=e){t.lineCap=l(i.borderCapStyle,e.borderCapStyle),t.setLineDash(l(i.borderDash,e.borderDash)),t.lineDashOffset=l(i.borderDashOffset,e.borderDashOffset),t.lineJoin=l(i.borderJoinStyle,e.borderJoinStyle),t.lineWidth=l(i.borderWidth,e.borderWidth),t.strokeStyle=l(i.borderColor,e.borderColor)}function Zn(t,e,i){t.lineTo(i.x,i.y)}function Jn(t,e,i={}){const s=t.length,{start:n=0,end:o=s-1}=i,{start:a,end:r}=e,l=Math.max(n,a),h=Math.min(o,r),c=n<a&&o<a||n>r&&o>r;return{count:s,start:l,loop:e.loop,ilen:h<l&&!c?s+h-l:h-l}}function Qn(t,e,i,s){const{points:n,options:o}=e,{count:a,start:r,loop:l,ilen:h}=Jn(n,i,s),c=function(t){return t.stepped?Fe:t.tension||"monotone"===t.cubicInterpolationMode?Ve:Zn}(o);let d,u,f,{move:g=!0,reverse:p}=s||{};for(d=0;d<=h;++d)u=n[(r+(p?h-d:d))%a],u.skip||(g?(t.moveTo(u.x,u.y),g=!1):c(t,f,u,p,o.stepped),f=u);return l&&(u=n[(r+(p?h:0))%a],c(t,f,u,p,o.stepped)),!!l}function to(t,e,i,s){const n=e.points,{count:o,start:a,ilen:r}=Jn(n,i,s),{move:l=!0,reverse:h}=s||{};let c,d,u,f,g,p,m=0,b=0;const x=t=>(a+(h?r-t:t))%o,_=()=>{f!==g&&(t.lineTo(m,g),t.lineTo(m,f),t.lineTo(m,p))};for(l&&(d=n[x(0)],t.moveTo(d.x,d.y)),c=0;c<=r;++c){if(d=n[x(c)],d.skip)continue;const e=d.x,i=d.y,s=0|e;s===u?(i<f?f=i:i>g&&(g=i),m=(b*m+e)/++b):(_(),t.lineTo(e,i),u=s,b=0,f=g=i),p=i}_()}function eo(t){const e=t.options,i=e.borderDash&&e.borderDash.length;return!(t._decimated||t._loop||e.tension||"monotone"===e.cubicInterpolationMode||e.stepped||i)?to:Qn}const io="function"==typeof Path2D;function so(t,e,i,s){io&&!e.options.segment?function(t,e,i,s){let n=e._path;n||(n=e._path=new Path2D,e.path(n,i,s)&&n.closePath()),Gn(t,e.options),t.stroke(n)}(t,e,i,s):function(t,e,i,s){const{segments:n,options:o}=e,a=eo(e);for(const r of n)Gn(t,o,r.style),t.beginPath(),a(t,e,r,{start:i,end:i+s-1})&&t.closePath(),t.stroke()}(t,e,i,s)}class no extends Hs{static id="line";static defaults={borderCapStyle:"butt",borderDash:[],borderDashOffset:0,borderJoinStyle:"miter",borderWidth:3,capBezierPoints:!0,cubicInterpolationMode:"default",fill:!1,spanGaps:!1,stepped:!1,tension:0};static defaultRoutes={backgroundColor:"backgroundColor",borderColor:"borderColor"};static descriptors={_scriptable:!0,_indexable:t=>"borderDash"!==t&&"fill"!==t};constructor(t){super(),this.animated=!0,this.options=void 0,this._chart=void 0,this._loop=void 0,this._fullLoop=void 0,this._path=void 0,this._points=void 0,this._segments=void 0,this._decimated=!1,this._pointsUpdated=!1,this._datasetIndex=void 0,t&&Object.assign(this,t)}updateControlPoints(t,e){const i=this.options;if((i.tension||"monotone"===i.cubicInterpolationMode)&&!i.stepped&&!this._pointsUpdated){const s=i.spanGaps?this._loop:this._fullLoop;hi(this._points,i,t,s,e),this._pointsUpdated=!0}}set points(t){this._points=t,delete this._segments,delete this._path,this._pointsUpdated=!1}get points(){return this._points}get segments(){return this._segments||(this._segments=zi(this,this.options.segment))}first(){const t=this.segments,e=this.points;return t.length&&e[t[0].start]}last(){const t=this.segments,e=this.points,i=t.length;return i&&e[t[i-1].end]}interpolate(t,e){const i=this.options,s=t[e],n=this.points,o=Ii(this,{property:e,start:s,end:s});if(!o.length)return;const a=[],r=function(t){return t.stepped?pi:t.tension||"monotone"===t.cubicInterpolationMode?mi:gi}(i);let l,h;for(l=0,h=o.length;l<h;++l){const{start:h,end:c}=o[l],d=n[h],u=n[c];if(d===u){a.push(d);continue}const f=r(d,u,Math.abs((s-d[e])/(u[e]-d[e])),i.stepped);f[e]=t[e],a.push(f)}return 1===a.length?a[0]:a}pathSegment(t,e,i){return eo(this)(t,this,e,i)}path(t,e,i){const s=this.segments,n=eo(this);let o=this._loop;e=e||0,i=i||this.points.length-e;for(const a of s)o&=n(t,this,a,{start:e,end:e+i-1});return!!o}draw(t,e,i,s){const n=this.options||{};(this.points||[]).length&&n.borderWidth&&(t.save(),so(t,this,i,s),t.restore()),this.animated&&(this._pointsUpdated=!1,this._path=void 0)}}function oo(t,e,i,s){const n=t.options,{[i]:o}=t.getProps([i],s);return Math.abs(e-o)<n.radius+n.hitRadius}function ao(t,e){const{x:i,y:s,base:n,width:o,height:a}=t.getProps(["x","y","base","width","height"],e);let r,l,h,c,d;return t.horizontal?(d=a/2,r=Math.min(i,n),l=Math.max(i,n),h=s-d,c=s+d):(d=o/2,r=i-d,l=i+d,h=Math.min(s,n),c=Math.max(s,n)),{left:r,top:h,right:l,bottom:c}}function ro(t,e,i,s){return t?0:J(e,i,s)}function lo(t){const e=ao(t),i=e.right-e.left,s=e.bottom-e.top,n=function(t,e,i){const s=t.options.borderWidth,n=t.borderSkipped,o=Mi(s);return{t:ro(n.top,o.top,0,i),r:ro(n.right,o.right,0,e),b:ro(n.bottom,o.bottom,0,i),l:ro(n.left,o.left,0,e)}}(t,i/2,s/2),a=function(t,e,i){const{enableBorderRadius:s}=t.getProps(["enableBorderRadius"]),n=t.options.borderRadius,a=wi(n),r=Math.min(e,i),l=t.borderSkipped,h=s||o(n);return{topLeft:ro(!h||l.top||l.left,a.topLeft,0,r),topRight:ro(!h||l.top||l.right,a.topRight,0,r),bottomLeft:ro(!h||l.bottom||l.left,a.bottomLeft,0,r),bottomRight:ro(!h||l.bottom||l.right,a.bottomRight,0,r)}}(t,i/2,s/2);return{outer:{x:e.left,y:e.top,w:i,h:s,radius:a},inner:{x:e.left+n.l,y:e.top+n.t,w:i-n.l-n.r,h:s-n.t-n.b,radius:{topLeft:Math.max(0,a.topLeft-Math.max(n.t,n.l)),topRight:Math.max(0,a.topRight-Math.max(n.t,n.r)),bottomLeft:Math.max(0,a.bottomLeft-Math.max(n.b,n.l)),bottomRight:Math.max(0,a.bottomRight-Math.max(n.b,n.r))}}}}function ho(t,e,i,s){const n=null===e,o=null===i,a=t&&!(n&&o)&&ao(t,s);return a&&(n||tt(e,a.left,a.right))&&(o||tt(i,a.top,a.bottom))}function co(t,e){t.rect(e.x,e.y,e.w,e.h)}function uo(t,e,i={}){const s=t.x!==i.x?-e:0,n=t.y!==i.y?-e:0,o=(t.x+t.w!==i.x+i.w?e:0)-s,a=(t.y+t.h!==i.y+i.h?e:0)-n;return{x:t.x+s,y:t.y+n,w:t.w+o,h:t.h+a,radius:t.radius}}var fo=Object.freeze({__proto__:null,ArcElement:class extends Hs{static id="arc";static defaults={borderAlign:"center",borderColor:"#fff",borderDash:[],borderDashOffset:0,borderJoinStyle:void 0,borderRadius:0,borderWidth:2,offset:0,spacing:0,angle:void 0,circular:!0};static defaultRoutes={backgroundColor:"backgroundColor"};static descriptors={_scriptable:!0,_indexable:t=>"borderDash"!==t};circumference;endAngle;fullCircles;innerRadius;outerRadius;pixelMargin;startAngle;constructor(t){super(),this.options=void 0,this.circumference=void 0,this.startAngle=void 0,this.endAngle=void 0,this.innerRadius=void 0,this.outerRadius=void 0,this.pixelMargin=0,this.fullCircles=0,t&&Object.assign(this,t)}inRange(t,e,i){const s=this.getProps(["x","y"],i),{angle:n,distance:o}=X(s,{x:t,y:e}),{startAngle:a,endAngle:r,innerRadius:h,outerRadius:c,circumference:d}=this.getProps(["startAngle","endAngle","innerRadius","outerRadius","circumference"],i),u=(this.options.spacing+this.options.borderWidth)/2,f=l(d,r-a)>=O||Z(n,a,r),g=tt(o,h+u,c+u);return f&&g}getCenterPoint(t){const{x:e,y:i,startAngle:s,endAngle:n,innerRadius:o,outerRadius:a}=this.getProps(["x","y","startAngle","endAngle","innerRadius","outerRadius"],t),{offset:r,spacing:l}=this.options,h=(s+n)/2,c=(o+a+l+r)/2;return{x:e+Math.cos(h)*c,y:i+Math.sin(h)*c}}tooltipPosition(t){return this.getCenterPoint(t)}draw(t){const{options:e,circumference:i}=this,s=(e.offset||0)/4,n=(e.spacing||0)/2,o=e.circular;if(this.pixelMargin="inner"===e.borderAlign?.33:0,this.fullCircles=i>O?Math.floor(i/O):0,0===i||this.innerRadius<0||this.outerRadius<0)return;t.save();const a=(this.startAngle+this.endAngle)/2;t.translate(Math.cos(a)*s,Math.sin(a)*s);const r=s*(1-Math.sin(Math.min(C,i||0)));t.fillStyle=e.backgroundColor,t.strokeStyle=e.borderColor,function(t,e,i,s,n){const{fullCircles:o,startAngle:a,circumference:r}=e;let l=e.endAngle;if(o){qn(t,e,i,s,l,n);for(let e=0;e<o;++e)t.fill();isNaN(r)||(l=a+(r%O||O))}qn(t,e,i,s,l,n),t.fill()}(t,this,r,n,o),Kn(t,this,r,n,o),t.restore()}},BarElement:class extends Hs{static id="bar";static defaults={borderSkipped:"start",borderWidth:0,borderRadius:0,inflateAmount:"auto",pointStyle:void 0};static defaultRoutes={backgroundColor:"backgroundColor",borderColor:"borderColor"};constructor(t){super(),this.options=void 0,this.horizontal=void 0,this.base=void 0,this.width=void 0,this.height=void 0,this.inflateAmount=void 0,t&&Object.assign(this,t)}draw(t){const{inflateAmount:e,options:{borderColor:i,backgroundColor:s}}=this,{inner:n,outer:o}=lo(this),a=(r=o.radius).topLeft||r.topRight||r.bottomLeft||r.bottomRight?He:co;var r;t.save(),o.w===n.w&&o.h===n.h||(t.beginPath(),a(t,uo(o,e,n)),t.clip(),a(t,uo(n,-e,o)),t.fillStyle=i,t.fill("evenodd")),t.beginPath(),a(t,uo(n,e)),t.fillStyle=s,t.fill(),t.restore()}inRange(t,e,i){return ho(this,t,e,i)}inXRange(t,e){return ho(this,t,null,e)}inYRange(t,e){return ho(this,null,t,e)}getCenterPoint(t){const{x:e,y:i,base:s,horizontal:n}=this.getProps(["x","y","base","horizontal"],t);return{x:n?(e+s)/2:e,y:n?i:(i+s)/2}}getRange(t){return"x"===t?this.width/2:this.height/2}},LineElement:no,PointElement:class extends Hs{static id="point";parsed;skip;stop;static defaults={borderWidth:1,hitRadius:1,hoverBorderWidth:1,hoverRadius:4,pointStyle:"circle",radius:3,rotation:0};static defaultRoutes={backgroundColor:"backgroundColor",borderColor:"borderColor"};constructor(t){super(),this.options=void 0,this.parsed=void 0,this.skip=void 0,this.stop=void 0,t&&Object.assign(this,t)}inRange(t,e,i){const s=this.options,{x:n,y:o}=this.getProps(["x","y"],i);return Math.pow(t-n,2)+Math.pow(e-o,2)<Math.pow(s.hitRadius+s.radius,2)}inXRange(t,e){return oo(this,t,"x",e)}inYRange(t,e){return oo(this,t,"y",e)}getCenterPoint(t){const{x:e,y:i}=this.getProps(["x","y"],t);return{x:e,y:i}}size(t){let e=(t=t||this.options||{}).radius||0;e=Math.max(e,e&&t.hoverRadius||0);return 2*(e+(e&&t.borderWidth||0))}draw(t,e){const i=this.options;this.skip||i.radius<.1||!Re(this,e,this.size(i)/2)||(t.strokeStyle=i.borderColor,t.lineWidth=i.borderWidth,t.fillStyle=i.backgroundColor,Le(t,i,this.x,this.y))}getRange(){const t=this.options||{};return t.radius+t.hitRadius}}});function go(t,e,i,s){const n=t.indexOf(e);if(-1===n)return((t,e,i,s)=>("string"==typeof e?(i=t.push(e)-1,s.unshift({index:i,label:e})):isNaN(e)&&(i=null),i))(t,e,i,s);return n!==t.lastIndexOf(e)?i:n}function po(t){const e=this.getLabels();return t>=0&&t<e.length?e[t]:t}function mo(t,e,{horizontal:i,minRotation:s}){const n=$(s),o=(i?Math.sin(n):Math.cos(n))||.001,a=.75*e*(""+t).length;return Math.min(e/o,a)}class bo extends Js{constructor(t){super(t),this.start=void 0,this.end=void 0,this._startValue=void 0,this._endValue=void 0,this._valueRange=0}parse(t,e){return s(t)||("number"==typeof t||t instanceof Number)&&!isFinite(+t)?null:+t}handleTickRangeOptions(){const{beginAtZero:t}=this.options,{minDefined:e,maxDefined:i}=this.getUserBounds();let{min:s,max:n}=this;const o=t=>s=e?s:t,a=t=>n=i?n:t;if(t){const t=F(s),e=F(n);t<0&&e<0?a(0):t>0&&e>0&&o(0)}if(s===n){let e=0===n?1:Math.abs(.05*n);a(n+e),t||o(s-e)}this.min=s,this.max=n}getTickLimit(){const t=this.options.ticks;let e,{maxTicksLimit:i,stepSize:s}=t;return s?(e=Math.ceil(this.max/s)-Math.floor(this.min/s)+1,e>1e3&&(console.warn(`scales.${this.id}.ticks.stepSize: ${s} would result generating up to ${e} ticks. Limiting to 1000.`),e=1e3)):(e=this.computeTickLimit(),i=i||11),i&&(e=Math.min(i,e)),e}computeTickLimit(){return Number.POSITIVE_INFINITY}buildTicks(){const t=this.options,e=t.ticks;let i=this.getTickLimit();i=Math.max(2,i);const n=function(t,e){const i=[],{bounds:n,step:o,min:a,max:r,precision:l,count:h,maxTicks:c,maxDigits:d,includeBounds:u}=t,f=o||1,g=c-1,{min:p,max:m}=e,b=!s(a),x=!s(r),_=!s(h),y=(m-p)/(d+1);let v,M,w,k,S=B((m-p)/g/f)*f;if(S<1e-14&&!b&&!x)return[{value:p},{value:m}];k=Math.ceil(m/S)-Math.floor(p/S),k>g&&(S=B(k*S/g/f)*f),s(l)||(v=Math.pow(10,l),S=Math.ceil(S*v)/v),"ticks"===n?(M=Math.floor(p/S)*S,w=Math.ceil(m/S)*S):(M=p,w=m),b&&x&&o&&H((r-a)/o,S/1e3)?(k=Math.round(Math.min((r-a)/S,c)),S=(r-a)/k,M=a,w=r):_?(M=b?a:M,w=x?r:w,k=h-1,S=(w-M)/k):(k=(w-M)/S,k=V(k,Math.round(k),S/1e3)?Math.round(k):Math.ceil(k));const P=Math.max(U(S),U(M));v=Math.pow(10,s(l)?P:l),M=Math.round(M*v)/v,w=Math.round(w*v)/v;let D=0;for(b&&(u&&M!==a?(i.push({value:a}),M<a&&D++,V(Math.round((M+D*S)*v)/v,a,mo(a,y,t))&&D++):M<a&&D++);D<k;++D){const t=Math.round((M+D*S)*v)/v;if(x&&t>r)break;i.push({value:t})}return x&&u&&w!==r?i.length&&V(i[i.length-1].value,r,mo(r,y,t))?i[i.length-1].value=r:i.push({value:r}):x&&w!==r||i.push({value:w}),i}({maxTicks:i,bounds:t.bounds,min:t.min,max:t.max,precision:e.precision,step:e.stepSize,count:e.count,maxDigits:this._maxDigits(),horizontal:this.isHorizontal(),minRotation:e.minRotation||0,includeBounds:!1!==e.includeBounds},this._range||this);return"ticks"===t.bounds&&j(n,this,"value"),t.reverse?(n.reverse(),this.start=this.max,this.end=this.min):(this.start=this.min,this.end=this.max),n}configure(){const t=this.ticks;let e=this.min,i=this.max;if(super.configure(),this.options.offset&&t.length){const s=(i-e)/Math.max(t.length-1,1)/2;e-=s,i+=s}this._startValue=e,this._endValue=i,this._valueRange=i-e}getLabelForValue(t){return ne(t,this.chart.options.locale,this.options.ticks.format)}}class xo extends bo{static id="linear";static defaults={ticks:{callback:ae.formatters.numeric}};determineDataLimits(){const{min:t,max:e}=this.getMinMax(!0);this.min=a(t)?t:0,this.max=a(e)?e:1,this.handleTickRangeOptions()}computeTickLimit(){const t=this.isHorizontal(),e=t?this.width:this.height,i=$(this.options.ticks.minRotation),s=(t?Math.sin(i):Math.cos(i))||.001,n=this._resolveTickFontOptions(0);return Math.ceil(e/Math.min(40,n.lineHeight/s))}getPixelForValue(t){return null===t?NaN:this.getPixelForDecimal((t-this._startValue)/this._valueRange)}getValueForPixel(t){return this._startValue+this.getDecimalForPixel(t)*this._valueRange}}const _o=t=>Math.floor(z(t)),yo=(t,e)=>Math.pow(10,_o(t)+e);function vo(t){return 1===t/Math.pow(10,_o(t))}function Mo(t,e,i){const s=Math.pow(10,i),n=Math.floor(t/s);return Math.ceil(e/s)-n}function wo(t,{min:e,max:i}){e=r(t.min,e);const s=[],n=_o(e);let o=function(t,e){let i=_o(e-t);for(;Mo(t,e,i)>10;)i++;for(;Mo(t,e,i)<10;)i--;return Math.min(i,_o(t))}(e,i),a=o<0?Math.pow(10,Math.abs(o)):1;const l=Math.pow(10,o),h=n>o?Math.pow(10,n):0,c=Math.round((e-h)*a)/a,d=Math.floor((e-h)/l/10)*l*10;let u=Math.floor((c-d)/Math.pow(10,o)),f=r(t.min,Math.round((h+d+u*Math.pow(10,o))*a)/a);for(;f<i;)s.push({value:f,major:vo(f),significand:u}),u>=10?u=u<15?15:20:u++,u>=20&&(o++,u=2,a=o>=0?1:a),f=Math.round((h+d+u*Math.pow(10,o))*a)/a;const g=r(t.max,f);return s.push({value:g,major:vo(g),significand:u}),s}class ko extends Js{static id="logarithmic";static defaults={ticks:{callback:ae.formatters.logarithmic,major:{enabled:!0}}};constructor(t){super(t),this.start=void 0,this.end=void 0,this._startValue=void 0,this._valueRange=0}parse(t,e){const i=bo.prototype.parse.apply(this,[t,e]);if(0!==i)return a(i)&&i>0?i:null;this._zero=!0}determineDataLimits(){const{min:t,max:e}=this.getMinMax(!0);this.min=a(t)?Math.max(0,t):null,this.max=a(e)?Math.max(0,e):null,this.options.beginAtZero&&(this._zero=!0),this._zero&&this.min!==this._suggestedMin&&!a(this._userMin)&&(this.min=t===yo(this.min,0)?yo(this.min,-1):yo(this.min,0)),this.handleTickRangeOptions()}handleTickRangeOptions(){const{minDefined:t,maxDefined:e}=this.getUserBounds();let i=this.min,s=this.max;const n=e=>i=t?i:e,o=t=>s=e?s:t;i===s&&(i<=0?(n(1),o(10)):(n(yo(i,-1)),o(yo(s,1)))),i<=0&&n(yo(s,-1)),s<=0&&o(yo(i,1)),this.min=i,this.max=s}buildTicks(){const t=this.options,e=wo({min:this._userMin,max:this._userMax},this);return"ticks"===t.bounds&&j(e,this,"value"),t.reverse?(e.reverse(),this.start=this.max,this.end=this.min):(this.start=this.min,this.end=this.max),e}getLabelForValue(t){return void 0===t?"0":ne(t,this.chart.options.locale,this.options.ticks.format)}configure(){const t=this.min;super.configure(),this._startValue=z(t),this._valueRange=z(this.max)-z(t)}getPixelForValue(t){return void 0!==t&&0!==t||(t=this.min),null===t||isNaN(t)?NaN:this.getPixelForDecimal(t===this.min?0:(z(t)-this._startValue)/this._valueRange)}getValueForPixel(t){const e=this.getDecimalForPixel(t);return Math.pow(10,this._startValue+e*this._valueRange)}}function So(t){const e=t.ticks;if(e.display&&t.display){const t=ki(e.backdropPadding);return l(e.font&&e.font.size,ue.font.size)+t.height}return 0}function Po(t,e,i,s,n){return t===s||t===n?{start:e-i/2,end:e+i/2}:t<s||t>n?{start:e-i,end:e}:{start:e,end:e+i}}function Do(t){const e={l:t.left+t._padding.left,r:t.right-t._padding.right,t:t.top+t._padding.top,b:t.bottom-t._padding.bottom},i=Object.assign({},e),s=[],o=[],a=t._pointLabels.length,r=t.options.pointLabels,l=r.centerPointLabels?C/a:0;for(let u=0;u<a;u++){const a=r.setContext(t.getPointLabelContext(u));o[u]=a.padding;const f=t.getPointPosition(u,t.drawingArea+o[u],l),g=Si(a.font),p=(h=t.ctx,c=g,d=n(d=t._pointLabels[u])?d:[d],{w:Oe(h,c.string,d),h:d.length*c.lineHeight});s[u]=p;const m=G(t.getIndexAngle(u)+l),b=Math.round(Y(m));Co(i,e,m,Po(b,f.x,p.w,0,180),Po(b,f.y,p.h,90,270))}var h,c,d;t.setCenterPoint(e.l-i.l,i.r-e.r,e.t-i.t,i.b-e.b),t._pointLabelItems=function(t,e,i){const s=[],n=t._pointLabels.length,o=t.options,{centerPointLabels:a,display:r}=o.pointLabels,l={extra:So(o)/2,additionalAngle:a?C/n:0};let h;for(let o=0;o<n;o++){l.padding=i[o],l.size=e[o];const n=Oo(t,o,l);s.push(n),"auto"===r&&(n.visible=Ao(n,h),n.visible&&(h=n))}return s}(t,s,o)}function Co(t,e,i,s,n){const o=Math.abs(Math.sin(i)),a=Math.abs(Math.cos(i));let r=0,l=0;s.start<e.l?(r=(e.l-s.start)/o,t.l=Math.min(t.l,e.l-r)):s.end>e.r&&(r=(s.end-e.r)/o,t.r=Math.max(t.r,e.r+r)),n.start<e.t?(l=(e.t-n.start)/a,t.t=Math.min(t.t,e.t-l)):n.end>e.b&&(l=(n.end-e.b)/a,t.b=Math.max(t.b,e.b+l))}function Oo(t,e,i){const s=t.drawingArea,{extra:n,additionalAngle:o,padding:a,size:r}=i,l=t.getPointPosition(e,s+n+a,o),h=Math.round(Y(G(l.angle+E))),c=function(t,e,i){90===i||270===i?t-=e/2:(i>270||i<90)&&(t-=e);return t}(l.y,r.h,h),d=function(t){if(0===t||180===t)return"center";if(t<180)return"left";return"right"}(h),u=function(t,e,i){"right"===i?t-=e:"center"===i&&(t-=e/2);return t}(l.x,r.w,d);return{visible:!0,x:l.x,y:c,textAlign:d,left:u,top:c,right:u+r.w,bottom:c+r.h}}function Ao(t,e){if(!e)return!0;const{left:i,top:s,right:n,bottom:o}=t;return!(Re({x:i,y:s},e)||Re({x:i,y:o},e)||Re({x:n,y:s},e)||Re({x:n,y:o},e))}function To(t,e,i){const{left:n,top:o,right:a,bottom:r}=i,{backdropColor:l}=e;if(!s(l)){const i=wi(e.borderRadius),s=ki(e.backdropPadding);t.fillStyle=l;const h=n-s.left,c=o-s.top,d=a-n+s.width,u=r-o+s.height;Object.values(i).some((t=>0!==t))?(t.beginPath(),He(t,{x:h,y:c,w:d,h:u,radius:i}),t.fill()):t.fillRect(h,c,d,u)}}function Lo(t,e,i,s){const{ctx:n}=t;if(i)n.arc(t.xCenter,t.yCenter,e,0,O);else{let i=t.getPointPosition(0,e);n.moveTo(i.x,i.y);for(let o=1;o<s;o++)i=t.getPointPosition(o,e),n.lineTo(i.x,i.y)}}class Eo extends bo{static id="radialLinear";static defaults={display:!0,animate:!0,position:"chartArea",angleLines:{display:!0,lineWidth:1,borderDash:[],borderDashOffset:0},grid:{circular:!1},startAngle:0,ticks:{showLabelBackdrop:!0,callback:ae.formatters.numeric},pointLabels:{backdropColor:void 0,backdropPadding:2,display:!0,font:{size:10},callback:t=>t,padding:5,centerPointLabels:!1}};static defaultRoutes={"angleLines.color":"borderColor","pointLabels.color":"color","ticks.color":"color"};static descriptors={angleLines:{_fallback:"grid"}};constructor(t){super(t),this.xCenter=void 0,this.yCenter=void 0,this.drawingArea=void 0,this._pointLabels=[],this._pointLabelItems=[]}setDimensions(){const t=this._padding=ki(So(this.options)/2),e=this.width=this.maxWidth-t.width,i=this.height=this.maxHeight-t.height;this.xCenter=Math.floor(this.left+e/2+t.left),this.yCenter=Math.floor(this.top+i/2+t.top),this.drawingArea=Math.floor(Math.min(e,i)/2)}determineDataLimits(){const{min:t,max:e}=this.getMinMax(!1);this.min=a(t)&&!isNaN(t)?t:0,this.max=a(e)&&!isNaN(e)?e:0,this.handleTickRangeOptions()}computeTickLimit(){return Math.ceil(this.drawingArea/So(this.options))}generateTickLabels(t){bo.prototype.generateTickLabels.call(this,t),this._pointLabels=this.getLabels().map(((t,e)=>{const i=d(this.options.pointLabels.callback,[t,e],this);return i||0===i?i:""})).filter(((t,e)=>this.chart.getDataVisibility(e)))}fit(){const t=this.options;t.display&&t.pointLabels.display?Do(this):this.setCenterPoint(0,0,0,0)}setCenterPoint(t,e,i,s){this.xCenter+=Math.floor((t-e)/2),this.yCenter+=Math.floor((i-s)/2),this.drawingArea-=Math.min(this.drawingArea/2,Math.max(t,e,i,s))}getIndexAngle(t){return G(t*(O/(this._pointLabels.length||1))+$(this.options.startAngle||0))}getDistanceFromCenterForValue(t){if(s(t))return NaN;const e=this.drawingArea/(this.max-this.min);return this.options.reverse?(this.max-t)*e:(t-this.min)*e}getValueForDistanceFromCenter(t){if(s(t))return NaN;const e=t/(this.drawingArea/(this.max-this.min));return this.options.reverse?this.max-e:this.min+e}getPointLabelContext(t){const e=this._pointLabels||[];if(t>=0&&t<e.length){const i=e[t];return function(t,e,i){return Ci(t,{label:i,index:e,type:"pointLabel"})}(this.getContext(),t,i)}}getPointPosition(t,e,i=0){const s=this.getIndexAngle(t)-E+i;return{x:Math.cos(s)*e+this.xCenter,y:Math.sin(s)*e+this.yCenter,angle:s}}getPointPositionForValue(t,e){return this.getPointPosition(t,this.getDistanceFromCenterForValue(e))}getBasePosition(t){return this.getPointPositionForValue(t||0,this.getBaseValue())}getPointLabelPosition(t){const{left:e,top:i,right:s,bottom:n}=this._pointLabelItems[t];return{left:e,top:i,right:s,bottom:n}}drawBackground(){const{backgroundColor:t,grid:{circular:e}}=this.options;if(t){const i=this.ctx;i.save(),i.beginPath(),Lo(this,this.getDistanceFromCenterForValue(this._endValue),e,this._pointLabels.length),i.closePath(),i.fillStyle=t,i.fill(),i.restore()}}drawGrid(){const t=this.ctx,e=this.options,{angleLines:i,grid:s,border:n}=e,o=this._pointLabels.length;let a,r,l;if(e.pointLabels.display&&function(t,e){const{ctx:i,options:{pointLabels:s}}=t;for(let n=e-1;n>=0;n--){const e=t._pointLabelItems[n];if(!e.visible)continue;const o=s.setContext(t.getPointLabelContext(n));To(i,o,e);const a=Si(o.font),{x:r,y:l,textAlign:h}=e;Ne(i,t._pointLabels[n],r,l+a.lineHeight/2,a,{color:o.color,textAlign:h,textBaseline:"middle"})}}(this,o),s.display&&this.ticks.forEach(((t,e)=>{if(0!==e){r=this.getDistanceFromCenterForValue(t.value);const i=this.getContext(e),a=s.setContext(i),l=n.setContext(i);!function(t,e,i,s,n){const o=t.ctx,a=e.circular,{color:r,lineWidth:l}=e;!a&&!s||!r||!l||i<0||(o.save(),o.strokeStyle=r,o.lineWidth=l,o.setLineDash(n.dash),o.lineDashOffset=n.dashOffset,o.beginPath(),Lo(t,i,a,s),o.closePath(),o.stroke(),o.restore())}(this,a,r,o,l)}})),i.display){for(t.save(),a=o-1;a>=0;a--){const s=i.setContext(this.getPointLabelContext(a)),{color:n,lineWidth:o}=s;o&&n&&(t.lineWidth=o,t.strokeStyle=n,t.setLineDash(s.borderDash),t.lineDashOffset=s.borderDashOffset,r=this.getDistanceFromCenterForValue(e.ticks.reverse?this.min:this.max),l=this.getPointPosition(a,r),t.beginPath(),t.moveTo(this.xCenter,this.yCenter),t.lineTo(l.x,l.y),t.stroke())}t.restore()}}drawBorder(){}drawLabels(){const t=this.ctx,e=this.options,i=e.ticks;if(!i.display)return;const s=this.getIndexAngle(0);let n,o;t.save(),t.translate(this.xCenter,this.yCenter),t.rotate(s),t.textAlign="center",t.textBaseline="middle",this.ticks.forEach(((s,a)=>{if(0===a&&!e.reverse)return;const r=i.setContext(this.getContext(a)),l=Si(r.font);if(n=this.getDistanceFromCenterForValue(this.ticks[a].value),r.showLabelBackdrop){t.font=l.string,o=t.measureText(s.label).width,t.fillStyle=r.backdropColor;const e=ki(r.backdropPadding);t.fillRect(-o/2-e.left,-n-l.size/2-e.top,o+e.width,l.size+e.height)}Ne(t,s.label,0,-n,l,{color:r.color,strokeColor:r.textStrokeColor,strokeWidth:r.textStrokeWidth})})),t.restore()}drawTitle(){}}const Ro={millisecond:{common:!0,size:1,steps:1e3},second:{common:!0,size:1e3,steps:60},minute:{common:!0,size:6e4,steps:60},hour:{common:!0,size:36e5,steps:24},day:{common:!0,size:864e5,steps:30},week:{common:!1,size:6048e5,steps:4},month:{common:!0,size:2628e6,steps:12},quarter:{common:!1,size:7884e6,steps:4},year:{common:!0,size:3154e7}},Io=Object.keys(Ro);function zo(t,e){return t-e}function Fo(t,e){if(s(e))return null;const i=t._adapter,{parser:n,round:o,isoWeekday:r}=t._parseOpts;let l=e;return"function"==typeof n&&(l=n(l)),a(l)||(l="string"==typeof n?i.parse(l,n):i.parse(l)),null===l?null:(o&&(l="week"!==o||!N(r)&&!0!==r?i.startOf(l,o):i.startOf(l,"isoWeek",r)),+l)}function Vo(t,e,i,s){const n=Io.length;for(let o=Io.indexOf(t);o<n-1;++o){const t=Ro[Io[o]],n=t.steps?t.steps:Number.MAX_SAFE_INTEGER;if(t.common&&Math.ceil((i-e)/(n*t.size))<=s)return Io[o]}return Io[n-1]}function Bo(t,e,i){if(i){if(i.length){const{lo:s,hi:n}=et(i,e);t[i[s]>=e?i[s]:i[n]]=!0}}else t[e]=!0}function Wo(t,e,i){const s=[],n={},o=e.length;let a,r;for(a=0;a<o;++a)r=e[a],n[r]=a,s.push({value:r,major:!1});return 0!==o&&i?function(t,e,i,s){const n=t._adapter,o=+n.startOf(e[0].value,s),a=e[e.length-1].value;let r,l;for(r=o;r<=a;r=+n.add(r,1,s))l=i[r],l>=0&&(e[l].major=!0);return e}(t,s,n,i):s}class No extends Js{static id="time";static defaults={bounds:"data",adapters:{},time:{parser:!1,unit:!1,round:!1,isoWeekday:!1,minUnit:"millisecond",displayFormats:{}},ticks:{source:"auto",callback:!1,major:{enabled:!1}}};constructor(t){super(t),this._cache={data:[],labels:[],all:[]},this._unit="day",this._majorUnit=void 0,this._offsets={},this._normalized=!1,this._parseOpts=void 0}init(t,e={}){const i=t.time||(t.time={}),s=this._adapter=new Rn._date(t.adapters.date);s.init(e),x(i.displayFormats,s.formats()),this._parseOpts={parser:i.parser,round:i.round,isoWeekday:i.isoWeekday},super.init(t),this._normalized=e.normalized}parse(t,e){return void 0===t?null:Fo(this,t)}beforeLayout(){super.beforeLayout(),this._cache={data:[],labels:[],all:[]}}determineDataLimits(){const t=this.options,e=this._adapter,i=t.time.unit||"day";let{min:s,max:n,minDefined:o,maxDefined:r}=this.getUserBounds();function l(t){o||isNaN(t.min)||(s=Math.min(s,t.min)),r||isNaN(t.max)||(n=Math.max(n,t.max))}o&&r||(l(this._getLabelBounds()),"ticks"===t.bounds&&"labels"===t.ticks.source||l(this.getMinMax(!1))),s=a(s)&&!isNaN(s)?s:+e.startOf(Date.now(),i),n=a(n)&&!isNaN(n)?n:+e.endOf(Date.now(),i)+1,this.min=Math.min(s,n-1),this.max=Math.max(s+1,n)}_getLabelBounds(){const t=this.getLabelTimestamps();let e=Number.POSITIVE_INFINITY,i=Number.NEGATIVE_INFINITY;return t.length&&(e=t[0],i=t[t.length-1]),{min:e,max:i}}buildTicks(){const t=this.options,e=t.time,i=t.ticks,s="labels"===i.source?this.getLabelTimestamps():this._generate();"ticks"===t.bounds&&s.length&&(this.min=this._userMin||s[0],this.max=this._userMax||s[s.length-1]);const n=this.min,o=nt(s,n,this.max);return this._unit=e.unit||(i.autoSkip?Vo(e.minUnit,this.min,this.max,this._getLabelCapacity(n)):function(t,e,i,s,n){for(let o=Io.length-1;o>=Io.indexOf(i);o--){const i=Io[o];if(Ro[i].common&&t._adapter.diff(n,s,i)>=e-1)return i}return Io[i?Io.indexOf(i):0]}(this,o.length,e.minUnit,this.min,this.max)),this._majorUnit=i.major.enabled&&"year"!==this._unit?function(t){for(let e=Io.indexOf(t)+1,i=Io.length;e<i;++e)if(Ro[Io[e]].common)return Io[e]}(this._unit):void 0,this.initOffsets(s),t.reverse&&o.reverse(),Wo(this,o,this._majorUnit)}afterAutoSkip(){this.options.offsetAfterAutoskip&&this.initOffsets(this.ticks.map((t=>+t.value)))}initOffsets(t=[]){let e,i,s=0,n=0;this.options.offset&&t.length&&(e=this.getDecimalForValue(t[0]),s=1===t.length?1-e:(this.getDecimalForValue(t[1])-e)/2,i=this.getDecimalForValue(t[t.length-1]),n=1===t.length?i:(i-this.getDecimalForValue(t[t.length-2]))/2);const o=t.length<3?.5:.25;s=J(s,0,o),n=J(n,0,o),this._offsets={start:s,end:n,factor:1/(s+1+n)}}_generate(){const t=this._adapter,e=this.min,i=this.max,s=this.options,n=s.time,o=n.unit||Vo(n.minUnit,e,i,this._getLabelCapacity(e)),a=l(s.ticks.stepSize,1),r="week"===o&&n.isoWeekday,h=N(r)||!0===r,c={};let d,u,f=e;if(h&&(f=+t.startOf(f,"isoWeek",r)),f=+t.startOf(f,h?"day":o),t.diff(i,e,o)>1e5*a)throw new Error(e+" and "+i+" are too far apart with stepSize of "+a+" "+o);const g="data"===s.ticks.source&&this.getDataTimestamps();for(d=f,u=0;d<i;d=+t.add(d,a,o),u++)Bo(c,d,g);return d!==i&&"ticks"!==s.bounds&&1!==u||Bo(c,d,g),Object.keys(c).sort(zo).map((t=>+t))}getLabelForValue(t){const e=this._adapter,i=this.options.time;return i.tooltipFormat?e.format(t,i.tooltipFormat):e.format(t,i.displayFormats.datetime)}format(t,e){const i=this.options.time.displayFormats,s=this._unit,n=e||i[s];return this._adapter.format(t,n)}_tickFormatFunction(t,e,i,s){const n=this.options,o=n.ticks.callback;if(o)return d(o,[t,e,i],this);const a=n.time.displayFormats,r=this._unit,l=this._majorUnit,h=r&&a[r],c=l&&a[l],u=i[e],f=l&&c&&u&&u.major;return this._adapter.format(t,s||(f?c:h))}generateTickLabels(t){let e,i,s;for(e=0,i=t.length;e<i;++e)s=t[e],s.label=this._tickFormatFunction(s.value,e,t)}getDecimalForValue(t){return null===t?NaN:(t-this.min)/(this.max-this.min)}getPixelForValue(t){const e=this._offsets,i=this.getDecimalForValue(t);return this.getPixelForDecimal((e.start+i)*e.factor)}getValueForPixel(t){const e=this._offsets,i=this.getDecimalForPixel(t)/e.factor-e.end;return this.min+i*(this.max-this.min)}_getLabelSize(t){const e=this.options.ticks,i=this.ctx.measureText(t).width,s=$(this.isHorizontal()?e.maxRotation:e.minRotation),n=Math.cos(s),o=Math.sin(s),a=this._resolveTickFontOptions(0).size;return{w:i*n+a*o,h:i*o+a*n}}_getLabelCapacity(t){const e=this.options.time,i=e.displayFormats,s=i[e.unit]||i.millisecond,n=this._tickFormatFunction(t,0,Wo(this,[t],this._majorUnit),s),o=this._getLabelSize(n),a=Math.floor(this.isHorizontal()?this.width/o.w:this.height/o.h)-1;return a>0?a:1}getDataTimestamps(){let t,e,i=this._cache.data||[];if(i.length)return i;const s=this.getMatchingVisibleMetas();if(this._normalized&&s.length)return this._cache.data=s[0].controller.getAllParsedValues(this);for(t=0,e=s.length;t<e;++t)i=i.concat(s[t].controller.getAllParsedValues(this));return this._cache.data=this.normalize(i)}getLabelTimestamps(){const t=this._cache.labels||[];let e,i;if(t.length)return t;const s=this.getLabels();for(e=0,i=s.length;e<i;++e)t.push(Fo(this,s[e]));return this._cache.labels=this._normalized?t:this.normalize(t)}normalize(t){return lt(t.sort(zo))}}function Ho(t,e,i){let s,n,o,a,r=0,l=t.length-1;i?(e>=t[r].pos&&e<=t[l].pos&&({lo:r,hi:l}=it(t,"pos",e)),({pos:s,time:o}=t[r]),({pos:n,time:a}=t[l])):(e>=t[r].time&&e<=t[l].time&&({lo:r,hi:l}=it(t,"time",e)),({time:s,pos:o}=t[r]),({time:n,pos:a}=t[l]));const h=n-s;return h?o+(a-o)*(e-s)/h:o}var jo=Object.freeze({__proto__:null,CategoryScale:class extends Js{static id="category";static defaults={ticks:{callback:po}};constructor(t){super(t),this._startValue=void 0,this._valueRange=0,this._addedLabels=[]}init(t){const e=this._addedLabels;if(e.length){const t=this.getLabels();for(const{index:i,label:s}of e)t[i]===s&&t.splice(i,1);this._addedLabels=[]}super.init(t)}parse(t,e){if(s(t))return null;const i=this.getLabels();return((t,e)=>null===t?null:J(Math.round(t),0,e))(e=isFinite(e)&&i[e]===t?e:go(i,t,l(e,t),this._addedLabels),i.length-1)}determineDataLimits(){const{minDefined:t,maxDefined:e}=this.getUserBounds();let{min:i,max:s}=this.getMinMax(!0);"ticks"===this.options.bounds&&(t||(i=0),e||(s=this.getLabels().length-1)),this.min=i,this.max=s}buildTicks(){const t=this.min,e=this.max,i=this.options.offset,s=[];let n=this.getLabels();n=0===t&&e===n.length-1?n:n.slice(t,e+1),this._valueRange=Math.max(n.length-(i?0:1),1),this._startValue=this.min-(i?.5:0);for(let i=t;i<=e;i++)s.push({value:i});return s}getLabelForValue(t){return po.call(this,t)}configure(){super.configure(),this.isHorizontal()||(this._reversePixels=!this._reversePixels)}getPixelForValue(t){return"number"!=typeof t&&(t=this.parse(t)),null===t?NaN:this.getPixelForDecimal((t-this._startValue)/this._valueRange)}getPixelForTick(t){const e=this.ticks;return t<0||t>e.length-1?null:this.getPixelForValue(e[t].value)}getValueForPixel(t){return Math.round(this._startValue+this.getDecimalForPixel(t)*this._valueRange)}getBasePixel(){return this.bottom}},LinearScale:xo,LogarithmicScale:ko,RadialLinearScale:Eo,TimeScale:No,TimeSeriesScale:class extends No{static id="timeseries";static defaults=No.defaults;constructor(t){super(t),this._table=[],this._minPos=void 0,this._tableRange=void 0}initOffsets(){const t=this._getTimestampsForTable(),e=this._table=this.buildLookupTable(t);this._minPos=Ho(e,this.min),this._tableRange=Ho(e,this.max)-this._minPos,super.initOffsets(t)}buildLookupTable(t){const{min:e,max:i}=this,s=[],n=[];let o,a,r,l,h;for(o=0,a=t.length;o<a;++o)l=t[o],l>=e&&l<=i&&s.push(l);if(s.length<2)return[{time:e,pos:0},{time:i,pos:1}];for(o=0,a=s.length;o<a;++o)h=s[o+1],r=s[o-1],l=s[o],Math.round((h+r)/2)!==l&&n.push({time:l,pos:o/(a-1)});return n}_generate(){const t=this.min,e=this.max;let i=super.getDataTimestamps();return i.includes(t)&&i.length||i.splice(0,0,t),i.includes(e)&&1!==i.length||i.push(e),i.sort(((t,e)=>t-e))}_getTimestampsForTable(){let t=this._cache.all||[];if(t.length)return t;const e=this.getDataTimestamps(),i=this.getLabelTimestamps();return t=e.length&&i.length?this.normalize(e.concat(i)):e.length?e:i,t=this._cache.all=t,t}getDecimalForValue(t){return(Ho(this._table,t)-this._minPos)/this._tableRange}getValueForPixel(t){const e=this._offsets,i=this.getDecimalForPixel(t)/e.factor-e.end;return Ho(this._table,i*this._tableRange+this._minPos,!0)}}});const $o=["rgb(54, 162, 235)","rgb(255, 99, 132)","rgb(255, 159, 64)","rgb(255, 205, 86)","rgb(75, 192, 192)","rgb(153, 102, 255)","rgb(201, 203, 207)"],Yo=$o.map((t=>t.replace("rgb(","rgba(").replace(")",", 0.5)")));function Uo(t){return $o[t%$o.length]}function Xo(t){return Yo[t%Yo.length]}function qo(t){let e=0;return(i,s)=>{const n=t.getDatasetMeta(s).controller;n instanceof jn?e=function(t,e){return t.backgroundColor=t.data.map((()=>Uo(e++))),e}(i,e):n instanceof $n?e=function(t,e){return t.backgroundColor=t.data.map((()=>Xo(e++))),e}(i,e):n&&(e=function(t,e){return t.borderColor=Uo(e),t.backgroundColor=Xo(e),++e}(i,e))}}function Ko(t){let e;for(e in t)if(t[e].borderColor||t[e].backgroundColor)return!0;return!1}var Go={id:"colors",defaults:{enabled:!0,forceOverride:!1},beforeLayout(t,e,i){if(!i.enabled)return;const{data:{datasets:s},options:n}=t.config,{elements:o}=n;if(!i.forceOverride&&(Ko(s)||(a=n)&&(a.borderColor||a.backgroundColor)||o&&Ko(o)))return;var a;const r=qo(t);s.forEach(r)}};function Zo(t){if(t._decimated){const e=t._data;delete t._decimated,delete t._data,Object.defineProperty(t,"data",{configurable:!0,enumerable:!0,writable:!0,value:e})}}function Jo(t){t.data.datasets.forEach((t=>{Zo(t)}))}var Qo={id:"decimation",defaults:{algorithm:"min-max",enabled:!1},beforeElementsUpdate:(t,e,i)=>{if(!i.enabled)return void Jo(t);const n=t.width;t.data.datasets.forEach(((e,o)=>{const{_data:a,indexAxis:r}=e,l=t.getDatasetMeta(o),h=a||e.data;if("y"===Pi([r,t.options.indexAxis]))return;if(!l.controller.supportsDecimation)return;const c=t.scales[l.xAxisID];if("linear"!==c.type&&"time"!==c.type)return;if(t.options.parsing)return;let{start:d,count:u}=function(t,e){const i=e.length;let s,n=0;const{iScale:o}=t,{min:a,max:r,minDefined:l,maxDefined:h}=o.getUserBounds();return l&&(n=J(it(e,o.axis,a).lo,0,i-1)),s=h?J(it(e,o.axis,r).hi+1,n,i)-n:i-n,{start:n,count:s}}(l,h);if(u<=(i.threshold||4*n))return void Zo(e);let f;switch(s(a)&&(e._data=h,delete e.data,Object.defineProperty(e,"data",{configurable:!0,enumerable:!0,get:function(){return this._decimated},set:function(t){this._data=t}})),i.algorithm){case"lttb":f=function(t,e,i,s,n){const o=n.samples||s;if(o>=i)return t.slice(e,e+i);const a=[],r=(i-2)/(o-2);let l=0;const h=e+i-1;let c,d,u,f,g,p=e;for(a[l++]=t[p],c=0;c<o-2;c++){let s,n=0,o=0;const h=Math.floor((c+1)*r)+1+e,m=Math.min(Math.floor((c+2)*r)+1,i)+e,b=m-h;for(s=h;s<m;s++)n+=t[s].x,o+=t[s].y;n/=b,o/=b;const x=Math.floor(c*r)+1+e,_=Math.min(Math.floor((c+1)*r)+1,i)+e,{x:y,y:v}=t[p];for(u=f=-1,s=x;s<_;s++)f=.5*Math.abs((y-n)*(t[s].y-v)-(y-t[s].x)*(o-v)),f>u&&(u=f,d=t[s],g=s);a[l++]=d,p=g}return a[l++]=t[h],a}(h,d,u,n,i);break;case"min-max":f=function(t,e,i,n){let o,a,r,l,h,c,d,u,f,g,p=0,m=0;const b=[],x=e+i-1,_=t[e].x,y=t[x].x-_;for(o=e;o<e+i;++o){a=t[o],r=(a.x-_)/y*n,l=a.y;const e=0|r;if(e===h)l<f?(f=l,c=o):l>g&&(g=l,d=o),p=(m*p+a.x)/++m;else{const i=o-1;if(!s(c)&&!s(d)){const e=Math.min(c,d),s=Math.max(c,d);e!==u&&e!==i&&b.push({...t[e],x:p}),s!==u&&s!==i&&b.push({...t[s],x:p})}o>0&&i!==u&&b.push(t[i]),b.push(a),h=e,m=0,f=g=l,c=d=u=o}}return b}(h,d,u,n);break;default:throw new Error(`Unsupported decimation algorithm \'${i.algorithm}\'`)}e._decimated=f}))},destroy(t){Jo(t)}};function ta(t,e,i,s){if(s)return;let n=e[t],o=i[t];return"angle"===t&&(n=G(n),o=G(o)),{property:t,start:n,end:o}}function ea(t,e,i){for(;e>t;e--){const t=i[e];if(!isNaN(t.x)&&!isNaN(t.y))break}return e}function ia(t,e,i,s){return t&&e?s(t[i],e[i]):t?t[i]:e?e[i]:0}function sa(t,e){let i=[],s=!1;return n(t)?(s=!0,i=t):i=function(t,e){const{x:i=null,y:s=null}=t||{},n=e.points,o=[];return e.segments.forEach((({start:t,end:e})=>{e=ea(t,e,n);const a=n[t],r=n[e];null!==s?(o.push({x:a.x,y:s}),o.push({x:r.x,y:s})):null!==i&&(o.push({x:i,y:a.y}),o.push({x:i,y:r.y}))})),o}(t,e),i.length?new no({points:i,options:{tension:0},_loop:s,_fullLoop:s}):null}function na(t){return t&&!1!==t.fill}function oa(t,e,i){let s=t[e].fill;const n=[e];let o;if(!i)return s;for(;!1!==s&&-1===n.indexOf(s);){if(!a(s))return s;if(o=t[s],!o)return!1;if(o.visible)return s;n.push(s),s=o.fill}return!1}function aa(t,e,i){const s=function(t){const e=t.options,i=e.fill;let s=l(i&&i.target,i);void 0===s&&(s=!!e.backgroundColor);if(!1===s||null===s)return!1;if(!0===s)return"origin";return s}(t);if(o(s))return!isNaN(s.value)&&s;let n=parseFloat(s);return a(n)&&Math.floor(n)===n?function(t,e,i,s){"-"!==t&&"+"!==t||(i=e+i);if(i===e||i<0||i>=s)return!1;return i}(s[0],e,n,i):["origin","start","end","stack","shape"].indexOf(s)>=0&&s}function ra(t,e,i){const s=[];for(let n=0;n<i.length;n++){const o=i[n],{first:a,last:r,point:l}=la(o,e,"x");if(!(!l||a&&r))if(a)s.unshift(l);else if(t.push(l),!r)break}t.push(...s)}function la(t,e,i){const s=t.interpolate(e,i);if(!s)return{};const n=s[i],o=t.segments,a=t.points;let r=!1,l=!1;for(let t=0;t<o.length;t++){const e=o[t],s=a[e.start][i],h=a[e.end][i];if(tt(n,s,h)){r=n===s,l=n===h;break}}return{first:r,last:l,point:s}}class ha{constructor(t){this.x=t.x,this.y=t.y,this.radius=t.radius}pathSegment(t,e,i){const{x:s,y:n,radius:o}=this;return e=e||{start:0,end:O},t.arc(s,n,o,e.end,e.start,!0),!i.bounds}interpolate(t){const{x:e,y:i,radius:s}=this,n=t.angle;return{x:e+Math.cos(n)*s,y:i+Math.sin(n)*s,angle:n}}}function ca(t){const{chart:e,fill:i,line:s}=t;if(a(i))return function(t,e){const i=t.getDatasetMeta(e),s=i&&t.isDatasetVisible(e);return s?i.dataset:null}(e,i);if("stack"===i)return function(t){const{scale:e,index:i,line:s}=t,n=[],o=s.segments,a=s.points,r=function(t,e){const i=[],s=t.getMatchingVisibleMetas("line");for(let t=0;t<s.length;t++){const n=s[t];if(n.index===e)break;n.hidden||i.unshift(n.dataset)}return i}(e,i);r.push(sa({x:null,y:e.bottom},s));for(let t=0;t<o.length;t++){const e=o[t];for(let t=e.start;t<=e.end;t++)ra(n,a[t],r)}return new no({points:n,options:{}})}(t);if("shape"===i)return!0;const n=function(t){const e=t.scale||{};if(e.getPointPositionForValue)return function(t){const{scale:e,fill:i}=t,s=e.options,n=e.getLabels().length,a=s.reverse?e.max:e.min,r=function(t,e,i){let s;return s="start"===t?i:"end"===t?e.options.reverse?e.min:e.max:o(t)?t.value:e.getBaseValue(),s}(i,e,a),l=[];if(s.grid.circular){const t=e.getPointPositionForValue(0,a);return new ha({x:t.x,y:t.y,radius:e.getDistanceFromCenterForValue(r)})}for(let t=0;t<n;++t)l.push(e.getPointPositionForValue(t,r));return l}(t);return function(t){const{scale:e={},fill:i}=t,s=function(t,e){let i=null;return"start"===t?i=e.bottom:"end"===t?i=e.top:o(t)?i=e.getPixelForValue(t.value):e.getBasePixel&&(i=e.getBasePixel()),i}(i,e);if(a(s)){const t=e.isHorizontal();return{x:t?s:null,y:t?null:s}}return null}(t)}(t);return n instanceof ha?n:sa(n,s)}function da(t,e,i){const s=ca(e),{line:n,scale:o,axis:a}=e,r=n.options,l=r.fill,h=r.backgroundColor,{above:c=h,below:d=h}=l||{};s&&n.points.length&&(Ie(t,i),function(t,e){const{line:i,target:s,above:n,below:o,area:a,scale:r}=e,l=i._loop?"angle":e.axis;t.save(),"x"===l&&o!==n&&(ua(t,s,a.top),fa(t,{line:i,target:s,color:n,scale:r,property:l}),t.restore(),t.save(),ua(t,s,a.bottom));fa(t,{line:i,target:s,color:o,scale:r,property:l}),t.restore()}(t,{line:n,target:s,above:c,below:d,area:i,scale:o,axis:a}),ze(t))}function ua(t,e,i){const{segments:s,points:n}=e;let o=!0,a=!1;t.beginPath();for(const r of s){const{start:s,end:l}=r,h=n[s],c=n[ea(s,l,n)];o?(t.moveTo(h.x,h.y),o=!1):(t.lineTo(h.x,i),t.lineTo(h.x,h.y)),a=!!e.pathSegment(t,r,{move:a}),a?t.closePath():t.lineTo(c.x,i)}t.lineTo(e.first().x,i),t.closePath(),t.clip()}function fa(t,e){const{line:i,target:s,property:n,color:o,scale:a}=e,r=function(t,e,i){const s=t.segments,n=t.points,o=e.points,a=[];for(const t of s){let{start:s,end:r}=t;r=ea(s,r,n);const l=ta(i,n[s],n[r],t.loop);if(!e.segments){a.push({source:t,target:l,start:n[s],end:n[r]});continue}const h=Ii(e,l);for(const e of h){const s=ta(i,o[e.start],o[e.end],e.loop),r=Ri(t,n,s);for(const t of r)a.push({source:t,target:e,start:{[i]:ia(l,s,"start",Math.max)},end:{[i]:ia(l,s,"end",Math.min)}})}}return a}(i,s,n);for(const{source:e,target:l,start:h,end:c}of r){const{style:{backgroundColor:r=o}={}}=e,d=!0!==s;t.save(),t.fillStyle=r,ga(t,a,d&&ta(n,h,c)),t.beginPath();const u=!!i.pathSegment(t,e);let f;if(d){u?t.closePath():pa(t,s,c,n);const e=!!s.pathSegment(t,l,{move:u,reverse:!0});f=u&&e,f||pa(t,s,h,n)}t.closePath(),t.fill(f?"evenodd":"nonzero"),t.restore()}}function ga(t,e,i){const{top:s,bottom:n}=e.chart.chartArea,{property:o,start:a,end:r}=i||{};"x"===o&&(t.beginPath(),t.rect(a,s,r-a,n-s),t.clip())}function pa(t,e,i,s){const n=e.interpolate(i,s);n&&t.lineTo(n.x,n.y)}var ma={id:"filler",afterDatasetsUpdate(t,e,i){const s=(t.data.datasets||[]).length,n=[];let o,a,r,l;for(a=0;a<s;++a)o=t.getDatasetMeta(a),r=o.dataset,l=null,r&&r.options&&r instanceof no&&(l={visible:t.isDatasetVisible(a),index:a,fill:aa(r,a,s),chart:t,axis:o.controller.options.indexAxis,scale:o.vScale,line:r}),o.$filler=l,n.push(l);for(a=0;a<s;++a)l=n[a],l&&!1!==l.fill&&(l.fill=oa(n,a,i.propagate))},beforeDraw(t,e,i){const s="beforeDraw"===i.drawTime,n=t.getSortedVisibleDatasetMetas(),o=t.chartArea;for(let e=n.length-1;e>=0;--e){const i=n[e].$filler;i&&(i.line.updateControlPoints(o,i.axis),s&&i.fill&&da(t.ctx,i,o))}},beforeDatasetsDraw(t,e,i){if("beforeDatasetsDraw"!==i.drawTime)return;const s=t.getSortedVisibleDatasetMetas();for(let e=s.length-1;e>=0;--e){const i=s[e].$filler;na(i)&&da(t.ctx,i,t.chartArea)}},beforeDatasetDraw(t,e,i){const s=e.meta.$filler;na(s)&&"beforeDatasetDraw"===i.drawTime&&da(t.ctx,s,t.chartArea)},defaults:{propagate:!0,drawTime:"beforeDatasetDraw"}};const ba=(t,e)=>{let{boxHeight:i=e,boxWidth:s=e}=t;return t.usePointStyle&&(i=Math.min(i,e),s=t.pointStyleWidth||Math.min(s,e)),{boxWidth:s,boxHeight:i,itemHeight:Math.max(e,i)}};class xa extends Hs{constructor(t){super(),this._added=!1,this.legendHitBoxes=[],this._hoveredItem=null,this.doughnutMode=!1,this.chart=t.chart,this.options=t.options,this.ctx=t.ctx,this.legendItems=void 0,this.columnSizes=void 0,this.lineWidths=void 0,this.maxHeight=void 0,this.maxWidth=void 0,this.top=void 0,this.bottom=void 0,this.left=void 0,this.right=void 0,this.height=void 0,this.width=void 0,this._margins=void 0,this.position=void 0,this.weight=void 0,this.fullSize=void 0}update(t,e,i){this.maxWidth=t,this.maxHeight=e,this._margins=i,this.setDimensions(),this.buildLabels(),this.fit()}setDimensions(){this.isHorizontal()?(this.width=this.maxWidth,this.left=this._margins.left,this.right=this.width):(this.height=this.maxHeight,this.top=this._margins.top,this.bottom=this.height)}buildLabels(){const t=this.options.labels||{};let e=d(t.generateLabels,[this.chart],this)||[];t.filter&&(e=e.filter((e=>t.filter(e,this.chart.data)))),t.sort&&(e=e.sort(((e,i)=>t.sort(e,i,this.chart.data)))),this.options.reverse&&e.reverse(),this.legendItems=e}fit(){const{options:t,ctx:e}=this;if(!t.display)return void(this.width=this.height=0);const i=t.labels,s=Si(i.font),n=s.size,o=this._computeTitleHeight(),{boxWidth:a,itemHeight:r}=ba(i,n);let l,h;e.font=s.string,this.isHorizontal()?(l=this.maxWidth,h=this._fitRows(o,n,a,r)+10):(h=this.maxHeight,l=this._fitCols(o,s,a,r)+10),this.width=Math.min(l,t.maxWidth||this.maxWidth),this.height=Math.min(h,t.maxHeight||this.maxHeight)}_fitRows(t,e,i,s){const{ctx:n,maxWidth:o,options:{labels:{padding:a}}}=this,r=this.legendHitBoxes=[],l=this.lineWidths=[0],h=s+a;let c=t;n.textAlign="left",n.textBaseline="middle";let d=-1,u=-h;return this.legendItems.forEach(((t,f)=>{const g=i+e/2+n.measureText(t.text).width;(0===f||l[l.length-1]+g+2*a>o)&&(c+=h,l[l.length-(f>0?0:1)]=0,u+=h,d++),r[f]={left:0,top:u,row:d,width:g,height:s},l[l.length-1]+=g+a})),c}_fitCols(t,e,i,s){const{ctx:n,maxHeight:o,options:{labels:{padding:a}}}=this,r=this.legendHitBoxes=[],l=this.columnSizes=[],h=o-t;let c=a,d=0,u=0,f=0,g=0;return this.legendItems.forEach(((t,o)=>{const{itemWidth:p,itemHeight:m}=function(t,e,i,s,n){const o=function(t,e,i,s){let n=t.text;n&&"string"!=typeof n&&(n=n.reduce(((t,e)=>t.length>e.length?t:e)));return e+i.size/2+s.measureText(n).width}(s,t,e,i),a=function(t,e,i){let s=t;"string"!=typeof e.text&&(s=_a(e,i));return s}(n,s,e.lineHeight);return{itemWidth:o,itemHeight:a}}(i,e,n,t,s);o>0&&u+m+2*a>h&&(c+=d+a,l.push({width:d,height:u}),f+=d+a,g++,d=u=0),r[o]={left:f,top:u,col:g,width:p,height:m},d=Math.max(d,p),u+=m+a})),c+=d,l.push({width:d,height:u}),c}adjustHitBoxes(){if(!this.options.display)return;const t=this._computeTitleHeight(),{legendHitBoxes:e,options:{align:i,labels:{padding:s},rtl:n}}=this,o=Oi(n,this.left,this.width);if(this.isHorizontal()){let n=0,a=ft(i,this.left+s,this.right-this.lineWidths[n]);for(const r of e)n!==r.row&&(n=r.row,a=ft(i,this.left+s,this.right-this.lineWidths[n])),r.top+=this.top+t+s,r.left=o.leftForLtr(o.x(a),r.width),a+=r.width+s}else{let n=0,a=ft(i,this.top+t+s,this.bottom-this.columnSizes[n].height);for(const r of e)r.col!==n&&(n=r.col,a=ft(i,this.top+t+s,this.bottom-this.columnSizes[n].height)),r.top=a,r.left+=this.left+s,r.left=o.leftForLtr(o.x(r.left),r.width),a+=r.height+s}}isHorizontal(){return"top"===this.options.position||"bottom"===this.options.position}draw(){if(this.options.display){const t=this.ctx;Ie(t,this),this._draw(),ze(t)}}_draw(){const{options:t,columnSizes:e,lineWidths:i,ctx:s}=this,{align:n,labels:o}=t,a=ue.color,r=Oi(t.rtl,this.left,this.width),h=Si(o.font),{padding:c}=o,d=h.size,u=d/2;let f;this.drawTitle(),s.textAlign=r.textAlign("left"),s.textBaseline="middle",s.lineWidth=.5,s.font=h.string;const{boxWidth:g,boxHeight:p,itemHeight:m}=ba(o,d),b=this.isHorizontal(),x=this._computeTitleHeight();f=b?{x:ft(n,this.left+c,this.right-i[0]),y:this.top+c+x,line:0}:{x:this.left+c,y:ft(n,this.top+x+c,this.bottom-e[0].height),line:0},Ai(this.ctx,t.textDirection);const _=m+c;this.legendItems.forEach(((y,v)=>{s.strokeStyle=y.fontColor,s.fillStyle=y.fontColor;const M=s.measureText(y.text).width,w=r.textAlign(y.textAlign||(y.textAlign=o.textAlign)),k=g+u+M;let S=f.x,P=f.y;r.setWidth(this.width),b?v>0&&S+k+c>this.right&&(P=f.y+=_,f.line++,S=f.x=ft(n,this.left+c,this.right-i[f.line])):v>0&&P+_>this.bottom&&(S=f.x=S+e[f.line].width+c,f.line++,P=f.y=ft(n,this.top+x+c,this.bottom-e[f.line].height));if(function(t,e,i){if(isNaN(g)||g<=0||isNaN(p)||p<0)return;s.save();const n=l(i.lineWidth,1);if(s.fillStyle=l(i.fillStyle,a),s.lineCap=l(i.lineCap,"butt"),s.lineDashOffset=l(i.lineDashOffset,0),s.lineJoin=l(i.lineJoin,"miter"),s.lineWidth=n,s.strokeStyle=l(i.strokeStyle,a),s.setLineDash(l(i.lineDash,[])),o.usePointStyle){const a={radius:p*Math.SQRT2/2,pointStyle:i.pointStyle,rotation:i.rotation,borderWidth:n},l=r.xPlus(t,g/2);Ee(s,a,l,e+u,o.pointStyleWidth&&g)}else{const o=e+Math.max((d-p)/2,0),a=r.leftForLtr(t,g),l=wi(i.borderRadius);s.beginPath(),Object.values(l).some((t=>0!==t))?He(s,{x:a,y:o,w:g,h:p,radius:l}):s.rect(a,o,g,p),s.fill(),0!==n&&s.stroke()}s.restore()}(r.x(S),P,y),S=gt(w,S+g+u,b?S+k:this.right,t.rtl),function(t,e,i){Ne(s,i.text,t,e+m/2,h,{strikethrough:i.hidden,textAlign:r.textAlign(i.textAlign)})}(r.x(S),P,y),b)f.x+=k+c;else if("string"!=typeof y.text){const t=h.lineHeight;f.y+=_a(y,t)+c}else f.y+=_})),Ti(this.ctx,t.textDirection)}drawTitle(){const t=this.options,e=t.title,i=Si(e.font),s=ki(e.padding);if(!e.display)return;const n=Oi(t.rtl,this.left,this.width),o=this.ctx,a=e.position,r=i.size/2,l=s.top+r;let h,c=this.left,d=this.width;if(this.isHorizontal())d=Math.max(...this.lineWidths),h=this.top+l,c=ft(t.align,c,this.right-d);else{const e=this.columnSizes.reduce(((t,e)=>Math.max(t,e.height)),0);h=l+ft(t.align,this.top,this.bottom-e-t.labels.padding-this._computeTitleHeight())}const u=ft(a,c,c+d);o.textAlign=n.textAlign(ut(a)),o.textBaseline="middle",o.strokeStyle=e.color,o.fillStyle=e.color,o.font=i.string,Ne(o,e.text,u,h,i)}_computeTitleHeight(){const t=this.options.title,e=Si(t.font),i=ki(t.padding);return t.display?e.lineHeight+i.height:0}_getLegendItemAt(t,e){let i,s,n;if(tt(t,this.left,this.right)&&tt(e,this.top,this.bottom))for(n=this.legendHitBoxes,i=0;i<n.length;++i)if(s=n[i],tt(t,s.left,s.left+s.width)&&tt(e,s.top,s.top+s.height))return this.legendItems[i];return null}handleEvent(t){const e=this.options;if(!function(t,e){if(("mousemove"===t||"mouseout"===t)&&(e.onHover||e.onLeave))return!0;if(e.onClick&&("click"===t||"mouseup"===t))return!0;return!1}(t.type,e))return;const i=this._getLegendItemAt(t.x,t.y);if("mousemove"===t.type||"mouseout"===t.type){const o=this._hoveredItem,a=(n=i,null!==(s=o)&&null!==n&&s.datasetIndex===n.datasetIndex&&s.index===n.index);o&&!a&&d(e.onLeave,[t,o,this],this),this._hoveredItem=i,i&&!a&&d(e.onHover,[t,i,this],this)}else i&&d(e.onClick,[t,i,this],this);var s,n}}function _a(t,e){return e*(t.text?t.text.length:0)}var ya={id:"legend",_element:xa,start(t,e,i){const s=t.legend=new xa({ctx:t.ctx,options:i,chart:t});as.configure(t,s,i),as.addBox(t,s)},stop(t){as.removeBox(t,t.legend),delete t.legend},beforeUpdate(t,e,i){const s=t.legend;as.configure(t,s,i),s.options=i},afterUpdate(t){const e=t.legend;e.buildLabels(),e.adjustHitBoxes()},afterEvent(t,e){e.replay||t.legend.handleEvent(e.event)},defaults:{display:!0,position:"top",align:"center",fullSize:!0,reverse:!1,weight:1e3,onClick(t,e,i){const s=e.datasetIndex,n=i.chart;n.isDatasetVisible(s)?(n.hide(s),e.hidden=!0):(n.show(s),e.hidden=!1)},onHover:null,onLeave:null,labels:{color:t=>t.chart.options.color,boxWidth:40,padding:10,generateLabels(t){const e=t.data.datasets,{labels:{usePointStyle:i,pointStyle:s,textAlign:n,color:o,useBorderRadius:a,borderRadius:r}}=t.legend.options;return t._getSortedDatasetMetas().map((t=>{const l=t.controller.getStyle(i?0:void 0),h=ki(l.borderWidth);return{text:e[t.index].label,fillStyle:l.backgroundColor,fontColor:o,hidden:!t.visible,lineCap:l.borderCapStyle,lineDash:l.borderDash,lineDashOffset:l.borderDashOffset,lineJoin:l.borderJoinStyle,lineWidth:(h.width+h.height)/4,strokeStyle:l.borderColor,pointStyle:s||l.pointStyle,rotation:l.rotation,textAlign:n||l.textAlign,borderRadius:a&&(r||l.borderRadius),datasetIndex:t.index}}),this)}},title:{color:t=>t.chart.options.color,display:!1,position:"center",text:""}},descriptors:{_scriptable:t=>!t.startsWith("on"),labels:{_scriptable:t=>!["generateLabels","filter","sort"].includes(t)}}};class va extends Hs{constructor(t){super(),this.chart=t.chart,this.options=t.options,this.ctx=t.ctx,this._padding=void 0,this.top=void 0,this.bottom=void 0,this.left=void 0,this.right=void 0,this.width=void 0,this.height=void 0,this.position=void 0,this.weight=void 0,this.fullSize=void 0}update(t,e){const i=this.options;if(this.left=0,this.top=0,!i.display)return void(this.width=this.height=this.right=this.bottom=0);this.width=this.right=t,this.height=this.bottom=e;const s=n(i.text)?i.text.length:1;this._padding=ki(i.padding);const o=s*Si(i.font).lineHeight+this._padding.height;this.isHorizontal()?this.height=o:this.width=o}isHorizontal(){const t=this.options.position;return"top"===t||"bottom"===t}_drawArgs(t){const{top:e,left:i,bottom:s,right:n,options:o}=this,a=o.align;let r,l,h,c=0;return this.isHorizontal()?(l=ft(a,i,n),h=e+t,r=n-i):("left"===o.position?(l=i+t,h=ft(a,s,e),c=-.5*C):(l=n-t,h=ft(a,e,s),c=.5*C),r=s-e),{titleX:l,titleY:h,maxWidth:r,rotation:c}}draw(){const t=this.ctx,e=this.options;if(!e.display)return;const i=Si(e.font),s=i.lineHeight/2+this._padding.top,{titleX:n,titleY:o,maxWidth:a,rotation:r}=this._drawArgs(s);Ne(t,e.text,0,0,i,{color:e.color,maxWidth:a,rotation:r,textAlign:ut(e.align),textBaseline:"middle",translation:[n,o]})}}var Ma={id:"title",_element:va,start(t,e,i){!function(t,e){const i=new va({ctx:t.ctx,options:e,chart:t});as.configure(t,i,e),as.addBox(t,i),t.titleBlock=i}(t,i)},stop(t){const e=t.titleBlock;as.removeBox(t,e),delete t.titleBlock},beforeUpdate(t,e,i){const s=t.titleBlock;as.configure(t,s,i),s.options=i},defaults:{align:"center",display:!1,font:{weight:"bold"},fullSize:!0,padding:10,position:"top",text:"",weight:2e3},defaultRoutes:{color:"color"},descriptors:{_scriptable:!0,_indexable:!1}};const wa=new WeakMap;var ka={id:"subtitle",start(t,e,i){const s=new va({ctx:t.ctx,options:i,chart:t});as.configure(t,s,i),as.addBox(t,s),wa.set(t,s)},stop(t){as.removeBox(t,wa.get(t)),wa.delete(t)},beforeUpdate(t,e,i){const s=wa.get(t);as.configure(t,s,i),s.options=i},defaults:{align:"center",display:!1,font:{weight:"normal"},fullSize:!0,padding:0,position:"top",text:"",weight:1500},defaultRoutes:{color:"color"},descriptors:{_scriptable:!0,_indexable:!1}};const Sa={average(t){if(!t.length)return!1;let e,i,s=0,n=0,o=0;for(e=0,i=t.length;e<i;++e){const i=t[e].element;if(i&&i.hasValue()){const t=i.tooltipPosition();s+=t.x,n+=t.y,++o}}return{x:s/o,y:n/o}},nearest(t,e){if(!t.length)return!1;let i,s,n,o=e.x,a=e.y,r=Number.POSITIVE_INFINITY;for(i=0,s=t.length;i<s;++i){const s=t[i].element;if(s&&s.hasValue()){const t=q(e,s.getCenterPoint());t<r&&(r=t,n=s)}}if(n){const t=n.tooltipPosition();o=t.x,a=t.y}return{x:o,y:a}}};function Pa(t,e){return e&&(n(e)?Array.prototype.push.apply(t,e):t.push(e)),t}function Da(t){return("string"==typeof t||t instanceof String)&&t.indexOf("\\n")>-1?t.split("\\n"):t}function Ca(t,e){const{element:i,datasetIndex:s,index:n}=e,o=t.getDatasetMeta(s).controller,{label:a,value:r}=o.getLabelAndValue(n);return{chart:t,label:a,parsed:o.getParsed(n),raw:t.data.datasets[s].data[n],formattedValue:r,dataset:o.getDataset(),dataIndex:n,datasetIndex:s,element:i}}function Oa(t,e){const i=t.chart.ctx,{body:s,footer:n,title:o}=t,{boxWidth:a,boxHeight:r}=e,l=Si(e.bodyFont),h=Si(e.titleFont),c=Si(e.footerFont),d=o.length,f=n.length,g=s.length,p=ki(e.padding);let m=p.height,b=0,x=s.reduce(((t,e)=>t+e.before.length+e.lines.length+e.after.length),0);if(x+=t.beforeBody.length+t.afterBody.length,d&&(m+=d*h.lineHeight+(d-1)*e.titleSpacing+e.titleMarginBottom),x){m+=g*(e.displayColors?Math.max(r,l.lineHeight):l.lineHeight)+(x-g)*l.lineHeight+(x-1)*e.bodySpacing}f&&(m+=e.footerMarginTop+f*c.lineHeight+(f-1)*e.footerSpacing);let _=0;const y=function(t){b=Math.max(b,i.measureText(t).width+_)};return i.save(),i.font=h.string,u(t.title,y),i.font=l.string,u(t.beforeBody.concat(t.afterBody),y),_=e.displayColors?a+2+e.boxPadding:0,u(s,(t=>{u(t.before,y),u(t.lines,y),u(t.after,y)})),_=0,i.font=c.string,u(t.footer,y),i.restore(),b+=p.width,{width:b,height:m}}function Aa(t,e,i,s){const{x:n,width:o}=i,{width:a,chartArea:{left:r,right:l}}=t;let h="center";return"center"===s?h=n<=(r+l)/2?"left":"right":n<=o/2?h="left":n>=a-o/2&&(h="right"),function(t,e,i,s){const{x:n,width:o}=s,a=i.caretSize+i.caretPadding;return"left"===t&&n+o+a>e.width||"right"===t&&n-o-a<0||void 0}(h,t,e,i)&&(h="center"),h}function Ta(t,e,i){const s=i.yAlign||e.yAlign||function(t,e){const{y:i,height:s}=e;return i<s/2?"top":i>t.height-s/2?"bottom":"center"}(t,i);return{xAlign:i.xAlign||e.xAlign||Aa(t,e,i,s),yAlign:s}}function La(t,e,i,s){const{caretSize:n,caretPadding:o,cornerRadius:a}=t,{xAlign:r,yAlign:l}=i,h=n+o,{topLeft:c,topRight:d,bottomLeft:u,bottomRight:f}=wi(a);let g=function(t,e){let{x:i,width:s}=t;return"right"===e?i-=s:"center"===e&&(i-=s/2),i}(e,r);const p=function(t,e,i){let{y:s,height:n}=t;return"top"===e?s+=i:s-="bottom"===e?n+i:n/2,s}(e,l,h);return"center"===l?"left"===r?g+=h:"right"===r&&(g-=h):"left"===r?g-=Math.max(c,u)+n:"right"===r&&(g+=Math.max(d,f)+n),{x:J(g,0,s.width-e.width),y:J(p,0,s.height-e.height)}}function Ea(t,e,i){const s=ki(i.padding);return"center"===e?t.x+t.width/2:"right"===e?t.x+t.width-s.right:t.x+s.left}function Ra(t){return Pa([],Da(t))}function Ia(t,e){const i=e&&e.dataset&&e.dataset.tooltip&&e.dataset.tooltip.callbacks;return i?t.override(i):t}const za={beforeTitle:e,title(t){if(t.length>0){const e=t[0],i=e.chart.data.labels,s=i?i.length:0;if(this&&this.options&&"dataset"===this.options.mode)return e.dataset.label||"";if(e.label)return e.label;if(s>0&&e.dataIndex<s)return i[e.dataIndex]}return""},afterTitle:e,beforeBody:e,beforeLabel:e,label(t){if(this&&this.options&&"dataset"===this.options.mode)return t.label+": "+t.formattedValue||t.formattedValue;let e=t.dataset.label||"";e&&(e+=": ");const i=t.formattedValue;return s(i)||(e+=i),e},labelColor(t){const e=t.chart.getDatasetMeta(t.datasetIndex).controller.getStyle(t.dataIndex);return{borderColor:e.borderColor,backgroundColor:e.backgroundColor,borderWidth:e.borderWidth,borderDash:e.borderDash,borderDashOffset:e.borderDashOffset,borderRadius:0}},labelTextColor(){return this.options.bodyColor},labelPointStyle(t){const e=t.chart.getDatasetMeta(t.datasetIndex).controller.getStyle(t.dataIndex);return{pointStyle:e.pointStyle,rotation:e.rotation}},afterLabel:e,afterBody:e,beforeFooter:e,footer:e,afterFooter:e};function Fa(t,e,i,s){const n=t[e].call(i,s);return void 0===n?za[e].call(i,s):n}class Va extends Hs{static positioners=Sa;constructor(t){super(),this.opacity=0,this._active=[],this._eventPosition=void 0,this._size=void 0,this._cachedAnimations=void 0,this._tooltipItems=[],this.$animations=void 0,this.$context=void 0,this.chart=t.chart,this.options=t.options,this.dataPoints=void 0,this.title=void 0,this.beforeBody=void 0,this.body=void 0,this.afterBody=void 0,this.footer=void 0,this.xAlign=void 0,this.yAlign=void 0,this.x=void 0,this.y=void 0,this.height=void 0,this.width=void 0,this.caretX=void 0,this.caretY=void 0,this.labelColors=void 0,this.labelPointStyles=void 0,this.labelTextColors=void 0}initialize(t){this.options=t,this._cachedAnimations=void 0,this.$context=void 0}_resolveAnimations(){const t=this._cachedAnimations;if(t)return t;const e=this.chart,i=this.options.setContext(this.getContext()),s=i.enabled&&e.options.animation&&i.animations,n=new Os(this.chart,s);return s._cacheable&&(this._cachedAnimations=Object.freeze(n)),n}getContext(){return this.$context||(this.$context=(t=this.chart.getContext(),e=this,i=this._tooltipItems,Ci(t,{tooltip:e,tooltipItems:i,type:"tooltip"})));var t,e,i}getTitle(t,e){const{callbacks:i}=e,s=Fa(i,"beforeTitle",this,t),n=Fa(i,"title",this,t),o=Fa(i,"afterTitle",this,t);let a=[];return a=Pa(a,Da(s)),a=Pa(a,Da(n)),a=Pa(a,Da(o)),a}getBeforeBody(t,e){return Ra(Fa(e.callbacks,"beforeBody",this,t))}getBody(t,e){const{callbacks:i}=e,s=[];return u(t,(t=>{const e={before:[],lines:[],after:[]},n=Ia(i,t);Pa(e.before,Da(Fa(n,"beforeLabel",this,t))),Pa(e.lines,Fa(n,"label",this,t)),Pa(e.after,Da(Fa(n,"afterLabel",this,t))),s.push(e)})),s}getAfterBody(t,e){return Ra(Fa(e.callbacks,"afterBody",this,t))}getFooter(t,e){const{callbacks:i}=e,s=Fa(i,"beforeFooter",this,t),n=Fa(i,"footer",this,t),o=Fa(i,"afterFooter",this,t);let a=[];return a=Pa(a,Da(s)),a=Pa(a,Da(n)),a=Pa(a,Da(o)),a}_createItems(t){const e=this._active,i=this.chart.data,s=[],n=[],o=[];let a,r,l=[];for(a=0,r=e.length;a<r;++a)l.push(Ca(this.chart,e[a]));return t.filter&&(l=l.filter(((e,s,n)=>t.filter(e,s,n,i)))),t.itemSort&&(l=l.sort(((e,s)=>t.itemSort(e,s,i)))),u(l,(e=>{const i=Ia(t.callbacks,e);s.push(Fa(i,"labelColor",this,e)),n.push(Fa(i,"labelPointStyle",this,e)),o.push(Fa(i,"labelTextColor",this,e))})),this.labelColors=s,this.labelPointStyles=n,this.labelTextColors=o,this.dataPoints=l,l}update(t,e){const i=this.options.setContext(this.getContext()),s=this._active;let n,o=[];if(s.length){const t=Sa[i.position].call(this,s,this._eventPosition);o=this._createItems(i),this.title=this.getTitle(o,i),this.beforeBody=this.getBeforeBody(o,i),this.body=this.getBody(o,i),this.afterBody=this.getAfterBody(o,i),this.footer=this.getFooter(o,i);const e=this._size=Oa(this,i),a=Object.assign({},t,e),r=Ta(this.chart,i,a),l=La(i,a,r,this.chart);this.xAlign=r.xAlign,this.yAlign=r.yAlign,n={opacity:1,x:l.x,y:l.y,width:e.width,height:e.height,caretX:t.x,caretY:t.y}}else 0!==this.opacity&&(n={opacity:0});this._tooltipItems=o,this.$context=void 0,n&&this._resolveAnimations().update(this,n),t&&i.external&&i.external.call(this,{chart:this.chart,tooltip:this,replay:e})}drawCaret(t,e,i,s){const n=this.getCaretPosition(t,i,s);e.lineTo(n.x1,n.y1),e.lineTo(n.x2,n.y2),e.lineTo(n.x3,n.y3)}getCaretPosition(t,e,i){const{xAlign:s,yAlign:n}=this,{caretSize:o,cornerRadius:a}=i,{topLeft:r,topRight:l,bottomLeft:h,bottomRight:c}=wi(a),{x:d,y:u}=t,{width:f,height:g}=e;let p,m,b,x,_,y;return"center"===n?(_=u+g/2,"left"===s?(p=d,m=p-o,x=_+o,y=_-o):(p=d+f,m=p+o,x=_-o,y=_+o),b=p):(m="left"===s?d+Math.max(r,h)+o:"right"===s?d+f-Math.max(l,c)-o:this.caretX,"top"===n?(x=u,_=x-o,p=m-o,b=m+o):(x=u+g,_=x+o,p=m+o,b=m-o),y=x),{x1:p,x2:m,x3:b,y1:x,y2:_,y3:y}}drawTitle(t,e,i){const s=this.title,n=s.length;let o,a,r;if(n){const l=Oi(i.rtl,this.x,this.width);for(t.x=Ea(this,i.titleAlign,i),e.textAlign=l.textAlign(i.titleAlign),e.textBaseline="middle",o=Si(i.titleFont),a=i.titleSpacing,e.fillStyle=i.titleColor,e.font=o.string,r=0;r<n;++r)e.fillText(s[r],l.x(t.x),t.y+o.lineHeight/2),t.y+=o.lineHeight+a,r+1===n&&(t.y+=i.titleMarginBottom-a)}}_drawColorBox(t,e,i,s,n){const a=this.labelColors[i],r=this.labelPointStyles[i],{boxHeight:l,boxWidth:h}=n,c=Si(n.bodyFont),d=Ea(this,"left",n),u=s.x(d),f=l<c.lineHeight?(c.lineHeight-l)/2:0,g=e.y+f;if(n.usePointStyle){const e={radius:Math.min(h,l)/2,pointStyle:r.pointStyle,rotation:r.rotation,borderWidth:1},i=s.leftForLtr(u,h)+h/2,o=g+l/2;t.strokeStyle=n.multiKeyBackground,t.fillStyle=n.multiKeyBackground,Le(t,e,i,o),t.strokeStyle=a.borderColor,t.fillStyle=a.backgroundColor,Le(t,e,i,o)}else{t.lineWidth=o(a.borderWidth)?Math.max(...Object.values(a.borderWidth)):a.borderWidth||1,t.strokeStyle=a.borderColor,t.setLineDash(a.borderDash||[]),t.lineDashOffset=a.borderDashOffset||0;const e=s.leftForLtr(u,h),i=s.leftForLtr(s.xPlus(u,1),h-2),r=wi(a.borderRadius);Object.values(r).some((t=>0!==t))?(t.beginPath(),t.fillStyle=n.multiKeyBackground,He(t,{x:e,y:g,w:h,h:l,radius:r}),t.fill(),t.stroke(),t.fillStyle=a.backgroundColor,t.beginPath(),He(t,{x:i,y:g+1,w:h-2,h:l-2,radius:r}),t.fill()):(t.fillStyle=n.multiKeyBackground,t.fillRect(e,g,h,l),t.strokeRect(e,g,h,l),t.fillStyle=a.backgroundColor,t.fillRect(i,g+1,h-2,l-2))}t.fillStyle=this.labelTextColors[i]}drawBody(t,e,i){const{body:s}=this,{bodySpacing:n,bodyAlign:o,displayColors:a,boxHeight:r,boxWidth:l,boxPadding:h}=i,c=Si(i.bodyFont);let d=c.lineHeight,f=0;const g=Oi(i.rtl,this.x,this.width),p=function(i){e.fillText(i,g.x(t.x+f),t.y+d/2),t.y+=d+n},m=g.textAlign(o);let b,x,_,y,v,M,w;for(e.textAlign=o,e.textBaseline="middle",e.font=c.string,t.x=Ea(this,m,i),e.fillStyle=i.bodyColor,u(this.beforeBody,p),f=a&&"right"!==m?"center"===o?l/2+h:l+2+h:0,y=0,M=s.length;y<M;++y){for(b=s[y],x=this.labelTextColors[y],e.fillStyle=x,u(b.before,p),_=b.lines,a&&_.length&&(this._drawColorBox(e,t,y,g,i),d=Math.max(c.lineHeight,r)),v=0,w=_.length;v<w;++v)p(_[v]),d=c.lineHeight;u(b.after,p)}f=0,d=c.lineHeight,u(this.afterBody,p),t.y-=n}drawFooter(t,e,i){const s=this.footer,n=s.length;let o,a;if(n){const r=Oi(i.rtl,this.x,this.width);for(t.x=Ea(this,i.footerAlign,i),t.y+=i.footerMarginTop,e.textAlign=r.textAlign(i.footerAlign),e.textBaseline="middle",o=Si(i.footerFont),e.fillStyle=i.footerColor,e.font=o.string,a=0;a<n;++a)e.fillText(s[a],r.x(t.x),t.y+o.lineHeight/2),t.y+=o.lineHeight+i.footerSpacing}}drawBackground(t,e,i,s){const{xAlign:n,yAlign:o}=this,{x:a,y:r}=t,{width:l,height:h}=i,{topLeft:c,topRight:d,bottomLeft:u,bottomRight:f}=wi(s.cornerRadius);e.fillStyle=s.backgroundColor,e.strokeStyle=s.borderColor,e.lineWidth=s.borderWidth,e.beginPath(),e.moveTo(a+c,r),"top"===o&&this.drawCaret(t,e,i,s),e.lineTo(a+l-d,r),e.quadraticCurveTo(a+l,r,a+l,r+d),"center"===o&&"right"===n&&this.drawCaret(t,e,i,s),e.lineTo(a+l,r+h-f),e.quadraticCurveTo(a+l,r+h,a+l-f,r+h),"bottom"===o&&this.drawCaret(t,e,i,s),e.lineTo(a+u,r+h),e.quadraticCurveTo(a,r+h,a,r+h-u),"center"===o&&"left"===n&&this.drawCaret(t,e,i,s),e.lineTo(a,r+c),e.quadraticCurveTo(a,r,a+c,r),e.closePath(),e.fill(),s.borderWidth>0&&e.stroke()}_updateAnimationTarget(t){const e=this.chart,i=this.$animations,s=i&&i.x,n=i&&i.y;if(s||n){const i=Sa[t.position].call(this,this._active,this._eventPosition);if(!i)return;const o=this._size=Oa(this,t),a=Object.assign({},i,this._size),r=Ta(e,t,a),l=La(t,a,r,e);s._to===l.x&&n._to===l.y||(this.xAlign=r.xAlign,this.yAlign=r.yAlign,this.width=o.width,this.height=o.height,this.caretX=i.x,this.caretY=i.y,this._resolveAnimations().update(this,l))}}_willRender(){return!!this.opacity}draw(t){const e=this.options.setContext(this.getContext());let i=this.opacity;if(!i)return;this._updateAnimationTarget(e);const s={width:this.width,height:this.height},n={x:this.x,y:this.y};i=Math.abs(i)<.001?0:i;const o=ki(e.padding),a=this.title.length||this.beforeBody.length||this.body.length||this.afterBody.length||this.footer.length;e.enabled&&a&&(t.save(),t.globalAlpha=i,this.drawBackground(n,t,s,e),Ai(t,e.textDirection),n.y+=o.top,this.drawTitle(n,t,e),this.drawBody(n,t,e),this.drawFooter(n,t,e),Ti(t,e.textDirection),t.restore())}getActiveElements(){return this._active||[]}setActiveElements(t,e){const i=this._active,s=t.map((({datasetIndex:t,index:e})=>{const i=this.chart.getDatasetMeta(t);if(!i)throw new Error("Cannot find a dataset at index "+t);return{datasetIndex:t,element:i.data[e],index:e}})),n=!f(i,s),o=this._positionChanged(s,e);(n||o)&&(this._active=s,this._eventPosition=e,this._ignoreReplayEvents=!0,this.update(!0))}handleEvent(t,e,i=!0){if(e&&this._ignoreReplayEvents)return!1;this._ignoreReplayEvents=!1;const s=this.options,n=this._active||[],o=this._getActiveElements(t,n,e,i),a=this._positionChanged(o,t),r=e||!f(o,n)||a;return r&&(this._active=o,(s.enabled||s.external)&&(this._eventPosition={x:t.x,y:t.y},this.update(!0,e))),r}_getActiveElements(t,e,i,s){const n=this.options;if("mouseout"===t.type)return[];if(!s)return e;const o=this.chart.getElementsAtEventForMode(t,n.mode,n,i);return n.reverse&&o.reverse(),o}_positionChanged(t,e){const{caretX:i,caretY:s,options:n}=this,o=Sa[n.position].call(this,t,e);return!1!==o&&(i!==o.x||s!==o.y)}}var Ba={id:"tooltip",_element:Va,positioners:Sa,afterInit(t,e,i){i&&(t.tooltip=new Va({chart:t,options:i}))},beforeUpdate(t,e,i){t.tooltip&&t.tooltip.initialize(i)},reset(t,e,i){t.tooltip&&t.tooltip.initialize(i)},afterDraw(t){const e=t.tooltip;if(e&&e._willRender()){const i={tooltip:e};if(!1===t.notifyPlugins("beforeTooltipDraw",{...i,cancelable:!0}))return;e.draw(t.ctx),t.notifyPlugins("afterTooltipDraw",i)}},afterEvent(t,e){if(t.tooltip){const i=e.replay;t.tooltip.handleEvent(e.event,i,e.inChartArea)&&(e.changed=!0)}},defaults:{enabled:!0,external:null,position:"average",backgroundColor:"rgba(0,0,0,0.8)",titleColor:"#fff",titleFont:{weight:"bold"},titleSpacing:2,titleMarginBottom:6,titleAlign:"left",bodyColor:"#fff",bodySpacing:2,bodyFont:{},bodyAlign:"left",footerColor:"#fff",footerSpacing:2,footerMarginTop:6,footerFont:{weight:"bold"},footerAlign:"left",padding:6,caretPadding:2,caretSize:5,cornerRadius:6,boxHeight:(t,e)=>e.bodyFont.size,boxWidth:(t,e)=>e.bodyFont.size,multiKeyBackground:"#fff",displayColors:!0,boxPadding:0,borderColor:"rgba(0,0,0,0)",borderWidth:0,animation:{duration:400,easing:"easeOutQuart"},animations:{numbers:{type:"number",properties:["x","y","width","height","caretX","caretY"]},opacity:{easing:"linear",duration:200}},callbacks:za},defaultRoutes:{bodyFont:"font",footerFont:"font",titleFont:"font"},descriptors:{_scriptable:t=>"filter"!==t&&"itemSort"!==t&&"external"!==t,_indexable:!1,callbacks:{_scriptable:!1,_indexable:!1},animation:{_fallback:!1},animations:{_fallback:"animation"}},additionalOptionScopes:["interaction"]};return An.register(Yn,jo,fo,t),An.helpers={...Wi},An._adapters=Rn,An.Animation=Cs,An.Animations=Os,An.animator=xt,An.controllers=en.controllers.items,An.DatasetController=Ns,An.Element=Hs,An.elements=fo,An.Interaction=Xi,An.layouts=as,An.platforms=Ss,An.Scale=Js,An.Ticks=ae,Object.assign(An,Yn,jo,fo,t,Ss),An.Chart=An,"undefined"!=typeof window&&(window.Chart=An),An}));\n//# sourceMappingURL=chart.umd.js.map\n'

# ── helpers ──────────────────────────────────────────────────────────────────

def _j(x):
    """Serialize to JSON-safe Python literal for embedding in <script>."""
    return json.dumps(x)

def _fmt(v, pct=False, dp=1):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    if pct:
        return f"{v*100:+.{dp}f}%"
    return f"{v:+.{dp}f}"

# ── parse daily report ────────────────────────────────────────────────────────

def load_daily_report() -> dict:
    reports = sorted(ROOT.glob("daily_reports/daily_*.md"))
    if not reports:
        return {}
    latest = reports[-1]
    txt = latest.read_text()
    date_str = latest.stem.replace("daily_", "")

    # HMM regime
    if "Bull" in txt:
        hmm = "BULL"
    elif "Sideways" in txt or "SIDEWAYS" in txt or "Neutral" in txt:
        hmm = "SIDEWAYS"
    else:
        hmm = "BEAR"
    macro_line = re.search(r"Macro Overlay:\s+\*\*(.+?)\*\*", txt)
    macro = macro_line.group(1) if macro_line else "NEUTRAL"

    # ── LONG / SHORT books: read from clean ranked CSVs, not the markdown ──────
    # (the markdown daily report's LONG/SHORT tables can carry nan scores and
    #  alphabetical — not score-ranked — tickers; daily_picks.csv / daily_shorts.csv
    #  are the authoritative, correctly-ranked books.)
    def _latest_prices() -> dict:
        try:
            pc = pd.read_csv(ROOT / "sp500_price_cache.csv", index_col=0)
            last = pc.iloc[-1]
            return {str(k): v for k, v in last.items()}
        except Exception:
            return {}

    def _z(v, center=50.0, scale=50.0):
        try:
            return (float(v) - center) / scale
        except Exception:
            return 0.0

    prices = _latest_prices()

    def _price_str(tk: str) -> str:
        p = prices.get(tk)
        try:
            if p is not None and not (isinstance(p, float) and np.isnan(p)):
                return f"{float(p):,.2f}"
        except Exception:
            pass
        return "—"

    # LONG book from daily_picks.csv (BUY / STRONG BUY, ranked by alpha)
    longs = []
    try:
        dp = pd.read_csv(ROOT / "daily_picks.csv")
        dp = dp[dp["ticker"].astype(str).str.match(r"^[A-Z][A-Z.\-]*$")]  # drop garbage like "1"
        for _, row in dp.iterrows():
            longs.append({
                "rank":   int(row.get("alpha_rank", 0) or 0),
                "ticker": str(row["ticker"]),
                "score":  _z(row.get("alpha_score"), 50.0, 25.0),   # ~z-scale
                "price":  _price_str(str(row["ticker"])),
                "ml":     _z(row.get("sig_regime_ml", 50)),
                "factor": _z(row.get("sig_momentum", 50)),
            })
    except Exception:
        pass

    # SHORT book from daily_shorts.csv (bottom alpha ranks)
    shorts = []
    etf_prefixes = {"XL","QQ","SO","SM","SP","IW","GD","TL","HY","UU","LQ","GL"}
    try:
        ds = pd.read_csv(ROOT / "daily_shorts.csv")
        ds = ds[ds["ticker"].astype(str).str.match(r"^[A-Z][A-Z.\-]*$")]
        for i, (_, row) in enumerate(ds.iterrows(), start=1):
            tk = str(row["ticker"])
            is_etf = tk[:2] in etf_prefixes or (len(tk) >= 4 and not tk.isupper())
            shorts.append({
                "rank":   i,
                "ticker": tk,
                "score":  _z(row.get("alpha_score"), 50.0, 25.0),
                "price":  _price_str(tk),
                "is_etf": is_etf,
            })
    except Exception:
        pass

    # Signal changes
    changes = {}
    for tag, key in [("NEW LONG", "new_long"), ("EXIT LONG", "exit_long"),
                     ("NEW SHORT", "new_short"), ("EXIT SHORT", "exit_short")]:
        m2 = re.search(rf"{tag}:\s+(.+)", txt)
        changes[key] = [t.strip() for t in m2.group(1).split(",")] if m2 else []

    return {"date": date_str, "hmm": hmm, "macro": macro,
            "longs": longs, "shorts": shorts, **changes}

# ── load OOS equity curve ─────────────────────────────────────────────────────

def load_oos_chart() -> dict:
    p = ROOT / "wf_oos_equity_curve.csv"
    if not p.exists():
        return {}
    df = pd.read_csv(p, parse_dates=["rebalance_date"])
    oos = df[df["period"] == "OOS"].copy()
    if oos.empty:
        return {}
    base_ml  = oos["ml_nav_oos"].iloc[0]
    base_spy = oos["spy_nav"].iloc[0]
    oos["ml_idx"]  = (oos["ml_nav_oos"] / base_ml  * 100).round(1)
    oos["spy_idx"] = (oos["spy_nav"]    / base_spy  * 100).round(1)
    return {
        "labels":  oos["rebalance_date"].dt.strftime("%b %Y").tolist(),
        "ml":      oos["ml_idx"].tolist(),
        "spy":     oos["spy_idx"].tolist(),
        "final_ml":  round(float(oos["ml_idx"].iloc[-1]), 0),
        "final_spy": round(float(oos["spy_idx"].iloc[-1]), 0),
    }

# ── load HMM regime ──────────────────────────────────────────────────────────

def load_hmm_regime() -> dict:
    p = ROOT / "hmm_regime_daily.csv"
    if not p.exists():
        return {}
    try:
        df = pd.read_csv(p, parse_dates=["date"])
        df = df.dropna(subset=["date"]).sort_values("date")
        if df.empty:
            return {}
        last = df.iloc[-1]
        last_date = last["date"].date()
        days_stale = (datetime.now().date() - last_date).days
        # regime col: 0=BULL (low-vol state), 1=BEAR (high-vol state)
        regime_val = int(last.get("regime", 0))
        regime_str = "BEAR" if regime_val == 1 else "BULL"
        prob_bull = float(last.get("prob_bull", 0.5))
        prob_bear = float(last.get("prob_bear", 0.5))
        return {
            "regime":     regime_str,
            "prob_bull":  round(prob_bull, 4),
            "prob_bear":  round(prob_bear, 4),
            "date":       last_date.isoformat(),
            "days_stale": days_stale,
            "stale":      days_stale > 3,
        }
    except Exception:
        return {}


def load_macro_regime_outlook() -> dict:
    """Load the forward-looking macro regime outlook from macro_regime_outlook.json."""
    p = ROOT / "macro_regime_outlook.json"
    if not p.exists():
        return {}
    try:
        import json as _json
        data = _json.loads(p.read_text())
        return data
    except Exception:
        return {}

# ── load OOS summary ──────────────────────────────────────────────────────────

def load_oos_summary() -> dict:
    p = ROOT / "wf_oos_summary.csv"
    if not p.exists():
        return {}
    df = pd.read_csv(p).set_index("metric")
    def g(k, col):
        try: return df.loc[k, col]
        except: return "—"
    return {
        "oos_ic":     float(g("Ensemble IC",        "out_of_sample")),
        "oos_t":      float(g("IC t-stat",          "out_of_sample")),
        "oos_sharpe": float(g("Annualised Sharpe",   "out_of_sample")),
        "oos_dd":     float(g("Max Drawdown %",      "out_of_sample")),
        "oos_wr":     float(g("Win Rate vs SPY %",   "out_of_sample")),
        "oos_ret":    float(g("ML Total Return %",   "out_of_sample")),
        "spy_ret":    float(g("SPY Total Return %",  "out_of_sample")),
        "is_sharpe":  float(g("Annualised Sharpe",   "in_sample")),
        "is_ic":      float(g("Ensemble IC",         "in_sample")),
        "is_t":       float(g("IC t-stat",           "in_sample")),
        "is_dd":      float(g("Max Drawdown %",      "in_sample")),
        "is_wr":      float(g("Win Rate vs SPY %",   "in_sample")),
    }

# ── load live / paper data ────────────────────────────────────────────────────

def load_live_data() -> dict:
    out = {"positions": [], "ic_rows": [], "ic_status": "", "days_acc": 0}

    # Paper trading log — show only the most recent day's positions
    # (skip if the book is stale >30 days: a frozen legacy snapshot is worse than empty)
    p = ROOT / "paper_trading_log.csv"
    if p.exists():
        df = pd.read_csv(p)
        # 修复日期解析: 首行带时间戳会让 pandas 按单一格式推断, 后续纯日期行全变 NaT。
        # 先截取前10位(YYYY-MM-DD)再解析, 保证全部 142 行都被识别。
        df["date"] = pd.to_datetime(df["date"].astype(str).str.slice(0, 10), errors="coerce")
        df = df.dropna(subset=["date"]).sort_values("date")
        _fresh = (not df.empty) and (pd.Timestamp.now() - df.iloc[-1]["date"]).days <= 30
        if _fresh:
            r = df.iloc[-1]   # latest row only
            longs  = str(r.get("long_stocks",  "")).split("|") if pd.notna(r.get("long_stocks"))  else []
            shorts = str(r.get("short_stocks", "")).split("|") if pd.notna(r.get("short_stocks")) else []
            price_cols = {c.replace("price_", ""): r[c] for c in df.columns if c.startswith("price_") and pd.notna(r[c])}
            for tk in longs:
                tk = tk.strip()
                if tk:
                    out["positions"].append({
                        "date": str(r["date"].date()), "ticker": tk, "side": "LONG",
                        "entry": price_cols.get(tk, "—"),
                        "regime": str(r.get("hmm_regime", "—")),
                    })
            for tk in shorts:
                tk = tk.strip()
                if tk:
                    out["positions"].append({
                        "date": str(r["date"].date()), "ticker": tk, "side": "SHORT",
                        "entry": price_cols.get(tk, "—"),
                        "regime": str(r.get("hmm_regime", "—")),
                    })

    obs_p = ROOT / "live_ic_history.csv"
    if obs_p.exists():
        df = pd.read_csv(obs_p)
        if "evaluation_status" in df.columns:
            done = df[df["evaluation_status"] == "COMPLETE_LOCAL_IC"]
            for _, r in done.iterrows():
                ic_val = r.get("ic", None)
                if pd.notna(ic_val):
                    out["ic_rows"].append({
                        "signal":  str(r.get("signal", "—")),
                        "horizon": int(r.get("hold_days", 0)),
                        "ic":      round(float(ic_val), 4),
                    })

    # IC report status
    rpt = ROOT / "live_ic_report.md"
    if rpt.exists():
        txt = rpt.read_text()
        m = re.search(r"Accumulated \*\*(\d+)\*\* scoring days", txt)
        if m:
            out["days_acc"] = int(m.group(1))
        out["ic_status"] = txt.strip()

    return out

# ── load accruals ─────────────────────────────────────────────────────────────

def load_accruals() -> list:
    p = ROOT / "accruals_snapshot.csv"
    if not p.exists():
        return []
    df = pd.read_csv(p).sort_values("accrual_quality", ascending=False)
    rows = []
    for _, r in df.iterrows():
        rows.append({
            "ticker": r["ticker"],
            "ratio":  round(float(r["accrual_ratio"]), 3),
            "quality": round(float(r["accrual_quality"]), 3),
        })
    return rows

# ── load squeeze ──────────────────────────────────────────────────────────────

def load_squeeze() -> list:
    p = ROOT / "short_squeeze_signal.csv"
    if not p.exists():
        return []
    df = pd.read_csv(p).sort_values(["conditions_met", "intensity_score"],
                                    ascending=[False, False])
    rows = []
    for _, r in df.head(15).iterrows():
        rows.append({
            "ticker": r["ticker"],
            "conds":  int(r["conditions_met"]),
            "score":  round(float(r["intensity_score"]), 3),
            "mom_vs_spy": round(float(r["momentum_vs_spy"]) * 100, 1),
        })
    return rows

# ── load backtest monthly returns ─────────────────────────────────────────────

def load_backtest_monthly() -> dict:
    p = ROOT / "backtest_5yr_monthly.csv"
    if not p.exists():
        return {}
    df = pd.read_csv(p)
    if "rebalance_date" not in df.columns:
        return {}
    df["rebalance_date"] = pd.to_datetime(df["rebalance_date"], errors="coerce")
    df = df.dropna(subset=["rebalance_date"]).sort_values("rebalance_date")
    labels  = df["rebalance_date"].dt.strftime("%b %Y").tolist()
    strat   = [round(float(v) * 100, 2) for v in df["strategy_ret"]]
    spy     = [round(float(v) * 100, 2) for v in df["spy_ret"]]
    alpha   = [round(float(v) * 100, 2) for v in df.get("alpha", pd.Series([0]*len(df)))]
    strat_cum  = [round(float(v) * 100, 1) for v in df["strategy_cum"]] if "strategy_cum" in df.columns else []
    bench_cum  = [round(float(v) * 100, 1) for v in df["bench_cum"]]   if "bench_cum"   in df.columns else []
    wins = sum(1 for s, b in zip(strat, spy) if s > b)
    total = len(strat)
    # Compute annual compounded returns per calendar year
    df["year"] = df["rebalance_date"].dt.year
    annual_rets = {}
    for yr, grp in df.groupby("year"):
        if "strategy_ret" in grp.columns:
            cum = 1.0
            for v in grp["strategy_ret"]:
                try:
                    cum *= (1 + float(v))
                except Exception:
                    pass
            annual_rets[int(yr)] = round((cum - 1) * 100, 1)
    return {
        "labels": labels, "strat": strat, "spy": spy, "alpha": alpha,
        "strat_cum": strat_cum, "bench_cum": bench_cum,
        "win_rate": round(wins / total * 100, 0) if total else 0,
        "total_months": total,
        "final_strat_cum": strat_cum[-1] if strat_cum else 0,
        "final_bench_cum": bench_cum[-1] if bench_cum else 0,
        "annual_rets": annual_rets,
    }

# ── load paper NAV chart ───────────────────────────────────────────────────────

def load_paper_nav_chart() -> dict:
    for fname in ("paper_nav_curve.csv", "paper_sim_nav.csv"):
        p = ROOT / fname
        if not p.exists():
            continue
        df = pd.read_csv(p)
        if "date" not in df.columns:
            continue
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"]).sort_values("date")
        nav_col = "nav" if "nav" in df.columns else df.columns[1]
        labels = df["date"].dt.strftime("%b %d").tolist()
        nav    = [round(float(v), 2) for v in df[nav_col]]
        hwm    = [round(float(v), 2) for v in df["hwm"]] if "hwm" in df.columns else []
        dd     = [round(float(v) * 100, 2) for v in df["drawdown_pct"]] if "drawdown_pct" in df.columns else []
        start  = nav[0] if nav else 100
        final  = nav[-1] if nav else 100
        gain   = round((final / start - 1) * 100, 2) if start else 0
        max_dd = round(min(dd), 2) if dd else 0
        return {
            "labels": labels, "nav": nav, "hwm": hwm, "dd": dd,
            "start": start, "final": final, "gain": gain, "max_dd": max_dd,
            "n_days": len(nav),
        }
    return {}

# ── new loaders ───────────────────────────────────────────────────────────────

def load_workflow_steps() -> list:
    p = ROOT / "daily_workflow_steps.csv"
    if not p.exists():
        return []
    df = pd.read_csv(p)
    return df.to_dict("records")

def load_workflow_queue() -> list:
    p = ROOT / "daily_workflow_queue.csv"
    if not p.exists():
        return []
    df = pd.read_csv(p)
    cols = ["priority_rank","priority","ticker","sector","workflow_bucket",
            "what_to_do","sector_cycle_state","best_horizon",
            "sector_adjusted_action","option_route","risk_action","event_gate"]
    cols = [c for c in cols if c in df.columns]
    df = df[cols].head(20)
    return df.to_dict("records")

def load_alpha_scores() -> list:
    p = ROOT / "alpha_scores.csv"
    if not p.exists():
        return []
    df = pd.read_csv(p)
    sig_cols = [c for c in df.columns if c.startswith("sig_")]
    keep = ["ticker","alpha_score","alpha_rank","regime","signal","sector","crowding_level"] + sig_cols[:6]
    keep = [c for c in keep if c in df.columns]
    df = df[keep].sort_values("alpha_score", ascending=False).head(500)
    return df.to_dict("records")

def load_risk_gate() -> list:
    p = ROOT / "final_risk_gate.csv"
    if not p.exists():
        return []
    df = pd.read_csv(p)
    cols = ["ticker","sector","current_weight_pct","final_risk_action",
            "recommended_risk_weight_pct","risk_reduction_pct_of_current","reason_stack"]
    cols = [c for c in cols if c in df.columns]
    return df[cols].to_dict("records")

def load_ticker_drilldown() -> list:
    p = ROOT / "action_readiness_ticker_drilldown.csv"
    if not p.exists():
        return []
    df = pd.read_csv(p)
    cols = ["ticker","sector","current_stage","readiness_score",
            "why_blocked_plain_english","first_blocking_gate","first_gate_status",
            "first_clear_condition","trigger_to_watch","decision_room_summary",
            "option_summary","risk_summary","monitor_summary"]
    cols = [c for c in cols if c in df.columns]
    return df[cols].head(15).to_dict("records")

def load_desk_monitor() -> list:
    p = ROOT / "desk_monitor_events.csv"
    if not p.exists():
        return []
    df = pd.read_csv(p)
    cols = ["date","monitor","ticker","severity","title","detail","action"]
    cols = [c for c in cols if c in df.columns]
    df = df[cols].sort_values("severity", ascending=True) if "severity" in df.columns else df[cols]
    return df.head(20).to_dict("records")

def load_sector_cycle() -> list:
    p = ROOT / "sector_cycle_state.csv"
    if not p.exists():
        return []
    df = pd.read_csv(p)
    cols = ["etf","sector","cycle_state","rotation_label","cycle_score",
            "ret_20d_pct","ret_63d_pct","portfolio_weight_pct",
            "top_headline","cycle_note","cap_status"]
    cols = [c for c in cols if c in df.columns]
    df = df[cols].sort_values("cycle_score", ascending=False) if "cycle_score" in df.columns else df[cols]
    return df.to_dict("records")


def load_news() -> list:
    """Load stock news from stock_news.json + catalysts from event_research_dossier.csv."""
    TONE_LABEL  = {"POSITIVE": "Bullish signal", "NEGATIVE": "Bearish signal", "NEUTRAL": "No clear direction"}
    TONE_COLOR  = {"POSITIVE": "#1B6F4A", "NEGATIVE": "#B83232", "NEUTRAL": "#c8b487"}
    TONE_CLASS  = {"POSITIVE": "pos", "NEGATIVE": "neg", "NEUTRAL": "neu"}
    items = []
    p = ROOT / "stock_news.json"
    if p.exists():
        raw = json.loads(p.read_text(encoding="utf-8"))
        news_dict = raw.get("news", {})
        for ticker, ticker_news in news_dict.items():
            for item in (ticker_news or [])[:3]:
                tone = str(item.get("market_tone", "NEUTRAL")).upper()
                impact = int(str(item.get("impact_score", 0) or 0).split(".")[0])
                raw_action = item.get("action_hint", "")
                # Simplify verbose action hints to one plain sentence
                if "Potential catalyst" in raw_action:
                    action_clean = "Watch this — could be a buying opportunity. Confirm before acting."
                elif "Potential risk" in raw_action:
                    action_clean = "Risk flag — consider reducing or avoid adding to this position."
                elif "Context only" in raw_action:
                    action_clean = "Background info only — no action needed right now."
                else:
                    action_clean = raw_action
                items.append({
                    "ticker": ticker,
                    "title": item.get("title", ""),
                    "summary": item.get("summary", ""),
                    "publisher": item.get("publisher", ""),
                    "published": item.get("published", ""),
                    "link": item.get("link", ""),
                    "bullish_reasons": item.get("bullish_reasons", []),
                    "bearish_reasons": item.get("bearish_reasons", []),
                    "tone": TONE_LABEL.get(tone, "No clear direction"),
                    "tone_color": TONE_COLOR.get(tone, "#c8b487"),
                    "tone_class": TONE_CLASS.get(tone, "neu"),
                    "logic": item.get("news_logic", ""),
                    "action_hint": action_clean,
                    "impact": impact,
                })
    # pull catalysts/risks from dossier for enrichment
    p2 = ROOT / "event_research_dossier.csv"
    catalysts_map = {}
    risks_map = {}
    if p2.exists():
        df2 = pd.read_csv(p2)
        for _, row in df2.iterrows():
            t = str(row.get("ticker", ""))
            catalysts_map[t] = str(row.get("catalysts", "")) if not pd.isna(row.get("catalysts", "")) else ""
            risks_map[t] = str(row.get("risks", "")) if not pd.isna(row.get("risks", "")) else ""
    for item in items:
        item["catalysts"] = catalysts_map.get(item["ticker"], "")
        item["risks"] = risks_map.get(item["ticker"], "")
    items.sort(key=lambda x: x["impact"], reverse=True)
    return items[:50]


def _clean_rec_action(text: str) -> str:
    """Convert raw recommended_action CSV text to plain English."""
    text = text.replace("✅ No earnings risk in 3 weeks", "No earnings report coming up in the next 3 weeks")
    text = text.replace("No earnings risk", "No earnings report coming up")
    text = text.replace("Consider exiting or hedging before earnings", "Consider reducing or protecting this position before the earnings report")
    text = text.replace("⚠️", "⚠").replace("✅", "✓")
    return text.strip()


def load_earnings_calendar() -> list:
    """Load upcoming earnings sorted by days_until. Include recent past (>=−5 days)."""
    ACTION_PLAIN = {
        "STRONG BUY": "Strong buy signal",
        "BUY": "Buy signal",
        "HOLD": "Hold — no change yet",
        "SELL": "Reduce or exit",
        "STRONG SELL": "Exit this position",
    }
    RISK_PLAIN = {
        "HIGH":   "Earnings report soon — be cautious",
        "LOW":    "Earnings report not imminent",
        "CLEAR":  "No earnings report in the next 3 weeks",
        "MEDIUM": "Earnings approaching — watch closely",
        "WATCH":  "Earnings window — monitor",
    }
    p = ROOT / "earnings_calendar.csv"
    if not p.exists():
        return []
    df = pd.read_csv(p)
    if "days_until" in df.columns:
        upcoming = df[df["days_until"] >= -5]
        df = upcoming.sort_values("days_until") if len(upcoming) > 0 else df.sort_values("days_until", ascending=False).head(20)
    out = []
    for _, row in df.head(30).iterrows():
        action_raw = str(row.get("action", "")).strip().upper()
        risk_raw   = str(row.get("risk_flag", "")).strip().upper()
        days = row.get("days_until", None)
        if pd.isna(days):
            days_label = "Date TBD"
        elif days == 0:
            days_label = "Reporting TODAY"
        elif days == 1:
            days_label = "Tomorrow"
        elif days < 0:
            days_label = f"{int(abs(days))} days ago"
        else:
            days_label = f"In {int(days)} days"
        out.append({
            "ticker": str(row.get("ticker", "")),
            "earnings_date": str(row.get("earnings_date", "")),
            "days_until": days,
            "days_label": days_label,
            "risk_flag": RISK_PLAIN.get(risk_raw, risk_raw),
            "risk_class": "high" if risk_raw == "HIGH" else "low",
            "alpha_score": float(row.get("alpha_score", 0) or 0),
            "action": ACTION_PLAIN.get(action_raw, action_raw),
            "recommended_action": _clean_rec_action(str(row.get("recommended_action", ""))),
        })
    return out


def load_macro_breadth() -> dict:
    """Load index breadth + sector rotation for macro context."""
    TREND_LABEL  = {"UPTREND": "Uptrend ↑", "DOWNTREND": "Downtrend ↓", "SIDEWAYS": "Sideways →"}
    TREND_CLASS  = {"UPTREND": "up", "DOWNTREND": "dn", "SIDEWAYS": "neu"}
    ROT_CLASS    = {"LEADER": "leader", "NEUTRAL": "neu", "LAGGARD": "lag"}
    ROT_LABEL    = {"LEADER": "Leading the market ↑", "NEUTRAL": "In line with market →", "LAGGARD": "Falling behind ↓"}
    result = {"breadth": [], "rotation": []}
    p1 = ROOT / "index_breadth_dashboard.csv"
    if p1.exists():
        df = pd.read_csv(p1)
        INDEX_NAMES = {"SPY": "S&P 500", "QQQ": "Nasdaq 100", "IWM": "Small Caps", "DIA": "Dow Jones",
                       "XLK": "Technology", "SMH": "Semiconductors"}
        for _, row in df.iterrows():
            t  = str(row.get("ticker", ""))
            ts = str(row.get("trend_state", "")).upper()
            r20 = float(row.get("ret_20d", 0) or 0)
            result["breadth"].append({
                "ticker": t,
                "name": INDEX_NAMES.get(t, t),
                "close": float(row.get("close", 0) or 0),
                "ret_20d": r20,
                "ret_20d_str": f"{r20*100:+.1f}%",
                "trend": TREND_LABEL.get(ts, ts),
                "trend_class": TREND_CLASS.get(ts, "neu"),
                "above_20dma": bool(row.get("above_20dma", False)),
                "above_50dma": bool(row.get("above_50dma", False)),
            })
    p2 = ROOT / "sector_rotation_scores.csv"
    if p2.exists():
        df2 = pd.read_csv(p2)
        if "rotation_score" in df2.columns:
            df2 = df2.sort_values("rotation_score", ascending=False)
        for _, row in df2.head(12).iterrows():
            rl = str(row.get("rotation_label", "")).upper()
            result["rotation"].append({
                "ticker": str(row.get("ticker", "")),
                "theme": str(row.get("theme", "")),
                "ret_20d": float(row.get("ret_20d", 0) or 0),
                "ret_63d": float(row.get("ret_63d", 0) or 0),
                "rotation_score": float(row.get("rotation_score", 0) or 0),
                "label": ROT_LABEL.get(rl, rl),
                "label_class": ROT_CLASS.get(rl, "neu"),
            })
    return result


_RISK_ACTION_PLAIN = {
    "SIZE_DOWN":      ("Reduce this position — it's too large",      "bad"),
    "REDUCE_ONLY":    ("Trim before adding anything new",             "bad"),
    "BLOCKED":        ("Blocked — do not act right now",              "bad"),
    "HOLD":           ("Hold — no change needed",                     "neu"),
    "REVIEW":         ("Needs manual review before acting",           "neu"),
    "CLEAR":          ("Ready — all gates green",                     "good"),
    "OK":             ("OK — within limits",                         "good"),
    "PASS":           ("Skip — signal not strong enough",             "neu"),
}

_STAGE_PLAIN = {
    "NON_RISK_GATES_REQUIRED":    "Risk fixed — other checks pending",
    "RISK_REPAIR_REQUIRED":       "Reduce risk first",
    "RISK_REPAIR_IN_PROGRESS":    "Risk repair underway",
    "READY":                      "Ready to research",
    "WATCH":                      "Under active monitoring",
    "BLOCKED":                    "Blocked — action required",
    "CLEAR":                      "All gates clear",
    "GATE_REVIEW":                "Gate review needed",
}

def _humanize_reason_stack(reason: str) -> str:
    """Convert 'master:SIZE_DOWN; single:REDUCE_ONLY; ...' into plain English."""
    if not reason or ":" not in reason:
        return reason[:80] if reason else "—"
    _GATE = {
        "master":       "overall size too large",
        "single":       "exceeds max per-stock limit",
        "earnings_gap": "earnings report coming up",
        "kelly":        "too large for the signal strength",
        "sector":       "sector already at limit",
        "liquidity":    "stock not liquid enough",
        "crisis":       "crash risk too high",
        "event":        "major event risk",
    }
    _STATUS = {
        "SIZE_DOWN":   "reduce size",
        "REDUCE_ONLY": "must cut",
        "CLEAR":       "clear",
        "OK":          "OK",
        "HOLD":        "hold",
        "REVIEW":      "needs review",
        "BLOCK_NEW":   "do not add",
        "BLOCKED":     "blocked",
    }
    failed = []
    for part in reason.split(";"):
        part = part.strip()
        if ":" not in part:
            continue
        gate, status = part.split(":", 1)
        gate = gate.strip().lower(); status = status.strip().upper()
        if status in ("CLEAR", "OK"):
            continue
        glabel = _GATE.get(gate, gate)
        slabel = _STATUS.get(status, status.lower())
        failed.append(f"{glabel} ({slabel})")
    if not failed:
        return "All gates clear"
    n = len(failed)
    return f"{n} gate{'s' if n>1 else ''} flagged — " + ", ".join(failed)

def load_position_pnl() -> list:
    """Per-position health check: entry price, current signal view, risk gate status."""
    live = load_live_data()
    positions = live.get("positions", [])

    price_map = {}
    p = ROOT / "price_refresh_desk.csv"
    if p.exists():
        df = pd.read_csv(p)
        for _, row in df.iterrows():
            lp = row.get("latest_price")
            price_map[str(row["ticker"])] = {
                "price": float(lp) if not pd.isna(lp) else None,
                "date":  str(row.get("latest_price_date", "")),
                "stale": float(row.get("days_stale_vs_today", 0) or 0),
            }

    alpha_map = {}
    p2 = ROOT / "alpha_scores.csv"
    if p2.exists():
        df2 = pd.read_csv(p2)
        for _, row in df2.iterrows():
            alpha_map[str(row["ticker"])] = {
                "score":    float(row.get("alpha_score", 0) or 0),
                "signal":   str(row.get("signal", "")),
                "crowding": str(row.get("crowding_level", "")),
                "sector":   str(row.get("sector", "")),
            }

    risk_map = {}
    p3 = ROOT / "final_risk_gate.csv"
    if p3.exists():
        df3 = pd.read_csv(p3)
        for _, row in df3.iterrows():
            action = str(row.get("final_risk_action", ""))
            plain, cls = _RISK_ACTION_PLAIN.get(action, (action, "neu"))
            risk_map[str(row["ticker"])] = {
                "action":  action,
                "plain":   plain,
                "cls":     cls,
                "reason":  _humanize_reason_stack(str(row.get("reason_stack", ""))),
            }

    attr_map = {}
    attr_path = ROOT / "alpaca_pnl_attribution.csv"
    if attr_path.exists():
        df_attr = pd.read_csv(attr_path)
        for _, row in df_attr.iterrows():
            attr_map[str(row.get("ticker", ""))] = {
                "predicted_mu":   float(row.get("predicted_mu", 0) or 0),
                "book":           str(row.get("book", "—")),
                "unrealized_pnl": float(row.get("unrealized_pnl", 0) or 0),
                "unrealized_ret": float(row.get("unrealized_ret", 0) or 0),
                "alpha_vs_pred":  float(row.get("alpha_vs_predicted", 0) or 0),
                "market_value":   float(row.get("market_value", 0) or 0),
            }

    action_map = {}
    wf_path = ROOT / "daily_workflow_queue.csv"
    if wf_path.exists():
        df_wf = pd.read_csv(wf_path)
        for _, row in df_wf.iterrows():
            t = str(row.get("ticker", ""))
            if t:
                action_map[t] = {
                    "priority":   str(row.get("priority", "—")),
                    "bucket":     str(row.get("workflow_bucket", "—")),
                    "what_to_do": str(row.get("what_to_do", "—")),
                }

    out = []
    for pos in positions:
        ticker    = str(pos["ticker"])
        side      = str(pos["side"])
        entry     = float(pos["entry"]) if pos.get("entry") not in ("—", None) else None
        pi        = price_map.get(ticker, {})
        curr      = pi.get("price")
        if entry and curr and entry > 0:
            pnl = (curr - entry) / entry * 100 if side == "LONG" else (entry - curr) / entry * 100
        else:
            pnl = None
        ai = alpha_map.get(ticker, {})
        ri = risk_map.get(ticker, {})
        at = attr_map.get(ticker, {})
        ac = action_map.get(ticker, {})
        sig = ai.get("signal", "")
        if side == "LONG":
            aligned = sig in ("BUY", "STRONG BUY") if sig else None
        else:
            aligned = sig in ("SELL", "SHORT", "STRONG SELL") if sig else None
        out.append({
            "ticker":          ticker,
            "side":            side,
            "entry_date":      str(pos.get("date", "")),
            "entry":           entry,
            "curr":            curr,
            "price_date":      pi.get("date", ""),
            "stale":           pi.get("stale", 0),
            "pnl":             pnl,
            "pnl_str":         f"{pnl:+.2f}%" if pnl is not None else "—",
            "alpha_score":     ai.get("score", 0),
            "signal":          sig,
            "crowding":        ai.get("crowding", ""),
            "sector":          ai.get("sector", ""),
            "risk_action":     ri.get("action", ""),
            "risk_plain":      ri.get("plain", ""),
            "risk_cls":        ri.get("cls", "neu"),
            "risk_reason":     ri.get("reason", ""),
            "aligned":         aligned,
            "predicted_mu":    at.get("predicted_mu", 0),
            "book":            at.get("book", "—"),
            "unrealized_pnl_usd": at.get("unrealized_pnl", 0),
            "alpha_vs_pred":   at.get("alpha_vs_pred", 0),
            "market_value":    at.get("market_value", 0),
            "action_priority": ac.get("priority", "—"),
            "action_bucket":   ac.get("bucket", "—"),
            "action_rec":      ac.get("what_to_do", "—"),
        })
    return out


def load_options_flow() -> dict:
    """Load unusual options flow data from options_flow.json."""
    import json as _j
    p = ROOT / "options_flow.json"
    if not p.exists():
        return {}
    try:
        return _j.loads(p.read_text())
    except Exception:
        return {}


def load_etf_flow() -> dict:
    """Load real-time ETF sector rotation data from etf_flow_daily.json."""
    import json as _j
    p = ROOT / "etf_flow_daily.json"
    if not p.exists():
        return {}
    try:
        return _j.loads(p.read_text())
    except Exception:
        return {}


def load_crowding_monitor() -> dict:
    """Portfolio-level crowding and factor exposure monitor."""
    result = {
        "factor_trend": [],
        "sector_concentration": {},
        "watch_tickers": [],
        "long_semis": [],
        "short_semis": [],
    }
    p = ROOT / "factor_exposure_monthly.csv"
    if p.exists():
        df = pd.read_csv(p)
        oos = df[df["period"] == "OOS"].tail(12) if "period" in df.columns else df.tail(12)
        for _, row in oos.iterrows():
            beta = float(row.get("port_beta", 0) or 0)
            mom  = float(row.get("port_mom", 0) or 0)
            result["factor_trend"].append({
                "date":     str(row.get("date", "")),
                "beta":     round(beta, 3),
                "momentum": round(mom, 3),
                "beta_warn":   beta > 1.0,
                "mom_warn":    mom > 1.0,
            })

    live = load_live_data()
    positions = live.get("positions", [])
    p2 = ROOT / "alpha_scores.csv"
    if p2.exists():
        df2 = pd.read_csv(p2)
        long_tks  = [p["ticker"] for p in positions if p["side"] == "LONG"]
        short_tks = [p["ticker"] for p in positions if p["side"] == "SHORT"]
        longs_df  = df2[df2["ticker"].isin(long_tks)]
        if "sector" in longs_df.columns:
            result["sector_concentration"] = longs_df["sector"].value_counts().to_dict()
        if "crowding_level" in df2.columns:
            watch = df2[(df2["ticker"].isin(long_tks + short_tks)) & (df2["crowding_level"] == "WATCH")]
            result["watch_tickers"] = watch["ticker"].tolist()

    SEMI_TICKERS = {"MU", "INTC", "AMD", "QCOM", "NVDA", "AVGO", "TXN", "SMH", "SOXX", "MCHP", "NXPI", "ADI"}
    result["long_semis"]  = [p["ticker"] for p in positions if p["side"] == "LONG"  and p["ticker"] in SEMI_TICKERS]
    result["short_semis"] = [p["ticker"] for p in positions if p["side"] == "SHORT" and p["ticker"] in SEMI_TICKERS]
    return result


def load_rolling_ic() -> dict:
    """Rolling IC monitor for the ML ensemble signal — last 24 months."""
    result = {"labels": [], "ic_3m": [], "ic_6m": [], "statuses": [],
              "target": 0.370, "current_3m": None, "current_status": "—",
              "factor_labels": [], "factor_ic": {}}
    p = ROOT / "rolling_ic_monitor.csv"
    if p.exists():
        df = pd.read_csv(p)
        ens = df[df["signal"] == "ml_ensemble"].sort_values("date").tail(24)
        result["labels"]   = ens["date"].tolist()
        result["ic_3m"]    = [round(float(v), 4) for v in ens["ic_3m"].fillna(0)]
        result["ic_6m"]    = [round(float(v), 4) for v in ens["ic_6m"].fillna(0)]
        result["statuses"] = ens["status"].tolist()
        if len(ens):
            result["current_3m"]     = float(ens["ic_3m"].iloc[-1])
            result["current_status"] = str(ens["status"].iloc[-1])
    p2 = ROOT / "factor_ic_history.csv"
    df2 = pd.DataFrame()
    if p2.exists() and p2.stat().st_size > 2:
        try:
            df2 = pd.read_csv(p2)
        except Exception:
            df2 = pd.DataFrame()
    if not df2.empty:
        date_col = "date" if "date" in df2.columns else ("rebalance_date" if "rebalance_date" in df2.columns else df2.columns[0])
        df2 = df2.sort_values(date_col).tail(24)
        if "factor" in df2.columns and "ic" in df2.columns:
            # long format: one row per factor per date
            for factor in ["Momentum", "LowVol", "Value"]:
                sub = df2[df2["factor"] == factor].tail(18)
                if result["factor_labels"] == []:
                    result["factor_labels"] = sub[date_col].tolist()
                result["factor_ic"][factor] = [round(float(v), 4) for v in sub["ic"].fillna(0)]
        else:
            # wide format: columns like ic_mom_12_1, ic_inv_vol, ic_above_200
            _WIDE_MAP = {
                "ic_mom_12_1": "Momentum",
                "ic_inv_vol":  "LowVol",
                "ic_above_200": "Value",
            }
            result["factor_labels"] = df2[date_col].tolist()
            for col, label in _WIDE_MAP.items():
                if col in df2.columns:
                    result["factor_ic"][label] = [round(float(v), 4) for v in df2[col].fillna(0)]
    return result


def load_signal_health() -> dict:
    """Load IC decay, cross-signal ICs, correlation monitor, joint beta."""
    result = {
        "ic_decay":    [],   # [{signal, h5, h10, h21, h42, h63, h126}]
        "cross_ic":    [],   # [{signal, ic_3m, ic_6m, status}]
        "corr_latest": {},   # {metric: value}
        "joint_beta":  {},   # {joint_beta, v9_beta, v11_beta, status}
    }
    # IC decay by horizon
    p = ROOT / "ic_decay_by_lag.csv"
    if p.exists():
        try:
            df = pd.read_csv(p)
            if "signal" in df.columns and "horizon" in df.columns and "ic" in df.columns:
                # Latest IC per (signal, horizon) pair
                piv = (df.sort_values("date")
                         .groupby(["signal", "horizon"])["ic"]
                         .last()
                         .unstack("horizon"))
                for sig in piv.index:
                    row = {"signal": sig}
                    for h in [5, 10, 21, 42, 63, 126]:
                        v = piv.loc[sig].get(h, float("nan"))
                        row[f"h{h}"] = round(float(v), 3) if pd.notna(v) else None
                    result["ic_decay"].append(row)
        except Exception:
            pass
    # Cross-signal rolling IC
    p2 = ROOT / "rolling_ic_monitor.csv"
    if p2.exists():
        try:
            df = pd.read_csv(p2)
            oos = df[df["period"] == "OOS"] if "period" in df.columns else df
            latest = oos.sort_values("date").groupby("signal").last().reset_index()
            for _, r in latest.iterrows():
                result["cross_ic"].append({
                    "signal":  str(r["signal"]),
                    "ic_3m":   round(float(r["ic_3m"]), 3) if pd.notna(r.get("ic_3m")) else None,
                    "ic_6m":   round(float(r["ic_6m"]), 3) if pd.notna(r.get("ic_6m")) else None,
                    "status":  str(r.get("status", "")),
                })
        except Exception:
            pass
    # Correlation monitor latest row
    p3 = ROOT / "correlation_monitor.csv"
    if p3.exists():
        try:
            df = pd.read_csv(p3)
            if not df.empty:
                latest = df.dropna(how="all").iloc[-1]
                for col in df.columns:
                    if col != "date" and pd.notna(latest.get(col)):
                        result["corr_latest"][col] = round(float(latest[col]), 4)
        except Exception:
            pass
    # Joint beta today
    p4 = ROOT / "joint_beta_today.csv"
    if p4.exists():
        try:
            df = pd.read_csv(p4)
            if not df.empty:
                r = df.iloc[0]
                result["joint_beta"] = {
                    "joint":  round(float(r.get("joint_beta", 0) or 0), 3),
                    "v9":     round(float(r.get("v9_beta",    0) or 0), 3),
                    "v11":    round(float(r.get("v11_beta",   0) or 0), 3),
                    "scale":  round(float(r.get("v9_scale",   1) or 1), 3),
                    "status": str(r.get("status", "OK")),
                }
        except Exception:
            pass

    # Earnings gate actions today
    eg_path = ROOT / "earnings_gate_today.csv"
    if eg_path.exists():
        try:
            eg = pd.read_csv(eg_path)
            removed   = eg[eg["action"] == "REMOVED"]["ticker"].tolist()
            penalized = eg[eg["action"].str.startswith("PENALIZED")]["ticker"].tolist()
            result["earnings_gate"] = {
                "removed":          removed,
                "penalized":        penalized,
                "n_removed":        len(removed),
                "n_penalized":      len(penalized),
            }
        except Exception:
            pass

    return result


def load_barra_risk() -> dict:
    """Load Barra factor risk model outputs (step88/89/90)."""
    result: dict = {
        "decomp": {},        # portfolio_risk_decomp.csv latest row
        "factor_exposures": [],  # top-factor exposures for current portfolio
        "neutralization":   {},  # factor_neutralization_today.csv
        "xbrl_top5":        [],  # top 5 stocks by fundamental score (step89)
        "factor_cov_diag":  [],  # diagonal of factor cov matrix (factor vols)
    }

    # Portfolio risk decomposition
    p = ROOT / "portfolio_risk_decomp.csv"
    if p.exists():
        try:
            df = pd.read_csv(p)
            if not df.empty:
                row = df.iloc[-1].to_dict()
                result["decomp"] = {k: row[k] for k in
                    ("total_annual_vol","factor_vol","specific_vol",
                     "factor_share","specific_share","n_positions")
                    if k in row}
        except Exception:
            pass

    # Factor covariance diagonal → per-factor annual vol
    fc_path = ROOT / "factor_cov.csv"
    if fc_path.exists():
        try:
            fc = pd.read_csv(fc_path, index_col=0)
            rows = []
            for f in fc.index:
                if "sector_" not in f:   # style factors only
                    rows.append({"factor": f,
                                 "annual_vol": round(float(fc.loc[f, f])**0.5, 4)})
            result["factor_cov_diag"] = sorted(rows,
                key=lambda x: x["annual_vol"], reverse=True)[:8]
        except Exception:
            pass

    # Factor neutralization report
    fn_path = ROOT / "factor_neutralization_today.csv"
    if fn_path.exists():
        try:
            df = pd.read_csv(fn_path)
            if not df.empty:
                result["neutralization"] = df.iloc[-1].to_dict()
        except Exception:
            pass

    # Regime-conditional weights
    rw_path = ROOT / "regime_weights_today.csv"
    if rw_path.exists():
        try:
            df = pd.read_csv(rw_path)
            if not df.empty:
                result["regime"]     = str(df.iloc[0].get("regime", "—"))
                result["vix_bucket"] = str(df.iloc[0].get("vix_bucket", "—"))
                # Top 4 movers (largest |delta|)
                df["abs_delta"] = pd.to_numeric(df.get("delta", 0), errors="coerce").abs()
                top_movers = df.nlargest(4, "abs_delta")[
                    ["signal", "base_weight", "regime_weight", "delta"]
                ].to_dict("records")
                result["regime_movers"] = top_movers
        except Exception:
            pass

    # MVO optimizer result
    opt_path = ROOT / "portfolio_weights_today.csv"
    if opt_path.exists():
        try:
            df = pd.read_csv(opt_path)
            if "ticker" in df.columns and "weight" in df.columns:
                result["opt_longs"]  = df[df["weight"] >  0.005].nlargest(8, "weight").to_dict("records")
                result["opt_shorts"] = df[df["weight"] < -0.005].nsmallest(8, "weight").to_dict("records")
        except Exception:
            pass

    # Optimizer report metadata
    opt_report = ROOT / "portfolio_optimizer_report.md"
    if opt_report.exists():
        try:
            txt = opt_report.read_text()
            for line in txt.splitlines():
                for key in ("Status", "Ex-ante Portfolio Vol", "Ex-ante Sharpe",
                            "Net Exposure", "Turnover"):
                    if f"| {key} |" in line:
                        val = line.split("|")[2].strip()
                        result[f"opt_{key.lower().replace(' ','_').replace('-','')}"] = val
        except Exception:
            pass

    # XBRL top-5 fundamental stocks
    xb_path = ROOT / "xbrl_fundamentals.csv"
    if xb_path.exists():
        try:
            df = pd.read_csv(xb_path)
            sig_col = "sig_fundamental"
            if sig_col in df.columns and "ticker" in df.columns:
                top = df.nlargest(5, sig_col)[
                    ["ticker", "earnings_yield", "roe", "gross_margin",
                     "revenue_growth_yoy", sig_col]
                ].fillna(float("nan"))
                result["xbrl_top5"] = top.to_dict("records")
        except Exception:
            pass

    return result


# ═══════════════════════════════════════════════════════════════════════════
# FT dark-panel design system — shared by all rewritten "core" panels so they
# match the new _fred_macro_panel / _pead_panel / _pnl_contrib_panel look:
# warm near-black card, salmon accent, serif tabular figures, uppercase eyebrow.
# ═══════════════════════════════════════════════════════════════════════════
FT = {
    "card": "#16140f", "inner": "#100e0a", "border": "#33301f", "border2": "#26241a",
    "ink": "#f0e9da", "mute": "#8f866f", "sub": "#b0a68f", "accent": "#c8b487",
    "pos": "#8faa9a", "neg": "#c68b83", "warn": "#cdbd8f", "faint": "#79715f",
    "serif": "'Baskerville','Hoefler Text','Iowan Old Style','Georgia',serif",
}


def _safe_panel(fn, *args, **kw) -> str:
    """Call a panel builder; on any error return a small dark placeholder instead
    of letting one panel crash the whole page f-string. Used for rewritten panels."""
    try:
        return fn(*args, **kw)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return (f'<div style="margin-bottom:26px;background:#16140f;border:1px solid #3a3128;'
                f'border-radius:8px;padding:16px 18px;color:#8a7f70;font-size:12px">'
                f'⚠ This panel failed to render (skipped; the rest of the page is unaffected): {type(e).__name__}</div>')


def _ft_open(eyebrow: str, meta: str = "", mb: int = 26) -> str:
    """Open an FT *editorial* dark card: small salmon kicker + big serif headline
    + hairline rule (the Financial Times masthead feel). Titles passed as
    "CATEGORY · Headline" split into kicker + serif headline."""
    if " · " in eyebrow:
        kicker, headline = eyebrow.split(" · ", 1)
    else:
        kicker, headline = "", eyebrow
    kicker_html = (f'<div style="font-size:10px;color:{FT["accent"]};text-transform:uppercase;'
                   f'letter-spacing:.18em;font-weight:400;margin-bottom:6px">{kicker}</div>' if kicker else "")
    return (f'<div style="margin-bottom:{mb}px;background:{FT["card"]};border:1px solid {FT["border"]};'
            f'border-radius:8px;padding:20px 24px;box-shadow:0 1px 3px rgba(0,0,0,.4)">'
            f'<div style="display:flex;justify-content:space-between;align-items:flex-end;flex-wrap:wrap;gap:12px;margin-bottom:12px">'
            f'<div>{kicker_html}<h3 style="font-family:{FT["serif"]};font-size:23px;font-weight:500;'
            f'color:{FT["ink"]};margin:0;line-height:1.12;letter-spacing:-.015em">{headline}</h3></div>'
            f'<span style="font-size:11px;color:{FT["mute"]};white-space:nowrap;padding-bottom:5px">{meta}</span></div>'
            f'<div style="height:2px;width:40px;background:{FT["accent"]};border-radius:1px;margin-bottom:16px"></div>')


def _ft_close(foot: str = "") -> str:
    f = f'<p style="color:{FT["faint"]};font-size:10.5px;margin-top:10px">{foot}</p>' if foot else ""
    return f + '</div>'


def _ft_stat(label: str, value, sub: str = "", color: str = None) -> str:
    """A single inner stat cell — serif tabular figure on the inner surface."""
    color = color or FT["ink"]
    return (f'<div style="padding:10px 14px;border:1px solid {FT["border2"]};border-radius:6px;background:{FT["inner"]}">'
            f'<div style="font-size:9.5px;color:{FT["mute"]};text-transform:uppercase;letter-spacing:.08em;line-height:1.25;min-height:22px">{label}</div>'
            f'<div style="font-size:20px;font-weight:400;color:{color};font-family:{FT["serif"]};font-variant-numeric:tabular-nums;margin-top:3px">{value}</div>'
            + (f'<div style="font-size:10px;color:{FT["sub"]};margin-top:2px">{sub}</div>' if sub else "")
            + '</div>')


def _ft_statgrid(cells: list, minw: int = 150) -> str:
    return (f'<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax({minw}px,1fr));gap:12px;margin-bottom:14px">'
            + "".join(cells) + '</div>')


def _ft_subhead(text: str) -> str:
    return (f'<div style="font-size:10px;color:{FT["accent"]};text-transform:uppercase;letter-spacing:.12em;'
            f'margin:6px 0 8px">{text}</div>')


def _ft_table(headers: list, rows_html: str, empty: str = "No data", minw: int = 0) -> str:
    """A dark FT table. `headers`: list of (label, align). rows_html: pre-built <tr>s."""
    ths = "".join(
        f'<th style="text-align:{a};padding:7px 12px;font-size:9.5px;text-transform:uppercase;'
        f'letter-spacing:.08em;color:{FT["mute"]};border-bottom:1px solid #453a2c">{h}</th>'
        for h, a in headers)
    body = rows_html or (f'<tr><td colspan="{len(headers)}" style="padding:10px 12px;color:{FT["faint"]}">{empty}</td></tr>')
    mw = f"min-width:{minw}px;" if minw else ""
    return (f'<div style="overflow-x:auto"><table style="width:100%;border-collapse:collapse;{mw}font-size:12.5px">'
            f'<thead><tr>{ths}</tr></thead><tbody>{body}</tbody></table></div>')


def _ft_td(val, align: str = "left", color: str = None, bold: bool = False) -> str:
    c = f"color:{color};" if color else ""
    w = "font-weight:400;" if bold else ""
    num = "font-variant-numeric:tabular-nums;" if align == "right" else ""
    return f'<td style="padding:6px 12px;text-align:{align};{c}{w}{num}border-bottom:1px solid #2a231b">{val}</td>'


def _build_barra_risk_panel(br: dict) -> str:
    """Render the Barra factor risk decomposition panel for the Risk tab."""
    if not br or not br.get("decomp"):
        return "<p style='color:#999;font-style:italic'>Factor risk model not yet run (execute step88 first).</p>"

    d0 = br["decomp"]  # keep a stable ref — the regime loop below reuses `d` as a float

    def _pct(v):
        try: return f"{float(v):.1%}"
        except: return "—"

    def _val(v, dp=2):
        try: return f"{float(v):.{dp}f}"
        except: return "—"

    # Factor share bar
    fshare = float(d0.get("factor_share", 0))
    sshare = float(d0.get("specific_share", 0))
    fshare_pct = round(fshare * 100)
    sshare_pct = 100 - fshare_pct

    share_color = FT["neg"] if fshare > 0.65 else FT["pos"] if fshare < 0.50 else FT["warn"]
    share_note  = ("⚠ High factor share — book tilted toward systematic risk"
                   if fshare > 0.65 else
                   "✓ Healthy split — sufficient stock-specific alpha"
                   if fshare < 0.55 else
                   "~ Borderline — consider factor-neutralizing large positions")

    # Factor volatility table
    fv_rows = ""
    for row in br.get("factor_cov_diag", []):
        fv_rows += ("<tr>" + _ft_td(row['factor'])
                    + _ft_td(f"{row['annual_vol']:.2%}", "right") + "</tr>")

    # Factor neutralization table
    fn = br.get("neutralization", {})
    fn_rows = ""
    for f in ("market_beta", "size", "momentum"):
        pre  = fn.get(f, "—")
        post = fn.get(f"post_{f}", "—")
        try:
            pre_f  = float(pre);  pre_s  = f"{pre_f:+.3f}"
            post_f = float(post); post_s = f"{post_f:+.3f}"
            arrow  = "✓" if abs(post_f) < abs(pre_f) else "~"
            color  = FT["pos"] if abs(post_f) <= 0.25 else FT["neg"]
        except:
            pre_s = post_s = "—"; arrow = ""; color = FT["faint"]
        fn_rows += ("<tr>" + _ft_td(f) + _ft_td(pre_s, "right")
                    + _ft_td(post_s, "right", color) + _ft_td(arrow, "right") + "</tr>")
    n_swaps = fn.get("n_swaps", "—")

    # Regime conditioning card
    regime     = br.get("regime", "—")
    vix_bucket = br.get("vix_bucket", "—")
    reg_label  = f"{regime} / VIX {vix_bucket}"
    reg_color  = (FT["pos"] if regime == "BULL" and vix_bucket == "LOW"
                  else FT["neg"] if regime == "BEAR" and vix_bucket == "HIGH"
                  else FT["warn"])
    regime_mover_rows = ""
    for r in br.get("regime_movers", []):
        dl = float(r.get("delta", 0))
        arr = "↑" if dl > 0 else "↓"
        col = FT["pos"] if dl > 0 else FT["neg"]
        regime_mover_rows += (
            "<tr>"
            + _ft_td(r.get('signal', ''))
            + _ft_td(f"{float(r.get('base_weight',0)):.3f}", "right")
            + _ft_td(f"{arr} {float(r.get('regime_weight',0)):.3f}", "right", col, bold=True)
            + _ft_td(f"{dl:+.3f}", "right", col)
            + "</tr>"
        )

    # MVO optimizer weights table
    opt_long_rows = opt_short_rows = ""
    for r in br.get("opt_longs", []):
        opt_long_rows  += ("<tr>" + _ft_td(r.get('ticker', ''), "left", FT["ink"], bold=True)
                           + _ft_td(f"{float(r.get('weight',0)):.2%}", "right", FT["pos"])
                           + _ft_td(r.get('sector', ''), "left", FT["sub"]) + "</tr>")
    for r in br.get("opt_shorts", []):
        opt_short_rows += ("<tr>" + _ft_td(r.get('ticker', ''), "left", FT["ink"], bold=True)
                           + _ft_td(f"{abs(float(r.get('weight',0))):.2%}", "right", FT["neg"])
                           + _ft_td(r.get('sector', ''), "left", FT["sub"]) + "</tr>")
    opt_vol    = br.get("opt_exante_portfolio_vol", "—")
    opt_sharpe = br.get("opt_exante_sharpe", "—")
    opt_net    = br.get("opt_net_exposure", "—")
    opt_turn   = br.get("opt_turnover", "—")

    # XBRL top-5 table
    xbrl_rows = ""
    for row in br.get("xbrl_top5", []):
        def _p(v): return f"{float(v):.1%}" if v == v else "—"
        xbrl_rows += (
            "<tr>" + _ft_td(row.get('ticker', ''), "left", FT["ink"], bold=True)
            + _ft_td(_p(row.get('earnings_yield')), "right")
            + _ft_td(_p(row.get('roe')), "right")
            + _ft_td(_p(row.get('gross_margin')), "right")
            + _ft_td(_p(row.get('revenue_growth_yoy')), "right")
            + _ft_td(row.get('sig_fundamental', '—'), "right", FT["pos"], bold=True) + "</tr>"
        )

    # ── Section 1: Factor risk decomposition ──
    comp_bar = (
        f'<div style="margin:4px 0 2px"><div style="font-size:11px;color:{FT["mute"]};margin-bottom:6px">'
        f'风险构成 — 系统性因子 vs Idiosyncratic (stock-picking)</div>'
        f'<div style="background:{FT["inner"]};border:1px solid {FT["border2"]};border-radius:5px;overflow:hidden;height:26px;display:flex">'
        f'<div style="background:{share_color};opacity:.85;width:{fshare_pct}%;display:flex;align-items:center;justify-content:center;color:#17130f;font-size:11px;font-weight:400">Factor {fshare_pct}%</div>'
        f'<div style="background:{FT["pos"]};opacity:.85;width:{sshare_pct}%;display:flex;align-items:center;justify-content:center;color:#17130f;font-size:11px;font-weight:400">Specific {sshare_pct}%</div>'
        f'</div><div style="font-size:11px;margin-top:6px;color:{share_color}">{share_note}</div></div>')
    fv_tbl = _ft_table([("Factor", "left"), ("Annual Vol", "right")], fv_rows, "No data")
    fn_tbl = _ft_table([("Factor", "left"), ("Before", "right"), ("After", "right"), ("Status", "right")], fn_rows, "No data")
    sec1 = (_ft_open("Risk · Barra Factor Risk Decomposition",
                     f"Systematic vs stock-picking · healthy factor share 40–55%")
            + _ft_statgrid([
                _ft_stat("Total Annual Vol", _pct(d0.get('total_annual_vol')), "Annualized portfolio volatility"),
                _ft_stat("Factor Vol", _pct(d0.get('factor_vol')), "Systematic (beta/size/momentum)", FT["warn"]),
                _ft_stat("Specific Vol", _pct(d0.get('specific_vol')), "Idiosyncratic (stock-picking)", FT["pos"]),
            ], 160)
            + comp_bar
            + '<div style="display:grid;grid-template-columns:1fr 1fr;gap:20px;margin-top:16px">'
            + f'<div>{_ft_subhead("Style Factor Annual Vol")}{fv_tbl}</div>'
            + f'<div>{_ft_subhead(f"Factor Neutralization ({n_swaps} swaps)")}{fn_tbl}</div></div>'
            + _ft_close("A healthy book runs 40–55% factor risk + 45–60% specific risk. Too much factor share = returns driven by the market, not signal quality."))

    # ── Section 2: Regime-conditional IC weights ──
    rm_tbl = _ft_table([("Signal", "left"), ("Base wt", "right"), ("Regime wt", "right"), ("Delta", "right")],
                       regime_mover_rows, "No regime adjustment (NEUTRAL/LOW)")
    sec2 = (_ft_open("Signal Weights · Regime-Conditional IC Weights", "HMM state + VIX bucket")
            + _ft_statgrid([
                _ft_stat("Current Regime", reg_label, "HMM state + VIX bucket", reg_color),
                _ft_stat("Active Multipliers", str(len(br.get("regime_movers", []))), "Signals reweighted today"),
            ], 200)
            + _ft_subhead("Biggest weight shifts today") + rm_tbl
            + _ft_close("Different signals lead in different regimes: momentum/ML dominate in calm bull markets; quality/accruals/crowding take over in bear/crisis. Multipliers stack on IC²-optimal weights, capped at ±60%."))

    # ── Section 3: MVO optimized weights ──
    ol_tbl = _ft_table([("Ticker", "left"), ("Weight", "right"), ("Sector", "left")], opt_long_rows, "Run step90 to populate")
    os_tbl = _ft_table([("Ticker", "left"), ("Weight", "right"), ("Sector", "left")], opt_short_rows, "Run step90 to populate")
    sec3 = (_ft_open("Optimization · MVO Mean-Variance Optimal Weights", "scipy SLSQP · Barra covariance")
            + _ft_statgrid([
                _ft_stat("Ex-ante Vol", str(opt_vol)),
                _ft_stat("Ex-ante Sharpe", str(opt_sharpe), "", FT["accent"]),
                _ft_stat("Net Exposure", str(opt_net)),
                _ft_stat("Turnover", str(opt_turn)),
            ], 130)
            + '<div style="display:grid;grid-template-columns:1fr 1fr;gap:20px">'
            + f'<div>{_ft_subhead("Longs")}{ol_tbl}</div><div>{_ft_subhead("Shorts")}{os_tbl}</div></div>'
            + _ft_close("权重最大化 IC 加权 alpha 减风险惩罚(λ=2.0×组合方差)。约束:单票 ±10%、行业 30% 上限、beta 中性 ±0.20、Turnover ≤60%。"))

    # ── Section 4: XBRL fundamentals ──
    xbrl_tbl = _ft_table([("Stock", "left"), ("E/P Yield", "right"), ("ROE", "right"),
                          ("Gross Margin", "right"), ("Rev Growth", "right"), ("Score", "right")],
                         xbrl_rows, "Run step89 to populate")
    sec4 = (_ft_open("Fundamentals · XBRL Filing Signal (top 5)", "Free SEC EDGAR 10-K/10-Q")
            + xbrl_tbl
            + _ft_close("Earnings yield / ROE / gross margin / revenue growth taken straight from SEC XBRL filings — replacing the Bloomberg/FactSet fundamentals institutions pay for."))

    return sec1 + sec2 + sec3 + sec4


def load_factor_attribution() -> dict:
    """Fama-French 5-factor decomposition + signal P&L attribution."""
    result = {"ff5": [], "signals": [], "summary": {}}
    p = ROOT / "factor_attribution.csv"
    if p.exists():
        df = pd.read_csv(p)
        WIN_LABEL = {"Full available": "Since 2018 (full history)",
                     "3 years": "Past 3 years",
                     "1 year": "Past 1 year"}
        for _, row in df.iterrows():
            win = str(row.get("window", ""))
            alpha = float(row.get("alpha_ann", 0) or 0)
            ir    = float(row.get("info_ratio", 0) or 0)
            r2    = float(row.get("r_squared", 0) or 0)
            result["ff5"].append({
                "window": WIN_LABEL.get(win, win),
                "alpha_ann": alpha,
                "alpha_str": f"{alpha*100:+.1f}%",
                "info_ratio": ir,
                "r_squared": r2,
                "beta_market": float(row.get("beta_Mkt_RF", 0) or 0),
                "beta_value":  float(row.get("beta_HML", 0) or 0),
                "beta_quality":float(row.get("beta_RMW", 0) or 0),
                "beta_size":   float(row.get("beta_SMB", 0) or 0),
                "beta_invest": float(row.get("beta_CMA", 0) or 0),
            })
    p2 = ROOT / "return_attribution_by_signal.csv"
    if p2.exists():
        df2 = pd.read_csv(p2)
        SIGNAL_NAMES = {
            "ml_ensemble": "ML Ensemble model",
            "regime_ml":   "Market regime filter",
            "squeeze":     "Volatility squeeze",
            "insider":     "Insider buying signal",
            "quality":     "Quality / profitability",
            "revision":    "Analyst revision",
            "sentiment":   "News sentiment",
            "surprise":    "Earnings surprise",
        }
        total = df2["attributed_pnl_pct"].abs().sum()
        for _, row in df2.iterrows():
            sig = str(row.get("signal", ""))
            pnl = float(row.get("attributed_pnl_pct", 0) or 0)
            share = float(row.get("signal_contribution_share", 0) or 0)
            result["signals"].append({
                "signal": SIGNAL_NAMES.get(sig, sig),
                "pnl": pnl,
                "pnl_str": f"{pnl:+.2f}%",
                "share": share,
                "share_pct": round(share * 100, 1),
                "bar_w": min(100, round(abs(pnl) / max(total, 0.01) * 100, 0)),
            })
    p3 = ROOT / "backtest_summary.csv"
    if p3.exists():
        df3 = pd.read_csv(p3)
        for _, row in df3.iterrows():
            key = str(row.get("metric", "")).strip()
            val = str(row.get("value", "")).strip()
            result["summary"][key] = val
    return result


def load_monthly_pnl() -> dict:
    """Monthly P&L decomposition — last 18 months."""
    result = {"labels": [], "net": [], "long_c": [], "short_c": [],
              "alpha_c": [], "mkt_c": [], "hit_rate": [],
              "avg_alpha": 0.0, "best_month": 0.0, "worst_month": 0.0,
              "long_win_months": 0, "total_months": 0}
    p = ROOT / "pnl_monthly_summary.csv"
    if not p.exists():
        return result
    df = pd.read_csv(p).dropna(subset=["net_ret"]).tail(18)
    result["labels"]   = df["month"].tolist()
    result["net"]      = [round(float(v)*100, 2) for v in df["net_ret"].fillna(0)]
    result["long_c"]   = [round(float(v)*100, 2) for v in df["long_contrib"].fillna(0)]
    result["short_c"]  = [round(float(v)*100, 2) for v in df["short_contrib"].fillna(0)]
    result["alpha_c"]  = [round(float(v)*100, 2) for v in df["alpha_contrib"].fillna(0)]
    result["mkt_c"]    = [round(float(v)*100, 2) for v in df["mkt_contrib"].fillna(0)]
    result["hit_rate"] = [round(float(v)*100, 1) for v in df["hit_rate"].fillna(0)]
    nets = df["net_ret"].dropna()
    result["avg_alpha"] = round(float(df["alpha_contrib"].mean()) * 100, 2) if "alpha_contrib" in df else 0
    result["best_month"]  = round(float(nets.max()) * 100, 2)
    result["worst_month"] = round(float(nets.min()) * 100, 2)
    result["long_win_months"] = int((df["net_ret"] > 0).sum())
    result["total_months"]    = len(df)
    return result


# ── Macro signal loader ───────────────────────────────────────────────────────

def load_macro_signal_snapshot() -> dict:
    """Load live macro signal values from macro_signals.json for display in macro section."""
    p = ROOT / "macro_signals.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


# ── v25.1 loaders ────────────────────────────────────────────────────────────

def load_v251_backtest() -> dict:
    """Load v25.1 OOS backtest stats from backtest_v25_gap3.csv."""
    import warnings; warnings.filterwarnings("ignore")
    p = ROOT / "backtest_v25_gap3.csv"
    if not p.exists():
        return {}
    try:
        import yfinance as yf
        df = pd.read_csv(p, parse_dates=["date"])
        rets = df["ret"].values
        RF_M = 0.026 / 12
        n = len(rets)
        cum = np.cumprod(1 + rets)
        ar = cum[-1] ** (12 / n) - 1
        ex = rets - RF_M
        sr = ex.mean() / ex.std(ddof=1) * np.sqrt(12)
        dd = cum / np.maximum.accumulate(cum) - 1
        mdd = dd.min()
        calmar = ar / abs(mdd) if mdd != 0 else 0
        dates = pd.DatetimeIndex(df["date"])

        # Fetch QQQ / SPY for annual comparison
        raw = yf.download(["QQQ", "SPY"], period="8y", progress=False, auto_adjust=True)["Close"]
        raw.index = pd.to_datetime(raw.index)

        def _monthly(series, dates):
            prev = None
            out = []
            for d in dates:
                sub = series.loc[:d]
                cur = float(sub.iloc[-1]) if len(sub) else None
                if prev is None:
                    p0 = series.loc[:dates[0] - pd.Timedelta(days=20)]
                    prev = float(p0.iloc[-1]) if len(p0) else cur
                out.append((cur / prev - 1) if (cur and prev) else 0.0)
                prev = cur
            return np.array(out)

        qqq_ret = _monthly(raw["QQQ"].dropna(), dates)
        spy_ret = _monthly(raw["SPY"].dropna(), dates)
        ann_rows, beat_years = [], 0
        for yr in range(2019, 2027):
            mask = dates.year == yr
            if not mask.any():
                continue
            s_ann = float(np.prod(1 + rets[mask]) - 1)
            q_ann = float(np.prod(1 + qqq_ret[mask]) - 1)
            p_ann = float(np.prod(1 + spy_ret[mask]) - 1)
            if s_ann > q_ann:
                beat_years += 1
            ann_rows.append({"year": yr, "strat": s_ann, "qqq": q_ann, "spy": p_ann, "beat": s_ann > q_ann})

        return {
            "ar": ar, "sharpe": sr, "mdd": mdd, "calmar": calmar,
            "cum_total": float(cum[-1]) - 1, "n_months": n, "beat_years": beat_years,
            "annual_rows": ann_rows,
            "cum_v251":  [round(float(v * 100), 1) for v in np.concatenate([[100.0], cum * 100])],
            "cum_qqq":   [round(float(v * 100), 1) for v in np.concatenate([[100.0], np.cumprod(1 + qqq_ret) * 100])],
            "cum_spy":   [round(float(v * 100), 1) for v in np.concatenate([[100.0], np.cumprod(1 + spy_ret) * 100])],
            "labels":    ["Jan 2019"] + df["date"].dt.strftime("%b %Y").tolist(),
        }
    except Exception as e:
        return {"error": str(e)}


def load_v251_regime() -> dict:
    """Fetch live VIX / QQQ / SPY and compute current v25.1 regime and TQQQ weight.
    Also reads HMM regime to apply a haircut when HMM=BEAR."""
    import warnings; warnings.filterwarnings("ignore")
    try:
        import yfinance as yf
        raw = yf.download(["QQQ", "^VIX", "SPY"], period="14mo", progress=False, auto_adjust=True)["Close"]
        raw.index = pd.to_datetime(raw.index)
        vix = float(raw["^VIX"].dropna().iloc[-1])
        qqq = float(raw["QQQ"].dropna().iloc[-1])
        spy = float(raw["SPY"].dropna().iloc[-1])
        qqq_s = raw["QQQ"].dropna()
        spy_s = raw["SPY"].dropna()
        qqq_ma200 = float(qqq_s.tail(200).mean()) if len(qqq_s) >= 200 else qqq
        spy_ma200 = float(spy_s.tail(200).mean()) if len(spy_s) >= 200 else spy
        qqq_3m    = float(qqq_s.iloc[-65]) if len(qqq_s) >= 65 else qqq
        gate_ma   = qqq > qqq_ma200
        gate_spy  = spy > spy_ma200
        gate_mom  = qqq > qqq_3m
        vix_tier  = "LOW" if vix < 20 else ("MID" if vix < 25 else "HIGH")
        vix_color = "#6BCCA0" if vix < 20 else ("#c8b487" if vix < 25 else "#EF9090")
        tqqq_base = 0.50 if (vix < 20) else (0.25 if vix < 25 else 0.00)
        gates_ok  = gate_spy and gate_ma and gate_mom
        if not gates_ok:
            tqqq_base = 0.00

        # Read HMM regime — if BEAR, cut TQQQ allocation by half
        hmm_regime = "UNKNOWN"
        hmm_prob_bear = None
        try:
            p = ROOT / "hmm_regime_daily.csv"
            if p.exists():
                import csv
                with open(p) as f:
                    rows = list(csv.reader(f))
                if len(rows) > 1:
                    hdr  = rows[0]
                    last = rows[-1]
                    regime_col = hdr.index("regime") if "regime" in hdr else 2
                    prob_bear_col = hdr.index("prob_bear") if "prob_bear" in hdr else 4
                    hmm_regime = "BEAR" if int(last[regime_col]) == 1 else "BULL"
                    hmm_prob_bear = float(last[prob_bear_col]) if len(last) > prob_bear_col else None
        except Exception:
            pass

        hmm_is_bear = hmm_regime == "BEAR"
        tqqq_wt = tqqq_base * 0.5 if hmm_is_bear else tqqq_base
        regime_label = "BULL" if gates_ok else "BEAR"

        return {
            "vix": vix, "vix_tier": vix_tier, "vix_color": vix_color,
            "qqq": qqq, "qqq_ma200": qqq_ma200, "spy": spy, "spy_ma200": spy_ma200,
            "gate_ma": gate_ma, "gate_spy": gate_spy, "gate_mom": gate_mom,
            "gates_ok": gates_ok, "tqqq_wt": tqqq_wt,
            "regime": regime_label,
            "regime_color": "#6BCCA0" if gates_ok else "#EF9090",
            "qqq_3m_ret": (qqq / qqq_3m - 1) * 100,
            "hmm_regime": hmm_regime,
            "hmm_is_bear": hmm_is_bear,
            "hmm_prob_bear": hmm_prob_bear,
            "tqqq_base": tqqq_base,
            "as_of": datetime.now().strftime("%Y-%m-%d %H:%M"),
        }
    except Exception:
        return {"vix": 0, "vix_tier": "—", "vix_color": "#888", "qqq": 0, "qqq_ma200": 0,
                "spy": 0, "spy_ma200": 0, "gate_ma": True, "gate_spy": True, "gate_mom": True,
                "gates_ok": True, "tqqq_wt": 0.25, "regime": "—", "regime_color": "#888",
                "qqq_3m_ret": 0, "hmm_regime": "UNKNOWN", "hmm_is_bear": False,
                "hmm_prob_bear": None, "tqqq_base": 0.25, "as_of": "fetch failed"}


def load_short_scanner() -> pd.DataFrame:
    """Load short technical scanner results from step_short_scanner.py."""
    p = ROOT / "short_scanner.csv"
    if not p.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(p)
    except Exception:
        return pd.DataFrame()


def load_dcf_valuation() -> pd.DataFrame:
    """Load DCF valuation results from step_dcf_valuation.py output."""
    p = ROOT / "dcf_valuation.csv"
    if not p.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(p)
        return df
    except Exception:
        return pd.DataFrame()


def load_deep_analysis() -> dict:
    """Load institutional deep-analysis results from deep_analysis_v3.json."""
    p = ROOT / "deep_analysis_v3.json"
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def load_economic_calendar() -> dict:
    """Load upcoming economic events from step_economic_calendar.py output."""
    p = ROOT / "economic_calendar.json"
    if not p.exists():
        return {"events": [], "count": 0}
    try:
        return json.loads(p.read_text())
    except Exception:
        return {"events": [], "count": 0}


def load_earnings_ai() -> pd.DataFrame:
    """Load AI-generated earnings summaries from step_earnings_ai.py output."""
    p = ROOT / "earnings_ai_summaries.csv"
    if not p.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(p)
    except Exception:
        return pd.DataFrame()


def load_watchlist() -> dict:
    """Load user watchlist from watchlist.json."""
    p = ROOT / "watchlist.json"
    if not p.exists():
        return {"tickers": [], "notes": {}}
    try:
        return json.loads(p.read_text())
    except Exception:
        return {"tickers": [], "notes": {}}


def load_famous_holdings() -> dict:
    """Load famous investor 13F holdings from famous_holdings.json."""
    p = ROOT / "famous_holdings.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}


def load_congressional_trades() -> dict:
    """Load congressional trading data from congressional_trades.json."""
    p = ROOT / "congressional_trades.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}


# ── Alert humanizer ───────────────────────────────────────────────────────────

def _humanize_desk_alert(r: dict) -> tuple[str, str, str, str]:
    """Convert raw desk_monitor row → (ticker, title, detail, action) in plain English."""
    import re as _re

    ticker_raw = str(r.get("ticker", ""))
    ticker     = "Portfolio" if ticker_raw.lower() in ("nan", "", "none") else ticker_raw.upper()
    title_raw  = str(r.get("title",  ""))
    detail_raw = str(r.get("detail", ""))
    action_raw = str(r.get("action", ""))
    monitor    = str(r.get("monitor", "")).upper()
    m1v        = str(r.get("metric_1_value", ""))

    _GATE_LABEL = {
        "master":       "overall position size",
        "single":       "max per stock limit",
        "earnings_gap": "earnings report coming up",
        "kelly":        "position too large for the signal strength",
        "sector":       "sector already at limit",
    }
    _STATUS_WORD = {
        "SIZE_DOWN":   "too large — reduce",
        "REDUCE_ONLY": "must cut size",
        "REVIEW":      "needs review",
        "BLOCK_NEW":   "blocked — do not add",
        "CLEAR":       "OK",
    }

    def _parse_nums(text: str):
        cm = _re.search(r"current=([\d]+(?:\.[\d]+)?)", text)
        lm = _re.search(r"limit=([\d]+(?:\.[\d]+)?)",   text)
        um = _re.search(r"used=([\d]+(?:\.[\d]+)?)",    text)
        return (float(cm.group(1)) if cm else None,
                float(lm.group(1)) if lm else None,
                float(um.group(1)) if um else None)

    if monitor == "RISK_LIMIT_BREACH":
        curr, lim, used = _parse_nums(detail_raw)

        if ticker == "Portfolio":
            tl = title_raw.lower()
            if "volatility" in tl:
                title  = "Overall portfolio swings are larger than the safety limit — reduce sizes"
                detail = (f"Current daily swing rate: {curr:.1%} per year. Safety limit: {lim:.1%}. "
                          f"Currently at {used:.0%} of the allowed swing budget."
                          if curr and lim and used else detail_raw)
                action = "Reduce all position sizes proportionally until the daily swings come back within the limit"
            elif "tail-risk" in tl or "single-name" in tl:
                title  = "One position is taking too much tail risk"
                detail = (f"Single-name tail risk at {curr:.2f}×, limit is {lim:.1f}×."
                          if curr and lim else detail_raw)
                action = "Trim the largest position — it's carrying disproportionate crash risk"
            elif "crisis" in tl or "correlation" in tl:
                title  = "Portfolio — correlated crash risk is too high"
                detail = (f"Crisis scenario risk at {curr:.2f}×, limit {lim:.1f}× "
                          f"(using {used:.0%} of budget)."
                          if curr and lim and used else detail_raw)
                action = "Reduce positions that tend to fall together in a market panic"
            else:
                budget_name = _re.sub(r"^risk budget breach:\s*", "", title_raw, flags=_re.IGNORECASE)
                title  = f"Portfolio — {budget_name} exceeded"
                detail = (f"Current {curr:.2f}×, limit {lim:.1f}×." if curr and lim else detail_raw)
                action = "Review and reduce positions"
        else:
            # Per-ticker: parse gate breakdown
            gates: dict[str, str] = {}
            for part in detail_raw.split(";"):
                part = part.strip()
                if ":" in part:
                    k, v = part.split(":", 1)
                    gates[k.strip()] = v.strip()

            failed   = {k: v for k, v in gates.items() if v.upper() not in ("CLEAR", "OK", "PASS")}
            n_total  = len(gates) if gates else 5
            n_failed = len(failed)

            worst = max(
                failed.values(),
                key=lambda s: {"BLOCK_NEW": 3, "REDUCE_ONLY": 2, "SIZE_DOWN": 1, "REVIEW": 0}.get(s.upper(), 0),
                default="SIZE_DOWN",
            )
            worst_plain = "position must be cut" if "REDUCE" in worst.upper() else "position is too large"

            title = f"{ticker} — {worst_plain} ({n_failed} of {n_total} checks failed)"
            if failed:
                parts  = [f"{_GATE_LABEL.get(k, k)}" for k, v in failed.items()]
                detail = "Issues: " + "; ".join(parts) + "."
            else:
                detail = detail_raw

            rec_m = _re.search(r"([\d.]+)%", action_raw)
            if rec_m:
                rec = rec_m.group(1)
                try:
                    curr_w = float(m1v)
                    cut_pct = (1 - float(rec) / curr_w) * 100
                    action  = f"Research sizing cap: {rec}%  (current {curr_w:.1f}% → needs {cut_pct:.0f}% cut)"
                except Exception:
                    action = f"Research sizing cap: {rec}%"
            else:
                action = action_raw

    elif monitor == "NEWS_SHOCK":
        clean = _re.sub(r"\s*\|\s*keywords:.*$", "", detail_raw, flags=_re.IGNORECASE)
        title  = title_raw
        detail = clean if clean.strip() else detail_raw
        action = action_raw

    elif monitor == "SQUEEZE_WATCH":
        title  = title_raw
        detail = _re.sub(r"rank_squeeze=[\d.]+\.\s*", "", detail_raw).strip().rstrip(".")
        action = action_raw

    elif monitor == "OPTIONS_ALERT":
        title  = title_raw
        detail = _re.sub(r"rank_options=[\d.]+,?\s*", "", detail_raw).strip()
        action = action_raw

    elif monitor == "PRICE_BREAK":
        # "Latest close 245.22 is below the prior 20-day low 245.78."
        # → "Price fell to $245.22 — lower than it's been in the past 4 weeks (prior low: $245.78)."
        close_m = _re.search(r"Latest close\s+([\d.]+)", detail_raw, _re.IGNORECASE)
        low_m   = _re.search(r"prior 20-?day low\s+([\d.]+)", detail_raw, _re.IGNORECASE)
        high_m  = _re.search(r"prior 20-?day high\s+([\d.]+)", detail_raw, _re.IGNORECASE)

        def _fv(m): return float(m.group(1).rstrip("."))

        if close_m and low_m:
            close_v = _fv(close_m)
            low_v   = _fv(low_m)
            title   = f"{ticker} — price broke below its 4-week low"
            detail  = (f"Price fell to ${close_v:,.2f}, which is lower than any close "
                       f"in the past 4 weeks (previous 4-week low was ${low_v:,.2f}).")
            action  = "Don't add to this position. Review your stop-loss level before taking any action."
        elif close_m and high_m:
            close_v = _fv(close_m)
            high_v  = _fv(high_m)
            title   = f"{ticker} — price broke above its 4-week high"
            detail  = (f"Price climbed to ${close_v:,.2f}, which is higher than any close "
                       f"in the past 4 weeks (previous 4-week high was ${high_v:,.2f}).")
            action  = "Watch for follow-through. This is not an automatic buy — wait for confirmation."
        else:
            # Fallback: clean up "prior 20-day" phrasing
            detail_clean = _re.sub(r"prior 20-?day (low|high)", r"4-week \1", detail_raw, flags=_re.IGNORECASE)
            detail_clean = _re.sub(r"Latest close", "Latest price", detail_clean, flags=_re.IGNORECASE)
            title  = title_raw
            detail = detail_clean
            action = action_raw

    else:
        title, detail, action = title_raw, detail_raw, action_raw

    return ticker, title, detail, action


# ── build HTML ────────────────────────────────────────────────────────────────

def _clean_trigger(text: str) -> str:
    """Convert raw trigger_to_watch text to plain English."""
    import re as _re
    text = _re.sub(r"prior 20d low", "4-week low", text, flags=_re.IGNORECASE)
    text = _re.sub(r"prior 20d high", "4-week high", text, flags=_re.IGNORECASE)
    text = _re.sub(r"Watch close above", "Buy signal if price closes above", text, flags=_re.IGNORECASE)
    text = _re.sub(r"Watch close below", "Warning if price closes below", text, flags=_re.IGNORECASE)
    text = _re.sub(r"near ([\d,.]+)", r"at $\1", text)
    return text.strip()


def _build_signal_health_section(sh: dict) -> str:
    """Render the Signal Health tab — rewritten to the FT dark-panel design."""
    parts = []

    def _iccol(v, lo=0.0, hi=0.05):
        return FT["pos"] if v > hi else FT["neg"] if v < lo else FT["mute"]

    # ── IC Decay by horizon ──
    ic_decay = sh.get("ic_decay", [])
    if ic_decay:
        rows = ""
        for r in ic_decay:
            cells = _ft_td(r['signal'], "left", FT["ink"], bold=True)
            for h in [5, 10, 21, 42, 63, 126]:
                v = r.get(f"h{h}")
                cells += _ft_td("—" if v is None else f"{v:+.3f}", "right",
                                FT["mute"] if v is None else _iccol(v, -0.02, 0.05))
            rows += f"<tr>{cells}</tr>"
        parts.append(_ft_open("Signal Health · IC Decay by Horizon", "IC>0.05 = predictive · faster decay = edge fades faster")
                     + _ft_table([("Signal", "left"), ("5d", "right"), ("10d", "right"),
                                  ("21d", "right"), ("42d", "right"), ("63d", "right"), ("126d", "right")],
                                 rows, "No data", minw=560)
                     + _ft_close("Information Coefficient at each forward-return horizon. Green = IC>0.05 (meaningfully predictive); decay speed = how fast the edge disappears."))

    # ── Live OOS IC by signal ──
    cross_ic = sh.get("cross_ic", [])
    if cross_ic:
        _SIG_NAMES = {"ml_ensemble": "AI model ensemble", "fear_vix": "Market fear (VIX)",
                      "google_trends": "Google search interest", "sec_filing_lag": "SEC filing-lag signal",
                      "accruals": "Earnings quality (accruals)", "momentum": "Momentum", "squeeze": "Short-squeeze potential"}
        rows = ""
        for r in cross_ic:
            sig_raw = r['signal']
            sig_name = _SIG_NAMES.get(sig_raw, sig_raw.replace("_", " ").title())
            st = str(r.get("status", ""))
            if "ALERT" in st: sc, stx = FT["neg"], "Alert — degraded"
            elif "WARN" in st: sc, stx = FT["warn"], "Watch"
            elif "OK" in st: sc, stx = FT["pos"], "OK"
            else: sc, stx = FT["mute"], st
            def _c(v):
                return _ft_td("—" if v is None else f"{v:+.3f}", "right",
                              FT["mute"] if v is None else _iccol(v, 0.0, 0.05))
            name_td = (f'<td style="padding:6px 12px;border-bottom:1px solid #2a231b">'
                       f'<span style="font-weight:400;color:{FT["ink"]}">{sig_name}</span>'
                       f'<br><span style="font-size:10px;color:{FT["faint"]}">{sig_raw}</span></td>')
            rows += (f"<tr>{name_td}" + _c(r.get("ic_3m")) + _c(r.get("ic_6m"))
                     + _ft_td(stx, "right", sc) + "</tr>")
        parts.append(_ft_open("Signal Health · Live OOS IC (walk-forward)", "Measured on real unseen data, not backtested")
                     + _ft_table([("Signal", "left"), ("3m IC", "right"), ("6m IC", "right"), ("Status", "right")],
                                 rows, "No data", minw=460)
                     + _ft_close("Out-of-sample IC from walk-forward validation — measured on real unseen data, not backtest-fitted."))

    # ── Correlation & beta monitor ──
    corr = sh.get("corr_latest", {})
    jb = sh.get("joint_beta", {})

    def _corr_row(label, key21, key63=None):
        v21 = corr.get(key21); v63 = corr.get(key63) if key63 else None
        def _cv(v):
            return _ft_td("—" if v is None else f"{v:+.3f}", "right",
                          FT["mute"] if v is None else (FT["neg"] if abs(v) > 0.80 else FT["warn"] if abs(v) > 0.65 else FT["pos"]))
        return f"<tr>{_ft_td(label)}{_cv(v21)}{_cv(v63)}</tr>"

    rows_corr = (_corr_row("Correlation vs SPY", "corr_vs_SPY_21d", "corr_vs_SPY_63d")
                 + _corr_row("Correlation vs QQQ", "corr_vs_QQQ_21d", "corr_vs_QQQ_63d")
                 + _corr_row("Beta vs SPY", "beta_vs_SPY_21d", "beta_vs_SPY_63d")
                 + _corr_row("Beta vs QQQ", "beta_vs_QQQ_21d", "beta_vs_QQQ_63d"))
    te = corr.get("tracking_error_vs_spy_21d")
    te_str = f"{te:.1%}" if te else "—"
    jb_val = jb.get("joint", None); jb_v9 = jb.get("v9", None)
    jb_v11 = jb.get("v11", None); jb_scale = jb.get("scale", 1)
    jb_col = FT["neg"] if (jb_val or 0) > 1.1 else FT["warn"] if (jb_val or 0) > 0.9 else FT["pos"]
    _has_corr = any(k for k in corr if "corr_vs" in k or "beta_vs" in k)
    note = ("" if _has_corr else
            f'<p style="color:{FT["warn"]};font-size:11.5px;background:{FT["inner"]};border:1px solid {FT["border2"]};padding:8px 12px;border-radius:5px;margin-bottom:12px">'
            "Rolling correlation not yet computable — the paper book has fewer than 21 days of returns. It will populate automatically once enough history accumulates.</p>")
    jb_section = ""
    if jb_val is not None:
        jb_section = ('<div style="margin-top:16px">' + _ft_subhead("Portfolio Beta · Live Estimate")
                      + _ft_statgrid([
                          _ft_stat("Joint Beta", f"{jb_val:.2f}", "", jb_col),
                          _ft_stat("v9 Beta", f"{(jb_v9 or 0):.2f}"),
                          _ft_stat("v11 Beta", f"{(jb_v11 or 0):.2f}"),
                          _ft_stat("v9 Scale", f"{jb_scale:.0%}"),
                      ], 130)
                      + f'<p style="font-size:11px;color:{FT["mute"]}">Joint beta &gt;1.1 → v9 positions auto-scaled down; &lt;0.9 → headroom to add risk.</p></div>')
    parts.append(_ft_open("Signal Health · Correlation & Beta Monitor", f"SPY tracking error (21d ann.) {te_str} · Caution >65% / Alert >80%")
                 + note
                 + _ft_table([("Metric", "left"), ("21-day", "right"), ("63-day", "right")], rows_corr, "No data")
                 + jb_section
                 + _ft_close("Rolling correlation/beta vs benchmarks. Green = low correlation (well diversified); red = high correlation (strategy moving with the index)."))

    # ── Earnings gate ──
    eg = sh.get("earnings_gate", {})
    n_removed = eg.get("n_removed", 0); n_penalized = eg.get("n_penalized", 0)
    if n_removed > 0 or n_penalized > 0:
        removed_str = ", ".join(eg.get("removed", [])[:8]) or "—"
        penalized_str = ", ".join(eg.get("penalized", [])[:8]) or "—"
        box = (lambda title, col, names: (
            f'<div style="background:{FT["inner"]};border:1px solid {col};border-radius:6px;padding:12px">'
            f'<div style="font-size:10px;text-transform:uppercase;letter-spacing:.1em;color:{col};margin-bottom:6px">{title}</div>'
            f'<div style="font-size:13px;font-weight:400;color:{FT["ink"]}">{names}</div></div>'))
        parts.append(_ft_open("Signal Health · Earnings Calendar Risk Gate", f"Today {n_removed + n_penalized} names filtered / downweighted")
                     + '<div style="display:grid;grid-template-columns:1fr 1fr;gap:16px">'
                     + box(f"Removed from picks ({n_removed})", FT["neg"], removed_str)
                     + box(f"Score penalized ({n_penalized})", FT["warn"], penalized_str) + '</div>'
                     + _ft_close("Names with earnings ≤3 days out are dropped from picks; those 4–21 days out have composite scores reduced — avoiding earnings-gap risk."))

    if not parts:
        parts.append(_ft_open("Signal Health") + f'<p style="color:{FT["mute"]};text-align:center;padding:30px">Run the daily pipeline to populate signal-health data.</p>' + _ft_close())

    return "\n".join(parts)


def _build_short_scanner_section(df: "pd.DataFrame") -> str:
    """Render Short Technical Scanner tab from short_scanner.csv."""
    if df is None or df.empty:
        return '<p style="color:#AAA;font-size:13px">Short scanner not yet run. It will populate after the next daily pipeline run (Step 373).</p>'

    today = str(pd.Timestamp.today().date())
    today_df = df[df["as_of"] == today] if "as_of" in df.columns else df
    if today_df.empty:
        today_df = df
    today_df = today_df.sort_values("score", ascending=False)

    as_of_str = today_df["as_of"].iloc[0] if "as_of" in today_df.columns else "—"
    n = len(today_df)

    now_df   = today_df[today_df["urgency"].str.startswith("NOW")]
    week_df  = today_df[today_df["urgency"].str.startswith("THIS WEEK")]
    watch_df = today_df[today_df["urgency"].str.startswith("WATCH")]

    def _urgency_badge(urgency: str) -> str:
        if urgency.startswith("NOW"):
            return f'<span style="background:#FEE2E2;color:#B83232;font-size:10px;font-weight:400;padding:2px 8px;border-radius:3px">NOW</span>'
        elif urgency.startswith("THIS WEEK"):
            return f'<span style="background:#FDF8EE;color:#c8b487;font-size:10px;font-weight:400;padding:2px 8px;border-radius:3px">THIS WEEK</span>'
        else:
            return f'<span style="background:#F3F4F6;color:#777;font-size:10px;font-weight:400;padding:2px 8px;border-radius:3px">WATCH</span>'

    def _score_bar(score: float) -> str:
        color = "#B83232" if score >= 60 else ("#c8b487" if score >= 45 else "#888")
        return (f'<div style="display:flex;align-items:center;gap:6px">'
                f'<div style="width:48px;height:4px;background:#241f18;border-radius:2px">'
                f'<div style="width:{score:.0f}%;height:4px;background:{color};border-radius:2px"></div></div>'
                f'<span style="font-size:11px;font-weight:400;color:{color}">{score:.0f}</span>'
                f'</div>')

    def _rsi_badge(rsi: float) -> str:
        if rsi >= 75:
            return f'<span style="color:#B83232;font-weight:400;font-variant-numeric:tabular-nums">{rsi:.0f}</span>'
        elif rsi >= 65:
            return f'<span style="color:#c8b487;font-weight:400;font-variant-numeric:tabular-nums">{rsi:.0f}</span>'
        else:
            return f'<span style="color:#888;font-variant-numeric:tabular-nums">{rsi:.0f}</span>'

    def _rr_badge(rr) -> str:
        if rr is None or (isinstance(rr, float) and pd.isna(rr)):
            return '<span style="color:#CCC">—</span>'
        rr = float(rr)
        if rr >= 3.0:
            return f'<span style="color:#1B6F4A;font-weight:400">{rr:.1f}x</span>'
        elif rr >= 2.0:
            return f'<span style="color:#c8b487;font-weight:400">{rr:.1f}x</span>'
        else:
            return f'<span style="color:#AAA">{rr:.1f}x</span>'

    def _row(r, rank: int) -> str:
        rr1 = r.get("rr_1"); rr2 = r.get("rr_2")
        signals_html = r.get("signals", "")
        ext = []
        if r.get("ext_20ma", 0) > 5:
            ext.append(f'+{r["ext_20ma"]:.0f}% vs 20MA')
        elif r.get("ext_20ma", 0) < -5:
            ext.append(f'{r["ext_20ma"]:.0f}% vs 20MA')
        if r.get("macd_cross"):
            ext.append("MACD✗")
        if r.get("vol_diverge"):
            ext.append("Vol÷")
        sub = " · ".join(ext[:2]) if ext else ""
        return (
            f'<tr>'
            f'<td style="font-size:11px;color:#999;text-align:center">{rank}</td>'
            f'<td><span style="font-weight:400;color:#c8b487;font-size:13px">{r["ticker"]}</span></td>'
            f'<td>{_score_bar(r["score"])}</td>'
            f'<td style="font-variant-numeric:tabular-nums;font-weight:400">${r["price"]:.2f}</td>'
            f'<td style="font-variant-numeric:tabular-nums">'
            f'<span style="color:#333;font-weight:400">${r["entry_low"]:.2f}</span>'
            f'<span style="color:#AAA;font-size:10px"> – </span>'
            f'<span style="color:#333;font-weight:400">${r["entry_high"]:.2f}</span>'
            f'</td>'
            f'<td style="color:#B83232;font-variant-numeric:tabular-nums;font-weight:400">${r["stop_loss"]:.2f}</td>'
            f'<td style="color:#1B6F4A;font-variant-numeric:tabular-nums">${r["target_1"]:.2f}</td>'
            f'<td style="color:#1B6F4A;font-variant-numeric:tabular-nums">${r["target_2"]:.2f}</td>'
            f'<td>{_rr_badge(rr1)} / {_rr_badge(rr2)}</td>'
            f'<td>{_rsi_badge(r["rsi"])}</td>'
            f'<td style="font-size:10px;color:#666;max-width:180px">{signals_html[:60]}</td>'
            f'<td>{_urgency_badge(r["urgency"])}</td>'
            f'</tr>'
        )

    all_rows = "".join(_row(r, i + 1) for i, (_, r) in enumerate(today_df.iterrows()))

    # Summary stats
    avg_rsi  = today_df["rsi"].mean() if "rsi" in today_df.columns else 0
    avg_rr   = today_df["rr_1"].mean() if "rr_1" in today_df.columns else 0
    high_rr  = today_df[today_df["rr_1"].notna() & (today_df["rr_1"] >= 3)]

    return f"""
{_ft_open("Short Technical Scanner · today's candidates", f"as of {as_of_str}")}
{_ft_statgrid([
    _ft_stat("Candidates Today", str(n), f"as of {as_of_str}", FT["accent"]),
    _ft_stat("Act Now (1-2d)", str(len(now_df)), "RSI ≥ 75 or MACD cross", FT["neg"]),
    _ft_stat("Avg RSI", f"{avg_rsi:.0f}", "across all candidates", FT["warn"]),
    _ft_stat("R/R ≥ 3x", str(len(high_rr)), "favorable setups", FT["pos"]),
], 150)}
{_ft_close()}

<div class="tbl-wrap" style="margin-bottom:28px">
  <p class="tbl-title">All Short Candidates — sorted by score</p>
  <div style="overflow-x:auto">
  <table>
    <thead><tr>
      <th style="width:28px">#</th>
      <th>Ticker</th>
      <th>Score</th>
      <th class="r">Price</th>
      <th>Entry Range (next 1-5d)</th>
      <th class="r">Stop Loss</th>
      <th class="r">Target 1</th>
      <th class="r">Target 2</th>
      <th class="r">R/R T1/T2</th>
      <th class="r">RSI</th>
      <th>Signals</th>
      <th>Urgency</th>
    </tr></thead>
    <tbody>{all_rows}</tbody>
  </table>
  </div>
  <p class="tbl-note">Entry Range = where to open the short over next 1-5 days. Stop Loss = exit if price reaches here. Target 1 = 20MA reversion. Target 2 = 50MA. R/R = reward-to-risk ratio (≥ 2x favorable). Green R/R = ≥ 3x.</p>
</div>

<div class="method-card" style="border-top:3px solid #B83232">
  <p style="font-size:10px;letter-spacing:1.5px;text-transform:uppercase;color:#999;font-weight:400;margin:0 0 14px">How to Use This Scanner</p>
  <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:20px">
    <div>
      <p style="font-size:11px;color:#333;font-weight:400;margin:0 0 4px">Entry</p>
      <p style="font-size:11px;color:#666;line-height:1.6;margin:0">Short anywhere in the entry range. The ideal entry is the higher end of the range (on a bounce). If the stock is already below the entry range, the setup has partially played out — assess whether it's still valid.</p>
    </div>
    <div>
      <p style="font-size:11px;color:#333;font-weight:400;margin:0 0 4px">Stop Loss</p>
      <p style="font-size:11px;color:#666;line-height:1.6;margin:0">Close the short if price closes above the stop loss level. Set the stop as a hard order — not a mental note. The stop is based on the recent 10-day high + 0.5× ATR to allow for normal volatility.</p>
    </div>
    <div>
      <p style="font-size:11px;color:#333;font-weight:400;margin:0 0 4px">Score Breakdown</p>
      <p style="font-size:11px;color:#666;line-height:1.6;margin:0">RSI overbought (25pts) + MACD bearish (20pts) + MA extension (20pts) + Bollinger Band (15pts) + Volume divergence (10pts) + Momentum (10pts). Score ≥ 60 = high conviction.</p>
    </div>
  </div>
</div>
"""


def _build_three_book_panel() -> str:
    """Three-book portfolio panel: backtest NAV + live Alpaca positions + IC signal health."""
    import json as _json
    from pathlib import Path as _Path
    ROOT_p = _Path(__file__).parent

    # ── Load backtest data ──────────────────────────────────────────────────
    bt_path = ROOT_p / "backtest_three_books.json"
    bt = {}
    if bt_path.exists():
        try:
            bt = _json.loads(bt_path.read_text())
        except Exception:
            pass

    # ── Load live Alpaca state ──────────────────────────────────────────────
    state_path = ROOT_p / "alpaca_book_state.json"
    book_state = {}
    if state_path.exists():
        try:
            book_state = _json.loads(state_path.read_text())
        except Exception:
            pass

    # ── Load IC multipliers ─────────────────────────────────────────────────
    sw_path = ROOT_p / "signal_weights.json"
    ic_mults = {}
    raw_ic   = {}
    if sw_path.exists():
        try:
            sw = _json.loads(sw_path.read_text())
            ic_mults = sw.get("ic_multipliers", {})
            raw_ic   = sw.get("raw_ic", {})
        except Exception:
            pass

    # ── Stats table rows ────────────────────────────────────────────────────
    BOOK_LABELS = {"SHORT": "SHORT (5d)", "MEDIUM": "MEDIUM (21d)", "LONG": "LONG (63d)", "SPY_BENCHMARK": "SPY"}
    BOOK_COLORS = {"SHORT": "#7c96a0", "MEDIUM": "#3ABA7A", "LONG": "#C8A040", "SPY_BENCHMARK": "#888"}

    stats_rows = ""
    for book in ("SHORT", "MEDIUM", "LONG", "SPY_BENCHMARK"):
        data = bt.get(book, {})
        s    = data.get("stats", {})
        cagr = s.get("cagr", 0)
        sh   = s.get("sharpe", 0)
        mdd  = s.get("max_drawdown", 0)
        cal  = s.get("calmar", 0)
        color = BOOK_COLORS.get(book, "#888")
        spy_cagr = bt.get("SPY_BENCHMARK", {}).get("stats", {}).get("cagr", 0)
        alpha_vs_spy = cagr - spy_cagr if book != "SPY_BENCHMARK" else 0
        alpha_color  = "#2A7A50" if alpha_vs_spy > 0 else "#7A3020"
        alpha_str    = (f'<span style="color:{alpha_color}">{alpha_vs_spy*100:+.1f}%</span>'
                        if book != "SPY_BENCHMARK" else "—")
        stats_rows += f"""
        <tr>
          <td><span style="display:inline-block;width:10px;height:10px;background:{color};border-radius:2px;margin-right:6px"></span>{BOOK_LABELS.get(book, book)}</td>
          <td style="text-align:right">{cagr*100:.1f}%</td>
          <td style="text-align:right">{sh:.2f}</td>
          <td style="text-align:right">{mdd*100:.1f}%</td>
          <td style="text-align:right">{cal:.2f}</td>
          <td style="text-align:right">{alpha_str}</td>
        </tr>"""

    # ── NAV chart data ──────────────────────────────────────────────────────
    nav_datasets = []
    for book, color in BOOK_COLORS.items():
        data = bt.get(book, {})
        nav  = data.get("nav", {})
        if not nav:
            continue
        # sample every 5 rows for chart performance
        items = list(nav.items())
        sampled = items[::5] + ([items[-1]] if items else [])
        labels_js = "[" + ",".join(f'"{d[:10]}"' for d, _ in sampled) + "]"
        vals_js   = "[" + ",".join(f"{v:.4f}" for _, v in sampled) + "]"
        dash = "[4,4]" if book == "SPY_BENCHMARK" else "[]"
        nav_datasets.append(
            f'{{"label":"{BOOK_LABELS.get(book,book)}","data":{vals_js},"borderColor":"{color}",'
            f'"borderDash":{dash},"fill":false,"pointRadius":0,"tension":0.2}}'
        )
    # labels from SPY or first book
    first_book_data = bt.get("SPY_BENCHMARK") or bt.get("SHORT") or {}
    first_nav = first_book_data.get("nav", {})
    first_items = list(first_nav.items())
    first_sampled = first_items[::5] + ([first_items[-1]] if first_items else [])
    chart_labels = "[" + ",".join(f'"{d[:10]}"' for d, _ in first_sampled) + "]"
    datasets_js  = "[" + ",".join(nav_datasets) + "]"

    # ── Live positions tables ───────────────────────────────────────────────
    pos_html = ""
    for book_name in ("SHORT", "MEDIUM", "LONG"):
        state = book_state.get(book_name, {})
        positions = state.get("positions", {})
        last_reb  = str(state.get("last_rebalance", "—"))[:10]
        eff_cap   = state.get("effective_capital", 0)
        reg_scale = state.get("regime_scale", 1.0)
        color = BOOK_COLORS.get(book_name, "#888")
        rows  = ""
        for i, (tk, val) in enumerate(sorted(positions.items(),
                                             key=lambda x: -x[1])[:10]):
            rows += f'<tr><td>{i+1}</td><td style="font-weight:400">{tk}</td><td style="text-align:right">${val:,.0f}</td></tr>'
        if not rows:
            rows = '<tr><td colspan="3" style="color:#888;font-style:italic">No active positions</td></tr>'
        pos_html += f"""
        <div>
          <p style="font-size:10px;font-weight:400;letter-spacing:1.5px;text-transform:uppercase;color:{color};margin-bottom:4px">{book_name} BOOK</p>
          <p style="font-size:11px;color:#888;margin-bottom:8px">Last rebal: {last_reb} · Capital: ${eff_cap:,.0f} · Regime scale: {reg_scale:.0%}</p>
          <table style="width:100%;font-size:11px">
            <thead><tr><th>#</th><th>Ticker</th><th style="text-align:right">Target $</th></tr></thead>
            <tbody>{rows}</tbody>
          </table>
        </div>"""

    # ── IC signal health bar ────────────────────────────────────────────────
    ic_bars = ""
    sig_order = sorted([(s, raw_ic.get(s)) for s in ic_mults], key=lambda x: (x[1] or 0), reverse=True)
    for sig, ic_val in sig_order[:10]:
        mult = ic_mults.get(sig, 1.0)
        ic_pct = min(max((ic_val or 0) * 5 + 0.5, 0), 1) * 100
        bar_color = "#2A7A50" if (ic_val or 0) > 0 else "#7A3020"
        ic_str = f"{ic_val:.3f}" if ic_val is not None else "null"
        mult_str = f"{mult:.2f}×"
        ic_bars += f"""
        <div style="margin-bottom:8px">
          <div style="display:flex;justify-content:space-between;font-size:10px;margin-bottom:3px">
            <span style="color:{FT['ink']};font-weight:400">{sig}</span>
            <span style="color:{FT['mute']}">IC {ic_str} → {mult_str}</span>
          </div>
          <div style="height:5px;background:{FT['inner']};border-radius:2px">
            <div style="height:5px;width:{ic_pct:.0f}%;background:{bar_color};border-radius:2px"></div>
          </div>
        </div>"""

    return f"""
<div id="three-book-panel" style="margin-bottom:26px;background:{FT['card']};border:1px solid {FT['border']};border-radius:8px;padding:16px 18px">
  <div style="display:flex;justify-content:space-between;align-items:baseline;flex-wrap:wrap;gap:8px;margin-bottom:14px">
    <span style="font-size:11px;color:{FT['mute']};text-transform:uppercase;letter-spacing:.14em">Live Portfolio · Three-Book Strategy · SHORT · MEDIUM · LONG</span>
    <span style="font-size:11px;color:{FT['mute']}">Alpaca paper trading + backtest</span>
  </div>

  <!-- Stats table -->
  <div class="tbl-wrap" style="margin-bottom:24px">
    <p class="tbl-title">2-Year Backtest Performance (last 501 trading days · 5 bps/trade)</p>
    <table>
      <thead><tr><th>Book</th><th style="text-align:right">CAGR</th><th style="text-align:right">Sharpe</th><th style="text-align:right">Max DD</th><th style="text-align:right">Calmar</th><th style="text-align:right">vs SPY</th></tr></thead>
      <tbody>{stats_rows}</tbody>
    </table>
  </div>

  <!-- NAV chart -->
  <div class="chart-box" style="margin-bottom:24px">
    <p class="chart-title">NAV Curves — Three Books vs SPY (normalised to 1.0)</p>
    <div class="chart-inner" style="height:240px"><canvas id="threeBookNavChart"></canvas></div>
  </div>

  <!-- Live positions -->
  <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:16px;margin-bottom:24px">
    {pos_html}
  </div>

  <!-- IC signal health -->
  <div class="method-card" style="margin-bottom:0">
    <p style="font-size:10px;font-weight:400;color:#c8b487;margin-bottom:12px;text-transform:uppercase;letter-spacing:1.5px">Live Signal IC Health — Rolling 60d Spearman IC → Weight Multipliers</p>
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px 24px">{ic_bars}</div>
  </div>
</div>

<script>
(function() {{
  var ctx = document.getElementById('threeBookNavChart');
  if (!ctx) return;
  new Chart(ctx, {{
    type: 'line',
    data: {{
      labels: {chart_labels},
      datasets: {datasets_js}
    }},
    options: {{
      responsive:true, maintainAspectRatio:false,
      interaction:{{mode:'index',intersect:false}},
      plugins:{{legend:{{position:'top',labels:{{font:{{size:10}},boxWidth:12}}}},tooltip:{{callbacks:{{label:function(c){{return c.dataset.label+': '+(c.parsed.y*100-100).toFixed(1)+'%'}}}}}}}},
      scales:{{
        x:{{ticks:{{maxTicksLimit:8,font:{{size:9}}}},grid:{{display:false}}}},
        y:{{ticks:{{callback:function(v){{return ((v-1)*100).toFixed(0)+'%'}},font:{{size:9}},color:'#8a7f70'}},grid:{{color:'#2a231b'}}}}
      }}
    }}
  }});
}})();
</script>"""


def _build_pnl_ic_panels() -> str:
    """
    Two side-by-side panels for Performance tab:
      Left:  Alpaca P&L attribution (per-book table + total)
      Right: Signal IC trend chart (signal_ic_history.csv time series)
    """
    import json as _json
    from pathlib import Path as _P

    ROOT_p = _P(__file__).parent

    # ── P&L data ──────────────────────────────────────────────────────────
    pnl_path = ROOT_p / "alpaca_pnl_summary.json"
    attr_path = ROOT_p / "alpaca_pnl_attribution.csv"

    pnl_rows_html = ""
    total_pnl     = 0.0
    ic_val        = None
    as_of         = ""
    if pnl_path.exists():
        try:
            pnl = _json.loads(pnl_path.read_text())
            total_pnl = pnl.get("total_pnl_$", 0.0)
            ic_val    = pnl.get("ic_mu_vs_realized")
            as_of     = pnl.get("as_of", "")
            for book in ("SHORT", "MEDIUM", "LONG"):
                s = pnl.get(book, {})
                if not s:
                    continue
                ret    = (s.get("avg_return") or 0) * 100
                pred   = (s.get("avg_pred_mu") or 0) * 100
                alpha  = (s.get("realized_alpha") or 0) * 100
                sign   = "+" if ret >= 0 else ""
                acol   = "#4ade80" if alpha >= 0 else "#f87171"
                pnl_rows_html += f"""
              <tr>
                <td>{book}</td>
                <td>{s.get('n_positions',0)}</td>
                <td style="color:{'#4ade80' if (s.get('total_pnl_$') or 0)>=0 else '#f87171'}">
                  ${s.get('total_pnl_$',0):+,.0f}</td>
                <td>{sign}{ret:.1f}%</td>
                <td>{pred:.1f}%</td>
                <td style="color:{acol};font-weight:400">{alpha:+.1f}%</td>
              </tr>"""
        except Exception:
            pass

    total_color = "#4ade80" if total_pnl >= 0 else "#f87171"
    ic_badge = ""
    if ic_val is not None:
        ic_color = "#4ade80" if ic_val > 0.05 else ("#facc15" if ic_val >= 0 else "#f87171")
        ic_badge = f'<span style="color:{ic_color};font-weight:400">IC={ic_val:.3f}</span>'

    if not pnl_rows_html:
        pnl_rows_html = '<tr><td colspan="6" style="color:#888;text-align:center">Run step_alpaca_pnl.py to populate</td></tr>'

    pnl_panel = f"""
<div class="tb-panel" style="margin-bottom:28px;background:{FT['card']};border:1px solid {FT['border']};border-radius:8px;padding:16px 18px">
  <div style="display:flex;justify-content:space-between;align-items:baseline;flex-wrap:wrap;gap:8px;margin-bottom:12px">
    <span style="font-size:11px;color:{FT['mute']};text-transform:uppercase;letter-spacing:.14em">Attribution · Alpaca Paper P&amp;L</span>
    <span style="font-size:11px;color:{FT['mute']}">{as_of} &nbsp; {ic_badge}</span>
  </div>
  <div style="overflow-x:auto">
    <table class="tbl-compact" style="width:100%;font-size:12px">
      <thead>
        <tr>
          <th>Book</th><th>Positions</th><th>Total P&amp;L</th>
          <th>Avg Ret</th><th>Pred μ</th><th>Alpha</th>
        </tr>
      </thead>
      <tbody>
        {pnl_rows_html}
        <tr style="border-top:1px solid #333;font-weight:400">
          <td colspan="2">Total</td>
          <td style="color:{total_color}">${total_pnl:+,.0f}</td>
          <td colspan="3"></td>
        </tr>
      </tbody>
    </table>
  </div>
</div>"""

    # ── IC bar chart — reads signal_weights.json (always available) ───────
    sw_path = ROOT_p / "signal_weights.json"
    ic_bars_html = ""
    ic_note = ""
    ic_updated = ""

    SIG_ORDER = [
        ("regime_ml",   "Regime ML"),
        ("momentum",    "Momentum"),
        ("ml_short",    "ML Short"),
        ("ml_medium",   "ML Medium"),
        ("ml_long",     "ML Long"),
        ("surprise",    "Earn Surprise"),
        ("sentiment",   "Sentiment"),
        ("quality",     "Quality"),
        ("squeeze",     "Short Squeeze"),
        ("revision",    "Revision"),
        ("insider",     "Insider"),
        ("accruals",    "Accruals"),
        ("piotroski",   "Piotroski"),
        ("options",     "Options"),
    ]

    if sw_path.exists():
        try:
            sw = _json.loads(sw_path.read_text())
            raw_ic   = sw.get("raw_ic", {})
            mults    = sw.get("ic_multipliers", {})
            ic_updated = sw.get("updated", "")
            ic_note  = f"as of {ic_updated}"

            bar_rows = []
            for key, label in SIG_ORDER:
                ic = raw_ic.get(key)
                if ic is None:
                    continue
                mult = mults.get(key, 1.0)
                pct  = max(-15, min(15, ic * 100))   # scale to ±15% display width
                bar_w = abs(pct) / 15 * 48            # max 48% of cell
                if ic >= 0.03:
                    bar_color, badge_color, status = "#4ade80", "#166534", "BOOST"
                elif ic >= 0:
                    bar_color, badge_color, status = "#60a5fa", "#1e3a5f", "OK"
                else:
                    bar_color, badge_color, status = "#f87171", "#7f1d1d", "DAMP"

                bar_dir = "right" if ic >= 0 else "left"
                bar_style = (
                    f"width:{bar_w:.1f}%;background:{bar_color};height:8px;"
                    f"border-radius:2px;display:inline-block;vertical-align:middle;"
                    f"margin-{'left' if ic < 0 else 'right'}:2px"
                )
                badge = (f'<span style="background:{badge_color};color:{bar_color};'
                         f'font-size:9px;padding:1px 5px;border-radius:3px;font-weight:400">'
                         f'{status}</span>')
                bar_rows.append(f"""
          <tr>
            <td style="color:#ccc;font-size:11px;white-space:nowrap">{label}</td>
            <td style="text-align:right;font-size:11px;color:{bar_color};font-variant-numeric:tabular-nums;padding-right:6px">{ic:+.3f}</td>
            <td style="width:55%">
              {'<span style="display:inline-block;width:48%;"></span>' if ic >= 0 else ''}
              <span style="{bar_style}"></span>
              {'<span style="display:inline-block;width:48%;"></span>' if ic < 0 else ''}
            </td>
            <td style="text-align:right;font-size:10px;color:#888">{mult:.2f}×&nbsp;{badge}</td>
          </tr>""")

            ic_bars_html = f"""<table style="width:100%;border-collapse:collapse">
        {''.join(bar_rows)}
        </table>"""
        except Exception as e:
            ic_bars_html = f'<p style="color:#888;font-size:12px;padding:12px">IC load error: {e}</p>'

    if not ic_bars_html:
        ic_bars_html = '<p style="color:#888;font-size:12px;padding:20px">Run step_rolling_ic.py first.</p>'

    ic_panel = f"""
<div class="tb-panel" style="margin-bottom:28px;background:{FT['card']};border:1px solid {FT['border']};border-radius:8px;padding:16px 18px">
  <div style="display:flex;justify-content:space-between;align-items:baseline;flex-wrap:wrap;gap:8px;margin-bottom:12px">
    <span style="font-size:11px;color:{FT['mute']};text-transform:uppercase;letter-spacing:.14em">Signal IC Health</span>
    <span style="font-size:11px;color:{FT['mute']}">{ic_note}</span>
  </div>
  <div style="overflow-y:auto;max-height:230px;padding:4px 0">
    {ic_bars_html}
  </div>
</div>"""

    return f"""
<div style="display:grid;grid-template-columns:1fr 1fr;gap:24px;margin-top:32px;margin-bottom:8px">
  <div>{pnl_panel}</div>
  <div>{ic_panel}</div>
</div>"""


def _build_dcf_section(df: "pd.DataFrame") -> str:
    """Render the DCF Valuation tab from dcf_valuation.csv."""
    if df is None or df.empty:
        return '<p style="color:#AAA;font-size:13px">DCF valuation not yet run. It will populate after the next daily pipeline run.</p>'

    today = str(pd.Timestamp.today().date())
    today_df = df[df["as_of"] == today] if "as_of" in df.columns else df
    if today_df.empty:
        today_df = df  # use whatever's there

    n_total   = len(today_df)
    as_of_str = today_df["as_of"].iloc[0] if "as_of" in today_df.columns else "—"
    rf_str    = f"{today_df['rf'].iloc[0]:.2f}%" if "rf" in today_df.columns else "—"
    erp_str   = f"{today_df['erp'].iloc[0]:.1f}%" if "erp" in today_df.columns else "—"

    # Stage distribution
    stage_counts = today_df["stage"].value_counts().to_dict() if "stage" in today_df.columns else {}
    stage_colors = {"High Growth": "#1B6F4A", "Maturing": "#c8b487", "Mature": "#888",
                    "Value Trap": "#B83232", "Unknown": "#CCC"}

    def _pct(x): return f"{x*100:.1f}%" if pd.notna(x) else "—"
    def _usd(x): return f"${x:,.0f}" if pd.notna(x) else "—"

    # Undervalued table (upside > 15%)
    undervalued = today_df[today_df["upside_pct"].notna() & (today_df["upside_pct"] > 15)].sort_values("upside_pct", ascending=False).head(25)
    # Overvalued table (upside < -15%)
    overvalued  = today_df[today_df["upside_pct"].notna() & (today_df["upside_pct"] < -15)].sort_values("upside_pct").head(25)
    # Top EVA generators
    top_eva     = today_df[today_df["eva_m"].notna() & (today_df["eva_m"] > 0)].sort_values("eva_m", ascending=False).head(15)
    # Capital destroyers
    destroyers  = today_df[today_df["roic_wacc_spread"].notna() & (today_df["roic_wacc_spread"] < 0)].sort_values("roic_wacc_spread").head(15)

    def _row_under(r):
        upside = r["upside_pct"]
        up_color = "#1B6F4A" if upside > 30 else ("#c8b487" if upside > 15 else "#888")
        roic = _pct(r.get("roic"))
        spread = r.get("roic_wacc_spread")
        sp_color = "#1B6F4A" if (spread or 0) > 0.05 else ("#c8b487" if (spread or 0) > 0 else "#B83232")
        stage = r.get("stage", "—")
        stage_color = stage_colors.get(stage, "#888")
        return (
            f'<tr>'
            f'<td style="font-weight:400;color:#c8b487">{r["ticker"]}</td>'
            f'<td style="color:#555;font-size:11px">{str(r.get("name",""))[:22]}</td>'
            f'<td style="color:#555;font-size:11px">{r.get("sector","")[:14]}</td>'
            f'<td style="font-variant-numeric:tabular-nums">${r.get("price",0):.0f}</td>'
            f'<td style="font-variant-numeric:tabular-nums">${r.get("iv_per_share",0):.0f}</td>'
            f'<td style="color:{up_color};font-weight:400;font-variant-numeric:tabular-nums">{upside:+.0f}%</td>'
            f'<td style="font-variant-numeric:tabular-nums">{roic}</td>'
            f'<td style="font-variant-numeric:tabular-nums">{_pct(r.get("wacc"))}</td>'
            f'<td style="color:{sp_color};font-weight:400;font-variant-numeric:tabular-nums">{_pct(spread)}</td>'
            f'<td style="color:{stage_color};font-size:11px;font-weight:400">{stage}</td>'
            f'<td style="font-variant-numeric:tabular-nums;color:#888">{r.get("pvgo_pct",0):.0f}%</td>'
            f'</tr>'
        )

    def _row_over(r):
        upside = r["upside_pct"]
        roic = _pct(r.get("roic"))
        stage = r.get("stage", "—")
        stage_color = stage_colors.get(stage, "#888")
        return (
            f'<tr>'
            f'<td style="font-weight:400;color:#c8b487">{r["ticker"]}</td>'
            f'<td style="color:#555;font-size:11px">{str(r.get("name",""))[:22]}</td>'
            f'<td style="color:#555;font-size:11px">{r.get("sector","")[:14]}</td>'
            f'<td style="font-variant-numeric:tabular-nums">${r.get("price",0):.0f}</td>'
            f'<td style="font-variant-numeric:tabular-nums">${r.get("iv_per_share",0):.0f}</td>'
            f'<td style="color:#B83232;font-weight:400;font-variant-numeric:tabular-nums">{upside:+.0f}%</td>'
            f'<td style="font-variant-numeric:tabular-nums">{roic}</td>'
            f'<td style="font-variant-numeric:tabular-nums">{_pct(r.get("wacc"))}</td>'
            f'<td style="color:{stage_color};font-size:11px;font-weight:400">{stage}</td>'
            f'</tr>'
        )

    def _row_eva(r):
        spread_color = "#1B6F4A" if (r.get("roic_wacc_spread") or 0) > 0.10 else "#c8b487"
        return (
            f'<tr>'
            f'<td style="font-weight:400;color:#c8b487">{r["ticker"]}</td>'
            f'<td style="color:#555;font-size:11px">{str(r.get("name",""))[:22]}</td>'
            f'<td style="color:{spread_color};font-weight:400;font-variant-numeric:tabular-nums">{_pct(r.get("roic"))}</td>'
            f'<td style="font-variant-numeric:tabular-nums">{_pct(r.get("wacc"))}</td>'
            f'<td style="color:{spread_color};font-weight:400">{_pct(r.get("roic_wacc_spread"))}</td>'
            f'<td style="font-variant-numeric:tabular-nums;color:#1B6F4A;font-weight:400">${r.get("eva_m",0):+,.0f}M</td>'
            f'<td style="color:#555;font-size:11px">{r.get("stage","")}</td>'
            f'</tr>'
        )

    def _row_destroy(r):
        return (
            f'<tr>'
            f'<td style="font-weight:400;color:#B83232">{r["ticker"]}</td>'
            f'<td style="color:#555;font-size:11px">{str(r.get("name",""))[:22]}</td>'
            f'<td style="font-variant-numeric:tabular-nums;color:#B83232">{_pct(r.get("roic"))}</td>'
            f'<td style="font-variant-numeric:tabular-nums">{_pct(r.get("wacc"))}</td>'
            f'<td style="color:#B83232;font-weight:400">{_pct(r.get("roic_wacc_spread"))}</td>'
            f'<td style="color:#B83232;font-variant-numeric:tabular-nums">${r.get("eva_m",0):+,.0f}M</td>'
            f'<td style="color:#555;font-size:11px">{r.get("stage","")}</td>'
            f'</tr>'
        )

    stage_pills = "".join(
        f'<span style="display:inline-block;padding:4px 12px;border-radius:3px;font-size:11px;font-weight:400;'
        f'color:{stage_colors.get(s,"#888")};border:1px solid {stage_colors.get(s,"#888")};margin:3px">'
        f'{s}: {cnt}</span>'
        for s, cnt in stage_counts.items()
    )

    under_rows  = "".join(_row_under(r) for _, r in undervalued.iterrows())
    over_rows   = "".join(_row_over(r) for _, r in overvalued.iterrows())
    eva_rows    = "".join(_row_eva(r) for _, r in top_eva.iterrows())
    dest_rows   = "".join(_row_destroy(r) for _, r in destroyers.iterrows())

    return f"""
<div class="method-card" style="margin-bottom:28px">
  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:18px">
    <div>
      <p style="font-size:10px;letter-spacing:1.5px;text-transform:uppercase;color:#c8b487;font-weight:400;margin:0 0 4px">Damodaran 3-Stage DCF · {n_total} stocks</p>
      <p style="font-size:12px;color:#555;margin:0">rf={rf_str} · ERP={erp_str} · Stable g=2.5% · as of {as_of_str}</p>
    </div>
    <div style="text-align:right">
      <p style="font-size:10px;color:#AAA;margin:0">Stage 1 yr1-5: current ROIC × blended growth</p>
      <p style="font-size:10px;color:#AAA;margin:0">Stage 2 yr6-10: linear fade · Stage 3: terminal ROIC=WACC+2%</p>
    </div>
  </div>
  <div style="margin-bottom:12px">{stage_pills}</div>
</div>

<div class="two-col-65">
  <div>
    <div class="tbl-wrap">
      <p class="tbl-title">Undervalued — DCF upside &gt; 15% ({len(undervalued)})</p>
      <div style="overflow-x:auto">
      <table>
        <thead><tr>
          <th>Ticker</th><th>Name</th><th>Sector</th>
          <th class="r">Price</th><th class="r">IV/sh</th><th class="r">Upside</th>
          <th class="r">ROIC</th><th class="r">WACC</th><th class="r">Spread</th>
          <th>Stage</th><th class="r">PVGO%</th>
        </tr></thead>
        <tbody>{under_rows if under_rows else "<tr><td colspan='11' style='color:#AAA;text-align:center'>None today</td></tr>"}</tbody>
      </table>
      </div>
      <p class="tbl-note">IV = intrinsic value per share. Spread = ROIC − WACC. PVGO = % of EV from growth options.</p>
    </div>
  </div>

  <div>
    <div class="tbl-wrap">
      <p class="tbl-title">EVA Champions — top economic value creators</p>
      <table>
        <thead><tr>
          <th>Ticker</th><th>Name</th>
          <th class="r">ROIC</th><th class="r">WACC</th><th class="r">Spread</th>
          <th class="r">EVA ($M)</th><th>Stage</th>
        </tr></thead>
        <tbody>{eva_rows if eva_rows else "<tr><td colspan='7' style='color:#AAA;text-align:center'>No data</td></tr>"}</tbody>
      </table>
      <p class="tbl-note">EVA = (ROIC − WACC) × Invested Capital. Positive = creating wealth.</p>
    </div>
  </div>
</div>

<div class="two-col-65" style="margin-top:28px">
  <div>
    <div class="tbl-wrap">
      <p class="tbl-title">Overvalued — DCF downside &gt; 15% ({len(overvalued)})</p>
      <div style="overflow-x:auto">
      <table>
        <thead><tr>
          <th>Ticker</th><th>Name</th><th>Sector</th>
          <th class="r">Price</th><th class="r">IV/sh</th><th class="r">Upside</th>
          <th class="r">ROIC</th><th class="r">WACC</th><th>Stage</th>
        </tr></thead>
        <tbody>{over_rows if over_rows else "<tr><td colspan='9' style='color:#AAA;text-align:center'>None today</td></tr>"}</tbody>
      </table>
      </div>
    </div>
  </div>

  <div>
    <div class="tbl-wrap">
      <p class="tbl-title">Capital Destroyers — ROIC &lt; WACC</p>
      <table>
        <thead><tr>
          <th>Ticker</th><th>Name</th>
          <th class="r">ROIC</th><th class="r">WACC</th><th class="r">Spread</th>
          <th class="r">EVA ($M)</th><th>Stage</th>
        </tr></thead>
        <tbody>{dest_rows if dest_rows else "<tr><td colspan='7' style='color:#AAA;text-align:center'>No data</td></tr>"}</tbody>
      </table>
      <p class="tbl-note">Growth compounds value destruction when ROIC &lt; WACC.</p>
    </div>
  </div>
</div>

<div class="method-card" style="margin-top:32px;border-top:3px solid #3a3128">
  <p style="font-size:10px;letter-spacing:1.5px;text-transform:uppercase;color:#999;font-weight:400;margin:0 0 12px">Methodology — Damodaran 5 Questions</p>
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:20px">
    <div>
      <p style="font-size:11px;color:#333;font-weight:400;margin:0 0 3px">1. Value of existing assets?</p>
      <p style="font-size:11px;color:#666;margin:0 0 10px">ROIC = NOPAT / Invested Capital. EVA = (ROIC−WACC)×IC. Positive EVA = existing operations create value.</p>
      <p style="font-size:11px;color:#333;font-weight:400;margin:0 0 3px">2. Value of growth assets?</p>
      <p style="font-size:11px;color:#666;margin:0 0 10px">PVGO = Enterprise Value − (NOPAT/WACC). Shows what % of market price is priced-in future growth, not current operations.</p>
      <p style="font-size:11px;color:#333;font-weight:400;margin:0 0 3px">3. Cash flow risk?</p>
      <p style="font-size:11px;color:#666;margin:0">WACC = Ke×(E/V) + Kd(1−t)×(D/V). Ke=rf+β×ERP. Higher WACC = higher risk = lower present value.</p>
    </div>
    <div>
      <p style="font-size:11px;color:#333;font-weight:400;margin:0 0 3px">4. When does it mature?</p>
      <p style="font-size:11px;color:#666;margin:0 0 10px">Lifecycle stage (High Growth/Maturing/Mature/Value Trap) based on revenue growth and ROIC-WACC spread. Stage 2 (yr 6-10) linearly fades to terminal.</p>
      <p style="font-size:11px;color:#333;font-weight:400;margin:0 0 3px">5. Equity value per share?</p>
      <p style="font-size:11px;color:#666;margin:0">EV = PV of Stage 1 + Stage 2 + Terminal FCFFs. Equity Value = EV − Net Debt. IV/share = Equity Value / Shares. Data: yfinance only.</p>
    </div>
  </div>
  <p style="font-size:10px;color:#AAA;margin:14px 0 0">Assumptions: rf={rf_str}, ERP={erp_str}, Stable g=2.5%, Terminal ROIC=WACC+2%. Banks/financials excluded (different model needed). This is a research tool, not a trading signal.</p>
</div>
"""


def _build_economic_calendar_widget(econ_cal: dict) -> str:
    events = econ_cal.get("events", [])
    if not events:
        return ""

    IMPACT_COLOR = {"high": "#B83232", "medium": "#c8b487"}
    TYPE_BG = {"FOMC": "#EFF6FF", "CPI": "#F0FDF4", "PPI": "#FFF7ED", "NFP": "#F5F3FF"}
    TYPE_TC = {"FOMC": "#495663", "CPI": "#166534", "PPI": "#9A3412", "NFP": "#544a66"}

    rows = []
    for ev in events[:12]:
        d_str  = ev["date"]
        days   = ev["days_until"]
        timing = "TODAY" if days == 0 else (f"Tomorrow" if days == 1 else f"in {days}d")
        bg     = TYPE_BG.get(ev["type"], "#F9FAFB")
        tc     = TYPE_TC.get(ev["type"], "#374151")
        ic     = IMPACT_COLOR.get(ev.get("impact", "medium"), "#c8b487")
        rows.append(
            f'<tr>'
            f'<td style="font-weight:400;color:#c8b487;white-space:nowrap">{d_str}</td>'
            f'<td style="color:{ic};font-size:11px;font-weight:400;white-space:nowrap">{timing}</td>'
            f'<td><span style="background:{bg};color:{tc};font-size:9px;font-weight:500;'
            f'letter-spacing:.8px;text-transform:uppercase;padding:2px 7px;border-radius:2px">'
            f'{ev["type"]}</span></td>'
            f'<td style="font-size:12px">{ev["emoji"]} {ev["name"]}</td>'
            f'<td><span style="font-size:10px;font-weight:400;color:{ic}">'
            f'{"●●●" if ev.get("impact")=="high" else "●●○"}</span></td>'
            f'</tr>'
        )

    return f"""
    <div class="mt36">
      <p class="eyebrow">Economic Calendar</p>
      <h3 style="font-family:'Playfair Display',serif;font-size:22px;font-weight:400;color:#1A1A1A;margin:8px 0 4px">
        Upcoming high-impact events — next 45 days</h3>
      <div class="rule"></div>
      <div style="overflow-x:auto">
      <table>
        <thead><tr>
          <th>Date</th><th>Timing</th><th>Type</th><th>Event</th><th>Impact</th>
        </tr></thead>
        <tbody>{''.join(rows)}</tbody>
      </table>
      </div>
      <p style="font-size:11px;color:#AAA;margin-top:8px">
        Sources: FOMC dates from federalreserve.gov · BLS release schedule.
        Run <code>step_economic_calendar.py</code> to refresh.</p>
    </div>"""


def _build_live_market_pulse() -> str:
    """Live auto-refreshing market pulse bar for the Today tab."""
    tickers_json = '["SPY","QQQ","IWM","VIX","GLD","TLT","HYG"]'
    return f"""
    <div class="mt36">
      <p class="eyebrow">Live Market Pulse</p>
      <h3 style="font-family:'Playfair Display',serif;font-size:22px;font-weight:400;color:#1A1A1A;margin:8px 0 4px">
        Real-time prices — auto-refreshes every 90 seconds</h3>
      <div class="rule"></div>
      <div id="pulse-bar" style="display:grid;grid-template-columns:repeat(7,1fr);gap:10px;margin-bottom:8px">
        <!-- filled by JS -->
        <div style="text-align:center;padding:12px 6px;background:#fff;border:1px solid #241f18;border-radius:4px;color:#CCC;font-size:11px">Loading…</div>
      </div>
      <p id="pulse-updated" style="font-size:10px;color:#BBB;text-align:right"></p>
    </div>

    <script>
    (function() {{
      var TICKERS = {tickers_json};
      var bar = document.getElementById('pulse-bar');
      var upd = document.getElementById('pulse-updated');

      function fetchPulse() {{
        fetch('/api/live?tickers=' + TICKERS.join(','))
          .then(function(r) {{ return r.json(); }})
          .then(function(data) {{
            var prices = data.prices || {{}};
            var cells = TICKERS.map(function(t) {{
              var p = prices[t];
              if (!p) return '<div style="text-align:center;padding:12px 6px;background:#fff;border:1px solid #241f18;border-radius:4px"><p style="font-size:11px;font-weight:400;color:#c8b487">' + t + '</p><p style="font-size:12px;color:#CCC">—</p></div>';
              var color = p.chg_pct >= 0 ? '#1B6F4A' : '#B83232';
              var bg    = p.chg_pct >= 0 ? '#F0FDF4' : '#FEF2F2';
              var sign  = p.chg_pct >= 0 ? '+' : '';
              var isVIX = t === 'VIX';
              if (isVIX) {{ color = p.chg_pct >= 0 ? '#B83232' : '#1B6F4A'; bg = p.chg_pct >= 0 ? '#FEF2F2' : '#F0FDF4'; }}
              return '<div style="text-align:center;padding:10px 6px;background:' + bg + ';border:1px solid ' + color + '33;border-radius:4px;border-top:2px solid ' + color + '">'
                + '<p style="font-size:10px;font-weight:500;letter-spacing:1px;color:#555;margin-bottom:4px">' + t + '</p>'
                + '<p style="font-size:16px;font-weight:400;color:#1A1A1A;margin-bottom:2px">$' + p.price.toFixed(2) + '</p>'
                + '<p style="font-size:11px;font-weight:400;color:' + color + '">' + sign + p.chg_pct.toFixed(2) + '%</p>'
                + '</div>';
            }});
            bar.innerHTML = cells.join('');
            if (upd) upd.textContent = 'Updated ' + (data.as_of || '') + ' ET · Next refresh in 90s';
          }})
          .catch(function() {{
            if (upd) upd.textContent = 'Live prices unavailable (server offline)';
          }});
      }}

      fetchPulse();
      setInterval(fetchPulse, 90000);
    }})();
    </script>"""


def _canyon_global_overlays() -> str:
    """Stock quick-look modal + global JS. Injected once near end of body."""
    return """
<!-- ═══ STOCK QUICK-LOOK MODAL ═══ -->
<div id="ql-overlay" onclick="if(event.target===this)canyonQL.close()"
  style="display:none;position:fixed;inset:0;background:rgba(0,0,0,.55);z-index:2000;
  align-items:center;justify-content:center;padding:20px">
  <div id="ql-modal" style="background:#fff;border-radius:8px;width:100%;max-width:680px;
    max-height:88vh;overflow-y:auto;box-shadow:0 24px 60px rgba(0,0,0,.35);position:relative">

    <!-- Header -->
    <div id="ql-header" style="background:#231a12;padding:20px 24px 16px;border-radius:8px 8px 0 0;position:sticky;top:0;z-index:1">
      <div style="display:flex;align-items:flex-start;justify-content:space-between">
        <div>
          <p id="ql-name" style="font-size:11px;color:rgba(255,255,255,.5);margin-bottom:4px"></p>
          <p id="ql-ticker-title" style="font-family:'Playfair Display',serif;font-size:26px;font-weight:400;color:#fff"></p>
        </div>
        <div style="text-align:right">
          <p id="ql-price" style="font-size:26px;font-weight:400;color:#fff"></p>
          <p id="ql-chg" style="font-size:13px;font-weight:400"></p>
        </div>
      </div>
      <!-- Sparkline canvas -->
      <canvas id="ql-sparkline" height="40" style="width:100%;margin-top:10px;display:block"></canvas>
      <button onclick="canyonQL.close()" style="position:absolute;top:12px;right:16px;background:rgba(255,255,255,.12);
        border:none;color:#fff;width:28px;height:28px;border-radius:50%;cursor:pointer;font-size:16px;
        display:flex;align-items:center;justify-content:center">×</button>
    </div>

    <!-- Body -->
    <div style="padding:20px 24px">
      <!-- Key metrics row -->
      <div id="ql-metrics" style="display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:20px"></div>

      <!-- Canyon signals -->
      <div id="ql-signals" style="margin-bottom:16px"></div>

      <!-- DCF -->
      <div id="ql-dcf" style="margin-bottom:16px"></div>

      <!-- Short scanner -->
      <div id="ql-short" style="margin-bottom:16px"></div>

      <!-- AI Summary -->
      <div id="ql-ai" style="margin-bottom:16px"></div>

      <!-- Ask AI button -->
      <div style="display:flex;gap:10px;margin-top:16px;border-top:1px solid #241f18;padding-top:16px">
        <button id="ql-ask-btn" onclick="canyonQL.askAI()"
          style="flex:1;background:#2a2418;color:#fff;border:none;padding:10px 16px;border-radius:4px;
          cursor:pointer;font-size:12px;font-weight:400">💬 Ask AI about this stock</button>
        <button onclick="canyonQL.addToWatchlist()"
          style="background:transparent;border:1px solid #c8b487;color:#c8b487;padding:10px 16px;border-radius:4px;
          cursor:pointer;font-size:12px;font-weight:400">⭐ Add to Watchlist</button>
      </div>
    </div>

    <!-- Loading overlay -->
    <div id="ql-loading" style="display:none;position:absolute;inset:0;background:#fff;border-radius:8px;
      align-items:center;justify-content:center">
      <div style="text-align:center">
        <div style="width:32px;height:32px;border:3px solid #241f18;border-top-color:#c8b487;
          border-radius:50%;animation:ql-spin 0.8s linear infinite;margin:0 auto 12px"></div>
        <p style="font-size:12px;color:#888">Loading live data…</p>
      </div>
    </div>
  </div>
</div>

<style>
@keyframes ql-spin {{ to {{ transform:rotate(360deg) }} }}
.canyon-ticker-link {{
  color:#c8b487;font-weight:400;cursor:pointer;text-decoration:none;
  border-bottom:1px dotted #c8b487;transition:color .15s;
}}
.canyon-ticker-link:hover {{ color:#c8b487 }}
/* Chat styles */
.chat-msg-user {{
  background:#2a2418;color:#fff;padding:10px 14px;border-radius:12px 12px 3px 12px;
  align-self:flex-end;max-width:75%;font-size:13px;line-height:1.5;word-break:break-word;
}}
.chat-msg-ai {{
  background:#fff;border:1px solid #241f18;padding:12px 14px;border-radius:3px 12px 12px 12px;
  max-width:85%;font-size:13px;line-height:1.65;word-break:break-word;
  white-space:pre-wrap;box-shadow:0 1px 3px rgba(0,0,0,.06);
}}
.chat-msg-ai strong {{ color:#c8b487 }}
.chat-msg-thinking {{
  display:flex;gap:4px;align-items:center;padding:10px 14px;
}}
.chat-dot {{
  width:7px;height:7px;border-radius:50%;background:#c8b487;
  animation:chat-bounce .9s infinite;
}}
.chat-dot:nth-child(2) {{ animation-delay:.15s }}
.chat-dot:nth-child(3) {{ animation-delay:.30s }}
@keyframes chat-bounce {{ 0%,60%,100% {{ transform:translateY(0) }} 30% {{ transform:translateY(-6px) }} }}
.chip-btn {{
  background:#191410;border:1px solid #2f281f;padding:6px 11px;border-radius:20px;
  cursor:pointer;font-size:11px;color:#a89c8c;white-space:nowrap;
  transition:background .15s,border-color .15s;
}}
.chip-btn:hover {{ background:#c8b487;color:#17130f;border-color:#c8b487 }}
</style>

<script>
// ══ Stock Quick-Look ══════════════════════════════════════════════════════════
var canyonQL = (function() {{
  var _cur = null;

  function open(ticker) {{
    _cur = ticker.toUpperCase();
    var overlay = document.getElementById('ql-overlay');
    overlay.style.display = 'flex';
    document.body.style.overflow = 'hidden';
    _reset();
    _load(_cur);
  }}

  function close() {{
    document.getElementById('ql-overlay').style.display = 'none';
    document.body.style.overflow = '';
  }}

  function _reset() {{
    document.getElementById('ql-ticker-title').textContent = _cur;
    document.getElementById('ql-name').textContent = 'Loading…';
    document.getElementById('ql-price').textContent = '—';
    document.getElementById('ql-chg').textContent = '';
    ['ql-metrics','ql-signals','ql-dcf','ql-short','ql-ai'].forEach(function(id) {{
      document.getElementById(id).innerHTML = '';
    }});
    document.getElementById('ql-loading').style.display = 'flex';
  }}

  function _load(ticker) {{
    fetch('/api/stockinfo/' + ticker)
      .then(function(r) {{ return r.json(); }})
      .then(function(d) {{
        document.getElementById('ql-loading').style.display = 'none';
        _render(d);
      }})
      .catch(function(e) {{
        document.getElementById('ql-loading').style.display = 'none';
        document.getElementById('ql-name').textContent = 'Error: ' + e.message;
      }});
  }}

  function _render(d) {{
    var live = d.live || {{}};
    var chgPct = live.chg_pct;
    var chgColor = chgPct === null ? '#888' : (chgPct >= 0 ? '#1B6F4A' : '#B83232');
    var sign = (chgPct !== null && chgPct >= 0) ? '+' : '';

    document.getElementById('ql-name').textContent = (live.name || _cur) + ' · ' + (live.sector || '') + ' · ' + (live.industry || '');
    document.getElementById('ql-ticker-title').textContent = _cur;
    document.getElementById('ql-price').textContent = live.price ? '$' + live.price.toFixed(2) : '—';
    var chgEl = document.getElementById('ql-chg');
    chgEl.textContent = chgPct !== null ? sign + chgPct.toFixed(2) + '%  today' : '';
    chgEl.style.color = chgColor;

    // Sparkline
    if (live.sparkline && live.sparkline.length > 1) {{
      var canvas = document.getElementById('ql-sparkline');
      canvas.style.display = 'block';
      var ctx = canvas.getContext('2d');
      var w = canvas.parentElement.clientWidth || 600;
      canvas.width = w;
      canvas.height = 44;
      var pts = live.sparkline;
      var min = Math.min.apply(null, pts), max = Math.max.apply(null, pts);
      var rng = max - min || 1;
      var xStep = w / (pts.length - 1);
      ctx.clearRect(0, 0, w, 44);
      // Fill
      ctx.beginPath();
      ctx.moveTo(0, 44 - ((pts[0] - min) / rng) * 38 - 3);
      pts.forEach(function(v, i) {{ ctx.lineTo(i * xStep, 44 - ((v - min) / rng) * 38 - 3); }});
      ctx.lineTo(w, 44); ctx.lineTo(0, 44); ctx.closePath();
      ctx.fillStyle = chgPct >= 0 ? 'rgba(27,111,74,.25)' : 'rgba(184,50,50,.25)';
      ctx.fill();
      // Line
      ctx.beginPath();
      ctx.moveTo(0, 44 - ((pts[0] - min) / rng) * 38 - 3);
      pts.forEach(function(v, i) {{ ctx.lineTo(i * xStep, 44 - ((v - min) / rng) * 38 - 3); }});
      ctx.strokeStyle = chgPct >= 0 ? '#1B6F4A' : '#B83232';
      ctx.lineWidth = 1.5;
      ctx.stroke();
    }} else {{
      document.getElementById('ql-sparkline').style.display = 'none';
    }}

    // Metrics
    var metrics = [
      ['P/E Fwd', live.pe_fwd ? live.pe_fwd.toFixed(1) + 'x' : '—'],
      ['Mkt Cap', live.market_cap_b ? '$' + live.market_cap_b.toFixed(1) + 'B' : '—'],
      ['Beta',    live.beta ? live.beta.toFixed(2) : '—'],
      ['Short %', live.short_float ? live.short_float.toFixed(1) + '%' : '—'],
      ['52W High','$' + (live.w52_high ? live.w52_high.toFixed(2) : '—')],
      ['52W Low', '$' + (live.w52_low  ? live.w52_low.toFixed(2)  : '—')],
      ['Target',  live.analyst_target ? '$' + live.analyst_target.toFixed(2) : '—'],
      ['Analysts', live.num_analysts ? live.num_analysts + ' · ' + (live.analyst_rating || '') : '—'],
    ];
    var metricGrid = document.getElementById('ql-metrics');
    metricGrid.style.gridTemplateColumns = 'repeat(4,1fr)';
    metricGrid.innerHTML = metrics.map(function(m) {{
      return '<div style="text-align:center;background:#241f18;border-radius:4px;padding:10px 6px">'
        + '<p style="font-size:9px;color:#AAA;text-transform:uppercase;letter-spacing:.8px;margin-bottom:4px">' + m[0] + '</p>'
        + '<p style="font-size:14px;font-weight:400;color:#c8b487">' + m[1] + '</p></div>';
    }}).join('');

    // Canyon signal
    var sig = d.canyon_signal;
    if (sig && sig.alpha_score !== null) {{
      var pct  = Math.min(100, Math.max(0, (sig.alpha_score + 3) / 6 * 100));
      var scolor = sig.alpha_score > 1 ? '#1B6F4A' : (sig.alpha_score < -1 ? '#B83232' : '#c8b487');
      document.getElementById('ql-signals').innerHTML =
        '<div style="background:#241f18;border-radius:4px;padding:12px 14px;border-left:3px solid #3a3128">'
        + '<p style="font-size:10px;text-transform:uppercase;letter-spacing:1px;color:#888;margin-bottom:6px">Canyon Alpha Signal</p>'
        + '<div style="display:flex;align-items:center;gap:12px">'
        + '<p style="font-size:22px;font-weight:400;color:' + scolor + '">' + sig.alpha_score.toFixed(2) + '</p>'
        + '<div style="flex:1"><div style="height:6px;background:#241f18;border-radius:3px"><div style="height:100%;width:' + pct + '%;background:' + scolor + ';border-radius:3px"></div></div>'
        + '<p style="font-size:10px;color:#AAA;margin-top:3px">Rank #' + sig.rank + ' of ' + sig.total_stocks + ' stocks</p></div></div></div>';
    }}

    // DCF
    var dcf = d.dcf;
    if (dcf && dcf.iv_per_share) {{
      var upside = dcf.upside_pct ? parseFloat(dcf.upside_pct) : null;
      var ucolor = upside > 0 ? '#1B6F4A' : '#B83232';
      var usign  = upside > 0 ? '+' : '';
      document.getElementById('ql-dcf').innerHTML =
        '<div style="background:#FDF8EE;border-radius:4px;padding:12px 14px;border-left:3px solid #c8b487">'
        + '<p style="font-size:10px;text-transform:uppercase;letter-spacing:1px;color:#c8b487;margin-bottom:8px">Damodaran DCF Valuation</p>'
        + '<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:8px">'
        + _qlMetric('Intrinsic Value', '$' + parseFloat(dcf.iv_per_share).toFixed(2))
        + _qlMetric('Upside/Downside', (upside !== null ? usign + upside.toFixed(1) + '%' : '—'), ucolor)
        + _qlMetric('WACC', dcf.wacc ? (parseFloat(dcf.wacc)*100).toFixed(1) + '%' : '—')
        + _qlMetric('ROIC', dcf.roic ? (parseFloat(dcf.roic)*100).toFixed(1) + '%' : '—')
        + _qlMetric('EVA ($M)', dcf.eva_m ? '$' + parseFloat(dcf.eva_m).toFixed(0) + 'M' : '—')
        + _qlMetric('Lifecycle Stage', dcf.stage || '—')
        + _qlMetric('PVGO %', dcf.pvgo_pct ? parseFloat(dcf.pvgo_pct).toFixed(1) + '%' : '—')
        + _qlMetric('Rev Growth', dcf.rev_growth_1y ? (parseFloat(dcf.rev_growth_1y)*100).toFixed(1) + '%' : '—')
        + '</div></div>';
    }}

    // Short
    var sh = d.short;
    if (sh && sh.score) {{
      var urgColor = sh.urgency && sh.urgency.indexOf('NOW') !== -1 ? '#B83232' : '#c8b487';
      document.getElementById('ql-short').innerHTML =
        '<div style="background:#FEF2F2;border-radius:4px;padding:12px 14px;border-left:3px solid #B83232">'
        + '<p style="font-size:10px;text-transform:uppercase;letter-spacing:1px;color:#B83232;margin-bottom:8px">Short Scanner</p>'
        + '<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:8px">'
        + _qlMetric('Short Score', parseFloat(sh.score).toFixed(0) + '/100', urgColor)
        + _qlMetric('RSI', sh.rsi ? parseFloat(sh.rsi).toFixed(1) : '—')
        + _qlMetric('Entry Range', sh.entry_low && sh.entry_high ? '$' + sh.entry_low + '–$' + sh.entry_high : '—')
        + _qlMetric('Stop Loss', sh.stop_loss ? '$' + sh.stop_loss : '—')
        + _qlMetric('Target 1', sh.target_1 ? '$' + sh.target_1 : '—')
        + _qlMetric('Target 2', sh.target_2 ? '$' + sh.target_2 : '—')
        + _qlMetric('R/R', sh.rr_1 ? sh.rr_1 + 'x' : '—')
        + _qlMetric('Urgency', sh.urgency || '—', urgColor)
        + '</div>'
        + (sh.signals ? '<p style="font-size:11px;color:#B83232;margin-top:8px">Signals: ' + sh.signals + '</p>' : '')
        + '</div>';
    }}

    // AI summary
    var ai = d.earnings_ai;
    if (ai && ai.summary && ai.summary.length > 20) {{
      document.getElementById('ql-ai').innerHTML =
        '<div style="background:#241f18;border-radius:4px;padding:12px 14px">'
        + '<p style="font-size:10px;text-transform:uppercase;letter-spacing:1px;color:#888;margin-bottom:8px">'
        + 'AI Earnings Analysis' + (ai.has_ai ? ' · <span style=\\"color:#1B6F4A\\">AI</span>' : '') + '</p>'
        + '<div style="font-size:12px;color:#333;line-height:1.7;white-space:pre-wrap">' + _esc(ai.summary) + '</div>'
        + '</div>';
    }}

    // Update ask-AI button
    document.getElementById('ql-ask-btn').onclick = function() {{
      close();
      showTab('chat');
      var chatTicker = document.getElementById('chat-ticker');
      if (chatTicker) chatTicker.value = _cur;
      setTimeout(function() {{
        var inp = document.getElementById('chat-input');
        if (inp) {{
          inp.value = 'Give me a comprehensive analysis of ' + _cur + ' — signals, valuation, risks, and your recommendation.';
          inp.focus();
        }}
      }}, 200);
    }};
  }}

  function _qlMetric(label, val, color) {{
    return '<div style="text-align:center;background:#fff;border-radius:3px;padding:8px 4px">'
      + '<p style="font-size:9px;color:#AAA;text-transform:uppercase;letter-spacing:.6px;margin-bottom:3px">' + label + '</p>'
      + '<p style="font-size:12px;font-weight:400;color:' + (color || '#3a3128') + '">' + val + '</p></div>';
  }}

  function _esc(s) {{
    return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
  }}

  function addToWatchlist() {{
    if (!_cur) return;
    fetch('/api/watchlist', {{
      method:'POST',
      headers:{{'Content-Type':'application/json'}},
      body:JSON.stringify({{ticker:_cur, note:''}})
    }}).then(function() {{
      alert(_cur + ' added to watchlist!');
    }});
  }}

  // Make all ticker-looking text clickable
  function bindTickers(root) {{
    root = root || document;
    root.querySelectorAll('[data-ticker]').forEach(function(el) {{
      if (el.dataset.qlBound) return;
      el.dataset.qlBound = '1';
      el.style.cursor = 'pointer';
      el.classList.add('canyon-ticker-link');
      el.addEventListener('click', function(e) {{
        e.stopPropagation();
        open(el.dataset.ticker);
      }});
    }});
  }}

  document.addEventListener('DOMContentLoaded', function() {{ bindTickers(); }});

  return {{ open:open, close:close, askAI: function() {{}}, addToWatchlist:addToWatchlist, bindTickers:bindTickers }};
}})();

// Keyboard: ESC closes modal
document.addEventListener('keydown', function(e) {{
  if (e.key === 'Escape') canyonQL.close();
}});
</script>"""


def _build_chat_section() -> str:
    return """
  <style>
    #chat-wrap {{ display:flex; gap:20px; align-items:stretch }}
    #chat-sidebar {{ width:260px; flex-shrink:0 }}
    #chat-main {{ flex:1; min-width:0; display:flex; flex-direction:column }}
    #chat-messages {{
      flex:1; background:#241f18; border:1px solid #241f18; border-radius:8px;
      padding:16px; min-height:400px; max-height:540px; overflow-y:auto;
      display:flex; flex-direction:column; gap:10px;
    }}
    #chat-form {{ display:flex; gap:8px; margin-top:10px }}
    #chat-input {{
      flex:1; border:1px solid #241f18; border-radius:6px;
      padding:11px 14px; font-size:13px; outline:none; background:#fff;
      transition:border-color .15s;
    }}
    #chat-input:focus {{ border-color:#c8b487 }}
    #chat-send {{
      background:#2a2418; color:#fff; border:none; padding:11px 22px;
      border-radius:6px; cursor:pointer; font-size:13px; font-weight:400;
      white-space:nowrap; transition:background .15s;
    }}
    #chat-send:hover {{ background:#3a3128 }}
    #chat-send:disabled {{ background:#CCC; cursor:default }}
    #chat-clear {{
      background:#fff; color:#999; border:1px solid #241f18;
      padding:11px 14px; border-radius:6px; cursor:pointer; font-size:12px;
    }}
    .chat-chips {{ display:flex; flex-wrap:wrap; gap:6px; margin-bottom:14px }}
    @media(max-width:768px) {{
      #chat-wrap {{ flex-direction:column }}
      #chat-sidebar {{ width:100% }}
      #chat-messages {{ min-height:280px; max-height:380px }}
    }}
  </style>

  <div class="container">
    <p class="eyebrow">Canyon AI · Powered by Claude</p>
    <h2 class="section-head">Research Assistant — Ask anything in plain English</h2>
    <div class="rule"></div>
    <p style="color:#666;font-size:13px;margin-bottom:20px">
      Canyon AI knows your signals, alpha scores, DCF valuations, short setups, macro regime, and earnings analysis.
      Ask about any S&amp;P 500 stock, your portfolio, or the market. Set <code>ANTHROPIC_API_KEY</code> to activate.</p>

    <!-- Quick suggestion chips -->
    <div class="chat-chips">
      <button class="chip-btn" onclick="canyonChat.setQ(this.dataset.q)"
        data-q="What are today's top 5 long signals and what makes them stand out?">📈 Top longs today</button>
      <button class="chip-btn" onclick="canyonChat.setQ(this.dataset.q)"
        data-q="Which stocks are most overbought and best short candidates right now?">📉 Best shorts</button>
      <button class="chip-btn" onclick="canyonChat.setQ(this.dataset.q)"
        data-q="What is the current market regime and 4-week bear probability? Should I be defensive?">🏛️ Market regime</button>
      <button class="chip-btn" onclick="canyonChat.setQ(this.dataset.q)"
        data-q="Give me the most undervalued stocks by DCF — highest upside to intrinsic value">💎 DCF undervalued</button>
      <button class="chip-btn" onclick="canyonChat.setQ(this.dataset.q)"
        data-q="What sectors are showing the strongest rotation momentum right now?">🔄 Sector rotation</button>
      <button class="chip-btn" onclick="canyonChat.setQ(this.dataset.q)"
        data-q="What are the main risks to my portfolio this week — macro, earnings, signals?">⚠️ Portfolio risks</button>
      <button class="chip-btn" onclick="canyonChat.setQ(this.dataset.q)"
        data-q="Which stocks have the best earnings quality — highest Piotroski score, low accruals?">📊 Earnings quality</button>
      <button class="chip-btn" onclick="canyonChat.setQ(this.dataset.q)"
        data-q="Compare NVDA, AMD, and INTC on valuation, growth, and Canyon signal">🔬 Compare stocks</button>
    </div>

    <div id="chat-wrap">
      <!-- Sidebar -->
      <div id="chat-sidebar">
        <div class="method-card acc" style="margin-bottom:14px">
          <p style="font-size:10px;text-transform:uppercase;letter-spacing:1.2px;color:#888;margin-bottom:8px;font-weight:400">Focus on Ticker</p>
          <div style="position:relative">
            <input id="chat-ticker" type="text" placeholder="e.g. NVDA"
              style="width:100%;border:1px solid #241f18;padding:8px 36px 8px 10px;border-radius:4px;font-size:14px;font-weight:400;background:#fff;letter-spacing:.5px"
              oninput="this.value=this.value.toUpperCase().replace(/[^A-Z]/g,'')" maxlength="6">
            <button onclick="var t=document.getElementById('chat-ticker').value;if(t)canyonQL.open(t)"
              style="position:absolute;right:6px;top:50%;transform:translateY(-50%);background:none;border:none;cursor:pointer;font-size:14px;color:#c8b487" title="Quick look">🔍</button>
          </div>
          <p style="font-size:10px;color:#BBB;margin-top:5px">Canyon loads full signal/DCF/short/AI data for this ticker as context.</p>
        </div>

        <div class="method-card acc" style="margin-bottom:14px">
          <p style="font-size:10px;text-transform:uppercase;letter-spacing:1.2px;color:#888;margin-bottom:8px;font-weight:400">Conversation History</p>
          <p id="chat-turn-count" style="font-size:12px;color:#666">0 messages</p>
          <button onclick="canyonChat.clear()"
            style="margin-top:8px;width:100%;background:#fff;border:1px solid #241f18;padding:7px;border-radius:4px;cursor:pointer;font-size:11px;color:#888">
            🗑 Clear conversation</button>
        </div>

        <div class="method-card acc">
          <p style="font-size:10px;text-transform:uppercase;letter-spacing:1.2px;color:#888;margin-bottom:8px;font-weight:400">Data Available in Context</p>
          <div style="display:flex;flex-direction:column;gap:4px;font-size:11px;color:#555">
            <p>✓ Alpha scores (all S&amp;P 500)</p>
            <p>✓ HMM regime + macro outlook</p>
            <p>✓ DCF valuations (200 stocks)</p>
            <p>✓ Short scanner scores</p>
            <p>✓ 13F crowding + sector rotation</p>
            <p id="ctx-ai-status">— AI summaries (run step 376)</p>
          </div>
        </div>
      </div>

      <!-- Main chat -->
      <div id="chat-main">
        <div id="chat-messages">
          <div id="chat-placeholder" style="display:flex;flex-direction:column;align-items:center;
            justify-content:center;height:100%;gap:12px;color:#AAA;text-align:center">
            <p style="font-size:32px">💬</p>
            <p style="font-size:13px;font-weight:400;color:#888">Ask Canyon anything</p>
            <p style="font-size:11px">Use the chips above or type a question below.<br>
               Click any ticker in the dashboard to open Quick Look, then ask AI from there.</p>
          </div>
        </div>

        <div id="chat-form">
          <input id="chat-input" type="text"
            placeholder="e.g. Why is NVDA a top signal? What does the DCF say about MSFT? Are we in BULL or BEAR?"
            onkeydown="if(event.key==='Enter'&&!event.shiftKey){{event.preventDefault();canyonChat.send()}}">
          <button id="chat-send" onclick="canyonChat.send()">Send ↵</button>
          <button id="chat-clear" onclick="canyonChat.clear()">✕</button>
        </div>
        <p style="font-size:10px;color:#BBB;margin-top:5px;text-align:right">
          Claude Haiku · Canyon data as context · Verify before acting on AI output</p>
      </div>
    </div>
  </div>

  <script>
  var canyonChat = (function() {{
    var history = [];
    var msgs    = document.getElementById('chat-messages');
    var ph      = document.getElementById('chat-placeholder');

    function _turnCount() {{
      var el = document.getElementById('chat-turn-count');
      if (el) el.textContent = history.length + ' messages';
    }}

    function _renderText(text) {{
      // Minimal markdown: **bold**, `code`, newlines
      return text
        .replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;')
        .replace(/\\*\\*(.+?)\\*\\*/g,'<strong>$1</strong>')
        .replace(/`([^`]+)`/g,'<code style="background:#241f18;padding:1px 4px;border-radius:3px;font-family:monospace;font-size:12px">$1</code>')
        .replace(/\\n/g,'<br>');
    }}

    function _addMsg(role, html, isHtml) {{
      if (ph) ph.style.display = 'none';
      var div = document.createElement('div');
      div.className = role === 'user' ? 'chat-msg-user' : 'chat-msg-ai';
      if (isHtml) {{ div.innerHTML = html; }} else {{ div.textContent = html; }}
      msgs.appendChild(div);
      msgs.scrollTop = msgs.scrollHeight;
      return div;
    }}

    function _addThinking() {{
      if (ph) ph.style.display = 'none';
      var div = document.createElement('div');
      div.className = 'chat-msg-ai chat-msg-thinking';
      div.innerHTML = '<div class="chat-dot"></div><div class="chat-dot"></div><div class="chat-dot"></div>';
      msgs.appendChild(div);
      msgs.scrollTop = msgs.scrollHeight;
      return div;
    }}

    function send() {{
      var inputEl  = document.getElementById('chat-input');
      var tickerEl = document.getElementById('chat-ticker');
      var sendBtn  = document.getElementById('chat-send');
      var q = (inputEl.value || '').trim();
      if (!q) return;

      inputEl.value = '';
      sendBtn.disabled = true;

      _addMsg('user', q);
      history.push({{role:'user', content:q}});
      _turnCount();

      var thinking = _addThinking();

      fetch('/api/chat', {{
        method: 'POST',
        headers: {{'Content-Type': 'application/json'}},
        body: JSON.stringify({{
          question: q,
          ticker:   (tickerEl ? tickerEl.value.trim() : ''),
          history:  history.slice(-10, -1)
        }})
      }})
      .then(function(r) {{ return r.json(); }})
      .then(function(data) {{
        var answer = data.answer || data.error || 'No response';
        msgs.removeChild(thinking);
        var rendered = _renderText(answer);
        _addMsg('assistant', rendered, true);
        history.push({{role:'assistant', content:answer}});
        _turnCount();
      }})
      .catch(function(err) {{
        msgs.removeChild(thinking);
        _addMsg('assistant', 'Connection error: ' + err.message);
      }})
      .finally(function() {{
        sendBtn.disabled = false;
        inputEl.focus();
      }});
    }}

    function setQ(q) {{
      var inp = document.getElementById('chat-input');
      if (inp) {{ inp.value = q; inp.focus(); }}
    }}

    function clear() {{
      history = [];
      msgs.innerHTML = '';
      if (ph) {{ ph.style.display = 'flex'; msgs.appendChild(ph); }}
      _turnCount();
    }}

    return {{ send:send, setQ:setQ, clear:clear }};
  }})();
  </script>"""


def _build_earnings_ai_section(df: "pd.DataFrame") -> str:
    if df is None or df.empty:
        return """
  <div style="text-align:center;padding:40px 20px;background:#241f18;border-radius:6px;border:1px dashed #241f18">
    <p style="font-size:32px;margin-bottom:12px">📋</p>
    <p style="font-size:14px;font-weight:400;color:#555;margin-bottom:6px">Earnings AI summaries not yet generated</p>
    <p style="font-size:12px;color:#AAA;margin-bottom:12px">
      Run: <code style="background:#241f18;padding:2px 6px;border-radius:3px">.venv/bin/python step_earnings_ai.py</code><br>
      Requires <code>ANTHROPIC_API_KEY</code> env var for AI narratives. Runs incrementally — skips stocks already done today.
    </p>
    <p style="font-size:11px;color:#BBB">Each stock generates a 5-section Damodaran-style analysis: Business Quality, Earnings Quality, Balance Sheet, Growth Drivers, Valuation Context.</p>
  </div>"""

    today_str = __import__('datetime').date.today().isoformat()
    fresh = df[df.get("as_of", pd.Series(dtype=str)) == today_str] if "as_of" in df.columns else df
    if fresh.empty:
        fresh = df

    def pct(val, green_if_positive=True):
        try:
            v = float(val)
            color = ("#1B6F4A" if v >= 0 else "#B83232") if green_if_positive else "#3a3128"
            sign = "+" if v >= 0 else ""
            return f'<span style="color:{color}">{sign}{v:.1%}</span>'
        except: return "—"

    def num(val, dec=1, prefix="", suffix=""):
        try: return f"{prefix}{float(val):.{dec}f}{suffix}"
        except: return "—"

    def trend_bar(val, lo=-0.5, hi=0.5, width=60):
        try:
            v = float(val)
            pct_pos = min(100, max(0, (v - lo) / (hi - lo) * 100))
            color = "#1B6F4A" if v >= 0 else "#B83232"
            return (f'<div style="display:flex;align-items:center;gap:6px">'
                    f'<div style="width:{width}px;height:5px;background:#241f18;border-radius:3px;flex-shrink:0">'
                    f'<div style="width:{pct_pos:.0f}%;height:100%;background:{color};border-radius:3px"></div></div>'
                    f'<span style="font-size:11px;color:{color};font-weight:400">{pct(val, True)}</span></div>')
        except: return "—"

    rating_colors = {"buy":"#1B6F4A","strong buy":"#1B6F4A","strong_buy":"#1B6F4A",
                     "hold":"#c8b487","neutral":"#c8b487",
                     "sell":"#B83232","underperform":"#B83232"}

    # Group by sector for sector rank + peer percentile computation
    sectors_seen = {}
    for _, r in fresh.iterrows():
        sec = str(r.get("sector", "Other"))
        score = float(r.get("alpha_score", 0) or 0)
        sectors_seen.setdefault(sec, []).append(score)
    sector_avg = {s: sum(v)/len(v) for s, v in sectors_seen.items()}
    sector_rank_order = sorted(sector_avg, key=sector_avg.get, reverse=True)

    # Peer percentile: for each metric, compute cross-sectional stats per sector
    PEER_METRICS = ["pe_fwd", "rev_growth_yoy", "op_margin", "roe", "ev_ebitda", "fcf_yield"]
    sector_stats: dict[str, dict] = {}
    for sec, grp in fresh.groupby("sector", observed=True):
        sector_stats[sec] = {}
        for col in PEER_METRICS:
            if col in grp.columns:
                vals = grp[col].dropna()
                if len(vals) > 1:
                    sector_stats[sec][col] = {"med": float(vals.median()),
                                               "lo": float(vals.quantile(0.1)),
                                               "hi": float(vals.quantile(0.9))}

    # Also load Canyon signal breakdown from alpha_scores.csv
    _sig_map: dict[str, dict] = {}
    try:
        _ascore_path = __import__('pathlib').Path(__file__).parent / "alpha_scores.csv"
        _adf = pd.read_csv(_ascore_path) if _ascore_path.exists() else pd.DataFrame()
        if not _adf.empty and "ticker" in _adf.columns:
            _sig_cols = [c for c in _adf.columns if c.startswith("sig_")]
            for _, row in _adf.iterrows():
                _sig_map[str(row["ticker"])] = {c: row[c] for c in _sig_cols if pd.notna(row.get(c))}
    except Exception:
        pass

    def peer_percentile_bar(val, sec, col):
        """Mini horizontal bar showing ticker's percentile vs sector peers."""
        try:
            v   = float(val)
            st  = sector_stats.get(sec, {}).get(col, {})
            if not st:
                return ""
            lo, hi = st["lo"], st["hi"]
            med    = st["med"]
            pct_pos = min(100, max(0, (v - lo) / max(hi - lo, 1e-9) * 100))
            med_pos = min(100, max(0, (med - lo) / max(hi - lo, 1e-9) * 100))
            # higher = better for growth/margin/roe/fcf; lower = better for pe/ev
            higher_better = col in ("rev_growth_yoy", "op_margin", "roe", "fcf_yield")
            color = "#1B6F4A" if (higher_better and v > med) or (not higher_better and v < med and v > 0) else "#B83232"
            return (f'<div title="vs {sec} sector median {med:.2f}" '
                    f'style="position:relative;height:4px;background:#241f18;border-radius:2px;margin-top:4px">'
                    f'<div style="position:absolute;left:0;width:{pct_pos:.0f}%;height:100%;'
                    f'background:{color};border-radius:2px;opacity:0.85"></div>'
                    f'<div style="position:absolute;left:{med_pos:.0f}%;width:2px;height:6px;top:-1px;'
                    f'background:#888;border-radius:1px"></div></div>')
        except Exception:
            return ""

    def signal_breakdown_html(ticker):
        """Top 4 driving signals for this ticker from alpha_scores.csv."""
        sigs = _sig_map.get(ticker, {})
        if not sigs:
            return ""
        SIG_LABELS = {
            "sig_regime_ml": "Regime ML", "sig_quality": "Quality",
            "sig_revision": "Revision", "sig_surprise": "Earnings Surprise",
            "sig_sentiment": "Sentiment", "sig_squeeze": "Short Squeeze",
            "sig_insider": "Insider", "sig_options": "Options",
            "sig_ml_ensemble": "ML Ensemble", "sig_momentum": "Momentum",
            "sig_accruals": "Accruals", "sig_piotroski": "Piotroski",
        }
        ranked = sorted(sigs.items(), key=lambda x: abs(float(x[1] or 0) - 50), reverse=True)[:4]
        bars = ""
        for sig, val in ranked:
            try:
                v = float(val)
                lbl = SIG_LABELS.get(sig, sig.replace("sig_", "").replace("_", " ").title())
                pct_w = min(100, max(0, v))
                color = "#1B6F4A" if v >= 65 else ("#B83232" if v <= 35 else "#c8b487")
                bars += (f'<div style="display:flex;align-items:center;gap:6px;margin-bottom:3px">'
                         f'<span style="width:90px;font-size:9px;color:#888;flex-shrink:0">{lbl}</span>'
                         f'<div style="flex:1;height:5px;background:#241f18;border-radius:2px">'
                         f'<div style="width:{pct_w:.0f}%;height:100%;background:{color};border-radius:2px"></div></div>'
                         f'<span style="font-size:9px;color:{color};width:28px;text-align:right">{v:.0f}</span></div>')
            except Exception:
                pass
        if not bars:
            return ""
        return (f'<details style="margin-top:8px"><summary style="font-size:10px;color:#888;cursor:pointer;'
                f'padding:4px 0;border-top:1px solid #241f18">📊 Signal breakdown (top 4 drivers)</summary>'
                f'<div style="padding:8px 0">{bars}</div></details>')

    # Sort by alpha score descending for display
    try:
        display_df = fresh.sort_values("alpha_score", ascending=False)
    except Exception:
        display_df = fresh

    rows = []
    for _, r in display_df.head(60).iterrows():
        ticker  = str(r.get("ticker", ""))
        sector  = str(r.get("sector", "Other"))
        summary = str(r.get("summary", "")).strip()
        has_ai  = bool(r.get("has_ai_summary", False))
        rating  = str(r.get("analyst_rating", "")).lower()
        rc      = rating_colors.get(rating, "#888")

        # Sector rank
        sec_rank = sector_rank_order.index(sector) + 1 if sector in sector_rank_order else "?"
        sec_total = len(sector_rank_order)

        # Alpha score bar
        try:
            ascore = float(r.get("alpha_score", 0) or 0)
            ascore_pct = min(100, max(0, (ascore + 3) / 6 * 100))
            ascore_color = "#1B6F4A" if ascore > 1 else ("#B83232" if ascore < -1 else "#c8b487")
        except Exception:
            ascore, ascore_pct, ascore_color = 0, 50, "#888"

        # EPS beat streak badge
        beats = r.get("beats_last_4q", None)
        beat_badge = ""
        try:
            b = int(beats)
            beat_badge = (f'<span style="background:#241f18;color:#1B6F4A;font-size:9px;padding:1px 5px;border-radius:3px;font-weight:400">'
                          f'{"★" * b} {b}/4 beats</span>') if b >= 3 else ""
        except Exception:
            pass

        # Analyst target upside
        target_upside = ""
        try:
            t = float(r.get("analyst_target", 0) or 0)
            p = float(r.get("price", 0) or 0)
            if t > 0 and p > 0:
                up = (t - p) / p * 100
                up_c = "#1B6F4A" if up > 0 else "#B83232"
                target_upside = f'<span style="font-size:10px;color:{up_c};font-weight:400">{"↑" if up>0 else "↓"} {abs(up):.0f}% to target</span>'
        except Exception:
            pass

        # Summary sections: split by known section headers
        summary_html = ""
        if summary and summary.lower() not in ("nan", "", "none"):
            # Light markdown: section headers like "**1. Business Quality**"
            summary_clean = (summary
                .replace("**1.", "<strong>1.")
                .replace("**2.", "<strong>2.")
                .replace("**3.", "<strong>3.")
                .replace("**4.", "<strong>4.")
                .replace("**5.", "<strong>5.")
                .replace("**\n", "</strong>\n")
                .replace("**:", "</strong>:")
                .replace("**", ""))
            summary_html = (
                f'<details style="margin-top:10px">'
                f'<summary style="font-size:11px;color:#c8b487;font-weight:400;cursor:pointer;'
                f'padding:6px 0;border-top:1px solid #241f18">{"🤖 AI Analysis" if has_ai else "📋 Analysis"} — click to expand</summary>'
                f'<div style="font-size:12px;color:#444;line-height:1.7;padding:10px 0;white-space:pre-wrap;border-left:2px solid {"#241f18" if has_ai else "#241f18"};padding-left:12px;margin-top:8px">'
                f'{summary_clean}</div></details>')

        rows.append(f"""
  <div class="method-card acc earn-card" data-sector="{sector}" style="margin-bottom:14px;border-left:3px solid {ascore_color}">
    <div style="display:flex;justify-content:space-between;align-items:flex-start;flex-wrap:wrap;gap:8px;margin-bottom:10px">
      <div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap">
        <span class="canyon-ticker-link" data-ticker="{ticker}" onclick="canyonQL.open('{ticker}')"
          style="font-family:'Playfair Display',serif;font-size:20px;font-weight:400;color:#c8b487">{ticker}</span>
        <span style="font-size:11px;color:#888">{sector} · #{sec_rank}/{sec_total} sector rank</span>
        {'<span style="font-size:9px;background:#241f18;color:#1B6F4A;padding:2px 6px;border-radius:10px;font-weight:400">🤖 AI</span>' if has_ai else ''}
        {beat_badge}
      </div>
      <div style="text-align:right">
        <span style="font-size:16px;font-weight:400;color:#c8b487">${num(r.get('price'), 2)}</span>
        <span style="display:block;font-size:10px;color:{rc};font-weight:400;text-transform:uppercase;margin-top:2px">{rating}</span>
        {target_upside}
      </div>
    </div>

    <!-- Metric grid: 8 key metrics -->
    <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin-bottom:10px">
      <div style="background:#241f18;border-radius:4px;padding:7px 8px">
        <p style="font-size:9px;color:#AAA;text-transform:uppercase;letter-spacing:.7px;margin-bottom:2px">P/E Fwd</p>
        <p style="font-size:13px;font-weight:400;color:#c8b487">{num(r.get('pe_fwd'))}x</p>
        {peer_percentile_bar(r.get('pe_fwd'), sector, 'pe_fwd')}
      </div>
      <div style="background:#241f18;border-radius:4px;padding:7px 8px">
        <p style="font-size:9px;color:#AAA;text-transform:uppercase;letter-spacing:.7px;margin-bottom:2px">Rev Growth YoY</p>
        {trend_bar(r.get('rev_growth_yoy'), -0.3, 0.5, 50)}
        {peer_percentile_bar(r.get('rev_growth_yoy'), sector, 'rev_growth_yoy')}
      </div>
      <div style="background:#241f18;border-radius:4px;padding:7px 8px">
        <p style="font-size:9px;color:#AAA;text-transform:uppercase;letter-spacing:.7px;margin-bottom:2px">Op Margin</p>
        <p style="font-size:13px;font-weight:400;color:#c8b487">{pct(r.get('op_margin'), False)}</p>
        {peer_percentile_bar(r.get('op_margin'), sector, 'op_margin')}
      </div>
      <div style="background:#241f18;border-radius:4px;padding:7px 8px">
        <p style="font-size:9px;color:#AAA;text-transform:uppercase;letter-spacing:.7px;margin-bottom:2px">ROE</p>
        <p style="font-size:13px;font-weight:400;color:#c8b487">{pct(r.get('roe'), False)}</p>
        {peer_percentile_bar(r.get('roe'), sector, 'roe')}
      </div>
      <div style="background:#241f18;border-radius:4px;padding:7px 8px">
        <p style="font-size:9px;color:#AAA;text-transform:uppercase;letter-spacing:.7px;margin-bottom:2px">Debt/Equity</p>
        <p style="font-size:13px;font-weight:400;color:#c8b487">{num(r.get('debt_equity'))}x</p>
      </div>
      <div style="background:#241f18;border-radius:4px;padding:7px 8px">
        <p style="font-size:9px;color:#AAA;text-transform:uppercase;letter-spacing:.7px;margin-bottom:2px">EV/EBITDA</p>
        <p style="font-size:13px;font-weight:400;color:#c8b487">{num(r.get('ev_ebitda'))}x</p>
        {peer_percentile_bar(r.get('ev_ebitda'), sector, 'ev_ebitda')}
      </div>
      <div style="background:#241f18;border-radius:4px;padding:7px 8px">
        <p style="font-size:9px;color:#AAA;text-transform:uppercase;letter-spacing:.7px;margin-bottom:2px">Canyon Alpha</p>
        <div style="display:flex;align-items:center;gap:5px">
          <div style="flex:1;height:5px;background:#241f18;border-radius:3px">
            <div style="width:{ascore_pct:.0f}%;height:100%;background:{ascore_color};border-radius:3px"></div>
          </div>
          <span style="font-size:11px;font-weight:400;color:{ascore_color}">{ascore:+.2f}</span>
        </div>
      </div>
      <div style="background:#241f18;border-radius:4px;padding:7px 8px">
        <p style="font-size:9px;color:#AAA;text-transform:uppercase;letter-spacing:.7px;margin-bottom:2px">FCF Yield</p>
        <p style="font-size:13px;font-weight:400;color:#c8b487">{pct(r.get('fcf_yield'), True)}</p>
        {peer_percentile_bar(r.get('fcf_yield'), sector, 'fcf_yield')}
      </div>
    </div>

    {signal_breakdown_html(ticker)}
    {summary_html}
  </div>""")

    ai_count = int(fresh["has_ai_summary"].sum()) if "has_ai_summary" in fresh.columns else 0
    sectors_list = "".join(f'<option value="{s}">{s}</option>' for s in sorted(sectors_seen.keys()))

    # Honest banner: when the AI narrative layer is off (no ANTHROPIC_API_KEY),
    # say so plainly. The fundamentals below are still live/real (yfinance).
    ai_off_banner = "" if ai_count > 0 else """
  <div style="background:#1b1710;border:1px solid #43391f;border-left:3px solid #c8b487;border-radius:6px;padding:12px 16px;margin-bottom:16px">
    <p style="font-size:12.5px;color:#c8b487;font-weight:400;letter-spacing:.02em;margin-bottom:3px">AI narrative layer — not enabled</p>
    <p style="font-size:12px;color:#b0a68f;line-height:1.5">The fundamentals below are <strong style="color:#f0e9da">live &amp; real</strong> (yfinance). The AI-written analysis is off because <code style="background:#241f18;padding:1px 5px;border-radius:3px;color:#cdbd8f">ANTHROPIC_API_KEY</code> is not set — no narrative is being shown or faked.</p>
  </div>"""

    header = ai_off_banner + f"""
  <div style="display:flex;align-items:center;gap:12px;flex-wrap:wrap;margin-bottom:16px">
    <p style="font-size:13px;color:#666;flex:1">
      <strong>{len(fresh)}</strong> stocks · <strong style="color:{'#1B6F4A' if ai_count > 0 else '#AAA'}">{ai_count} with AI narratives</strong>
    </p>
    <input id="earn-search" type="text" placeholder="🔍 Filter ticker…" oninput="earnFilter()"
      style="border:1px solid #241f18;padding:7px 10px;border-radius:4px;font-size:12px;width:150px">
    <select id="earn-sector" onchange="earnFilter()"
      style="border:1px solid #241f18;padding:7px 10px;border-radius:4px;font-size:12px;background:#fff">
      <option value="">All sectors</option>
      {sectors_list}
    </select>
  </div>
  <div id="earn-cards">"""

    footer = """</div>
  <script>
  function earnFilter() {{
    var q  = (document.getElementById('earn-search').value || '').toUpperCase();
    var sec = document.getElementById('earn-sector').value;
    document.querySelectorAll('#earn-cards .earn-card').forEach(function(el) {{
      var t = (el.querySelector('.canyon-ticker-link') || el).textContent.toUpperCase();
      var s = el.dataset.sector || '';
      el.style.display = (!q || t.includes(q)) && (!sec || s === sec) ? '' : 'none';
    }});
  }}
  </script>"""

    return header + "\n".join(rows) + footer


def _build_regime_gauge(hmm: str, bear_prob: float | None) -> str:
    """SVG arc gauge showing regime state + bear probability needle."""
    prob = bear_prob if bear_prob is not None else 0.0
    # needle angle: 0% = -150deg (far left), 100% = +150deg (far right)
    angle = -150 + prob * 3.0  # degrees
    rad = angle * 3.14159 / 180
    import math
    nx = 100 + 72 * math.sin(rad)
    ny = 100 - 72 * math.cos(rad)
    if hmm == "BULL":
        zone_color, zone_text, zone_bg = "#1B6F4A", "BULL — Long bias", "#F0FDF4"
    elif hmm == "BEAR":
        zone_color, zone_text, zone_bg = "#B83232", "BEAR — Defensive", "#FEF2F2"
    else:
        zone_color, zone_text, zone_bg = "#c8b487", "TRANSITION — Caution", "#FFFBEB"
    needle_color = "#B83232" if prob >= 60 else ("#c8b487" if prob >= 35 else "#1B6F4A")
    return f"""
    <div style="background:{zone_bg};border:1px solid {zone_color}33;border-radius:6px;
      padding:16px 20px;display:flex;align-items:center;gap:24px;margin-bottom:20px;border-top:3px solid {zone_color}">
      <svg viewBox="0 0 200 110" width="160" height="88" style="flex-shrink:0">
        <!-- Background arc zones -->
        <path d="M 28 100 A 72 72 0 0 1 74 30" stroke="#1B6F4A" stroke-width="10" fill="none" stroke-linecap="round" opacity=".35"/>
        <path d="M 74 30 A 72 72 0 0 1 126 30" stroke="#c8b487" stroke-width="10" fill="none" stroke-linecap="round" opacity=".35"/>
        <path d="M 126 30 A 72 72 0 0 1 172 100" stroke="#B83232" stroke-width="10" fill="none" stroke-linecap="round" opacity=".35"/>
        <!-- Labels -->
        <text x="18" y="108" font-size="9" fill="#1B6F4A" font-weight="700">BULL</text>
        <text x="92" y="22" font-size="9" fill="#c8b487" font-weight="700" text-anchor="middle">MID</text>
        <text x="168" y="108" font-size="9" fill="#B83232" font-weight="700" text-anchor="end">BEAR</text>
        <!-- Needle -->
        <line x1="100" y1="100" x2="{nx:.1f}" y2="{ny:.1f}" stroke="{needle_color}" stroke-width="2.5" stroke-linecap="round"/>
        <circle cx="100" cy="100" r="4" fill="{needle_color}"/>
        <!-- Center label -->
        <text x="100" y="96" font-size="14" font-weight="800" fill="{needle_color}" text-anchor="middle">{prob:.0f}%</text>
      </svg>
      <div>
        <p style="font-size:10px;text-transform:uppercase;letter-spacing:1px;color:#888;margin-bottom:4px">Regime Gauge · 4-Week Bear Probability</p>
        <p style="font-size:18px;font-weight:400;color:{zone_color};margin-bottom:2px">{zone_text}</p>
        <p style="font-size:12px;color:#666">Needle shows {prob:.0f}% probability of bear market in next 4 weeks.<br>
        Below 35% = lean long · 35-60% = reduce risk · Above 60% = defensive posture.</p>
      </div>
    </div>"""


def _build_earnings_this_week(earnings_cal: list) -> str:
    """Show earnings announcements in the next 7 days as a prominent row."""
    if not earnings_cal:
        return ""
    from datetime import date
    today = date.today()
    soon = [e for e in earnings_cal
            if isinstance(e.get("days_until"), (int, float)) and 0 <= e["days_until"] <= 7]
    soon = sorted(soon, key=lambda x: x.get("days_until", 99))[:12]
    if not soon:
        return ""
    cards = []
    for e in soon:
        t = e.get("ticker", "")
        d = int(e.get("days_until", 0))
        action = e.get("action", "")
        risk = e.get("risk_flag", "")
        day_text = "TODAY" if d == 0 else ("TOMORROW" if d == 1 else f"in {d}d")
        action_color = "#1B6F4A" if "BUY" in action.upper() else ("#B83232" if "AVOID" in action.upper() else "#888")
        risk_badge = f'<span style="background:#FEE2E2;color:#B83232;font-size:9px;padding:1px 4px;border-radius:3px;font-weight:400">⚠️ HIGH RISK</span>' if risk == "HIGH" else ""
        cards.append(f"""
        <div onclick="canyonQL.open('{t}')" style="cursor:pointer;background:#fff;border:1px solid #241f18;border-radius:4px;
          padding:10px 12px;min-width:100px;flex-shrink:0;border-top:2px solid {'#B83232' if risk=='HIGH' else '#c8b487'};
          transition:box-shadow .15s" onmouseover="this.style.boxShadow='0 2px 8px rgba(0,0,0,.12)'" onmouseout="this.style.boxShadow=''">
          <p style="font-size:9px;color:#c8b487;font-weight:500;text-transform:uppercase;margin-bottom:3px">{day_text}</p>
          <p style="font-size:15px;font-weight:500;color:#c8b487;margin-bottom:2px">{t}</p>
          <p style="font-size:10px;color:{action_color};font-weight:400">{action}</p>
          {risk_badge}
        </div>""")
    return f"""
    <div style="background:#FDF8EE;border:1px solid #241f18;border-radius:6px;padding:14px 16px;margin-bottom:20px">
      <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:10px">
        <p style="font-size:10px;text-transform:uppercase;letter-spacing:1.2px;color:#c8b487;font-weight:400">
          📅 Earnings This Week — next 7 days</p>
        <p style="font-size:10px;color:#BBB">{len(soon)} upcoming</p>
      </div>
      <div style="display:flex;gap:8px;overflow-x:auto;padding-bottom:4px">
        {''.join(cards)}
      </div>
    </div>"""


# ── Famous Holdings Tab ───────────────────────────────────────────────────────

def _build_congressional_section(ct: dict) -> str:
    """Deep congressional trading STOCK Act analysis — committee conflict, timing patterns, trade log."""
    if not ct or not ct.get("members"):
        return """
<div style="border:1px solid #241f18;border-radius:8px;padding:20px;background:#fff;margin-bottom:24px">
  <p style="font-size:13px;color:#888">Run <code>python step_congressional_trading.py</code> to load congressional trade data.</p>
</div>"""

    members = ct.get("members", {})
    hot     = ct.get("hot_tickers", [])
    alerts  = ct.get("conflict_alert_summary", [])
    as_of   = ct.get("as_of", "—")

    # Honest provenance: is this live, a curated fallback, and/or stale?
    _src_raw   = str(ct.get("source", ""))
    _is_fallbk = "fallback" in _src_raw.lower() or "blocked" in _src_raw.lower()
    _stale_days = None
    try:
        import datetime as _dt
        _stale_days = (_dt.date.today() - _dt.date.fromisoformat(str(as_of))).days
    except Exception:
        pass
    _is_stale = _stale_days is not None and _stale_days >= 2
    _src_label = ("Curated fallback (live API blocked) — public STOCK Act archives"
                  if _is_fallbk else "STOCK Act public filings")
    _prov_banner = ""
    if _is_fallbk or _is_stale:
        _bits = []
        if _is_fallbk: _bits.append("live disclosure API is currently blocked; showing curated data from public STOCK Act archives")
        if _is_stale:  _bits.append(f"as of {as_of}, {_stale_days} days old")
        _prov_banner = f"""
  <div style="background:#1b1710;border-bottom:1px solid #43391f;border-left:3px solid #c8b487;padding:10px 24px">
    <p style="font-size:11.5px;color:#cdbd8f;letter-spacing:.02em">Data note — {' · '.join(_bits)}. Not real-time; treat as reference, not a live signal.</p>
  </div>"""

    PARTY_BG  = {"D": "#5f7480", "R": "#B71C1C"}
    PARTY_LT  = {"D": "#202832", "R": "#FFEBEE"}
    PARTY_TXT = {"D": "#495663", "R": "#7F1111"}

    # ── Conflict Alerts banner ───────────────────────────────────────────────
    alert_pills = ""
    for a in alerts[:6]:
        ticker_col = "#c8b487"
        alert_pills += f"""
<div style="display:inline-block;margin:3px;padding:6px 10px;background:#FFF8E6;border:1px solid #43391f;border-radius:6px;font-size:9px;line-height:1.4">
  <span style="font-weight:500;color:#c8b487">{a['member'].split()[-1]}</span>
  <span style="margin:0 4px;color:#888">→</span>
  <span style="font-weight:500;color:{ticker_col}">{a['ticker']}</span>
  <span style="color:#C0392B;font-weight:400;margin-left:4px">⏱ {a['lead_days']}d lead</span>
  <div style="font-size:8px;color:#666;margin-top:2px;max-width:240px">{a['event'][:60]}…</div>
</div>"""

    # ── Member deep-dive cards ────────────────────────────────────────────────
    member_cards = ""
    for name, m in list(members.items())[:6]:
        party     = m.get("party", "?")
        hdr_bg    = PARTY_BG.get(party, "#555")
        lt_bg     = PARTY_LT.get(party, "#F5F5F5")
        lt_txt    = PARTY_TXT.get(party, "#333")

        # Committee badges
        cmte_badges = "".join(
            f'<span style="display:inline-block;padding:2px 7px;margin:1px;border-radius:3px;background:{lt_bg};color:{lt_txt};font-size:8px;font-weight:400">{c[:28]}</span>'
            for c in m.get("committees", [])
        )

        # Conflict tickers
        conflict_badges = "".join(
            f'<span style="display:inline-block;padding:1px 5px;margin:1px;border-radius:20px;background:#FFF3E0;color:#E65100;font-size:8px;font-weight:400">{tk}</span>'
            for tk in m.get("conflict_tickers", [])[:7]
        )

        # Sector breakdown mini-bars
        sector_bars = ""
        for sec, pct in sorted(m.get("sector_breakdown", {}).items(), key=lambda x: -x[1])[:4]:
            sec_col = {"Technology":"#5f7480","Defense":"#8B3A3A","Energy":"#8B6914",
                       "Defense/Industrials":"#8B3A3A","Semiconductors":"#5a5470",
                       "Financials":"#4c5f65","Enterprise Cloud":"#1A6B3C"}.get(sec, "#888")
            sector_bars += f"""
<div style="display:flex;align-items:center;gap:6px;margin-bottom:3px">
  <div style="width:100px;font-size:8px;color:#555;text-align:right;flex-shrink:0">{sec[:16]}</div>
  <div style="flex:1;background:#EEE;border-radius:2px;height:7px">
    <div style="height:7px;border-radius:2px;background:{sec_col};width:{min(pct*1.8,100):.0f}%;opacity:.8"></div>
  </div>
  <div style="width:28px;font-size:8px;color:#888;font-variant-numeric:tabular-nums">{pct:.0f}%</div>
</div>"""

        # Top buys/sells
        buy_pills = " ".join(
            f'<span style="display:inline-block;padding:2px 6px;border-radius:20px;background:#241f18;color:#1B7A3B;font-size:9px;font-weight:400;margin:1px">{tk} <span style="opacity:.7">${v:,.0f}K</span></span>'
            for tk, v in (m.get("top_buys") or [])[:5]
        ) or "—"

        sell_pills = " ".join(
            f'<span style="display:inline-block;padding:2px 6px;border-radius:20px;background:#FFEBEE;color:#C0392B;font-size:9px;font-weight:400;margin:1px">{tk} <span style="opacity:.7">${v:,.0f}K</span></span>'
            for tk, v in (m.get("top_sells") or [])[:4]
        ) or "—"

        # Trade log with context column
        recent = m.get("recent_trades", [])[:6]
        trade_rows = ""
        for t in recent:
            act_col = "#1B7A3B" if t.get("action","") == "BUY" else "#C0392B"
            dt = str(t.get("date",""))
            dt_date = dt[:10]
            dt_time = dt[11:16] if len(dt) > 10 else ""
            ctx = t.get("context", "")[:55]
            trade_rows += f"""
<tr style="border-bottom:1px solid #241f18">
  <td style="padding:5px 6px;white-space:nowrap">
    <div style="font-weight:500;font-size:11px;color:#c8b487">{t.get('ticker','')}</div>
  </td>
  <td style="padding:5px 6px">
    <span style="font-size:9px;font-weight:400;padding:1px 5px;border-radius:3px;
      background:{'#241f18' if t.get('action')=='BUY' else '#FFEBEE'};color:{act_col}">{t.get('action','')}</span>
  </td>
  <td style="padding:5px 6px;font-size:9px;color:#444;font-variant-numeric:tabular-nums;white-space:nowrap">{t.get('amount','')[:20]}</td>
  <td style="padding:5px 6px;white-space:nowrap">
    <div style="font-size:9px;font-weight:400;color:#c8b487">{dt_date}</div>
    <div style="font-size:8px;color:#888">{dt_time} ET</div>
  </td>
  <td style="padding:5px 6px;font-size:8px;color:#555;max-width:200px">{ctx}{'…' if len(t.get('context',''))>55 else ''}</td>
</tr>"""

        if not trade_rows:
            trade_rows = '<tr><td colspan="5" style="padding:10px;font-size:10px;color:#aaa;text-align:center">No recent trades</td></tr>'

        # Alpha badge
        alpha = m.get("est_alpha_vs_spy", "")
        alpha_col = "#1B7A3B" if alpha.startswith("+") else "#C0392B"
        alpha_badge = f'<span style="font-size:10px;font-weight:400;color:{alpha_col}">{alpha} vs SPY</span>' if alpha else ""

        member_cards += f"""
<div style="border:1px solid #241f18;border-radius:10px;overflow:hidden;background:#fff;
  box-shadow:0 2px 6px rgba(0,0,0,.06)">

  <!-- Member header -->
  <div style="background:{hdr_bg};color:#fff;padding:14px 18px">
    <div style="display:flex;align-items:center;justify-content:space-between">
      <div>
        <div style="font-size:14px;font-weight:500">{name}</div>
        <div style="font-size:10px;opacity:.75;margin-top:2px">{m.get('chamber','?')} · {m.get('state','?')} · {m.get('n_trades',0)} disclosed trades</div>
      </div>
      <div style="text-align:right">
        {alpha_badge}
        <div style="font-size:9px;opacity:.65;margin-top:2px">${m.get('portfolio_est_m',0):.0f}M est. portfolio</div>
      </div>
    </div>
    <!-- Committee badges -->
    <div style="margin-top:8px">{cmte_badges}</div>
  </div>

  <!-- Body: 2-column -->
  <div style="display:grid;grid-template-columns:1fr 1fr;border-bottom:1px solid #241f18">

    <!-- Left: Committee context + conflict analysis -->
    <div style="padding:12px 14px;border-right:1px solid #241f18">
      <div style="font-size:9px;font-weight:400;text-transform:uppercase;letter-spacing:.8px;color:#888;margin-bottom:6px">Information Advantage Analysis</div>
      <div style="font-size:10px;color:#333;line-height:1.55;margin-bottom:10px;padding:8px;background:#FAFAFA;border-radius:4px;border-left:3px solid {hdr_bg}">{m.get('committee_context','')[:220]}{'…' if len(m.get('committee_context',''))>220 else ''}</div>
      <div style="font-size:9px;font-weight:400;text-transform:uppercase;letter-spacing:.8px;color:#E65100;margin-bottom:4px">Conflict-of-Interest Tickers</div>
      <div style="margin-bottom:10px">{conflict_badges or '<span style="font-size:9px;color:#aaa">None flagged</span>'}</div>
      <div style="font-size:9px;font-weight:400;text-transform:uppercase;letter-spacing:.8px;color:#888;margin-bottom:4px">Trading Pattern</div>
      <div style="font-size:9px;color:#555;line-height:1.5;padding:6px 8px;background:#FFFBF0;border-radius:4px;border:1px solid #43391f">{m.get('pattern_context','')[:180]}{'…' if len(m.get('pattern_context',''))>180 else ''}</div>
      <div style="margin-top:8px;font-size:9px;color:#888">🏆 {m.get('top_known_win','')[:80]}</div>
      <div style="margin-top:4px;font-size:8px;color:#aaa">Disclosure lag: {m.get('trade_lag_days','—')}</div>
    </div>

    <!-- Right: Holdings + sector breakdown -->
    <div style="padding:12px 14px">
      <div style="font-size:9px;font-weight:400;text-transform:uppercase;letter-spacing:.8px;color:#888;margin-bottom:4px">Top Buys (cumulative $K)</div>
      <div style="margin-bottom:10px;line-height:1.8">{buy_pills}</div>
      <div style="font-size:9px;font-weight:400;text-transform:uppercase;letter-spacing:.8px;color:#C0392B;margin-bottom:4px">Top Sells</div>
      <div style="margin-bottom:10px;line-height:1.8">{sell_pills}</div>
      <div style="font-size:9px;font-weight:400;text-transform:uppercase;letter-spacing:.8px;color:#888;margin-bottom:6px">Sector Breakdown</div>
      {sector_bars}
    </div>

  </div>

  <!-- Trade log with context -->
  <div style="overflow-x:auto">
    <div style="padding:8px 14px 4px;font-size:9px;font-weight:400;text-transform:uppercase;letter-spacing:.8px;color:#888">Recent Trade Log — timestamp, amount, congressional context</div>
    <table style="width:100%;border-collapse:collapse;min-width:600px">
      <thead>
        <tr style="background:#FAFAFA">
          <th style="padding:4px 6px;font-size:8px;color:#999;text-align:left">Ticker</th>
          <th style="padding:4px 6px;font-size:8px;color:#999;text-align:left">Action</th>
          <th style="padding:4px 6px;font-size:8px;color:#999;text-align:left">Amount</th>
          <th style="padding:4px 6px;font-size:8px;color:#999;text-align:left">Date/Time (ET)</th>
          <th style="padding:4px 6px;font-size:8px;color:#999;text-align:left">Congressional Context</th>
        </tr>
      </thead>
      <tbody>{trade_rows}</tbody>
    </table>
  </div>

</div>"""

    # ── Hot tickers with context ──────────────────────────────────────────────
    hot_rows = ""
    for h in hot[:10]:
        cx_star = "★" if h.get("canyon_owns") else ""
        cx_col  = "color:#1B7A3B;font-weight:400" if h.get("canyon_owns") else "color:#ccc"
        avg_ld  = h.get("avg_lead_days", 0)
        ld_col  = "#C0392B" if avg_ld >= 15 else ("#c8b487" if avg_ld >= 7 else "#888")
        hot_context = h.get("context","")[:70]
        hot_rows += f"""
<tr style="border-bottom:1px solid #241f18">
  <td style="padding:6px 8px;font-weight:500;font-size:12px;color:#c8b487;white-space:nowrap">{h['ticker']}</td>
  <td style="padding:6px 8px;font-size:11px;text-align:center;font-variant-numeric:tabular-nums">{h.get('n_members',0)}</td>
  <td style="padding:6px 8px;font-size:11px;text-align:center;font-variant-numeric:tabular-nums">{h.get('n_trades',0)}</td>
  <td style="padding:6px 8px;font-size:10px;font-weight:400;color:{ld_col}">{f'{avg_ld}d' if avg_ld else '—'}</td>
  <td style="padding:6px 8px;font-size:8px;color:#555;max-width:180px">{hot_context}{'…' if len(h.get('context',''))>70 else ''}</td>
  <td style="padding:6px 8px;font-size:12px;{cx_col}">{cx_star}</td>
</tr>"""

    return f"""
<div style="border:1px solid #241f18;border-radius:12px;overflow:hidden;background:#fff;box-shadow:0 2px 12px rgba(0,0,0,.08)">

  <!-- Header -->
  <div style="background:linear-gradient(135deg,#2a2418 0%,#3a3128 100%);color:#fff;padding:18px 24px">
    <div style="display:flex;align-items:center;gap:16px">
      <span style="font-size:28px">🏛️</span>
      <div>
        <div style="font-size:16px;font-weight:500;letter-spacing:.3px">Congressional Trading — STOCK Act Deep Analysis</div>
        <div style="font-size:10px;opacity:.7;margin-top:3px">
          {len(members)} members · {sum(m.get('n_trades',0) for m in members.values())} total disclosed trades
          · Data as of {as_of} · Source: {_src_label}
        </div>
      </div>
      <div style="margin-left:auto;text-align:right;font-size:10px;opacity:.6">
        Pelosi · Tuberville · Crenshaw<br>Khanna · Warner · Gottheimer
      </div>
    </div>
  </div>
  {_prov_banner}

  <!-- Conflict Alerts -->
  {f"""
  <div style="padding:14px 20px;background:#FFFBF0;border-bottom:1px solid #43391f">
    <div style="font-size:9px;font-weight:400;text-transform:uppercase;letter-spacing:1px;color:#c8b487;margin-bottom:8px">
      ⚠ Conflict-of-Interest Alerts — Trade preceded public event by significant lead time
    </div>
    <div style="display:flex;flex-wrap:wrap;gap:4px">{alert_pills}</div>
    <div style="font-size:8px;color:#888;margin-top:6px">Lead time = days between trade date and public event announcement. Higher lead time → greater information asymmetry concern.</div>
  </div>""" if alerts else ""}

  <!-- Member deep-dive cards -->
  <div style="padding:16px 20px">
    <div style="display:grid;grid-template-columns:repeat(auto-fill,minmax(640px,1fr));gap:16px">
      {member_cards}
    </div>
  </div>

  <!-- Congress-wide hot tickers -->
  <div style="padding:0 20px 20px">
    <div style="font-size:10px;font-weight:400;text-transform:uppercase;letter-spacing:1px;color:#888;margin-bottom:8px">Most-Bought Tickers Across All Tracked Members</div>
    <div style="overflow-x:auto">
      <table style="width:100%;border-collapse:collapse;min-width:600px">
        <thead>
          <tr style="background:#FAFAFA">
            <th style="padding:6px 8px;font-size:9px;color:#888;text-align:left">Ticker</th>
            <th style="padding:6px 8px;font-size:9px;color:#888;text-align:center">Members</th>
            <th style="padding:6px 8px;font-size:9px;color:#888;text-align:center">Trades</th>
            <th style="padding:6px 8px;font-size:9px;color:#888">Avg Lead</th>
            <th style="padding:6px 8px;font-size:9px;color:#888">Pattern Context</th>
            <th style="padding:6px 8px;font-size:9px;color:#888">Canyon</th>
          </tr>
        </thead>
        <tbody>{hot_rows}</tbody>
      </table>
    </div>
  </div>

</div>"""


def _build_sector_planet_map() -> str:
    """Full-width interactive S&P 500 sector planet mind-map with logic links."""
    import json as _json_pm, pathlib as _pl_pm
    canyon_raw: dict = {}
    try:
        _adf_path = _pl_pm.Path(__file__).parent / "alpha_scores.csv"
        if _adf_path.exists():
            import csv as _csv_pm
            with open(_adf_path) as _f:
                for row in _csv_pm.DictReader(_f):
                    t = row.get("ticker", "").strip()
                    v = row.get("alpha_score") or row.get("alpha_rank") or "0"
                    try:
                        canyon_raw[t] = round(float(v) / 100, 3)
                    except Exception:
                        pass
    except Exception:
        pass
    canyon_json = _json_pm.dumps(canyon_raw)

    return f"""
<div style="background:#231a12;border-radius:12px;padding:22px 26px 18px;margin-bottom:28px;position:relative;overflow:visible">
  <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:14px">
    <div>
      <p style="font-size:10px;letter-spacing:2px;text-transform:uppercase;color:#c8b487;margin:0 0 3px">S&P 500 Sector Intelligence — Interactive Planet Map</p>
      <h3 style="color:#fff;font-family:'Playfair Display',serif;font-size:19px;margin:0;font-weight:400">AI &amp; Compute · Robotics · Energy · Healthcare · Financials — with Logical Connections</h3>
    </div>
    <div style="font-size:10px;color:rgba(255,255,255,.4);text-align:right;flex-shrink:0;padding-left:16px">
      <div>Hover any node for deep detail</div>
      <div style="color:#c8b487;margin-top:2px">★ = Canyon signal active</div>
    </div>
  </div>

  <div style="position:relative">
    <canvas id="sp-planet-map" width="1028" height="640"
      style="width:100%;height:auto;display:block;border-radius:8px;cursor:crosshair"></canvas>
    <div id="sp-planet-tip"
      style="display:none;position:absolute;background:rgba(10,20,46,.97);
             border:1px solid #c8b487;border-radius:8px;padding:12px 16px;
             color:#fff;font-size:12px;line-height:1.7;pointer-events:none;
             max-width:240px;z-index:99;box-shadow:0 4px 24px rgba(0,0,0,.5)">
    </div>
  </div>

  <div style="display:flex;flex-wrap:wrap;gap:20px;margin-top:12px">
    <div style="display:flex;align-items:center;gap:6px"><span style="width:11px;height:11px;border-radius:50%;background:#1B6F4A;display:inline-block"></span><span style="font-size:11px;color:rgba(255,255,255,.55)">Outperforming SPY YTD</span></div>
    <div style="display:flex;align-items:center;gap:6px"><span style="width:11px;height:11px;border-radius:50%;background:#7A2020;display:inline-block"></span><span style="font-size:11px;color:rgba(255,255,255,.55)">Underperforming SPY YTD</span></div>
    <div style="display:flex;align-items:center;gap:6px"><span style="width:11px;height:11px;border-radius:50%;background:#c8b487;display:inline-block"></span><span style="font-size:11px;color:rgba(255,255,255,.55)">Canyon ★ active signal</span></div>
    <div style="display:flex;align-items:center;gap:6px"><span style="border-top:1px dashed rgba(255,255,255,.4);width:22px;display:inline-block"></span><span style="font-size:11px;color:rgba(255,255,255,.55)">Logic connection</span></div>
  </div>
</div>

<script>
(function(){{
  var canvas = document.getElementById('sp-planet-map');
  if (!canvas || !canvas.getContext) return;
  var ctx = canvas.getContext('2d');
  var W = canvas.width, H = canvas.height;
  var CX = W/2, CY = H/2;
  var tip = document.getElementById('sp-planet-tip');

  var CANYON = {canyon_json};

  var SECTORS = [
    {{
      id:'ai', label:'AI & Compute', emoji:'🤖', color:'#7C6FEC',
      angle:-90,
      desc:'Foundation models, GPU compute, cloud AI — the enabling brain of the new economy.',
      layer:'上游 → 中游 (Infra to Application)',
      stocks:[
        {{t:'NVDA',name:'NVIDIA',      ytd:0.85, note:'H200/B200 GPU monopoly · 85% AI accelerator share · $3.3T cap'}},
        {{t:'MSFT',name:'Microsoft',   ytd:0.18, note:'Azure OpenAI · Copilot · 49% OpenAI stake · $3.1T cap'}},
        {{t:'GOOGL',name:'Alphabet',   ytd:0.32, note:'Gemini Ultra · TPU custom silicon · Google Cloud AI'}},
        {{t:'META',name:'Meta',        ytd:0.45, note:'Llama open-source · 3.5B DAU data moat · FAIR lab'}},
        {{t:'AMZN',name:'Amazon',      ytd:0.28, note:'Trainium/Inferentia custom chips · Bedrock AI marketplace'}},
        {{t:'TSM', name:'TSMC',        ytd:0.55, note:'3nm/2nm node monopoly · makes 90% of advanced AI chips'}},
        {{t:'AVGO',name:'Broadcom',    ytd:0.42, note:'Custom AI ASICs (Google XPU, Meta MTIA) · revenue ×5'}},
        {{t:'AMD', name:'AMD',         ytd:-0.05,note:'MI300X vs H100 · gaining hyperscaler share · $250B cap'}},
        {{t:'PLTR',name:'Palantir',    ytd:0.90, note:'AIP enterprise AI · US Gov contracts · fast-growing'}},
      ]
    }},
    {{
      id:'robotics', label:'Robotics & Auto', emoji:'🦾', color:'#10B981',
      angle:-18,
      desc:'Physical AI: surgical robots, humanoids, industrial automation — the embodied intelligence layer.',
      layer:'中游 → 下游 (Physical deployment)',
      stocks:[
        {{t:'ISRG',name:'Intuitive Surg', ytd:0.22, note:'da Vinci 4.0 · 2M+ procedures/yr · 70% gross margin'}},
        {{t:'ETN', name:'Eaton Corp',     ytd:0.35, note:'Power management for EVs & data centers · infra play'}},
        {{t:'HON', name:'Honeywell',      ytd:0.08, note:'Industrial AI sensors · building automation · defense'}},
        {{t:'CAT', name:'Caterpillar',    ytd:0.15, note:'Autonomous mining trucks · global infrastructure build'}},
        {{t:'DE',  name:'Deere',          ytd:-0.08,note:'Autonomous farm equipment · See & Spray precision AI'}},
        {{t:'AXON',name:'Axon',           ytd:0.68, note:'AI-powered law enforcement + body cam analytics'}},
        {{t:'ROK', name:'Rockwell Auto',  ytd:-0.12,note:'Factory automation PLC leader · smart manufacturing'}},
      ]
    }},
    {{
      id:'energy', label:'Energy & Power', emoji:'⚡', color:'#F59E0B',
      angle:54,
      desc:'O&G incumbents + clean energy transition + nuclear renaissance powering AI data centers.',
      layer:'基础设施 (Critical infrastructure)',
      stocks:[
        {{t:'XOM', name:'ExxonMobil',   ytd:0.05, note:'Largest US E&P · Permian Basin · LNG export growth'}},
        {{t:'CVX', name:'Chevron',      ytd:0.02, note:'LNG + deepwater · clean hydrogen pilot projects'}},
        {{t:'NEE', name:'NextEra',      ytd:0.08, note:'#1 wind & solar operator · AI data center supplier'}},
        {{t:'CEG', name:'Constellation',ytd:0.65, note:'Nuclear power to AI data centers · MSFT 20yr deal'}},
        {{t:'VST', name:'Vistra',       ytd:1.20, note:'Nuclear+gas peaker · highest YTD in S&P 500 energy'}},
        {{t:'FSLR',name:'First Solar',  ytd:0.15, note:'US-made CdTe solar · IRA domestic content premium'}},
        {{t:'COP', name:'ConocoPhillips',ytd:-0.02,note:'Pure-play E&P · Marathon acquisition · low-cost barrel'}},
      ]
    }},
    {{
      id:'health', label:'Healthcare & Bio', emoji:'🏥', color:'#EC4899',
      angle:162,
      desc:'GLP-1 obesity revolution + AI drug discovery + managed care — the decade\'s alpha theme.',
      layer:'应用层 (Consumer + B2B)',
      stocks:[
        {{t:'LLY', name:'Eli Lilly',    ytd:0.55, note:'Mounjaro/Zepbound GLP-1 blockbuster · $800B+ cap'}},
        {{t:'UNH', name:'UnitedHealth', ytd:-0.35,note:'Optum AI + managed care · fraud probe headwinds'}},
        {{t:'ABBV',name:'AbbVie',       ytd:0.18, note:'Humira successor Skyrizi/Rinvoq · strong pipeline'}},
        {{t:'MRK', name:'Merck',        ytd:-0.05,note:'Keytruda $25B/yr · oncology pipeline + oncology AI'}},
        {{t:'VRTX',name:'Vertex',       ytd:0.08, note:'CF monopoly (Trikafta) · CRISPR gene therapy next'}},
        {{t:'AMGN',name:'Amgen',        ytd:0.12, note:'MariTide GLP-1 challenger · biosimilar portfolio'}},
        {{t:'REGN',name:'Regeneron',    ytd:0.02, note:'Dupixent $14B/yr · cancer pipeline expansion'}},
      ]
    }},
    {{
      id:'fintech', label:'Financials & Cap', emoji:'🏦', color:'#3B82F6',
      angle:198,
      desc:'Banks, asset managers, payment rails — AI transformation of trading, risk, and credit.',
      layer:'资本层 (Capital allocation)',
      stocks:[
        {{t:'BRK', name:'Berkshire',  ytd:0.18, note:'Buffett compounding machine · AAPL 42% of portfolio'}},
        {{t:'JPM', name:'JPMorgan',   ytd:0.28, note:'AI trading desk · IB fee rebound · largest US bank'}},
        {{t:'V',   name:'Visa',       ytd:0.12, note:'3.9B cards · 80%+ margins · network moat · $530B cap'}},
        {{t:'MA',  name:'Mastercard', ytd:0.14, note:'Cross-border growth · tokenization & embedded finance'}},
        {{t:'GS',  name:'Goldman',    ytd:0.35, note:'AI M&A advisory · capital markets recovery · quant desk'}},
        {{t:'BLK', name:'BlackRock',  ytd:0.22, note:'Aladdin AI risk platform · $10T AUM · ETF dominance'}},
        {{t:'AXP', name:'Amex',       ytd:0.25, note:'Premium cardholder loyalty · spend per card +9% YoY'}},
      ]
    }},
  ];

  var LINKS = [
    {{from:'ai',      to:'energy',   label:'Data center power +500% by 2030',  color:'rgba(245,158,11,0.65)'}},
    {{from:'ai',      to:'robotics', label:'Foundation models → embodied AI',   color:'rgba(16,185,129,0.65)'}},
    {{from:'ai',      to:'health',   label:'Drug discovery AI — 10× faster',    color:'rgba(236,72,153,0.65)'}},
    {{from:'ai',      to:'fintech',  label:'Algo trading + risk AI',             color:'rgba(59,130,246,0.65)'}},
    {{from:'energy',  to:'robotics', label:'Factory electrification wave',       color:'rgba(245,158,11,0.40)'}},
    {{from:'health',  to:'fintech',  label:'Biotech M&A — $200B+ pipeline',     color:'rgba(59,130,246,0.40)'}},
  ];

  var ORBIT_R   = 218;   // planet distance from center
  var SECTOR_R  = 46;    // planet circle radius
  var MOON_ORB  = 82;    // stock orbit radius from planet center
  var MOON_R    = 17;    // stock circle radius
  var SUN_R     = 54;    // central S&P 500 circle radius

  // Compute positions
  SECTORS.forEach(function(s) {{
    var rad = s.angle * Math.PI / 180;
    s.x = CX + ORBIT_R * Math.cos(rad);
    s.y = CY + ORBIT_R * Math.sin(rad);
    s.stocks.forEach(function(st, j) {{
      var stRad = (j / s.stocks.length) * 2 * Math.PI - Math.PI/2;
      st.x = s.x + MOON_ORB * Math.cos(stRad);
      st.y = s.y + MOON_ORB * Math.sin(stRad);
      st.canyon = CANYON[st.t] || 0;
    }});
  }});

  var hovered = null, hovSt = null, frame = 0;

  function bg() {{
    // Deep space background
    ctx.fillStyle = '#0A1628';
    ctx.fillRect(0,0,W,H);
    // Subtle star grid
    var starPts = [[80,35],[180,90],[320,18],[490,580],[620,28],[750,595],[920,120],[1005,45],[55,310],[1018,380],[430,590],[150,490],[870,530],[230,555]];
    starPts.forEach(function(p) {{
      var sz = 0.8 + 0.7*Math.random();
      ctx.beginPath();
      ctx.arc(p[0], p[1], sz, 0, Math.PI*2);
      ctx.fillStyle = 'rgba(255,255,255,'+(0.15+0.2*Math.sin(frame*0.04+p[0]*0.01))+')';
      ctx.fill();
    }});
    // Center glow
    var grd = ctx.createRadialGradient(CX,CY,0,CX,CY,280);
    grd.addColorStop(0,'rgba(184,148,63,0.06)');
    grd.addColorStop(1,'rgba(0,0,0,0)');
    ctx.fillStyle = grd;
    ctx.fillRect(0,0,W,H);
  }}

  function drawConnections() {{
    LINKS.forEach(function(lk) {{
      var s = null, t = null;
      SECTORS.forEach(function(x) {{ if(x.id===lk.from) s=x; if(x.id===lk.to) t=x; }});
      if(!s||!t) return;
      var mx = (s.x+t.x)/2 + (t.y-s.y)*0.12;
      var my = (s.y+t.y)/2 - (t.x-s.x)*0.12;
      ctx.beginPath();
      ctx.moveTo(s.x,s.y); ctx.quadraticCurveTo(mx,my,t.x,t.y);
      ctx.strokeStyle = lk.color;
      ctx.lineWidth = 1.5;
      ctx.setLineDash([4,7]);
      ctx.stroke();
      ctx.setLineDash([]);
      // Label
      var lx = 0.25*s.x + 0.5*mx + 0.25*t.x;
      var ly = 0.25*s.y + 0.5*my + 0.25*t.y;
      ctx.font = '9px sans-serif';
      var tw = ctx.measureText(lk.label).width;
      ctx.fillStyle = 'rgba(10,22,40,0.85)';
      ctx.fillRect(lx-tw/2-5, ly-8, tw+10, 16);
      ctx.fillStyle = lk.color.replace(/[0-9.]+[)]/,'0.95)');
      ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
      ctx.fillText(lk.label, lx, ly);
    }});
  }}

  function drawOrbitRing() {{
    ctx.beginPath();
    ctx.arc(CX,CY,ORBIT_R,0,Math.PI*2);
    ctx.strokeStyle = 'rgba(184,148,63,0.10)';
    ctx.lineWidth = 1;
    ctx.setLineDash([3,8]);
    ctx.stroke();
    ctx.setLineDash([]);
  }}

  function drawSun() {{
    var pulse = 1 + 0.04*Math.sin(frame*0.04);
    // Outer glow
    var grd = ctx.createRadialGradient(CX,CY,0,CX,CY,SUN_R*2*pulse);
    grd.addColorStop(0,'rgba(184,148,63,0.28)');
    grd.addColorStop(1,'rgba(184,148,63,0)');
    ctx.beginPath(); ctx.arc(CX,CY,SUN_R*2*pulse,0,Math.PI*2);
    ctx.fillStyle=grd; ctx.fill();
    // Sun disk
    ctx.beginPath(); ctx.arc(CX,CY,SUN_R,0,Math.PI*2);
    ctx.fillStyle='#c8b487'; ctx.fill();
    ctx.strokeStyle='rgba(255,255,255,0.22)'; ctx.lineWidth=2; ctx.stroke();
    // Text
    ctx.textAlign='center'; ctx.textBaseline='middle';
    ctx.font='bold 12px sans-serif'; ctx.fillStyle='#fff';
    ctx.fillText('S&P 500', CX, CY-10);
    ctx.font='9px sans-serif'; ctx.fillStyle='rgba(255,255,255,0.7)';
    ctx.fillText('503 companies', CX, CY+4);
    ctx.fillText('$46T market cap', CX, CY+16);
  }}

  function drawSectors() {{
    SECTORS.forEach(function(s) {{
      var isHov = hovered && hovered.id===s.id;
      // Moon orbit ring
      ctx.beginPath(); ctx.arc(s.x,s.y,MOON_ORB,0,Math.PI*2);
      ctx.strokeStyle = isHov ? s.color+'55' : 'rgba(255,255,255,0.05)';
      ctx.lineWidth=1; ctx.setLineDash([2,6]); ctx.stroke(); ctx.setLineDash([]);
      // Planet glow on hover
      if(isHov) {{
        var grd=ctx.createRadialGradient(s.x,s.y,0,s.x,s.y,SECTOR_R*2.2);
        grd.addColorStop(0,s.color+'40'); grd.addColorStop(1,'transparent');
        ctx.beginPath(); ctx.arc(s.x,s.y,SECTOR_R*2.2,0,Math.PI*2);
        ctx.fillStyle=grd; ctx.fill();
      }}
      // Planet disk
      var pr = isHov ? SECTOR_R*1.10 : SECTOR_R;
      ctx.beginPath(); ctx.arc(s.x,s.y,pr,0,Math.PI*2);
      ctx.fillStyle=s.color; ctx.fill();
      ctx.strokeStyle='rgba(255,255,255,'+(isHov?'0.45':'0.18')+')';
      ctx.lineWidth=isHov?2.5:1.5; ctx.stroke();
      // Planet label
      ctx.textAlign='center'; ctx.textBaseline='middle';
      ctx.font='14px sans-serif'; ctx.fillStyle='#fff';
      ctx.fillText(s.emoji, s.x, s.y-9);
      ctx.font='bold 7.5px sans-serif'; ctx.fillStyle='rgba(255,255,255,0.9)';
      var words=s.label.split(' & ');
      if(words.length>1){{
        ctx.fillText(words[0], s.x, s.y+3);
        ctx.fillText('& '+words[1], s.x, s.y+12);
      }} else {{
        ctx.fillText(s.label, s.x, s.y+7);
      }}
      // Stock moons
      s.stocks.forEach(function(st) {{
        var isSt = hovSt && hovSt.t===st.t;
        var mr = isSt ? MOON_R*1.3 : MOON_R;
        // Color logic
        var fc;
        if(st.canyon>0.55) fc='#c8b487';
        else if(st.ytd>0.10) fc='#1B6F4A';
        else if(st.ytd>-0.05) fc='#2A5A3A';
        else fc='#7A2020';
        ctx.beginPath(); ctx.arc(st.x,st.y,mr,0,Math.PI*2);
        ctx.fillStyle=fc; ctx.fill();
        ctx.strokeStyle=isSt?'rgba(255,255,255,0.9)':'rgba(255,255,255,0.22)';
        ctx.lineWidth=isSt?2:1; ctx.stroke();
        // Canyon star ring
        if(st.canyon>0.55) {{
          ctx.beginPath();
          ctx.arc(st.x,st.y,mr+3,0,Math.PI*2);
          ctx.strokeStyle='rgba(184,148,63,0.6)';
          ctx.lineWidth=1.5; ctx.setLineDash([2,3]); ctx.stroke(); ctx.setLineDash([]);
        }}
        // Ticker text
        ctx.textAlign='center'; ctx.textBaseline='middle';
        ctx.font='bold '+(isSt?8:7)+'px sans-serif';
        ctx.fillStyle='#fff';
        if(st.canyon>0.55) {{
          ctx.font='8px sans-serif';
          ctx.fillText('★', st.x, st.y-4);
          ctx.font='bold 6.5px sans-serif';
          ctx.fillText(st.t, st.x, st.y+5);
        }} else {{
          ctx.fillText(st.t, st.x, st.y);
        }}
        // YTD micro label
        ctx.font='6px sans-serif';
        ctx.fillStyle=st.ytd>=0?'rgba(107,204,160,0.85)':'rgba(200,80,80,0.85)';
        ctx.fillText((st.ytd>=0?'+':'')+Math.round(st.ytd*100)+'%', st.x, st.y+mr+8);
      }});
    }});
  }}

  function getPos(e) {{
    var r=canvas.getBoundingClientRect();
    return {{x:(e.clientX-r.left)*(W/r.width), y:(e.clientY-r.top)*(H/r.height)}};
  }}

  function showTip(canvasX, canvasY, html) {{
    tip.innerHTML=html;
    tip.style.display='block';
    var r=canvas.getBoundingClientRect();
    var cx=(canvasX/W)*r.width, cy=(canvasY/H)*r.height;
    var lft=cx+16, top=cy-10;
    var tw=tip.offsetWidth||240, th=tip.offsetHeight||160;
    if(lft+tw>r.width-10) lft=cx-tw-16;
    if(top+th>r.height-10) top=Math.max(5, r.height-th-10);
    tip.style.left=lft+'px'; tip.style.top=top+'px';
  }}

  canvas.addEventListener('mousemove', function(e) {{
    var p=getPos(e);
    hovered=null; hovSt=null;
    var found=false;
    // Check moons first
    SECTORS.forEach(function(s) {{
      if(found) return;
      s.stocks.forEach(function(st) {{
        if(found) return;
        var dx=p.x-st.x,dy=p.y-st.y;
        if(Math.sqrt(dx*dx+dy*dy)<MOON_R+8) {{
          found=true; hovSt=st; hovered=s;
          var ytdStr=(st.ytd>=0?'+':'')+Math.round(st.ytd*100)+'%';
          var ytdClr=st.ytd>=0?'#6BCCA0':'#E06060';
          var canyonLine=st.canyon>0.4?'<div style="margin-top:7px;padding:5px 8px;background:rgba(184,148,63,0.15);border-left:2px solid #c8b487;border-radius:3px"><span style="color:#c8b487;font-size:10px">Canyon ★ Signal: <strong>'+(st.canyon*100).toFixed(0)+'% conviction</strong></span></div>':'';
          showTip(p.x,p.y,
            '<div style="font-size:9px;letter-spacing:1.5px;text-transform:uppercase;color:'+s.color+';margin-bottom:5px">'+s.label+'</div>'+
            '<div style="font-size:15px;font-weight:400;margin-bottom:2px">'+st.t+'</div>'+
            '<div style="font-size:11px;color:#aaa;margin-bottom:5px">'+st.name+'</div>'+
            '<div style="font-size:12px;color:'+ytdClr+';font-weight:400">'+ytdStr+' YTD vs SPY</div>'+
            '<div style="font-size:11px;color:#ccc;margin-top:7px;line-height:1.65">'+st.note+'</div>'+
            canyonLine
          );
        }}
      }});
    }});
    if(!found) {{
      SECTORS.forEach(function(s) {{
        if(found) return;
        var dx=p.x-s.x,dy=p.y-s.y;
        if(Math.sqrt(dx*dx+dy*dy)<SECTOR_R+12) {{
          found=true; hovered=s;
          var stList=s.stocks.map(function(st){{return '<span style="display:inline-block;margin:2px 3px;padding:1px 5px;background:'+s.color+'33;border-radius:3px;font-size:10px">'+st.t+'</span>';}}).join('');
          showTip(p.x,p.y,
            '<div style="font-size:14px;font-weight:400;color:'+s.color+';margin-bottom:5px">'+s.emoji+' '+s.label+'</div>'+
            '<div style="font-size:11px;color:#ccc;line-height:1.65;margin-bottom:8px">'+s.desc+'</div>'+
            '<div style="font-size:10px;color:#888;margin-bottom:5px">'+s.layer+'</div>'+
            '<div style="margin-top:5px">'+stList+'</div>'
          );
        }}
      }});
    }}
    if(!found) {{
      var dx=p.x-CX,dy=p.y-CY;
      if(Math.sqrt(dx*dx+dy*dy)<SUN_R+5) {{
        found=true;
        showTip(p.x,p.y,
          '<div style="font-size:14px;font-weight:400;color:#c8b487;margin-bottom:6px">S&P 500 — The Universe</div>'+
          '<div style="font-size:11px;color:#ccc;line-height:1.65">503 companies · $46T market cap<br>5 key theme sectors shown with logical connections between them</div>'+
          '<div style="font-size:10px;color:#888;margin-top:8px">Hover any planet or stock for detail</div>'
        );
      }}
    }}
    if(!found) {{ tip.style.display='none'; }}
    canvas.style.cursor=found?'pointer':'crosshair';
  }});

  canvas.addEventListener('mouseleave', function() {{
    hovered=null; hovSt=null; tip.style.display='none';
  }});

  function render() {{
    frame++;
    bg();
    drawOrbitRing();
    drawConnections();
    drawSun();
    drawSectors();
    requestAnimationFrame(render);
  }}
  render();
}})();
</script>
"""


def _build_flow_tab(options_flow: dict, etf_flow: dict, econ_cal: dict) -> str:
    """Market Intelligence / Flow tab: options unusual flow + ETF rotation + econ calendar."""
    import json as _j, html as _html_mod
    def _esc(s): return _html_mod.escape(str(s)) if s is not None else ""

    # ── ETF Sector Rotation panel ─────────────────────────────────────────────
    sectors = etf_flow.get("sectors", [])
    etf_as_of = etf_flow.get("as_of", "—")
    etf_updated = etf_flow.get("updated", "")

    def _ret_bar(val, max_abs=6.0):
        pct = min(100, abs(val) / max_abs * 50)
        col = "#1B6F4A" if val >= 0 else "#B83232"
        if val >= 0:
            return f'<div style="display:flex;align-items:center;gap:6px"><div style="width:50%;text-align:right"><div style="height:12px;background:{col};border-radius:2px;width:{pct:.0f}%;margin-left:auto"></div></div><span style="font-size:12px;font-weight:400;color:{col};width:52px">{val:+.2f}%</span><div style="width:50%"></div></div>'
        else:
            return f'<div style="display:flex;align-items:center;gap:6px"><div style="width:50%"></div><span style="font-size:12px;font-weight:400;color:{col};width:52px;text-align:right">{val:+.2f}%</span><div style="width:50%;text-align:left"><div style="height:12px;background:{col};border-radius:2px;width:{pct:.0f}%"></div></div></div>'

    etf_rows = ""
    for s in sectors:
        flow = s.get("flow_signal", "NEUTRAL")
        flow_col = "#1B6F4A" if flow == "INFLOW" else ("#B83232" if flow == "OUTFLOW" else "#999")
        flow_bg  = "#EAF5EE" if flow == "INFLOW" else ("#FEF0EF" if flow == "OUTFLOW" else "#241f18")
        is_spy   = s["etf"] in ("SPY", "QQQ")
        row_style = 'border-top:2px solid #241f18;font-style:italic' if is_spy else ''
        etf_rows += f"""<tr style="{row_style}">
          <td style="padding:7px 8px;font-weight:400;font-size:12px;color:#c8b487">{_esc(s['etf'])}</td>
          <td style="padding:7px 8px;font-size:11.5px;color:#555">{_esc(s['name'])}</td>
          <td style="padding:7px 8px">{_ret_bar(s['ret_1d'])}</td>
          <td style="padding:7px 8px">{_ret_bar(s['ret_5d'])}</td>
          <td style="padding:7px 8px">{_ret_bar(s['ret_1m'])}</td>
          <td style="padding:7px 8px">{_ret_bar(s['ret_3m'])}</td>
          <td style="padding:7px 8px;text-align:center"><span style="font-size:10px;font-weight:400;color:{flow_col};background:{flow_bg};padding:2px 7px;border-radius:3px">{flow}</span></td>
        </tr>"""

    if not etf_rows:
        etf_rows = '<tr><td colspan="7" style="color:#AAA;padding:20px;text-align:center">Run step_etf_flow_rt.py to load sector data</td></tr>'

    etf_panel = f"""<div style="overflow-x:auto">
      <table style="width:100%;border-collapse:collapse;font-size:12px">
        <thead><tr style="background:#241f18;border-bottom:2px solid #241f18">
          <th style="padding:8px;text-align:left;font-size:10px;color:#888;font-weight:400;letter-spacing:.5px">ETF</th>
          <th style="padding:8px;text-align:left;font-size:10px;color:#888;font-weight:400;letter-spacing:.5px">Sector</th>
          <th style="padding:8px;text-align:center;font-size:10px;color:#888;font-weight:400;letter-spacing:.5px">1 Day</th>
          <th style="padding:8px;text-align:center;font-size:10px;color:#888;font-weight:400;letter-spacing:.5px">5 Day</th>
          <th style="padding:8px;text-align:center;font-size:10px;color:#888;font-weight:400;letter-spacing:.5px">1 Month</th>
          <th style="padding:8px;text-align:center;font-size:10px;color:#888;font-weight:400;letter-spacing:.5px">3 Month</th>
          <th style="padding:8px;text-align:center;font-size:10px;color:#888;font-weight:400;letter-spacing:.5px">Flow Signal</th>
        </tr></thead>
        <tbody>{etf_rows}</tbody>
      </table>
    </div>
    <p style="font-size:11px;color:#AAA;margin-top:6px">Flow = 5-day avg volume vs 20-day avg volume ratio. Updated {etf_as_of} {etf_updated}</p>"""

    # ── Options Unusual Flow panel ─────────────────────────────────────────────
    top_flows = options_flow.get("top_flows", [])
    sentiments = options_flow.get("ticker_sentiment", [])
    opt_as_of = options_flow.get("as_of", "—")
    opt_count  = options_flow.get("unusual_count", 0)
    opt_scanned = options_flow.get("tickers_scanned", 0)

    flow_rows = ""
    for f in top_flows[:25]:
        side = f.get("side", "")
        side_col = "#1B6F4A" if side == "CALL" else "#B83232"
        side_bg  = "#EAF5EE" if side == "CALL" else "#FEF0EF"
        prem = f.get("premium_est", 0)
        prem_str = f"${prem/1_000:.0f}K" if prem < 1_000_000 else f"${prem/1_000_000:.1f}M"
        flow_rows += f"""<tr style="border-bottom:1px solid #241f18">
          <td style="padding:7px 8px;font-weight:400;font-size:12px;color:#c8b487">{_esc(f.get('ticker',''))}</td>
          <td style="padding:7px 8px"><span style="font-size:10px;font-weight:500;color:{side_col};background:{side_bg};padding:2px 8px;border-radius:3px;letter-spacing:.5px">{side}</span></td>
          <td style="padding:7px 8px;font-size:12px;color:#555">${_esc(str(f.get('strike','')))}</td>
          <td style="padding:7px 8px;font-size:11px;color:#888">{_esc(f.get('expiry',''))} <span style="color:#c8b487">({f.get('days_to_exp','?')}d)</span></td>
          <td style="padding:7px 8px;font-size:12px;font-weight:400;color:#c8b487;font-variant-numeric:tabular-nums">{f.get('volume',0):,}</td>
          <td style="padding:7px 8px;font-size:11px;color:#999">{f.get('open_interest',0):,}</td>
          <td style="padding:7px 8px;font-size:12px;font-weight:400;color:#c8b487">{f.get('vol_oi_ratio',0):.1f}×</td>
          <td style="padding:7px 8px;font-size:12px;font-weight:400;color:{side_col}">{prem_str}</td>
          <td style="padding:7px 8px;font-size:11px;color:#999">{f.get('iv',0)*100:.0f}%</td>
        </tr>"""

    if not flow_rows:
        flow_rows = '<tr><td colspan="9" style="color:#AAA;padding:20px;text-align:center">Run step_options_flow.py to scan options</td></tr>'

    # Sentiment summary chips
    sent_chips = ""
    for s in sentiments[:12]:
        bias = s.get("bias","NEUTRAL")
        chip_col = "#1B6F4A" if bias == "BULLISH" else ("#B83232" if bias == "BEARISH" else "#999")
        chip_bg  = "#EAF5EE" if bias == "BULLISH" else ("#FEF0EF" if bias == "BEARISH" else "#241f18")
        call_pct = s.get("call_pct", 50)
        prem_tot = s.get("total_prem", 0)
        prem_str = f"${prem_tot/1000:.0f}K" if prem_tot < 1_000_000 else f"${prem_tot/1_000_000:.1f}M"
        sent_chips += f"""<div style="display:flex;flex-direction:column;align-items:center;padding:10px 12px;background:{chip_bg};border:1px solid {chip_col}33;border-radius:8px;min-width:80px">
          <span style="font-size:13px;font-weight:500;color:#c8b487">{_esc(s['ticker'])}</span>
          <span style="font-size:10px;font-weight:400;color:{chip_col};margin-top:2px">{bias}</span>
          <div style="width:60px;height:6px;background:#EEE;border-radius:3px;margin-top:4px;overflow:hidden">
            <div style="height:6px;background:#1B6F4A;border-radius:3px;width:{call_pct:.0f}%"></div>
          </div>
          <span style="font-size:9px;color:#AAA;margin-top:2px">C:{call_pct:.0f}% P:{100-call_pct:.0f}%</span>
          <span style="font-size:9px;color:#888;margin-top:1px">{prem_str}</span>
        </div>"""

    options_panel = f"""
    <div style="margin-bottom:16px">
      <div style="display:flex;gap:8px;flex-wrap:wrap;margin-bottom:14px">{sent_chips}</div>
      <p style="font-size:11px;color:#AAA;margin-bottom:10px">Sentiment bar = Call % of premium. Green = bullish money, red = bearish.</p>
    </div>
    <div style="overflow-x:auto;border:1px solid #241f18;border-radius:6px">
      <table style="width:100%;border-collapse:collapse">
        <thead><tr style="background:#241f18;border-bottom:2px solid #241f18">
          <th style="padding:8px;text-align:left;font-size:10px;color:#888;font-weight:400">Ticker</th>
          <th style="padding:8px;text-align:left;font-size:10px;color:#888;font-weight:400">Side</th>
          <th style="padding:8px;text-align:left;font-size:10px;color:#888;font-weight:400">Strike</th>
          <th style="padding:8px;text-align:left;font-size:10px;color:#888;font-weight:400">Expiry</th>
          <th style="padding:8px;text-align:right;font-size:10px;color:#888;font-weight:400">Volume</th>
          <th style="padding:8px;text-align:right;font-size:10px;color:#888;font-weight:400">OI</th>
          <th style="padding:8px;text-align:right;font-size:10px;color:#888;font-weight:400">Vol/OI</th>
          <th style="padding:8px;text-align:right;font-size:10px;color:#888;font-weight:400">Premium</th>
          <th style="padding:8px;text-align:right;font-size:10px;color:#888;font-weight:400">IV</th>
        </tr></thead>
        <tbody>{flow_rows}</tbody>
      </table>
    </div>
    <p style="font-size:11px;color:#AAA;margin-top:6px">{opt_count} unusual flows across top {opt_scanned} tickers · Vol/OI &gt; 1.5 · Premium &gt; $50K · {opt_as_of}</p>"""

    # ── Economic Calendar panel ───────────────────────────────────────────────
    events = econ_cal.get("events", [])
    cal_as_of = econ_cal.get("as_of", "—")

    impact_col = {"high": "#B83232", "medium": "#c8b487", "low": "#1B6F4A"}
    impact_bg  = {"high": "#FEF0EF", "medium": "#FEF9EC", "low": "#EAF5EE"}

    cal_cards = ""
    for ev in events:
        imp = ev.get("impact", "low")
        ic  = impact_col.get(imp, "#999")
        ib  = impact_bg.get(imp, "#241f18")
        d   = ev.get("days_until", "?")
        d_str = "TODAY" if d == 0 else (f"in {d}d" if isinstance(d, int) else str(d))
        d_col = "#B83232" if isinstance(d, int) and d <= 3 else ("#c8b487" if isinstance(d, int) and d <= 7 else "#555")
        cal_cards += f"""<div style="display:flex;gap:14px;align-items:flex-start;padding:12px 14px;background:{ib};border:1px solid {ic}33;border-left:4px solid {ic};border-radius:6px">
          <div style="text-align:center;min-width:52px">
            <div style="font-size:18px">{_esc(ev.get('emoji','📅'))}</div>
            <div style="font-size:11px;font-weight:400;color:{d_col}">{d_str}</div>
          </div>
          <div>
            <div style="font-size:13px;font-weight:400;color:#1A1A1A">{_esc(ev.get('name',''))}</div>
            <div style="font-size:11px;color:#888;margin-top:2px">{_esc(ev.get('date',''))} &nbsp;·&nbsp; <span style="font-weight:400;color:{ic};text-transform:uppercase">{imp} impact</span></div>
          </div>
        </div>"""

    if not cal_cards:
        cal_cards = '<p style="color:#AAA;font-size:13px">No upcoming events — run step_economic_calendar.py</p>'

    cal_panel = f"""<div style="display:flex;flex-direction:column;gap:10px">{cal_cards}</div>
    <p style="font-size:11px;color:#AAA;margin-top:8px">As of {cal_as_of}</p>"""

    return f"""<section id="sec-flow" class="tab-section">
  <div class="container">
    <p class="eyebrow">Market Intelligence</p>
    <h2 class="section-head">Options flow · Sector rotation · Economic events</h2>
    <div class="rule"></div>

    <div class="two-col-even" style="gap:32px">
      <div style="flex:2;min-width:0">
        <p class="eyebrow" style="margin-bottom:8px">ETF Sector Rotation</p>
        <h3 style="font-family:'Playfair Display',serif;font-size:18px;font-weight:400;color:#1A1A1A;margin:0 0 12px">Which sectors are getting money?</h3>
        {etf_panel}
      </div>
      <div style="flex:1;min-width:220px">
        <p class="eyebrow" style="margin-bottom:8px">Economic Calendar</p>
        <h3 style="font-family:'Playfair Display',serif;font-size:18px;font-weight:400;color:#1A1A1A;margin:0 0 12px">Upcoming macro events</h3>
        {cal_panel}
      </div>
    </div>

    <div class="mt36">
      <p class="eyebrow" style="margin-bottom:8px">Options Unusual Flow</p>
      <h3 style="font-family:'Playfair Display',serif;font-size:18px;font-weight:400;color:#1A1A1A;margin:0 0 4px">Large option bets — who's betting and which direction?</h3>
      <p class="lead" style="margin:0 0 14px">Vol/OI &gt; 1.5 means new money is entering (not existing holders). Premium &gt; $50K filters out noise. Green = call (bullish bet), red = put (bearish or hedge).</p>
      {options_panel}
    </div>

  </div>
</section>"""


def _build_quant_qc_tab() -> str:
    """Quantitative quality control — IC audit, stress test, factor correlation, beta, extended backtest."""
    import json as _json, pathlib as _pl
    ROOT = _pl.Path(__file__).parent

    def _load(fname):
        p = ROOT / fname
        if p.exists():
            try:
                with open(p) as f: return _json.load(f)
            except Exception: pass
        return {}

    ic    = _load("ic_audit_report.json")
    st    = _load("stress_test_results.json")
    fc    = _load("factor_corr_matrix.json")
    tc    = _load("tc_verification.json")
    ext   = _load("backtest_extended.json")

    def _esc(s): import html as _h; return _h.escape(str(s)) if s is not None else ""

    # ── IC Audit card ─────────────────────────────────────────────────────────
    ic_verdict  = ic.get("verdict", "UNKNOWN")
    ic_color    = "#C0392B" if "NOT" in ic_verdict else ("#F39C12" if "LOW" in ic_verdict else "#27AE60")
    ic_n        = ic.get("n_periods", "?")
    ic_val      = ic.get("ic_value")
    ic_ci_lo    = ic.get("ic_ci_low")
    ic_ci_hi    = ic.get("ic_ci_high")
    ic_val_str  = f"{ic_val:+.4f}" if isinstance(ic_val, float) else "N/A"
    ic_ci_str   = f"[{ic_ci_lo:.3f}, {ic_ci_hi:.3f}]" if isinstance(ic_ci_lo, float) else "N/A"
    ic_recs     = ic.get("recommendations", [])
    ic_rec_html = "".join(f'<li style="margin-bottom:6px">{_esc(r)}</li>' for r in ic_recs)

    ic_card = f"""
    <div style="background:#fff;border:1px solid #241f18;border-radius:8px;padding:20px;margin-bottom:20px;border-left:5px solid {ic_color}">
      <div style="display:flex;align-items:center;gap:12px;margin-bottom:12px">
        <div style="background:{ic_color};color:#fff;font-size:11px;font-weight:400;padding:4px 10px;border-radius:12px;letter-spacing:.8px">
          {'⚠ ' if 'NOT' in ic_verdict else ''}IC AUDIT
        </div>
        <span style="font-size:18px;font-weight:500;color:{ic_color}">{_esc(ic_verdict)}</span>
      </div>
      <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:16px;margin-bottom:16px">
        <div><p style="font-size:10px;color:#888;text-transform:uppercase;margin-bottom:3px">Observations</p>
          <p style="font-size:22px;font-weight:500;color:{'#C0392B' if isinstance(ic_n,int) and ic_n<60 else '#1B6F4A'}">{ic_n}</p>
          <p style="font-size:10px;color:#888">need 60+</p></div>
        <div><p style="font-size:10px;color:#888;text-transform:uppercase;margin-bottom:3px">Computed IC</p>
          <p style="font-size:22px;font-weight:500;color:{ic_color}">{ic_val_str}</p>
          <p style="font-size:10px;color:#888">dashboard shows +0.370</p></div>
        <div><p style="font-size:10px;color:#888;text-transform:uppercase;margin-bottom:3px">95% CI</p>
          <p style="font-size:14px;font-weight:400;color:#555;margin-top:5px">{ic_ci_str}</p>
          <p style="font-size:10px;color:#888">includes zero → not sig.</p></div>
        <div><p style="font-size:10px;color:#888;text-transform:uppercase;margin-bottom:3px">Look-ahead Risk</p>
          <p style="font-size:14px;font-weight:400;color:#1B6F4A;margin-top:5px">LOW</p>
          <p style="font-size:10px;color:#888">no systematic leakage found</p></div>
      </div>
      <div style="background:#FEF9F0;border-radius:6px;padding:12px">
        <p style="font-size:11px;font-weight:400;color:#c8b487;margin-bottom:8px">What To Do</p>
        <ul style="font-size:12px;color:#555;padding-left:18px;margin:0">{ic_rec_html}</ul>
      </div>
    </div>"""

    # ── Transaction Costs card ────────────────────────────────────────────────
    tc_color = "#27AE60" if tc.get("is_tc_netted") else "#C0392B"
    tc_card = f"""
    <div style="background:#fff;border:1px solid #241f18;border-radius:8px;padding:20px;margin-bottom:20px;border-left:5px solid {tc_color}">
      <div style="display:flex;align-items:center;gap:12px;margin-bottom:12px">
        <div style="background:{tc_color};color:#fff;font-size:11px;font-weight:400;padding:4px 10px;border-radius:12px">TC AUDIT</div>
        <span style="font-size:18px;font-weight:500;color:{tc_color}">{'✓ CONFIRMED NETTED' if tc.get('is_tc_netted') else '✗ NOT NETTED'}</span>
      </div>
      <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:16px">
        <div><p style="font-size:10px;color:#888;text-transform:uppercase;margin-bottom:3px">Total TC Paid</p>
          <p style="font-size:22px;font-weight:500;color:#555">{tc.get('total_tc_bps','?')} bps</p></div>
        <div><p style="font-size:10px;color:#888;text-transform:uppercase;margin-bottom:3px">Net CAGR</p>
          <p style="font-size:22px;font-weight:500;color:#1B6F4A">{tc.get('net_cagr','?')}%</p></div>
        <div><p style="font-size:10px;color:#888;text-transform:uppercase;margin-bottom:3px">Gross CAGR</p>
          <p style="font-size:22px;font-weight:500;color:#555">{tc.get('gross_cagr_est','?')}%</p></div>
        <div><p style="font-size:10px;color:#888;text-transform:uppercase;margin-bottom:3px">TC Drag/Year</p>
          <p style="font-size:22px;font-weight:500;color:#B83232">−{tc.get('tc_drag_pct','?')}%</p></div>
      </div>
    </div>"""

    # ── Stress Test table ─────────────────────────────────────────────────────
    disclaimer = _esc(st.get("disclaimer", "Simulation only — current rankings applied retroactively."))
    periods    = st.get("periods", [])
    st_rows = ""
    for p in periods:
        sr   = p.get("strategy_ret", 0) or 0
        spr  = p.get("spy_ret",      0) or 0
        smdd = p.get("strategy_mdd", 0) or 0
        alpha_val = sr - spr
        sc = "#1B6F4A" if sr > 0 else "#C0392B"
        ac = "#1B6F4A" if alpha_val > 0 else "#C0392B"
        st_rows += f"""<tr>
          <td style="font-weight:400;font-size:12px">{_esc(p.get('name',''))}</td>
          <td style="color:#888;font-size:11px">{_esc(p.get('start',''))} → {_esc(p.get('end',''))}</td>
          <td style="color:{sc};font-weight:400;font-variant-numeric:tabular-nums">{sr*100:+.1f}%</td>
          <td style="font-variant-numeric:tabular-nums">{spr*100:+.1f}%</td>
          <td style="color:{ac};font-weight:400;font-variant-numeric:tabular-nums">{alpha_val*100:+.1f}%</td>
          <td style="color:#B83232;font-variant-numeric:tabular-nums">{smdd*100:.1f}%</td>
        </tr>"""
    st_card = f"""
    <div style="background:#fff;border:1px solid #241f18;border-radius:8px;padding:20px;margin-bottom:20px">
      <p style="font-size:13px;font-weight:400;color:#c8b487;margin-bottom:4px">Stress Tests — 2020 &amp; 2022 Crisis Periods</p>
      <p style="font-size:10px;color:#c8b487;margin-bottom:14px">⚠ {disclaimer}</p>
      <div style="overflow-x:auto"><table style="width:100%;border-collapse:collapse;font-size:12px">
        <thead><tr style="background:#F5F4F0">
          <th style="padding:8px;text-align:left">Period</th><th style="padding:8px;text-align:left">Dates</th>
          <th style="padding:8px;text-align:right">Strategy</th><th style="padding:8px;text-align:right">SPY</th>
          <th style="padding:8px;text-align:right">Alpha</th><th style="padding:8px;text-align:right">Max DD</th>
        </tr></thead>
        <tbody>{st_rows}</tbody>
      </table></div>
    </div>"""

    # ── Factor Correlation card ───────────────────────────────────────────────
    corr_m     = fc.get("correlation_matrix", {})
    high_pairs = fc.get("high_corr_pairs", [])
    vif_scores = fc.get("vif_scores", {})
    factors    = list(corr_m.keys()) if corr_m else []
    # Build mini correlation table
    header_cells = "".join(f'<th style="padding:4px 8px;font-size:9px;background:#F5F4F0;text-align:center;max-width:60px;overflow:hidden">{f[:6]}</th>' for f in factors)
    corr_rows_html = ""
    for f1 in factors:
        def _cv(f1, f2):
            v = corr_m.get(f1, {}).get(f2)
            return float(v) if v is not None else 0.0
        row_cells = "".join(
            f'<td style="padding:4px 6px;text-align:center;font-size:10px;font-variant-numeric:tabular-nums;background:{_corr_bg(_cv(f1,f2))}">{_cv(f1,f2):.2f}</td>'
            for f2 in factors
        ) if corr_m else ""
        corr_rows_html += f'<tr><td style="padding:4px 8px;font-size:10px;font-weight:400;white-space:nowrap;background:#F5F4F0">{f1[:12]}</td>{row_cells}</tr>'
    hp_html = "".join(f'<span style="background:#FEE2E2;color:#B83232;padding:2px 8px;border-radius:10px;font-size:11px;margin:2px">{_esc(hp)}</span>' for hp in high_pairs) or '<span style="color:#1B6F4A;font-size:11px">None above 0.60 threshold</span>'
    fc_card = f"""
    <div style="background:#fff;border:1px solid #241f18;border-radius:8px;padding:20px;margin-bottom:20px">
      <p style="font-size:13px;font-weight:400;color:#c8b487;margin-bottom:4px">Factor Correlation &amp; Redundancy</p>
      <div style="margin-bottom:12px"><span style="font-size:11px;color:#888">High correlation pairs (|r|&gt;0.60): </span>{hp_html}</div>
      <div style="overflow-x:auto;margin-bottom:14px"><table style="border-collapse:collapse">
        <thead><tr><th style="padding:4px 8px;font-size:9px;background:#F5F4F0"></th>{header_cells}</tr></thead>
        <tbody>{corr_rows_html}</tbody>
      </table></div>
      <div style="display:flex;gap:12px;flex-wrap:wrap">
        {''.join(f'<div style="background:#F5F4F0;border-radius:6px;padding:8px 14px"><p style="font-size:9px;color:#888;text-transform:uppercase">{_esc(k)} VIF</p><p style="font-size:16px;font-weight:400;color:{"#C0392B" if isinstance(v,float) and v==v and v>5 else "#3a3128"}">{f"{v:.2f}" if isinstance(v,float) and v==v else "—"}</p></div>' for k, v in vif_scores.items())}
      </div>
    </div>"""

    # ── Extended Backtest year table ──────────────────────────────────────────
    yearly  = ext.get("yearly_stats", [])
    ext_disclaimer = _esc(ext.get("disclaimer", ""))
    ext_rows = ""
    for y in yearly:
        sr   = y.get("strategy_ret", 0) or 0
        spr  = y.get("spy_ret",      0) or 0
        alp  = y.get("alpha",        0) or 0
        mdd  = y.get("mdd",          0) or 0
        sc   = "#1B6F4A" if sr > 0 else "#C0392B"
        ac   = "#1B6F4A" if alp > 0 else "#C0392B"
        ext_rows += f"""<tr style="border-bottom:1px solid #241f18">
          <td style="padding:8px;font-weight:400">{_esc(str(y.get('year','')))}</td>
          <td style="padding:8px;color:{sc};font-weight:400;font-variant-numeric:tabular-nums">{sr*100:+.1f}%</td>
          <td style="padding:8px;font-variant-numeric:tabular-nums">{spr*100:+.1f}%</td>
          <td style="padding:8px;color:{ac};font-weight:400;font-variant-numeric:tabular-nums">{alp*100:+.1f}%</td>
          <td style="padding:8px;color:#B83232;font-variant-numeric:tabular-nums">{mdd*100:.1f}%</td>
        </tr>"""
    ext_card = f"""
    <div style="background:#fff;border:1px solid #241f18;border-radius:8px;padding:20px;margin-bottom:20px">
      <p style="font-size:13px;font-weight:400;color:#c8b487;margin-bottom:4px">Extended Backtest 2019–2026 (Year-by-Year)</p>
      <p style="font-size:10px;color:#c8b487;margin-bottom:14px">⚠ {ext_disclaimer}</p>
      <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:16px;margin-bottom:16px">
        <div style="background:#F5F4F0;border-radius:6px;padding:12px">
          <p style="font-size:10px;color:#888;text-transform:uppercase;margin-bottom:3px">Strategy CAGR</p>
          <p style="font-size:22px;font-weight:500;color:#c8b487">{ext.get('cagr','?')}%</p></div>
        <div style="background:#F5F4F0;border-radius:6px;padding:12px">
          <p style="font-size:10px;color:#888;text-transform:uppercase;margin-bottom:3px">SPY CAGR</p>
          <p style="font-size:22px;font-weight:500;color:#555">{ext.get('spy_cagr','?')}%</p></div>
        <div style="background:#F5F4F0;border-radius:6px;padding:12px">
          <p style="font-size:10px;color:#888;text-transform:uppercase;margin-bottom:3px">Max Drawdown</p>
          <p style="font-size:22px;font-weight:500;color:#B83232">{ext.get('mdd','?')}%</p></div>
      </div>
      <div style="overflow-x:auto"><table style="width:100%;border-collapse:collapse;font-size:12px">
        <thead><tr style="background:#F5F4F0">
          <th style="padding:8px;text-align:left">Year</th>
          <th style="padding:8px;text-align:right">Strategy</th><th style="padding:8px;text-align:right">SPY</th>
          <th style="padding:8px;text-align:right">Alpha</th><th style="padding:8px;text-align:right">Max DD</th>
        </tr></thead>
        <tbody>{ext_rows}</tbody>
      </table></div>
    </div>"""

    # ── Honest performance card: rigorous backtest + live-IC significance ──
    honest_card = ""
    try:
        import json as _json
        rb = _json.load(open(ROOT / "rigorous_backtest.json")) if (ROOT / "rigorous_backtest.json").exists() else {}
        lo = (rb.get("long_only") or {}); ls = (rb.get("long_short") or {})
        lof = lo.get("full_net", {}); lsf = ls.get("full_net", {}); spy = lo.get("spy_buy_hold", {})
        def _pc(x):
            try: return f"{float(x)*100:+.1f}%"
            except Exception: return "—"
        # live IC verdict
        icv = "—"
        icp = ROOT / "live_ic_report.md"
        if icp.exists():
            for _l in icp.read_text().splitlines():
                if "SIGNIFICANT" in _l or "CONCLUSIVE" in _l or "NO SIGNIFICANT" in _l:
                    icv = _l.replace("**", "").strip(); break
        # PEAD research result (first real-edge attempt)
        pead = _json.load(open(ROOT / "pead_results.json")) if (ROOT / "pead_results.json").exists() else {}
        pead_line = ""
        if pead and "sue_ic_t" in pead:
            _pt = pead.get("sue_ic_t"); _pa = pead.get("alpha_annual_after_costs")
            _verdict = "no significant edge in large-cap (t≈0) — as theory predicts; the anomaly lives in small/under-covered names" if (_pt is not None and abs(_pt) < 2) else "shows signal"
            pead_line = (f'<p style="font-size:12px;color:#a89c8c;margin:8px 0 0"><strong style="color:#c8b487">'
                f'Edge research — PEAD (earnings drift):</strong> SUE→return IC t={_pt}, '
                f'alpha {(_pa or 0)*100:+.1f}%/yr → {_verdict}. '
                f'<span style="color:#8faa9a">Framework correctly falsified a dead edge.</span></p>')
        if lof:
            honest_card = f"""
    <div style="background:#16140f;border:1px solid #453a2c;border-left:4px solid #c8b487;border-radius:8px;padding:20px 24px;margin-bottom:24px">
      <div style="font-size:11px;letter-spacing:1.5px;text-transform:uppercase;color:#c8b487;font-weight:400;margin-bottom:6px">Honest Backtest — realistic costs · no look-ahead · survivorship-controlled · {lo.get('period','')}</div>
      <p style="font-size:12px;color:#a89c8c;margin:0 0 14px">Point-in-time price signals over 16y deep history, restricted to names actually in the S&amp;P 500 at each date (PIT membership). Replaces the old inflated headline (Sharpe ~5 = look-ahead + no costs + survivorship bias).</p>
      <div style="overflow-x:auto"><table style="width:100%;border-collapse:collapse;font-size:13px">
        <thead><tr style="border-bottom:2px solid #453a2c">
          <th style="text-align:left;padding:6px 10px;color:#8a7f70;font-size:10px;text-transform:uppercase">Config</th>
          <th style="text-align:right;padding:6px 10px;color:#8a7f70;font-size:10px;text-transform:uppercase">Sharpe</th>
          <th style="text-align:right;padding:6px 10px;color:#8a7f70;font-size:10px;text-transform:uppercase">CAGR</th>
          <th style="text-align:right;padding:6px 10px;color:#8a7f70;font-size:10px;text-transform:uppercase">Max DD</th>
          <th style="text-align:right;padding:6px 10px;color:#8a7f70;font-size:10px;text-transform:uppercase">Beta</th>
          <th style="text-align:right;padding:6px 10px;color:#8a7f70;font-size:10px;text-transform:uppercase">Alpha/yr</th>
        </tr></thead><tbody>
        <tr><td style="padding:6px 10px;color:#f4ecdf;font-weight:400">Long-Only (top 30)</td>
          <td style="text-align:right;padding:6px 10px;color:#8faa9a;font-weight:400">{lof.get('sharpe','—')}</td>
          <td style="text-align:right;padding:6px 10px">{_pc(lof.get('cagr'))}</td>
          <td style="text-align:right;padding:6px 10px;color:#c68b83">{_pc(lof.get('max_dd'))}</td>
          <td style="text-align:right;padding:6px 10px">{lo.get('beta_to_spy','—')}</td>
          <td style="text-align:right;padding:6px 10px;color:#c8b487;font-weight:400">{_pc(lo.get('alpha_annual_after_costs'))}</td></tr>
        <tr><td style="padding:6px 10px;color:#f4ecdf">Long-Short (top/bot 30)</td>
          <td style="text-align:right;padding:6px 10px;color:#c68b83;font-weight:400">{lsf.get('sharpe','—')}</td>
          <td style="text-align:right;padding:6px 10px;color:#c68b83">{_pc(lsf.get('cagr'))}</td>
          <td style="text-align:right;padding:6px 10px;color:#c68b83">{_pc(lsf.get('max_dd'))}</td>
          <td style="text-align:right;padding:6px 10px">{ls.get('beta_to_spy','—')}</td>
          <td style="text-align:right;padding:6px 10px;color:#c68b83">{_pc(ls.get('alpha_annual_after_costs'))}</td></tr>
        <tr><td style="padding:6px 10px;color:#9a8e80">SPY buy &amp; hold</td>
          <td style="text-align:right;padding:6px 10px">{spy.get('sharpe','—')}</td>
          <td style="text-align:right;padding:6px 10px">{_pc(spy.get('cagr'))}</td>
          <td style="text-align:right;padding:6px 10px">{_pc(spy.get('max_dd'))}</td>
          <td style="text-align:right;padding:6px 10px">1.00</td>
          <td style="text-align:right;padding:6px 10px">0.0%</td></tr>
      </tbody></table></div>
      <p style="font-size:12px;color:#a89c8c;margin:14px 0 0"><strong style="color:#c68b83">Short book loses money</strong> (now dropped — long-only recommended book). Once survivorship bias is removed, <strong>long-only alpha ≈ 0</strong> (R²={lo.get('r2_market','—')} to market) — the strategy is essentially a beta play, not stock-selection edge.</p>
      <p style="font-size:12px;color:#a89c8c;margin:8px 0 0"><strong style="color:#c8b487">Live signal edge:</strong> {icv}</p>
      <p style="font-size:12px;color:#a89c8c;margin:8px 0 0"><strong style="color:#c8b487">Overfitting check:</strong> a single momentum signal matches/beats the 4-signal blend — the extra signals add no edge (evidence to simplify the live 10+ signal stack).</p>
      {pead_line}
    </div>"""
    except Exception:
        honest_card = ""

    return f"""<section id="sec-qc" class="tab-section">
  <div class="container">
    <p class="eyebrow">System · Quant QC</p>
    <h2 class="section-head">Quantitative Quality Control — Model Integrity Report</h2>
    <div class="rule"></div>
    <p style="color:#666;font-size:13px;margin-bottom:24px">Independent honesty checks on the Canyon system — including where the model falls short.</p>
    {honest_card}
    {ic_card}
    {tc_card}
    {st_card}
    {fc_card}
    {ext_card}
  </div>
</section>"""


def _intraday_panel() -> str:
    """日内感知: 盘中牛熊 + 盘中新闻 (Intraday News) + 盘中入场时机 (Intraday Entry Timing)。"""
    p = ROOT / "intraday_signals.json"
    if not p.exists():
        return ""
    try:
        j = json.load(open(p))
    except Exception:
        return ""
    reg = j.get("regime", {})
    rname = reg.get("regime", "—")
    rcol = "#8faa9a" if rname == "进攻" else "#c68b83" if rname == "避险" else "#c0a878"
    news = j.get("news", [])[:6]
    entry = j.get("entry", [])
    nrows = "".join(
        f'<div style="display:flex;gap:8px;padding:2px 0;font-size:11.5px"><span style="color:#8a7f70;white-space:nowrap">{n.get("mins_ago","?")}分前</span>'
        f'<span style="color:#c8b487;font-weight:400">{n.get("ticker","")}</span>'
        f'<span style="color:#cabeae">{n.get("title","")}</span></div>' for n in news) or '<div style="color:#8a7f70;font-size:12px">近8小时无新闻</div>'
    erows = ""
    for e in entry:
        sig = str(e.get("信号", ""))
        scol = "#8faa9a" if "可入场" in sig else "#c0a878" if "等回调" in sig or "偏强" in sig else "#c68b83"
        erows += (f'<tr><td style="padding:5px 10px;color:#f4ecdf;font-weight:400">{e.get("ticker","")}</td>'
                  f'<td style="padding:5px 10px;text-align:right;color:#a89c8c;font-size:11px;font-variant-numeric:tabular-nums">{e.get("px","")}</td>'
                  f'<td style="padding:5px 10px;text-align:right;color:#8a7f70;font-size:11px;font-variant-numeric:tabular-nums">{e.get("vwap","")}</td>'
                  f'<td style="padding:5px 10px;color:{scol};font-size:11px">{sig}</td></tr>')
    return f"""
    <div style="margin-bottom:26px;background:#16140f;border:1px solid #3a3128;border-radius:8px;padding:16px 18px">
      <div style="display:flex;justify-content:space-between;align-items:baseline;flex-wrap:wrap;gap:8px;margin-bottom:12px">
        <span style="font-size:11px;color:#8a7f70;text-transform:uppercase;letter-spacing:.14em">日内感知层 (Intraday Awareness Layer) · 盘中实时(免费日内数据)</span>
        <span style="font-size:11px;color:#8a7f70">{j.get('updated','')} · 收盘后=最后交易日</span>
      </div>
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:18px">
        <div>
          <div style="font-size:10px;color:#8a7f70;text-transform:uppercase;letter-spacing:.12em;margin-bottom:4px">① 盘中牛熊判读 (Intraday Bull/Bear Read)</div>
          <div style="font-size:24px;font-weight:400;color:{rcol};font-family:'Financier Display',Georgia,serif">{rname}<span style="font-size:13px;color:#8a7f70;margin-left:8px">分{reg.get('score','')}</span></div>
          <div style="font-size:11px;color:#a89c8c;margin-top:4px">QQQ {reg.get('qqq_px','')} vs VWAP {reg.get('vwap','')} · 当日{reg.get('day_chg_%','')}% · VIX {reg.get('vix','')}</div>
          <div style="font-size:11px;color:#8a7f70;margin-top:2px">{' / '.join(reg.get('reasons',[]))}</div>
          <div style="font-size:10px;color:#8a7f70;text-transform:uppercase;letter-spacing:.12em;margin:12px 0 4px">③ 盘中入场时机 (Intraday Entry Timing)(集中清单)</div>
          <div style="overflow-x:auto"><table style="width:100%;border-collapse:collapse"><tbody>{erows}</tbody></table></div>
        </div>
        <div>
          <div style="font-size:10px;color:#8a7f70;text-transform:uppercase;letter-spacing:.12em;margin-bottom:6px">② 盘中新闻 (Intraday News)(近8小时·分钟级)</div>
          {nrows}
        </div>
      </div>
      <p style="color:#746a5d;font-size:10.5px;margin-top:10px">诚实: 免费日内仅近7-60天历史, 可实时监控/择时, 无法深度回测验证。是盘中工具, 非"验证过必赚"。</p>
    </div>"""


def _ten_layer_matrix_panel() -> str:
    """The Canyon 10-layer decision matrix — for each name, where every layer L1..L10
    stands, plus the master action & reason. From master_10_layer_decision_matrix.csv."""
    p = ROOT / "master_10_layer_decision_matrix.csv"
    if not p.exists():
        return ""
    try:
        df = pd.read_csv(p)
    except Exception:
        return ""
    if df.empty or "ticker" not in df.columns:
        return ""
    import html as _h
    def _esc(s): return _h.escape(str(s)) if s is not None else ""
    C_CARD, C_INK, C_MUTE, C_SUB, C_GOLD = "#16140f", "#f0e9da", "#8f866f", "#b0a68f", "#c8b487"
    # status → color
    def _scol(v):
        v = str(v).upper()
        if v in ("OK", "CLEAR", "PASS", "READY", "GREEN"): return "#8faa9a"
        if v in ("PARTIAL", "WAIT", "PENDING_MANUAL_CHECKS", "PAPER_ONLY"): return "#cdbd8f"
        if v in ("BLOCKED", "RED"): return "#c68b83"
        return "#4a4433"  # NO_DATA / SKIP / TRACE / ALREADY_CLOSED / other = dim
    LAYERS = [("L1_data", "L1"), ("L2_macro", "L2"), ("L3_sector", "L3"), ("L4_fundamental", "L4"),
              ("L5_event", "L5"), ("L6_price", "L6"), ("L7_options", "L7"), ("L8_risk", "L8"),
              ("L9_execution", "L9"), ("L10_learning", "L10")]
    head_cells = "".join(f'<th style="padding:5px 6px;font-size:9px;color:{C_MUTE};font-weight:400;text-align:center">{lbl}</th>' for _, lbl in LAYERS)
    body = ""
    for _, r in df.iterrows():
        tk = _esc(str(r.get("ticker", "")))
        cells = ""
        for col, _lbl in LAYERS:
            v = str(r.get(col, ""))
            c = _scol(v)
            cells += f'<td title="{_esc(v)}" style="padding:4px;text-align:center"><span style="display:inline-block;width:11px;height:11px;border-radius:3px;background:{c}"></span></td>'
        action = _esc(str(r.get("master_action", "")))
        reason = _esc(str(r.get("master_reason", ""))[:90])
        body += (f'<tr style="border-top:1px solid #241f18">'
                 f'<td style="padding:6px 8px;font-size:12.5px;color:{C_GOLD};font-weight:400;white-space:nowrap">{tk}</td>'
                 f'{cells}'
                 f'<td style="padding:6px 8px;font-size:10.5px;color:{C_SUB};white-space:nowrap">{action}</td></tr>')
    legend = ('<div style="display:flex;gap:14px;flex-wrap:wrap;font-size:10px;color:#8f866f;margin-top:10px">'
              '<span><span style="display:inline-block;width:9px;height:9px;border-radius:2px;background:#8faa9a"></span> clear</span>'
              '<span><span style="display:inline-block;width:9px;height:9px;border-radius:2px;background:#cdbd8f"></span> partial/wait</span>'
              '<span><span style="display:inline-block;width:9px;height:9px;border-radius:2px;background:#c68b83"></span> blocked</span>'
              '<span><span style="display:inline-block;width:9px;height:9px;border-radius:2px;background:#4a4433"></span> no data</span></div>')
    return (f'<div style="margin-bottom:26px;background:{C_CARD};border:1px solid #241f18;border-radius:8px;padding:18px 20px">'
            f'<div style="font-size:10px;letter-spacing:.14em;text-transform:uppercase;color:{C_GOLD};margin-bottom:2px">10-Layer Decision Matrix</div>'
            f'<div style="font-size:19px;font-family:\'Baskerville\',Georgia,serif;color:{C_INK};margin-bottom:6px">Every layer, every name — where the build actually stands</div>'
            f'<div style="overflow-x:auto"><table style="border-collapse:collapse;width:100%;min-width:640px">'
            f'<thead><tr><th style="padding:5px 8px;font-size:9px;color:{C_MUTE};font-weight:400;text-align:left">Name</th>{head_cells}'
            f'<th style="padding:5px 8px;font-size:9px;color:{C_MUTE};font-weight:400;text-align:left">Master action</th></tr></thead>'
            f'<tbody>{body}</tbody></table></div>{legend}'
            f'<div style="font-size:10px;color:{C_MUTE};margin-top:8px">L1 Data · L2 Macro · L3 Sector · L4 Fundamental · L5 Event · L6 Price · L7 Options · L8 Risk · L9 Execution · L10 Learning</div>'
            f'</div>')


def _research_memo_panel() -> str:
    """Per-name research memos — verdict, what it is, why it matters, what would change
    my mind, and the multi-horizon view. From ticker_research_memo.csv."""
    p = ROOT / "ticker_research_memo.csv"
    if not p.exists():
        return ""
    try:
        df = pd.read_csv(p)
    except Exception:
        return ""
    if df.empty or "ticker" not in df.columns:
        return ""
    import html as _h
    def _esc(s): return _h.escape(str(s)) if s is not None else ""
    C_CARD, C_INK, C_MUTE, C_SUB = "#16140f", "#f0e9da", "#8f866f", "#b0a68f"
    C_GOLD, C_POS, C_NEG = "#c8b487", "#8faa9a", "#c68b83"
    try:
        df = df.sort_values("memo_rank")
    except Exception:
        pass
    rows = ""
    for _, r in df.head(12).iterrows():
        tk     = _esc(str(r.get("ticker", "")))
        verdict = _esc(str(r.get("current_verdict", ""))[:200])
        status = str(r.get("memo_status", ""))
        v_col  = C_NEG if ("block" in status.lower() or "block" in verdict.lower()) else C_POS
        whatis = _esc(str(r.get("what_is_this", ""))[:160])
        whym   = _esc(str(r.get("why_it_matters", ""))[:180])
        says   = _esc(str(r.get("what_source_says", ""))[:200])
        change = _esc(str(r.get("what_would_change_my_mind", ""))[:180])
        st     = _esc(str(r.get("short_term_view", ""))[:90])
        mt     = _esc(str(r.get("medium_term_view", ""))[:90])
        lt     = _esc(str(r.get("long_term_view", ""))[:90])
        def _line(lbl, txt, col=C_SUB):
            return (f'<div style="font-size:11.5px;color:{col};line-height:1.5;margin-top:3px"><b style="color:#a89c8c">{lbl}:</b> {txt}</div>'
                    if txt and txt != "nan" else "")
        horizons = ""
        if any(x and x != "nan" for x in (st, mt, lt)):
            horizons = (f'<div style="display:flex;gap:14px;flex-wrap:wrap;font-size:10.5px;color:{C_MUTE};margin-top:5px">'
                        f'<span><b style="color:#a89c8c">ST</b> {st}</span><span><b style="color:#a89c8c">MT</b> {mt}</span><span><b style="color:#a89c8c">LT</b> {lt}</span></div>')
        rows += (f'<div style="padding:13px 0;border-top:1px solid #241f18">'
                 f'<div style="display:flex;justify-content:space-between;align-items:baseline;gap:10px">'
                 f'<span style="font-size:15px;color:{C_GOLD};font-family:\'Baskerville\',Georgia,serif">{tk}</span>'
                 f'<span style="font-size:10px;color:{v_col};text-transform:uppercase;letter-spacing:.06em">{_esc(status)}</span></div>'
                 f'<div style="font-size:12px;color:{C_INK};line-height:1.5;margin-top:3px">{verdict}</div>'
                 + _line("What", whatis) + _line("Why it matters", whym) + _line("Source says", says)
                 + _line("Would change my mind", change, C_POS) + horizons + '</div>')
    return (f'<div style="margin-bottom:26px;background:{C_CARD};border:1px solid #241f18;border-radius:8px;padding:18px 20px">'
            f'<div style="font-size:10px;letter-spacing:.14em;text-transform:uppercase;color:{C_GOLD};margin-bottom:2px">Research Memos</div>'
            f'<div style="font-size:19px;font-family:\'Baskerville\',Georgia,serif;color:{C_INK};margin-bottom:6px">The written case per name — and what would change it</div>'
            f'{rows}</div>')


def _morning_brief_panel() -> str:
    """The desk's daily executive answer — read before looking at any ticker.
    Surfaces pm_morning_brief_cards.csv (card / value / why_it_matters / color)."""
    p = ROOT / "pm_morning_brief_cards.csv"
    if not p.exists():
        return ""
    try:
        df = pd.read_csv(p)
    except Exception:
        return ""
    if df.empty:
        return ""
    import html as _h
    def _esc(s): return _h.escape(str(s)) if s is not None else ""
    C_CARD, C_INK, C_SUB, C_GOLD = "#16140f", "#f0e9da", "#b0a68f", "#c8b487"
    CMAP = {"red": "#c68b83", "green": "#8faa9a", "amber": "#cdbd8f", "yellow": "#cdbd8f",
            "orange": "#cdbd8f", "blue": "#8fa8d8", "gray": "#8f866f", "grey": "#8f866f"}
    cards = ""
    for _, r in df.iterrows():
        col   = CMAP.get(str(r.get("color", "")).strip().lower(), C_GOLD)
        card  = _esc(str(r.get("card", "")))
        val   = _esc(str(r.get("value", "")))
        why   = _esc(str(r.get("why_it_matters", "")))
        cards += (f'<div style="padding:13px 16px;border-left:3px solid {col};background:#14110b;border-radius:6px;margin-bottom:8px">'
                  f'<div style="font-size:9.5px;text-transform:uppercase;letter-spacing:.12em;color:{col};margin-bottom:3px">{card}</div>'
                  f'<div style="font-size:14px;color:{C_INK};line-height:1.5;margin-bottom:3px">{val}</div>'
                  f'<div style="font-size:11.5px;color:{C_SUB};line-height:1.45">{why}</div></div>')
    return (f'<div style="margin-bottom:26px;background:{C_CARD};border:1px solid #241f18;border-radius:8px;padding:18px 20px">'
            f'<div style="font-size:10px;letter-spacing:.14em;text-transform:uppercase;color:{C_GOLD};margin-bottom:2px">PM Morning Brief</div>'
            f'<div style="font-size:19px;font-family:\'Baskerville\',Georgia,serif;color:{C_INK};margin-bottom:10px">The desk answer — read this first</div>'
            f'{cards}</div>')


def _sector_theme_panel() -> str:
    """Sector / theme depth — ranked cycle theses with stance, score, evidence,
    the risk against it, and the next research question. From sector_theme_depth_thesis.csv."""
    p = ROOT / "sector_theme_depth_thesis.csv"
    if not p.exists():
        return ""
    try:
        df = pd.read_csv(p)
    except Exception:
        return ""
    if df.empty or "cycle_thesis" not in df.columns:
        return ""
    import html as _h
    def _esc(s): return _h.escape(str(s)) if s is not None else ""
    C_CARD, C_INK, C_MUTE, C_SUB = "#16140f", "#f0e9da", "#8f866f", "#b0a68f"
    C_GOLD, C_POS, C_NEG = "#c8b487", "#8faa9a", "#c68b83"
    try:
        df = df.sort_values("thesis_score_0_100", ascending=False)
    except Exception:
        pass
    rows = ""
    for _, r in df.head(10).iterrows():
        theme = _esc(str(r.get("theme_or_subsector", "")))
        stance = _esc(str(r.get("stance", "")))
        thesis = _esc(str(r.get("cycle_thesis", "")))
        try:
            sc = float(r.get("thesis_score_0_100", 0))
        except Exception:
            sc = 0.0
        sc_col = C_POS if sc >= 66 else (C_GOLD if sc >= 40 else C_NEG)
        evid = _esc(str(r.get("supporting_evidence", ""))[:200])
        risk = _esc(str(r.get("contradiction_or_risk", ""))[:160])
        nextq = _esc(str(r.get("next_research_question", ""))[:140])
        r20 = r.get("avg_ret_20d_pct", ""); r63 = r.get("avg_ret_63d_pct", "")
        perf = ""
        try:
            perf = f'<span style="color:{C_MUTE};font-size:11px">20d {float(r20):+.1f}% · 63d {float(r63):+.1f}%</span>'
        except Exception:
            pass
        rows += (f'<div style="padding:13px 0;border-top:1px solid #241f18">'
                 f'<div style="display:flex;justify-content:space-between;align-items:baseline;gap:12px">'
                 f'<span style="font-size:13.5px;color:{C_INK}">{theme}</span>'
                 f'<span style="font-size:13px;color:{sc_col};font-variant-numeric:tabular-nums">{sc:.0f}<span style="font-size:10px;color:{C_MUTE}">/100</span></span></div>'
                 f'<div style="font-size:10px;text-transform:uppercase;letter-spacing:.08em;color:{C_GOLD};margin:2px 0 4px">{stance} · {perf}</div>'
                 f'<div style="font-size:12px;color:{C_SUB};line-height:1.5">{thesis}</div>'
                 f'<div style="font-size:11px;color:{C_MUTE};margin-top:4px;line-height:1.45"><b style="color:#a89c8c">Evidence:</b> {evid}</div>'
                 + (f'<div style="font-size:11px;color:{C_NEG};margin-top:2px;line-height:1.45"><b>Risk:</b> {risk}</div>' if risk and risk != "nan" else "")
                 + (f'<div style="font-size:11px;color:{C_MUTE};margin-top:2px"><b style="color:#a89c8c">Next:</b> {nextq}</div>' if nextq and nextq != "nan" else "")
                 + '</div>')
    return (f'<div style="margin-bottom:26px;background:{C_CARD};border:1px solid #241f18;border-radius:8px;padding:18px 20px">'
            f'<div style="font-size:10px;letter-spacing:.14em;text-transform:uppercase;color:{C_GOLD};margin-bottom:2px">Sector / Theme Depth</div>'
            f'<div style="font-size:19px;font-family:\'Baskerville\',Georgia,serif;color:{C_INK};margin-bottom:6px">Where the cycle favors capital — and the risk against each</div>'
            f'{rows}</div>')


def _strategy_thesis_panel() -> str:
    """Per-name strategy thesis with the four scenarios (base / bull / bear / no-trade),
    conviction, horizon, and the trigger to watch. From institutional_strategy_thesis_board.csv."""
    p = ROOT / "institutional_strategy_thesis_board.csv"
    if not p.exists():
        return ""
    try:
        df = pd.read_csv(p)
    except Exception:
        return ""
    if df.empty or "current_strategy_thesis" not in df.columns:
        return ""
    import html as _h
    def _esc(s): return _h.escape(str(s)) if s is not None else ""
    C_CARD, C_INK, C_MUTE, C_SUB = "#16140f", "#f0e9da", "#8f866f", "#b0a68f"
    C_GOLD, C_POS, C_NEG, C_WARN = "#c8b487", "#8faa9a", "#c68b83", "#cdbd8f"
    try:
        df = df.sort_values("thesis_quality_score", ascending=False)
    except Exception:
        pass
    rows = ""
    for _, r in df.head(12).iterrows():
        tk    = _esc(str(r.get("ticker", "")))
        sleeve = _esc(str(r.get("strategy_sleeve", "")))
        posture = _esc(str(r.get("strategy_posture", "")))
        conv   = _esc(str(r.get("conviction_tier", "")))
        horizon = _esc(str(r.get("best_horizon", "")))
        thesis = _esc(str(r.get("current_strategy_thesis", ""))[:260])
        base = _esc(str(r.get("base_case", ""))[:160])
        bull = _esc(str(r.get("bull_case", ""))[:150])
        bear = _esc(str(r.get("bear_case", ""))[:150])
        trig = _esc(str(r.get("trigger_to_watch", ""))[:160])
        p_col = C_NEG if ("de-risk" in posture.lower() or "preserv" in posture.lower()) else (C_POS if ("add" in posture.lower() or "accumulat" in posture.lower()) else C_GOLD)
        def _case(lbl, txt, col):
            return (f'<div style="font-size:11px;color:{C_SUB};line-height:1.45;margin-top:2px"><b style="color:{col}">{lbl}:</b> {txt}</div>'
                    if txt and txt != "nan" else "")
        rows += (f'<div style="padding:13px 0;border-top:1px solid #241f18">'
                 f'<div style="display:flex;justify-content:space-between;align-items:baseline;gap:10px">'
                 f'<span style="font-size:15px;color:{C_GOLD};font-family:\'Baskerville\',Georgia,serif">{tk}</span>'
                 f'<span style="font-size:10px;color:{p_col};text-transform:uppercase;letter-spacing:.06em">{posture}</span></div>'
                 f'<div style="font-size:10px;color:{C_MUTE};margin:1px 0 5px">{sleeve} · {conv} · best horizon {horizon}</div>'
                 f'<div style="font-size:12px;color:{C_INK};line-height:1.5">{thesis}</div>'
                 + _case("Base", base, C_MUTE) + _case("Bull", bull, C_POS) + _case("Bear", bear, C_NEG)
                 + (f'<div style="font-size:11px;color:{C_WARN};margin-top:3px"><b>Watch:</b> {trig}</div>' if trig and trig != "nan" else "")
                 + '</div>')
    return (f'<div style="margin-bottom:26px;background:{C_CARD};border:1px solid #241f18;border-radius:8px;padding:18px 20px">'
            f'<div style="font-size:10px;letter-spacing:.14em;text-transform:uppercase;color:{C_GOLD};margin-bottom:2px">Strategy Thesis Board</div>'
            f'<div style="font-size:19px;font-family:\'Baskerville\',Georgia,serif;color:{C_INK};margin-bottom:6px">Every name\'s case — base, bull, bear, and what would change it</div>'
            f'{rows}</div>')


def _macro_deep_panel() -> str:
    """Deep macro read — synthesizes the 4-week regime outlook into a written thesis:
    per-signal value, trend, plain-English interpretation, and the trigger to watch.
    Turns the rich-but-buried macro_regime_outlook.json into an analyst-style read."""
    p = ROOT / "macro_regime_outlook.json"
    if not p.exists():
        return ""
    try:
        d = json.load(open(p))
    except Exception:
        return ""
    import html as _h
    def _esc(s): return _h.escape(str(s)) if s is not None else ""

    C_CARD, C_INK, C_MUTE, C_SUB = "#16140f", "#f0e9da", "#8f866f", "#b0a68f"
    C_GOLD, C_POS, C_NEG, C_WARN = "#c8b487", "#8faa9a", "#c68b83", "#cdbd8f"

    comp    = d.get("composite", {})
    bp      = comp.get("bear_prob")
    label   = str(comp.get("label", ""))
    delta   = comp.get("bear_prob_delta")
    signals = d.get("signals", {})
    as_of   = _esc(str(d.get("as_of", "")))

    hp_col = C_POS if (bp is not None and bp < 25) else (C_WARN if (bp is not None and bp < 50) else C_NEG)
    delta_txt = ""
    if delta is not None:
        d_col   = C_POS if delta < 0 else (C_NEG if delta > 0 else C_MUTE)
        d_arrow = "▼" if delta < 0 else ("▲" if delta > 0 else "→")
        d_word  = "improving" if delta < 0 else ("deteriorating" if delta > 0 else "flat")
        delta_txt = f'<span style="color:{d_col};font-size:13px"> {d_arrow} {abs(delta):.1f}pp WoW ({d_word})</span>'

    # Synthesize a one-line thesis from the signal scores.
    ok_sigs  = [s for s in signals.values() if s.get("ok", True)]
    supportive = [s.get("name", "") for s in ok_sigs
                  if float(s.get("bear_score", 0)) <= float(s.get("max_score", 2) or 2) * 0.25]
    watch_now  = [s.get("name", "") for s in ok_sigs
                  if float(s.get("bear_score", 0)) >= float(s.get("max_score", 2) or 2) * 0.6]
    thesis = f"{len(supportive)} of {len(ok_sigs)} macro pillars supportive"
    thesis += f"; the strain is in {', '.join(watch_now)}" if watch_now else "; no pillar is flashing red"

    rows = ""
    for s in ok_sigs:
        nm     = _esc(s.get("name", ""))
        disp   = _esc(s.get("display", str(s.get("value", ""))))
        trend  = _esc(s.get("trend_label", ""))
        interp = _esc(s.get("interpretation", ""))
        watch  = _esc(s.get("watch_for", ""))
        bs = float(s.get("bear_score", 0)); ms = float(s.get("max_score", 2) or 2)
        frac = max(0.0, min(1.0, bs / ms))
        bar_col = C_POS if frac < 0.25 else (C_WARN if frac < 0.6 else C_NEG)
        rows += (f'<div style="padding:12px 0;border-top:1px solid #241f18">'
                 f'<div style="display:flex;justify-content:space-between;align-items:baseline;gap:12px">'
                 f'<span style="font-size:13.5px;color:{C_INK}">{nm}</span>'
                 f'<span style="font-size:13px;color:{C_GOLD};font-variant-numeric:tabular-nums">{disp} '
                 f'<span style="color:{C_MUTE};font-size:11px">· {trend}</span></span></div>'
                 f'<div style="height:3px;background:#241f18;border-radius:2px;margin:7px 0 6px">'
                 f'<div style="height:3px;width:{frac*100:.0f}%;background:{bar_col};border-radius:2px"></div></div>'
                 f'<div style="font-size:12px;color:{C_SUB};line-height:1.5">{interp}</div>'
                 f'<div style="font-size:11px;color:{C_MUTE};margin-top:3px">Watch: {watch}</div></div>')

    bp_disp = f"{bp:.1f}%" if bp is not None else "—"
    return (f'<div style="margin-bottom:26px;background:{C_CARD};border:1px solid #241f18;border-radius:8px;padding:18px 20px">'
            f'<div style="font-size:10px;letter-spacing:.14em;text-transform:uppercase;color:{C_GOLD};margin-bottom:2px">Macro Regime · 4-Week Outlook</div>'
            f'<div style="display:flex;align-items:baseline;gap:14px;flex-wrap:wrap;margin-bottom:4px">'
            f'<span style="font-size:34px;color:{hp_col};font-family:\'Baskerville\',Georgia,serif;line-height:1">{bp_disp}</span>'
            f'<span style="font-size:15px;color:{hp_col}">{_esc(label)} · recession-risk (4wk)</span>{delta_txt}</div>'
            f'<div style="font-size:12.5px;color:{C_SUB};line-height:1.55;margin-bottom:2px">{thesis}. As of {as_of}.</div>'
            f'{rows}</div>')


def _insider_ls_panel() -> str:
    """The combined long/short insider book — the actual product's honest numbers.
    Reads insider_ls_backtest.json (from canyon_insider_ls_backtest.py)."""
    import json as _json, html as _h
    def _esc(s): return _h.escape(str(s)) if s is not None else ""
    C_CARD, C_INK, C_MUTE, C_SUB = "#16140f", "#f0e9da", "#8f866f", "#b0a68f"
    C_GOLD, C_POS, C_NEG = "#c8b487", "#8faa9a", "#c68b83"
    p = ROOT / "insider_ls_backtest.json"
    if not p.exists() or p.stat().st_size < 20:
        return ""
    try:
        m = _json.loads(p.read_text())
    except Exception:
        return ""
    if "error" in m:
        return ""
    c = m.get("combined_long_short", {}); ln = m.get("long_market_neutral", {})
    def _row(label, d, hi=False):
        col = C_GOLD if hi else C_SUB
        return (f'<div style="display:grid;grid-template-columns:1.6fr 1fr 1fr 1fr;gap:8px;padding:8px 0;'
                f'border-top:1px solid #241f18;font-size:12.5px">'
                f'<span style="color:{col};font-weight:{"600" if hi else "400"}">{label}</span>'
                f'<span style="color:{C_INK};text-align:right;font-variant-numeric:tabular-nums">{d.get("alpha_annual",0):.1%}</span>'
                f'<span style="color:{C_INK};text-align:right;font-variant-numeric:tabular-nums">{d.get("sharpe","—")}</span>'
                f'<span style="color:{C_NEG};text-align:right;font-variant-numeric:tabular-nums">{d.get("max_dd",0):.1%}</span></div>')
    hdr = ('<div style="display:grid;grid-template-columns:1.6fr 1fr 1fr 1fr;gap:8px;font-size:9.5px;'
           f'text-transform:uppercase;letter-spacing:.08em;color:{C_MUTE};padding-bottom:2px">'
           '<span>Book</span><span style="text-align:right">Alpha/yr</span>'
           '<span style="text-align:right">Sharpe</span><span style="text-align:right">Max DD</span></div>')
    return (f'<div style="margin-bottom:26px;background:{C_CARD};border:1px solid #241f18;border-radius:8px;padding:18px 20px">'
            f'<div style="font-size:10px;letter-spacing:.14em;text-transform:uppercase;color:{C_GOLD};margin-bottom:2px">Combined Book &middot; Long / Short</div>'
            f'<div style="font-size:19px;font-family:\'Baskerville\',Georgia,serif;color:{C_INK};margin-bottom:4px">The actual product &mdash; honest backtest</div>'
            f'<div style="font-size:12px;color:{C_SUB};margin-bottom:12px;line-height:1.5">Long insider buy-dips, short only insider <b style="color:#a89c8c">cluster</b>-sells '
            f'(&ge;2 insiders dumping) &mdash; the only short that earns its keep. Dollar-neutral, realistic {m.get("cost_bps_one_side","44")}bps, HAC t. '
            f'<span style="color:{C_MUTE}">Adding the short lifts Sharpe (0.95&rarr;{c.get("sharpe","—")}) but not drawdown; it&rsquo;s pure alpha, not a free lunch. Paper-validating live.</span></div>'
            f'{hdr}{_row("Combined long/short (the product)", c, hi=True)}{_row("Long market-neutral alone", ln)}</div>')


def _insider_short_panel() -> str:
    """Small-cap insider-SELL SHORT watchlist (the book's short leg). Reads
    insider_short_today.csv. Only borrowable names are actually shortable."""
    import html as _h
    def _esc(s): return _h.escape(str(s)) if s is not None else ""
    C_CARD, C_INK, C_MUTE, C_SUB = "#16140f", "#f0e9da", "#8f866f", "#b0a68f"
    C_GOLD, C_NEG, C_POS = "#c8b487", "#c68b83", "#8faa9a"
    p = ROOT / "insider_short_today.csv"
    eyebrow = ('<div style="font-size:10px;letter-spacing:.14em;text-transform:uppercase;'
               f'color:{C_NEG};margin-bottom:2px">Insider Sell &middot; Short Leg</div>'
               '<div style="font-size:19px;font-family:\'Baskerville\',Georgia,serif;'
               f'color:{C_INK};margin-bottom:4px">Where insiders are selling &mdash; short candidates</div>')
    thesis = (f'<div style="font-size:12px;color:{C_SUB};margin-bottom:12px;line-height:1.5">'
              'Small-caps with heavy recent insider <b style="color:#a89c8c">open-market selling</b>. Validated '
              '(6,542 real sells, reversal-neutralized): short gross +22.7%/t=5.3, <b style="color:'
              f'{C_POS}">+12.7%/t=3.0 net of 10% borrow</b> &mdash; but DIES at 25% borrow. So only '
              '<b>borrowable</b> names qualify (hard-to-borrow = high fee = the death zone). '
              '<span style="color:#8f866f">Shorting is unlimited-loss + squeeze risk; confirm the borrow fee before any order.</span></div>')
    if not p.exists() or p.stat().st_size < 20:
        return (f'<div style="margin-bottom:26px;background:{C_CARD};border:1px solid #241f18;'
                f'border-radius:8px;padding:18px 20px">{eyebrow}{thesis}'
                f'<div style="font-size:12px;color:{C_MUTE};padding:8px 0">Short scanner not run yet.</div></div>')
    try:
        df = pd.read_csv(p)
    except Exception:
        return ""
    borrow = df[df.get("tradable_short", False) == True] if "tradable_short" in df.columns else df
    if borrow.empty:
        body = (f'<div style="font-size:12px;color:{C_MUTE};padding:8px 0">Insider selling detected but '
                'no names are currently easy-to-borrow &mdash; nothing shortable without a high fee. '
                'A real "stand down".</div>')
        return (f'<div style="margin-bottom:26px;background:{C_CARD};border:1px solid #241f18;'
                f'border-radius:8px;padding:18px 20px">{eyebrow}{thesis}{body}</div>')
    rows = ""
    for _, r in borrow.head(20).iterrows():
        def _chip(txt, col):
            return (f'<span style="font-size:9px;text-transform:uppercase;letter-spacing:.06em;color:{col};'
                    f'border:1px solid {col};border-radius:3px;padding:1px 5px;margin-left:5px">{txt}</span>')
        tags = ""
        if bool(r.get("cluster")): tags += _chip("Cluster", C_MUTE)
        if bool(r.get("large")):   tags += _chip("Large", C_GOLD)
        if bool(r.get("cxo_involved")): tags += _chip("CEO/CFO", C_SUB)
        try:
            usd = f"${int(r.get('total_usd', 0)):,}"
        except Exception:
            usd = "-"
        sellers = int(r.get("sellers", 0))
        rows += (f'<div style="padding:10px 0;border-top:1px solid #241f18;display:flex;'
                 f'justify-content:space-between;align-items:baseline;gap:10px">'
                 f'<div><span style="font-size:15px;color:{C_INK};font-weight:600">{_esc(r.get("ticker"))}</span>'
                 f'<span style="font-size:9px;color:{C_POS};border:1px solid {C_POS};border-radius:3px;'
                 f'padding:1px 5px;margin-left:6px">borrowable</span>{tags}</div>'
                 f'<div style="text-align:right"><span style="font-size:11px;color:{C_NEG}">{usd} sold</span>'
                 f'<span style="font-size:10px;color:{C_MUTE};display:block">{sellers} seller(s) &middot; sold {_esc(r.get("latest_sell"))}</span></div></div>')
    return (f'<div style="margin-bottom:26px;background:{C_CARD};border:1px solid #241f18;'
            f'border-radius:8px;padding:18px 20px">{eyebrow}{thesis}'
            f'<div style="font-size:11px;color:{C_MUTE};margin-bottom:2px">{len(borrow)} borrowable short candidate(s)</div>'
            f'{rows}</div>')


def _insider_scan_panel() -> str:
    """Small-cap insider-BUY watchlist — the one validated edge. Reads
    insider_scan_today.csv (from canyon_insider_scanner.py). Names with a fresh
    Form 4 open-market buy still inside the 10-trading-day hold window, ranked by
    signal strength (cluster > large > single). Honest: research signal, not orders."""
    import html as _h, datetime as _dt
    def _esc(s): return _h.escape(str(s)) if s is not None else ""
    C_CARD, C_INK, C_MUTE, C_SUB = "#16140f", "#f0e9da", "#8f866f", "#b0a68f"
    C_GOLD, C_POS, C_NEG = "#c8b487", "#8faa9a", "#c68b83"
    p = ROOT / "insider_scan_today.csv"

    eyebrow = ('<div style="font-size:10px;letter-spacing:.14em;text-transform:uppercase;'
               f'color:{C_GOLD};margin-bottom:2px">Insider Buy Signal &middot; Small-Cap</div>'
               '<div style="font-size:19px;font-family:\'Baskerville\',Georgia,serif;'
               f'color:{C_INK};margin-bottom:4px">Where insiders are buying &mdash; the one validated edge</div>')
    thesis = (f'<div style="font-size:12px;color:{C_SUB};margin-bottom:12px;line-height:1.5">'
              'S&amp;P 600 small-caps with a fresh Form 4 <b style="color:#a89c8c">open-market buy</b> '
              'still inside the <b>10-trading-day</b> hold window. Backtested on real Form 4, market-neutral, '
              'realistic small-cap costs (44bps), HAC t, out-of-sample, holds in every sub-period incl. the '
              f'calm 2010-19 bull: <b style="color:{C_POS}">+31%/yr net &middot; t=3.7 &middot; the &#9733;dip subset (insider '
              'bought after a fall) is strongest</b> &mdash; the only signal in the system that survives every honest '
              f'test. <span style="color:{C_MUTE}">It is a distress/turnaround play, not value; small capacity; '
              'a research watchlist, not orders.</span></div>')

    if not p.exists() or p.stat().st_size < 20:
        body = (f'<div style="font-size:12px;color:{C_MUTE};padding:10px 0">'
                'Scanner not run yet today &mdash; run <code style="color:#a89c8c">canyon_insider_scanner.py</code> '
                'to populate. No active signal shown rather than a stale one.</div>')
        return (f'<div style="margin-bottom:26px;background:{C_CARD};border:1px solid #241f18;'
                f'border-radius:8px;padding:18px 20px">{eyebrow}{thesis}{body}</div>')
    try:
        df = pd.read_csv(p)
    except Exception:
        return ""
    if df.empty:
        body = (f'<div style="font-size:12px;color:{C_MUTE};padding:10px 0">No small-cap insider '
                'is inside a live 10-day buy window right now. That is a real "nothing to do" &mdash; '
                'the edge is episodic.</div>')
        return (f'<div style="margin-bottom:26px;background:{C_CARD};border:1px solid #241f18;'
                f'border-radius:8px;padding:18px 20px">{eyebrow}{thesis}{body}</div>')

    rows = ""
    for _, r in df.head(30).iterrows():
        tags = ""
        def _chip(txt, col):
            return (f'<span style="font-size:9px;text-transform:uppercase;letter-spacing:.06em;'
                    f'color:{col};border:1px solid {col};border-radius:3px;padding:1px 5px;margin-left:5px">{txt}</span>')
        if bool(r.get("big_dip")):  tags += _chip("★ Deep-dip", C_NEG)
        elif bool(r.get("dip")):    tags += _chip("★ Dip", C_NEG)
        if bool(r.get("cluster")): tags += _chip("Cluster", C_POS)
        if bool(r.get("large")):   tags += _chip("Large", C_GOLD)
        if bool(r.get("cxo_involved")): tags += _chip("CEO/CFO", C_SUB)
        try:
            usd = f"${int(r.get('total_usd', 0)):,}"
        except Exception:
            usd = "-"
        left = int(r.get("approx_days_left", 0))
        ins = int(r.get("insiders", 0))
        bar = int(round((10 - left) / 10 * 100)) if left is not None else 0
        rows += (f'<div style="padding:11px 0;border-top:1px solid #241f18">'
                 f'<div style="display:flex;justify-content:space-between;align-items:baseline;gap:10px">'
                 f'<div><span style="font-size:15px;color:{C_INK};font-weight:600;letter-spacing:.02em">{_esc(r.get("ticker"))}</span>{tags}</div>'
                 f'<span style="font-size:11px;color:{C_GOLD}">{usd}</span></div>'
                 f'<div style="display:flex;justify-content:space-between;align-items:center;gap:10px;margin-top:4px">'
                 f'<span style="font-size:11px;color:{C_SUB}">{ins} insider{"s" if ins!=1 else ""} &middot; latest buy {_esc(r.get("latest_buy"))}</span>'
                 f'<span style="font-size:10.5px;color:{C_MUTE}">~{left}d left in 10d hold</span></div>'
                 # progress bar of the 10-day clock
                 f'<div style="height:3px;background:#241f18;border-radius:2px;margin-top:5px">'
                 f'<div style="height:3px;width:{bar}%;background:{C_POS};border-radius:2px"></div></div></div>')

    note = (f'<div style="font-size:10.5px;color:{C_MUTE};margin-top:12px;line-height:1.5">'
            'Ranked by strength: <b style="color:#c68b83">★ Dip</b> (insider bought AFTER the stock fell &mdash; '
            'validated strongest: +20%/yr vs +12.5% baseline, t=3.4) &gt; cluster (&ge;2 insiders/30d) &gt; '
            'large (&ge;$100k) &gt; single. Green bar = elapsed of the 10-day window. Exit near bar-full.</div>')
    return (f'<div style="margin-bottom:26px;background:{C_CARD};border:1px solid #241f18;'
            f'border-radius:8px;padding:18px 20px">{eyebrow}{thesis}'
            f'<div style="font-size:11px;color:{C_MUTE};margin-bottom:2px">{len(df)} active name(s) &middot; 10-day hold clock</div>'
            f'{rows}{note}</div>')


def _news_deep_panel() -> str:
    """Deep news read — top stories by daily value (recency x impact x model-alpha),
    each with source + read-through + the name it hits. Filterable by 'my holdings'
    and by sector. Reads news_impact_targets.csv + holdings from paper book/picks."""
    import datetime as _dt, urllib.parse as _up, json as _json, html as _h
    p = ROOT / "news_impact_targets.csv"
    if not p.exists():
        return ""
    try:
        df = pd.read_csv(p)
    except Exception:
        return ""
    if df.empty or "headline" not in df.columns:
        return ""
    def _esc(s): return _h.escape(str(s)) if s is not None else ""
    C_CARD, C_INK, C_MUTE, C_SUB = "#16140f", "#f0e9da", "#8f866f", "#b0a68f"
    C_GOLD, C_POS, C_NEG = "#c8b487", "#8faa9a", "#c68b83"

    # holdings = paper book positions + today's picks
    holds = set()
    try:
        _bk = _json.loads((ROOT / "alpaca_book_state.json").read_text())
        for _b in ("SHORT", "MEDIUM", "LONG"):
            _pos = _bk.get(_b, {}).get("positions", [])
            if isinstance(_pos, dict):
                _pos = list(_pos.keys())
            for _pp in _pos:
                _t = _pp if isinstance(_pp, str) else (_pp.get("ticker") or _pp.get("symbol"))
                if _t:
                    holds.add(str(_t).upper())
    except Exception:
        pass
    for _f in ("daily_picks.csv", "daily_final.csv"):
        try:
            _dp = pd.read_csv(ROOT / _f)
            for _c in ("ticker", "symbol", "Ticker"):
                if _c in _dp.columns:
                    holds |= set(_dp[_c].dropna().astype(str).str.upper())
                    break
        except Exception:
            pass

    def _norm_sector(s):
        s = str(s or "").strip()
        m = {"Technology": "Tech", "Information Technology": "Tech",
             "Consumer Disc": "Consumer", "Consumer Discretionary": "Consumer",
             "Consumer Staples": "Consumer", "Communication Services": "Comm Svcs",
             "Aerospace / Space": "Aerospace", "Health Care": "Healthcare"}
        return m.get(s, s) if s and s != "nan" else "Other"

    df = df.copy()
    df["_imp"]   = pd.to_numeric(df.get("impact_score", 0), errors="coerce").fillna(0)
    df["_alpha"] = pd.to_numeric(df.get("alpha_score", 0), errors="coerce").fillna(0)
    df["_pub"]   = pd.to_datetime(df.get("published", ""), errors="coerce")
    _today = pd.Timestamp(_dt.date.today())
    df["_daysago"] = (_today - df["_pub"]).dt.days
    _imp_n   = (df["_imp"].abs() / 5.0).clip(0, 1)
    _alpha_n = ((df["_alpha"] - 33.0) / 48.0).clip(0, 1)
    _rec_n   = (1.0 - df["_daysago"].fillna(30) / 30.0).clip(0, 1)
    df["_value"] = 0.42 * _imp_n + 0.30 * _alpha_n + 0.28 * _rec_n
    df = df.sort_values("_value", ascending=False).drop_duplicates(subset=["headline"]).head(50)
    _latest = df["_pub"].max()
    _latest_s = _latest.strftime("%b %d") if pd.notna(_latest) else "-"

    def _domain(u):
        try:
            return _up.urlparse(u).netloc.replace("www.", "")
        except Exception:
            return ""

    def _datebadge(days):
        if pd.isna(days):
            return ""
        days = int(days)
        if days <= 0:  return "Today"
        if days == 1:  return "Yesterday"
        if days < 7:   return f"{days}d ago"
        if days < 35:  return f"{days//7}w ago"
        return f"{days}d ago"

    _secs = df["target_sector"].map(_norm_sector)
    _sec_counts = _secs.value_counts()
    _top_secs = [s for s in _sec_counts.index if s != "Other"][:7]

    rows = ""
    n_hold = 0
    for _, r in df.iterrows():
        tone  = str(r.get("market_tone", "")).upper()
        t_col = C_NEG if "NEG" in tone else (C_POS if "POS" in tone else C_MUTE)
        head  = _esc(str(r.get("headline", ""))[:150])
        tgt   = str(r.get("target_ticker", "")).upper()
        rel   = _esc(str(r.get("target_relation", "")))
        logic = _esc(str(r.get("news_logic", ""))[:230])
        action = _esc(str(r.get("action_hint", ""))[:150])
        sec   = _norm_sector(r.get("target_sector"))
        pub   = _esc(str(r.get("publisher", "")).strip() or "source")
        imp   = float(r["_imp"])
        badge = _datebadge(r.get("_daysago"))
        link  = str(r.get("link", ""))
        dom   = _esc(_domain(link))
        is_hold = 1 if tgt in holds else 0
        if is_hold:
            n_hold += 1
        star = '<span style="color:#c8b487">&#9733;</span> ' if is_hold else ""
        head_html = (f'<a href="{_esc(link)}" target="_blank" style="color:{C_INK};text-decoration:none;border-bottom:1px solid #3a3128">{head}</a>'
                     if link.startswith("http") else head)
        src_html = (f'<a href="{_esc(link)}" target="_blank" style="color:{C_GOLD};text-decoration:none">{pub}</a>'
                    if link.startswith("http") else f'<span style="color:{C_GOLD}">{pub}</span>')
        if dom and dom.lower() not in pub.lower():
            src_html += f'<span style="color:{C_MUTE};font-size:10px"> &middot; {dom}</span>'
        rows += (f'<div class="cnews-item" data-hold="{is_hold}" data-sec="{_esc(sec)}" style="padding:13px 0;border-top:1px solid #241f18">'
                 f'<div style="display:flex;justify-content:space-between;gap:12px;align-items:baseline">'
                 f'<span style="font-size:9.5px;text-transform:uppercase;letter-spacing:.08em;color:{t_col}">{_esc(tone.title())} &middot; impact {imp:+.0f}</span>'
                 f'<span style="font-size:10px;color:{C_MUTE}">{badge} &middot; {_esc(sec)}</span></div>'
                 f'<div style="font-size:13.5px;color:{C_INK};line-height:1.45;margin:5px 0 3px">{star}{head_html}</div>'
                 f'<div style="font-size:11px;margin-bottom:4px">Source: {src_html}</div>'
                 f'<div style="font-size:11.5px;color:{C_SUB};line-height:1.5"><b style="color:#a89c8c">Read-through:</b> {logic}</div>'
                 + (f'<div style="font-size:11px;color:{C_SUB};line-height:1.45;margin-top:2px"><b style="color:#a89c8c">Do:</b> {action}</div>' if action and action != "nan" else "")
                 + f'<div style="font-size:11px;color:{C_GOLD};margin-top:3px">&rarr; hits <b>{_esc(tgt)}</b> ({rel})</div></div>')

    def _chip(label, kind, val, active=False):
        bg = "#2a2418" if active else "transparent"
        return (f'<button class="cnews-chip" onclick="cnewsFilter(this,\'{kind}\',\'{_esc(val)}\')" '
                f'style="cursor:pointer;font-size:11px;padding:4px 10px;border-radius:14px;'
                f'border:1px solid #3a3128;background:{bg};color:#c8b487;letter-spacing:.02em">{label}</button>')
    chips = _chip("All", "all", "", True) + _chip(f"&#9733; My holdings ({n_hold})", "hold", "1")
    chips += '<span style="width:1px;height:16px;background:#3a3128;display:inline-block;margin:0 4px"></span>'
    for s in _top_secs:
        chips += _chip(f"{s} ({int(_sec_counts.get(s, 0))})", "sec", s)

    _js = ("<script>function cnewsFilter(btn,kind,val){"
           "var box=btn.closest('.cnews-box');"
           "box.querySelectorAll('.cnews-chip').forEach(function(c){c.style.background='transparent';});"
           "btn.style.background='#2a2418';"
           "box.querySelectorAll('.cnews-item').forEach(function(it){"
           "var show=true;"
           "if(kind==='hold'){show=it.getAttribute('data-hold')==='1';}"
           "else if(kind==='sec'){show=it.getAttribute('data-sec')===val;}"
           "it.style.display=show?'block':'none';});}</script>")

    n = len(df)
    return (f'<div class="cnews-box" style="margin-bottom:26px;background:{C_CARD};border:1px solid #241f18;border-radius:8px;padding:18px 20px">'
            f'<div style="font-size:10px;letter-spacing:.14em;text-transform:uppercase;color:{C_GOLD};margin-bottom:2px">News Impact &middot; Read-Through</div>'
            f'<div style="font-size:19px;font-family:\'Baskerville\',Georgia,serif;color:{C_INK};margin-bottom:4px">What\'s moving &mdash; and who it hits</div>'
            f'<div style="font-size:12px;color:{C_SUB};margin-bottom:10px">Top {n} of ~1,800 recent stories by <b style="color:#a89c8c">daily value</b> (recency &times; impact &times; model-alpha). &#9733; = hits a name you hold. Latest: {_latest_s}.</div>'
            f'<div style="display:flex;gap:6px;flex-wrap:wrap;margin-bottom:6px">{chips}</div>'
            f'{rows}{_js}</div>')



def _alphavantage_news_panel() -> str:
    """Alpha Vantage market news sentiment (free-tier NEWS_SENTIMENT, 1 call/day).
    Honest states: not-enabled (no key), error (rate-limit/network), or live feed."""
    p = ROOT / "alphavantage_news_sentiment.json"
    if not p.exists():
        return ""
    try:
        d = json.load(open(p))
    except Exception:
        return ""

    import html as _h
    def _esc(s): return _h.escape(str(s)) if s is not None else ""

    C_CARD, C_INK, C_MUTE, C_SUB = "#16140f", "#f0e9da", "#8f866f", "#b0a68f"
    C_GOLD, C_POS, C_NEG = "#c8b487", "#8faa9a", "#c68b83"

    def _wrap(inner: str) -> str:
        return (f'<div style="margin-bottom:26px;background:{C_CARD};border:1px solid #241f18;'
                f'border-radius:8px;padding:16px 18px">'
                f'<div style="font-size:10px;letter-spacing:.14em;text-transform:uppercase;'
                f'color:{C_GOLD};margin-bottom:2px">Alpha Vantage</div>'
                f'<div style="font-size:19px;font-family:\'Baskerville\',\'Hoefler Text\',Georgia,serif;'
                f'color:{C_INK};margin-bottom:12px">Market News Sentiment</div>{inner}</div>')

    # State 1: not enabled (no key) — honest, not faked.
    if not d.get("enabled", False):
        reason = _esc(str(d.get("reason", "Not enabled.")))
        return _wrap(
            f'<div style="border-left:3px solid {C_GOLD};padding:10px 14px;background:#1b1710;border-radius:6px">'
            f'<p style="font-size:12.5px;color:{C_GOLD};margin-bottom:3px">Not enabled</p>'
            f'<p style="font-size:12px;color:{C_SUB};line-height:1.5">{reason}</p></div>')

    # State 2: enabled but this run failed (rate limit / network) — honest.
    if not d.get("ok", False):
        err = _esc(str(d.get("error", "fetch failed")))
        return _wrap(
            f'<div style="border-left:3px solid {C_NEG};padding:10px 14px;background:#1b1310;border-radius:6px">'
            f'<p style="font-size:12.5px;color:{C_NEG};margin-bottom:3px">No data this run</p>'
            f'<p style="font-size:12px;color:{C_SUB};line-height:1.5">{err}</p></div>')

    # State 3: live feed.
    avg = d.get("avg_sentiment")
    label = str(d.get("market_label", "Neutral"))
    lab_col = C_POS if ("Bull" in label) else (C_NEG if "Bear" in label else C_MUTE)
    n = int(d.get("n_articles", 0) or 0)
    as_of = _esc(str(d.get("as_of", "")))
    head = (f'<div style="display:flex;align-items:baseline;gap:14px;flex-wrap:wrap;margin-bottom:12px">'
            f'<span style="font-size:26px;color:{lab_col};font-family:\'Baskerville\',Georgia,serif">{label}</span>'
            f'<span style="font-size:13px;color:{C_MUTE}">avg score '
            f'<b style="color:{lab_col}">{avg if avg is not None else "—"}</b> · {n} articles · {as_of}</span></div>')
    rows = ""
    for a in (d.get("top_articles") or [])[:10]:
        s_lab = str(a.get("sentiment", ""))
        s_col = C_POS if ("Bull" in s_lab) else (C_NEG if "Bear" in s_lab else C_MUTE)
        title = _esc(str(a.get("title", "")))
        src = _esc(str(a.get("source", "")))
        url = _esc(str(a.get("url", "")))
        t_html = f'<a href="{url}" target="_blank" style="color:{C_INK};text-decoration:none">{title}</a>' if url else title
        rows += (f'<div style="padding:9px 0;border-top:1px solid #241f18;display:flex;gap:12px;align-items:flex-start">'
                 f'<span style="flex-shrink:0;min-width:96px;font-size:10px;color:{s_col};text-transform:uppercase;'
                 f'letter-spacing:.05em">{_esc(s_lab)}</span>'
                 f'<span style="flex:1;font-size:12.5px;line-height:1.45;color:{C_SUB}">{t_html}'
                 f'<span style="color:{C_MUTE};font-size:11px"> — {src}</span></span></div>')
    return _wrap(head + rows)


def _fred_macro_panel() -> str:
    """第1层真实宏观数据盘: FRED 官方序列(利率/信用/就业/通胀/风险)。"""
    p = ROOT / "fred_macro_latest.json"
    if not p.exists():
        return ""
    try:
        d = json.load(open(p))
    except Exception:
        return ""
    # 分组: (标题, [(key, 单位后缀, 好方向 up/down/None)])
    groups = [
        ("Rates / Curve", [("T10Y2Y", "", None), ("DGS10", "%", None), ("DGS2", "%", None),
                       ("DFII10", "%", None), ("FEDFUNDS", "%", None)]),
        ("Credit Spreads", [("BAMLH0A0HYM2", "%", "down"), ("BAMLC0A0CM", "%", "down")]),
        ("Jobs / Growth", [("UNRATE", "%", "down"), ("ICSA", "K", "down"), ("PAYEMS", "M", "up")]),
        ("Inflation / Risk", [("CPILFESL", "", "down"), ("VIXCLS", "", "down"), ("UMCSENT", "", "up")]),
    ]

    def _cell(key):
        v = d.get(key)
        if not isinstance(v, dict) or not v.get("ok", True) or v.get("value") is None:
            return ""
        val = v.get("value")
        lbl = v.get("label", key)
        chg = v.get("chg_1m")
        good = None
        # 数值格式
        if key == "ICSA":
            disp = f"{val/1000:,.0f}K"
        elif key == "PAYEMS":
            disp = f"{val/1000:,.1f}M"
        elif key == "M2SL":
            disp = f"${val/1000:,.1f}T"
        elif abs(val) >= 100:
            disp = f"{val:,.1f}"
        else:
            disp = f"{val:.2f}"
        chg_html = ""
        if chg is not None:
            arrow = "▲" if chg > 0 else "▼" if chg < 0 else "＝"
            ccol = "#8a7f70"
            chg_html = f'<span style="font-size:10px;color:{ccol};margin-left:6px">{arrow}{abs(chg):.2f} 1m</span>'
        return (f'<div style="padding:8px 12px;border:1px solid #2f281f;border-radius:6px;background:#191410">'
                f'<div style="font-size:9.5px;color:#8a7f70;text-transform:uppercase;letter-spacing:.08em;line-height:1.25;min-height:24px">{lbl}</div>'
                f'<div style="font-size:19px;font-weight:400;color:#f4ecdf;font-family:\'Financier Display\',Georgia,serif;font-variant-numeric:tabular-nums;margin-top:3px">{disp}{chg_html}</div>'
                f'<div style="font-size:9px;color:#5f574a;margin-top:2px">as of {v.get("as_of","")}</div></div>')

    cols = ""
    for title, keys in groups:
        cells = "".join(c for c in (_cell(k) for k, *_ in keys) if c)
        if not cells:
            continue
        cols += (f'<div><div style="font-size:10px;color:#c0a878;text-transform:uppercase;letter-spacing:.12em;margin-bottom:8px">{title}</div>'
                 f'<div style="display:grid;grid-template-columns:1fr;gap:8px">{cells}</div></div>')
    meta = d.get("_meta", {})
    upd = meta.get("as_of", "") if isinstance(meta, dict) else ""
    # 简要读数: 曲线是否倒挂 / 信用是否紧
    yc = d.get("T10Y2Y", {}).get("value")
    hy = d.get("BAMLH0A0HYM2", {}).get("value")
    read = []
    if yc is not None:
        read.append("Curve inverted (recession signal)" if yc < 0 else f"Curve normal (+{yc:.2f})")
    if hy is not None:
        read.append("Credit tight" if hy > 5 else "Credit easy/normal")
    return f"""
    <div style="margin-bottom:26px;background:#16140f;border:1px solid #3a3128;border-radius:8px;padding:16px 18px">
      <div style="display:flex;justify-content:space-between;align-items:baseline;flex-wrap:wrap;gap:8px;margin-bottom:14px">
        <span style="font-size:11px;color:#8a7f70;text-transform:uppercase;letter-spacing:.14em">Layer 1 · Real Macro Dashboard · FRED official series (free)</span>
        <span style="font-size:11px;color:#8a7f70">{' · '.join(read)} · {upd}</span>
      </div>
      <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:18px">{cols}</div>
    </div>"""


def _pnl_contrib_panel() -> str:
    """谁在赚钱: 历史累计 P&L 贡献榜(真实归因, 非模拟)。"""
    p = ROOT / "pnl_top_contributors.csv"
    if not p.exists() or p.stat().st_size < 20:
        return ""
    try:
        df = pd.read_csv(p)
    except Exception:
        return ""
    if df.empty or "total_contribution" not in df.columns:
        return ""
    df = df.sort_values("total_contribution", ascending=False).head(12)
    mx = max(abs(df["total_contribution"].max()), abs(df["total_contribution"].min()), 1e-9)
    rows = ""
    for _, r in df.iterrows():
        c = float(r["total_contribution"])
        pct = c * 100
        w = min(abs(c) / mx * 100, 100)
        col = "#8faa9a" if c >= 0 else "#c68b83"
        rows += (f'<div style="display:flex;align-items:center;gap:10px;padding:3px 0">'
                 f'<span style="width:52px;font-size:12px;font-weight:400;color:#f4ecdf">{r["ticker"]}</span>'
                 f'<div style="flex:1;height:14px;background:#191410;border-radius:3px;overflow:hidden"><div style="height:100%;width:{w:.0f}%;background:{col};opacity:.75"></div></div>'
                 f'<span style="width:64px;text-align:right;font-size:11.5px;color:{col};font-variant-numeric:tabular-nums">{pct:+.1f}%</span>'
                 f'<span style="width:44px;text-align:right;font-size:10px;color:#5f574a;font-variant-numeric:tabular-nums">{int(r.get("trading_days",0))}d</span></div>')
    return f"""
    <div style="margin-bottom:26px;background:#16140f;border:1px solid #3a3128;border-radius:8px;padding:16px 18px">
      <div style="display:flex;justify-content:space-between;align-items:baseline;flex-wrap:wrap;gap:8px;margin-bottom:12px">
        <span style="font-size:11px;color:#8a7f70;text-transform:uppercase;letter-spacing:.14em">Attribution · Cumulative P&L Contributors (who is making money)</span>
        <span style="font-size:11px;color:#8a7f70">bar = share of top contribution · right = trading days</span>
      </div>
      {rows}
      <p style="color:#746a5d;font-size:10.5px;margin-top:10px">Ranked by each name's contribution to cumulative portfolio return (real attribution). Concentrate firepower on the persistent positive contributors.</p>
    </div>"""


def _pead_panel() -> str:
    """事件驱动核心验证: 财报后漂移(PEAD)——盈利超预期后是否延续。"""
    p = ROOT / "pead_summary.json"
    if not p.exists():
        return ""
    try:
        d = json.load(open(p))
    except Exception:
        return ""
    bw = d.get("by_window", [])
    if not bw:
        return ""
    cells = ""
    for w in bw:
        wd = w.get("window_days")
        alpha = w.get("avg_alpha", 0) * 100
        hit = w.get("hit_rate", 0) * 100
        acol = "#8faa9a" if alpha >= 0 else "#c68b83"
        cells += (f'<div style="padding:10px 14px;border:1px solid #2f281f;border-radius:6px;background:#191410;text-align:center">'
                  f'<div style="font-size:10px;color:#8a7f70;text-transform:uppercase;letter-spacing:.1em">+{wd}-day window</div>'
                  f'<div style="font-size:20px;font-weight:400;color:{acol};font-family:\'Financier Display\',Georgia,serif;font-variant-numeric:tabular-nums;margin-top:3px">{alpha:+.2f}%</div>'
                  f'<div style="font-size:10px;color:#a89c8c;margin-top:2px">alpha · hit {hit:.0f}%</div></div>')
    n = d.get("n_events", "?")
    return f"""
    <div style="margin-bottom:26px;background:#16140f;border:1px solid #3a3128;border-radius:8px;padding:16px 18px">
      <div style="display:flex;justify-content:space-between;align-items:baseline;flex-wrap:wrap;gap:8px;margin-bottom:12px">
        <span style="font-size:11px;color:#8a7f70;text-transform:uppercase;letter-spacing:.14em">Event validation · Post-Earnings Drift (PEAD) · last {n} earnings events</span>
        <span style="font-size:11px;color:#8a7f70">{d.get('as_of','')}</span>
      </div>
      <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(120px,1fr));gap:12px">{cells}</div>
      <p style="color:#746a5d;font-size:10.5px;margin-top:10px">Honest read: large-cap PEAD drift is weak (as theory predicts). This is a reality check on whether events persist — not a guarantee. The real edge is in concentrated stock selection and event-type filtering, not mechanically chasing earnings.</p>
    </div>"""


def _build_event_engine_tab() -> str:
    """事件驱动主动投资系统 — 作战台 (War Room) (第0/5/7/9/10层)."""
    cand_p = ROOT / "event_candidates.csv"
    if not cand_p.exists() or cand_p.stat().st_size < 5:
        body = ('<p style="color:#a89c8c">尚未生成候选。请先建 <code style="color:#c8b487">event_pool.csv</code> '
                '(股票池 + L/N/M/P/C 打分),再运行 <code style="color:#c8b487">canyon_event_system.py</code>。</p>')
        return f'<section id="sec-eventengine" class="tab-section"><div class="container"><p class="eyebrow">利润发动机 (Profit Engine) · 事件驱动</p><h2 class="section-head">作战台 (War Room)</h2><div class="rule"></div>{body}</div></section>'

    df = pd.read_csv(cand_p)
    # 第1层宏观情报评分卡 (Macro Intel Scorecard)
    intel_html = ""
    ip = ROOT / "macro_intel_scorecard.json"
    if ip.exists():
        try:
            intel = json.load(open(ip))
            mods = intel.get("情报模块", {})
            heat_rows = ""
            for nm, v in sorted(mods.items(), key=lambda x: -x[1].get("heat", 0)):
                h = int(v.get("heat", 0))
                col = "#c68b83" if h >= 3 else "#c0a878" if h == 2 else "#746a5d"
                bars = "".join(f'<span style="display:inline-block;width:14px;height:8px;border-radius:2px;margin-right:2px;background:{col if i<h else "#2a231b"}"></span>' for i in range(4))
                heat_rows += f'<div style="display:flex;align-items:center;gap:10px;padding:3px 0"><span style="width:110px;font-size:12px;color:#a89c8c">{nm}</span>{bars}<span style="color:{col};font-size:11px;margin-left:4px">{v.get("hits",0)}条</span></div>'
            active = intel.get("激活事件池 (Active Event Pool)", [])
            intel_html = f"""
    <div style="background:#16140f;border:1px solid #453a2c;border-radius:8px;padding:20px 24px;margin-bottom:24px">
      <div style="display:flex;justify-content:space-between;flex-wrap:wrap;gap:12px;margin-bottom:14px">
        <div><span style="font-size:11px;color:#8a7f70;text-transform:uppercase;letter-spacing:1px">第1层 · 宏观情报评分卡 (Macro Intel Scorecard)</span>
          <span style="font-size:12px;color:#746a5d;margin-left:10px">扫描新闻 {intel.get('news_scanned','?')} 条 · {intel.get('updated','')}</span></div>
        <div style="font-size:13px;color:#c8b487;font-weight:400">宏观模式 (Macro Mode): {intel.get('宏观模式 (Macro Mode)','—')} · 总仓位 {intel.get('总仓位制度 (Total Position Regime)','—')}</div>
      </div>
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:20px">
        <div>{heat_rows}</div>
        <div style="font-size:12px;color:#a89c8c;line-height:1.9">
          <div><strong style="color:#8faa9a">重点受益链条 (Key Beneficiary Chain):</strong> {', '.join(intel.get('重点受益链条 (Key Beneficiary Chain)',[])) or '—'}</div>
          <div><strong style="color:#c68b83">风险链条 (Risk Chain):</strong> {', '.join(intel.get('风险链条 (Risk Chain)',[])) or '—'}</div>
          <div style="margin-top:8px"><strong style="color:#c8b487">激活事件池 (Active Event Pool):</strong> {', '.join(active) or '—'}</div>
        </div>
      </div>
    </div>"""
        except Exception:
            intel_html = ""
    n_pe = int((df["pool"] == "利润发动机 (Profit Engine)").sum())
    n_ev = int((df["pool"] == "事件型爆发池").sum())
    n_active = n_pe + n_ev
    mf = float(df["MacroFilter"].iloc[0]) if "MacroFilter" in df.columns and len(df) else 0.85
    if n_active <= 2:
        sw_state, sw_pos, sw_color = "机会稀缺", "20%–45%", "#c68b83"
    elif n_active <= 5:
        sw_state, sw_pos, sw_color = "机会一般", "45%–70%", "#c0a878"
    else:
        sw_state, sw_pos, sw_color = "机会丰富", "70%–100%", "#8faa9a"

    # 行业板块轮动信号 (Sector Rotation Signal)
    rot_html = ""
    rotp = ROOT / "sector_rotation.csv"
    if rotp.exists():
        try:
            rdf = pd.read_csv(rotp)
            SEC_CN = {"Technology":"科技","Communication Services":"通信","Financials":"金融",
                      "Health Care":"医疗","Industrials":"工业","Energy":"能源","Materials":"材料",
                      "Utilities":"公用","Real Estate":"地产","Consumer Discretionary":"可选消费",
                      "Consumer Staples":"必需消费"}
            rrows = ""
            for _, r in rdf.iterrows():
                sig = str(r["signal"]); score = float(r["rotation_score"])
                col = "#8faa9a" if "超配" in sig else "#c68b83" if "低配" in sig else "#9a8e80"
                dr = str(r["direction"]); dcol = "#8faa9a" if "▲" in dr else "#c68b83" if "▼" in dr else "#9a8e80"
                rs = float(r["rel_strength"]) * 100; rscol = "#8faa9a" if rs >= 0 else "#c68b83"
                bar = int(round(score))
                secname = SEC_CN.get(str(r["sector"]), str(r["sector"]))
                rrows += f"""<tr>
                  <td style="padding:7px 10px;font-weight:400;color:#f4ecdf">{secname}<span style="color:#746a5d;font-size:10px;margin-left:5px">{r['n']}只</span></td>
                  <td style="padding:7px 10px;min-width:130px"><div style="background:#2a231b;border-radius:3px;height:16px;position:relative"><div style="background:{col};height:16px;width:{bar}%;border-radius:3px"></div><span style="position:absolute;left:6px;top:0;font-size:10px;line-height:16px;color:#0b0f17;font-weight:400">{score:.0f}</span></div></td>
                  <td style="padding:7px 10px"><span style="color:{col};font-size:11px;font-weight:400">{sig}</span></td>
                  <td style="padding:7px 10px;color:{dcol};font-size:11px">{dr}</td>
                  <td style="padding:7px 10px;text-align:right;color:{rscol};font-size:11px;font-variant-numeric:tabular-nums">{rs:+.1f}%</td>
                  <td style="padding:7px 10px;text-align:right;color:#a89c8c;font-size:11px;font-variant-numeric:tabular-nums">{float(r['breadth_50d'])*100:.0f}%</td>
                  <td style="padding:7px 10px;color:#8a7f70;font-size:11px">{r.get('event_type','—')}</td>
                </tr>"""
            lead = [SEC_CN.get(s, s) for s in rdf[rdf["signal"].str.contains("超配")]["sector"]]
            lag = [SEC_CN.get(s, s) for s in rdf[rdf["signal"].str.contains("低配")]["sector"]]
            rot_html = f"""
    <div style="margin-bottom:26px">
      <div style="display:flex;justify-content:space-between;flex-wrap:wrap;gap:10px;margin-bottom:12px">
        <span style="font-size:11px;color:#8a7f70;text-transform:uppercase;letter-spacing:1px">行业板块轮动信号 (Sector Rotation Signal) · 标普500全体聚合(资金在轮入/轮出哪些板块)</span>
        <span style="font-size:12px"><span style="color:#8faa9a">轮入 {', '.join(lead)}</span> <span style="color:#746a5d">|</span> <span style="color:#c68b83">轮出 {', '.join(lag)}</span></span>
      </div>
      <div style="overflow-x:auto"><table style="width:100%;border-collapse:collapse;font-size:12px;min-width:620px">
        <thead><tr style="border-bottom:1px solid #453a2c">
          <th style="text-align:left;padding:6px 10px;font-size:10px;text-transform:uppercase;color:#8a7f70">板块</th>
          <th style="text-align:left;padding:6px 10px;font-size:10px;text-transform:uppercase;color:#8a7f70">轮动分</th>
          <th style="text-align:left;padding:6px 10px;font-size:10px;text-transform:uppercase;color:#8a7f70">信号</th>
          <th style="text-align:left;padding:6px 10px;font-size:10px;text-transform:uppercase;color:#8a7f70">方向</th>
          <th style="text-align:right;padding:6px 10px;font-size:10px;text-transform:uppercase;color:#8a7f70">相对强度 (Relative Strength)</th>
          <th style="text-align:right;padding:6px 10px;font-size:10px;text-transform:uppercase;color:#8a7f70">广度</th>
          <th style="text-align:left;padding:6px 10px;font-size:10px;text-transform:uppercase;color:#8a7f70">事件类型 (Event Type)</th>
        </tr></thead><tbody>{rrows}</tbody></table></div>
      <p style="color:#746a5d;font-size:11px;margin-top:8px">轮动分综合: 相对强度 (Relative Strength)(vs大盘) + 加速度(1M vs 3M) + 广度(站上50日线占比) + 6月Momentum。方向 ▲加速=资金流入 / ▼退潮=流出。</p>
    </div>"""
        except Exception:
            rot_html = ""

    # 板块龙头 (Sector Leaders) ETF 指标 (可交易工具视角)
    etf_html = ""
    etfp = ROOT / "sector_etf_indicators.csv"
    if etfp.exists():
        try:
            edf = pd.read_csv(etfp)
            erows = ""
            for _, r in edf.iterrows():
                rs = float(r["rel_strength"]) * 100 if pd.notna(r["rel_strength"]) else 0.0
                rscol = "#8faa9a" if rs >= 0 else "#c68b83"
                m1 = float(r["mom_1m"]) * 100; m1c = "#8faa9a" if m1 >= 0 else "#c68b83"
                m3 = float(r["mom_3m"]) * 100; m3c = "#8faa9a" if m3 >= 0 else "#c68b83"
                dr = str(r["direction"]); dcol = "#8faa9a" if "▲" in dr else "#c68b83" if "▼" in dr else "#9a8e80"
                trend = str(r["trend"]); tcol = "#8faa9a" if "Longs" in trend else "#c68b83" if "偏空" in trend else "#c0a878"
                erows += f"""<tr>
                  <td style="padding:7px 10px;font-weight:400;color:#c8b487">{r['etf']}</td>
                  <td style="padding:7px 10px;color:#cabeae;font-size:11px">{r['name']}</td>
                  <td style="padding:7px 10px;text-align:right;color:#f4ecdf;font-size:11px;font-variant-numeric:tabular-nums">${r['price']:.2f}</td>
                  <td style="padding:7px 10px;text-align:right;color:{m1c};font-size:11px;font-variant-numeric:tabular-nums">{m1:+.1f}%</td>
                  <td style="padding:7px 10px;text-align:right;color:{m3c};font-size:11px;font-variant-numeric:tabular-nums">{m3:+.1f}%</td>
                  <td style="padding:7px 10px;text-align:right;color:{rscol};font-size:11px;font-weight:400;font-variant-numeric:tabular-nums">{rs:+.1f}%</td>
                  <td style="padding:7px 10px;color:{dcol};font-size:11px">{dr}</td>
                  <td style="padding:7px 10px;color:{tcol};font-size:11px">{trend}</td>
                </tr>"""
            etf_html = f"""
    <div style="margin-bottom:26px">
      <div style="font-size:11px;color:#8a7f70;text-transform:uppercase;letter-spacing:1px;margin-bottom:10px">板块龙头 (Sector Leaders) ETF 指标 · 可交易工具视角(按相对强度 (Relative Strength)排序,基准 SPY)</div>
      <div style="overflow-x:auto"><table style="width:100%;border-collapse:collapse;font-size:12px;min-width:640px">
        <thead><tr style="border-bottom:1px solid #453a2c">
          <th style="text-align:left;padding:6px 10px;font-size:10px;text-transform:uppercase;color:#8a7f70">ETF</th>
          <th style="text-align:left;padding:6px 10px;font-size:10px;text-transform:uppercase;color:#8a7f70">板块</th>
          <th style="text-align:right;padding:6px 10px;font-size:10px;text-transform:uppercase;color:#8a7f70">价格</th>
          <th style="text-align:right;padding:6px 10px;font-size:10px;text-transform:uppercase;color:#8a7f70">1M</th>
          <th style="text-align:right;padding:6px 10px;font-size:10px;text-transform:uppercase;color:#8a7f70">3M</th>
          <th style="text-align:right;padding:6px 10px;font-size:10px;text-transform:uppercase;color:#8a7f70">相对强度 (Relative Strength)</th>
          <th style="text-align:left;padding:6px 10px;font-size:10px;text-transform:uppercase;color:#8a7f70">方向</th>
          <th style="text-align:left;padding:6px 10px;font-size:10px;text-transform:uppercase;color:#8a7f70">趋势</th>
        </tr></thead><tbody>{erows}</tbody></table></div>
      <p style="color:#746a5d;font-size:11px;margin-top:8px">相对强度 (Relative Strength) = ETF 3月涨幅 − SPY 3月涨幅(跑赢大盘为正)。与上方成分股聚合互补: 这里是可直接买卖的板块工具。</p>
    </div>"""
        except Exception:
            etf_html = ""

    # 真实数据源 (Real Data Sources): SEC EDGAR 8-K 事件 + CFTC COT 商品持仓
    src_html = ""
    try:
        ev_cards = ""
        ep = ROOT / "edgar_events.csv"
        if ep.exists():
            ed = pd.read_csv(ep)
            hi = ed[pd.to_numeric(ed.get("8k_severity", 0), errors="coerce").fillna(0) >= 2].copy()
            hi = hi.sort_values("8k_severity", ascending=False).head(8)
            n8k = int((pd.to_numeric(ed.get("n_8k_30d", 0), errors="coerce").fillna(0) > 0).sum())
            nins = int((pd.to_numeric(ed.get("insider_active", 0), errors="coerce").fillna(0) > 0).sum())
            evrows = ""
            for _, r in hi.iterrows():
                sev = int(float(r.get("8k_severity", 0)))
                scol = "#c68b83" if sev >= 4 else "#c0a878" if sev >= 3 else "#8a7f70"
                evrows += f'<div style="display:flex;justify-content:space-between;gap:10px;padding:3px 0;font-size:12px"><span><strong style="color:#f4ecdf">{r["ticker"]}</strong> <span style="color:#8a7f70">{r.get("8k_desc","")}</span></span><span style="color:{scol};white-space:nowrap">严重度{sev} · {r.get("latest_8k_date","")}</span></div>'
            ev_cards = f"""
      <div style="background:#16140f;border:1px solid #3a3128;border-left:3px solid #c8b487;padding:14px 18px;border-radius:8px">
        <div style="font-size:12px;font-weight:400;color:#c8b487;margin-bottom:2px">SEC EDGAR 真实事件流</div>
        <div style="font-size:11px;color:#8a7f70;margin-bottom:8px">{n8k} 家近期8-K · {nins} 家内部人活跃(Form4)· 官方近实时</div>
        {evrows or '<div style="color:#8a7f70;font-size:12px">近期无高严重度8-K</div>'}
      </div>"""
        cot_card = ""
        cp = ROOT / "cot_positioning.csv"
        if cp.exists():
            cd = pd.read_csv(cp)
            cotrows = ""
            for _, r in cd.iterrows():
                idx = float(r.get("cot_index", 50))
                icol = "#8faa9a" if idx < 25 else "#c68b83" if idx > 80 else "#a89c8c"
                fav = float(r.get("cot_boost", 0)) > 0
                cotrows += f'<div style="display:flex;justify-content:space-between;gap:8px;padding:3px 0;font-size:12px"><span style="color:#cabeae">{r["commodity"]}{" 🔵" if fav else ""}</span><span style="color:{icol};font-variant-numeric:tabular-nums">COT {idx:.0f} · {r.get("setup","")[:12]}</span></div>'
            cot_card = f"""
      <div style="background:#16140f;border:1px solid #3a3128;border-left:3px solid #8aa6a6;padding:14px 18px;border-radius:8px">
        <div style="font-size:12px;font-weight:400;color:#8aa6a6;margin-bottom:2px">CFTC 商品持仓(COT)</div>
        <div style="font-size:11px;color:#8a7f70;margin-bottom:8px">投机资金极值 → 商品供需错配 setup · 官方周报</div>
        {cotrows}
      </div>"""
        if ev_cards or cot_card:
            src_html = f"""
    <div style="margin-bottom:26px">
      <div style="font-size:11px;color:#8a7f70;text-transform:uppercase;letter-spacing:.14em;margin-bottom:10px">真实数据源 (Real Data Sources) · SEC 备案 + CFTC 持仓(超越 yfinance 的官方事件流)</div>
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:14px">{ev_cards}{cot_card}</div>
    </div>"""
    except Exception:
        src_html = ""

    # 第2层生命周期 (Lifecycle) + 第4层功能池 映射
    life_m, conf_m = {}, {}
    lp = ROOT / "lifecycle_style.csv"
    if lp.exists():
        try:
            for _, r in pd.read_csv(lp).iterrows():
                life_m[str(r["ticker"])] = f"{r.get('lifecycle','')}/{r.get('style','')}"
        except Exception:
            pass
    fp_html = ""
    fpp = ROOT / "functional_pools.csv"
    if fpp.exists():
        try:
            fdf = pd.read_csv(fpp)
            pool_meta = {
                "利润发动机 (Profit Engine)储备池": ("#c8b487", "预备役 · 逼近发动机门槛,前瞻确认到位即升入"),
                "核心储备池":     ("#8faa9a", "压舱石 · 稳健趋势/防御,提供 beta 与稳定性"),
                "主题链条观察池": ("#8aa6a6", "挂单观察 · 有事件信号但执行/结构未就位"),
                "回收观察池":     ("#c68b83", "逻辑走坏 · 衰退期/执行分低,待剔除"),
            }
            cards = ""
            for pool, (col, desc) in pool_meta.items():
                sub = fdf[fdf["func_pool"] == pool]
                names = " ".join(f'<span style="color:#cabeae">{t}</span>' for t in sub["ticker"].head(10))
                cards += f"""
      <div style="background:#16140f;border:1px solid #3a3128;border-left:4px solid {col};padding:16px 18px;border-radius:8px">
        <div style="display:flex;justify-content:space-between;align-items:baseline"><span style="font-size:13px;font-weight:400;color:{col}">{pool}</span><span style="font-size:20px;font-weight:400;color:#f4ecdf;font-variant-numeric:tabular-nums">{len(sub)}</span></div>
        <div style="font-size:11px;color:#8a7f70;margin:4px 0 8px">{desc}</div>
        <div style="font-size:11px;line-height:1.9;word-spacing:4px">{names or '—'}</div>
      </div>"""
            fp_html = f"""
    <div style="margin-bottom:26px">
      <div style="font-size:11px;color:#8a7f70;text-transform:uppercase;letter-spacing:1px;margin-bottom:10px">第4层 · 功能分层(标普500全体 {len(fdf)} 只的角色归属)</div>
      <div style="display:grid;grid-template-columns:repeat(2,1fr);gap:14px">{cards}</div>
    </div>"""
        except Exception:
            fp_html = ""

    # 集中冲锋清单 (冲击跑赢纳指)
    conc_html = ""
    ccp = ROOT / "concentrated_portfolio.csv"
    if ccp.exists():
        try:
            cc = pd.read_csv(ccp)
            cs = json.load(open(ROOT / "concentrated_summary.json")) if (ROOT / "concentrated_summary.json").exists() else {}
            crows = ""
            for _, r in cc.iterrows():
                w = float(r["weight_pct"])
                crows += f"""<tr>
                  <td style="padding:7px 10px;font-weight:400;color:#f4ecdf">{r['ticker']}</td>
                  <td style="padding:7px 10px;min-width:110px"><div style="background:#1a140e;border-radius:3px;height:15px;position:relative"><div style="background:#c8b487;height:15px;width:{min(w/16*100,100):.0f}%;border-radius:3px"></div><span style="position:absolute;left:6px;top:0;font-size:10px;line-height:15px;color:#17130f;font-weight:400">{w:.1f}%</span></div></td>
                  <td style="padding:7px 10px;color:#a89c8c;font-size:11px">{r.get('event_type','')}</td>
                  <td style="padding:7px 10px;color:#a89c8c;font-size:11px">{r.get('sector','')}</td>
                  <td style="padding:7px 10px;text-align:right;color:#8faa9a;font-size:12px;font-weight:400;font-variant-numeric:tabular-nums">{r.get('FES','')}</td>
                </tr>"""
            conc_html = f"""
    <div style="margin:26px 0">
      <div style="font-size:11px;color:#8a7f70;text-transform:uppercase;letter-spacing:.14em;margin-bottom:6px">★ 每日集中操作清单 · 核心策略(集中10只验证过的事件标的 · 去偏差后温和跑赢纳指)</div>
      <div style="display:grid;grid-template-columns:repeat(4,1fr);border-top:1px solid #3a3128;border-bottom:1px solid #3a3128;margin-bottom:14px">
        <div style="padding:12px 16px 12px 0"><div style="font-size:10px;color:#8a7f70;text-transform:uppercase;letter-spacing:.12em">持股/投入</div><div style="font-size:22px;font-weight:400;color:#f4ecdf;font-family:'Financier Display',Georgia,serif;line-height:1.1">{cs.get('n','')}只 <span style="color:#8a7f70;font-size:14px">{cs.get('invested_pct',0):.0f}%</span></div></div>
        <div style="padding:12px 16px;border-left:1px solid #3a3128"><div style="font-size:10px;color:#8a7f70;text-transform:uppercase;letter-spacing:.12em">组合波动</div><div style="font-size:22px;font-weight:400;color:#f4ecdf;font-family:'Financier Display',Georgia,serif;line-height:1.1">{cs.get('portfolio_vol_est',0)*100:.0f}%</div></div>
        <div style="padding:12px 16px;border-left:1px solid #3a3128"><div style="font-size:10px;color:#8a7f70;text-transform:uppercase;letter-spacing:.12em">最大行业</div><div style="font-size:15px;font-weight:400;color:#f4ecdf;line-height:1.6">{cs.get('top_sector','')} {cs.get('top_sector_pct',0):.0f}%</div></div>
        <div style="padding:12px 0 12px 16px;border-left:1px solid #3a3128"><div style="font-size:10px;color:#8a7f70;text-transform:uppercase;letter-spacing:.12em">去偏差回测(集中10)</div><div style="font-size:15px;font-weight:400;color:#8faa9a;line-height:1.6">24.4% · 夏普1.4<span style="color:#8a7f70;font-size:11px"> vs 纳指21%/0.99</span></div></div>
      </div>
      <div style="overflow-x:auto"><table style="width:100%;border-collapse:collapse;font-size:12px;min-width:520px">
        <thead><tr style="border-bottom:1px solid #3a3128">
          <th style="text-align:left;padding:6px 10px;font-size:10px;text-transform:uppercase;color:#8a7f70">标的</th>
          <th style="text-align:left;padding:6px 10px;font-size:10px;text-transform:uppercase;color:#8a7f70">目标仓位 (Target Position)</th>
          <th style="text-align:left;padding:6px 10px;font-size:10px;text-transform:uppercase;color:#8a7f70">事件类型 (Event Type)</th>
          <th style="text-align:left;padding:6px 10px;font-size:10px;text-transform:uppercase;color:#8a7f70">行业</th>
          <th style="text-align:right;padding:6px 10px;font-size:10px;text-transform:uppercase;color:#8a7f70">FES</th>
        </tr></thead><tbody>{crows}</tbody></table></div>
      <p style="color:#8faa9a;font-size:11.5px;margin-top:8px"><strong>操作:</strong> 这是系统每日更新的核心清单 —— 按目标仓位 (Target Position)买入这 {cs.get('n','')} 只(集中=edge的来源, 别摊太散)。去偏差(PIT成分股)回测: <strong>年化24.4% · 夏普1.39 · 回撤减半</strong>, 温和跑赢纳指。</p>
      <p style="color:#746a5d;font-size:11px;margin-top:4px">⚠ 诚实边界: 2年样本+仍缺324只退市股 → 真实值会打折(现实约看20%上下); edge在"集中选股"不在择时/做空(16年验证: 择时只护回撤不加收益, 裸空指数长期稳输)。是"真实但温和的edge", 非圣杯。</p>
    </div>"""
        except Exception:
            conc_html = ""

    # 持仓管理 (Position Management): 止损 + 抽本金 (Pull Principal) + 移动止损 (Trailing Stop)
    posmgr_html = ""
    pap = ROOT / "position_actions.csv"
    if pap.exists():
        try:
            pa = pd.read_csv(pap)
            arows = ""
            for _, r in pa.iterrows():
                act = str(r.get("action", ""))
                acol = ("#c68b83" if "止损清仓 (Stop-Loss Exit)" in act else "#c0a878" if "移动止损 (Trailing Stop)" in act
                        else "#8faa9a" if ("抽回本金" in act or "大赢" in act) else "#a89c8c")
                ret = float(r.get("ret_%", 0)); rcol = "#8faa9a" if ret >= 0 else "#c68b83"
                arows += (f'<tr><td style="padding:5px 10px;color:#f4ecdf;font-weight:400">{r.get("ticker","")}</td>'
                          f'<td style="padding:5px 10px;text-align:right;color:{rcol};font-size:11px;font-variant-numeric:tabular-nums">{ret:+.1f}%</td>'
                          f'<td style="padding:5px 10px;color:{acol};font-size:11px;font-weight:400">{act}</td>'
                          f'<td style="padding:5px 10px;color:#8a7f70;font-size:11px">{r.get("detail","")}</td></tr>')
            posmgr_html = f"""
    <div style="margin:26px 0">
      <div style="font-size:11px;color:#8a7f70;text-transform:uppercase;letter-spacing:.14em;margin-bottom:6px">持仓管理 (Position Management) · 止损 / 抽本金 (Pull Principal) / 移动止损 (Trailing Stop)(把"暴雷"变"小伤")</div>
      <p style="color:#a89c8c;font-size:11.5px;margin-bottom:10px">铁律: <strong style="color:#c68b83">跌破入场-15%→止损清仓 (Stop-Loss Exit)</strong>(防暴雷) · <strong style="color:#8faa9a">涨+50%→抽回本金只留利润</strong>(house money) · <strong style="color:#c0a878">从高点回撤-15%→移动止损 (Trailing Stop)锁利</strong></p>
      <div style="overflow-x:auto"><table style="width:100%;border-collapse:collapse;font-size:12px;min-width:520px">
        <thead><tr style="border-bottom:1px solid #3a3128">
          <th style="text-align:left;padding:6px 10px;font-size:10px;text-transform:uppercase;color:#8a7f70">标的</th>
          <th style="text-align:right;padding:6px 10px;font-size:10px;text-transform:uppercase;color:#8a7f70">盈亏</th>
          <th style="text-align:left;padding:6px 10px;font-size:10px;text-transform:uppercase;color:#8a7f70">今日动作</th>
          <th style="text-align:left;padding:6px 10px;font-size:10px;text-transform:uppercase;color:#8a7f70">说明</th>
        </tr></thead><tbody>{arows}</tbody></table></div>
      <p style="color:#746a5d;font-size:11px;margin-top:8px">诚实: 纸面账本, 入场价=首次进清单当日价; 真用需连券商/手动记成交价。这一层让单只暴雷≤-15%封顶, 赢家抽本金 (Pull Principal)后用利润博。</p>
    </div>"""
        except Exception:
            posmgr_html = ""

    # 第6层 仓位构建 + 执行成本 (作战部署)
    deploy_html = ""
    ppp = ROOT / "position_plan_event.csv"
    if ppp.exists():
        try:
            pp = pd.read_csv(ppp)
            ps = json.load(open(ROOT / "position_plan_summary.json")) if (ROOT / "position_plan_summary.json").exists() else {}
            ec = pd.read_csv(ROOT / "execution_cost_plan.csv") if (ROOT / "execution_cost_plan.csv").exists() else pd.DataFrame()
            es = json.load(open(ROOT / "execution_cost_summary.json")) if (ROOT / "execution_cost_summary.json").exists() else {}
            cost_map = {str(r["ticker"]): r for _, r in ec.iterrows()} if not ec.empty else {}
            prows = ""
            for _, r in pp.iterrows():
                tk = str(r["ticker"])
                is_probe = "试探" in str(r.get("rationale", ""))
                tr = "发动机轨" if "利润发动机 (Profit Engine)" in str(r.get("track", "")) else "普通轨"
                trcol = "#c8b487" if tr == "发动机轨" else "#8a7f70"
                badge = "试探仓" if is_probe else "正式仓"
                bcol = "#c0a878" if is_probe else "#8faa9a"
                c = cost_map.get(tk)
                rt = f"{float(c['roundtrip_bps']):.0f}bps" if c is not None else "—"
                net = f"{float(c['net_edge_%']):+.0f}%" if c is not None else "—"
                wbar = float(r["weight_pct"])
                prows += f"""<tr>
                  <td style="padding:8px 10px;font-weight:400;color:#f4ecdf">{tk}</td>
                  <td style="padding:8px 10px"><span style="color:{trcol};font-size:11px">{tr}</span></td>
                  <td style="padding:8px 10px"><span style="background:#241f16;color:{bcol};font-size:10px;padding:2px 7px;border-radius:3px">{badge}</span></td>
                  <td style="padding:8px 10px;min-width:110px"><div style="background:#1a140e;border-radius:3px;height:15px;position:relative"><div style="background:#c8b487;height:15px;width:{min(wbar/14*100,100):.0f}%;border-radius:3px"></div><span style="position:absolute;left:6px;top:0;font-size:10px;line-height:15px;color:#17130f;font-weight:400">{wbar:.1f}%</span></div></td>
                  <td style="padding:8px 10px;color:#a89c8c;font-size:11px">{r.get('event_type','')}</td>
                  <td style="padding:8px 10px;color:#a89c8c;font-size:11px">{r.get('sector','')}</td>
                  <td style="padding:8px 10px;text-align:right;color:#8a7f70;font-size:11px;font-variant-numeric:tabular-nums">{rt}</td>
                  <td style="padding:8px 10px;text-align:right;color:#8faa9a;font-size:11px;font-variant-numeric:tabular-nums">{net}</td>
                </tr>"""
            band = ps.get("band", "—")
            deploy_html = f"""
    <div style="margin:26px 0">
      <div style="font-size:11px;color:#8a7f70;text-transform:uppercase;letter-spacing:.14em;margin-bottom:12px">第6层 · 作战部署(仓位构建 × 执行成本)</div>
      <div style="display:grid;grid-template-columns:repeat(4,1fr);border-top:1px solid #3a3128;border-bottom:1px solid #3a3128;margin-bottom:16px">
        <div style="padding:14px 18px 14px 0">
          <div style="font-size:10px;color:#8a7f70;text-transform:uppercase;letter-spacing:.12em">总部署 / 现金</div>
          <div style="font-size:23px;font-weight:400;color:#f4ecdf;font-family:'Financier Display',Georgia,serif;line-height:1.1">{ps.get('invested_pct',0):.0f}% <span style="color:#8a7f70;font-size:15px">/ {ps.get('cash_pct',0):.0f}%</span></div>
          <div style="font-size:11px;color:#a89c8c;margin-top:3px">{band} · 预算 {ps.get('total_budget_pct',0):.0f}%</div>
        </div>
        <div style="padding:14px 18px;border-left:1px solid #3a3128">
          <div style="font-size:10px;color:#8a7f70;text-transform:uppercase;letter-spacing:.12em">组合波动估计</div>
          <div style="font-size:23px;font-weight:400;color:#f4ecdf;font-family:'Financier Display',Georgia,serif;line-height:1.1">{ps.get('portfolio_vol_est',0)*100:.0f}%</div>
          <div style="font-size:11px;color:#a89c8c;margin-top:3px">{ps.get('n_positions',0)} 仓 · 最大行业 {ps.get('top_sector_pct',0):.0f}%</div>
        </div>
        <div style="padding:14px 18px;border-left:1px solid #3a3128">
          <div style="font-size:10px;color:#8a7f70;text-transform:uppercase;letter-spacing:.12em">往返摩擦</div>
          <div style="font-size:23px;font-weight:400;color:#f4ecdf;font-family:'Financier Display',Georgia,serif;line-height:1.1">{es.get('blended_roundtrip_bps',0):.0f}<span style="font-size:14px;color:#8a7f70"> bps</span></div>
          <div style="font-size:11px;color:#a89c8c;margin-top:3px">${es.get('total_roundtrip_cost_usd',0):,} @ ${es.get('nav_usd',0)/1e6:.0f}M</div>
        </div>
        <div style="padding:14px 0 14px 18px;border-left:1px solid #3a3128">
          <div style="font-size:10px;color:#8a7f70;text-transform:uppercase;letter-spacing:.12em">净edge覆盖</div>
          <div style="font-size:23px;font-weight:400;color:#8faa9a;font-family:'Financier Display',Georgia,serif;line-height:1.1">{es.get('positions_edge_survives',0)}/{es.get('positions_total',0)}</div>
          <div style="font-size:11px;color:#a89c8c;margin-top:3px">扣成本后仍正</div>
        </div>
      </div>
      <div style="overflow-x:auto"><table style="width:100%;border-collapse:collapse;font-size:12px;min-width:680px">
        <thead><tr style="border-bottom:1px solid #3a3128">
          <th style="text-align:left;padding:6px 10px;font-size:10px;text-transform:uppercase;color:#8a7f70">标的</th>
          <th style="text-align:left;padding:6px 10px;font-size:10px;text-transform:uppercase;color:#8a7f70">轨</th>
          <th style="text-align:left;padding:6px 10px;font-size:10px;text-transform:uppercase;color:#8a7f70">类型</th>
          <th style="text-align:left;padding:6px 10px;font-size:10px;text-transform:uppercase;color:#8a7f70">目标仓位 (Target Position)</th>
          <th style="text-align:left;padding:6px 10px;font-size:10px;text-transform:uppercase;color:#8a7f70">事件类型 (Event Type)</th>
          <th style="text-align:left;padding:6px 10px;font-size:10px;text-transform:uppercase;color:#8a7f70">行业</th>
          <th style="text-align:right;padding:6px 10px;font-size:10px;text-transform:uppercase;color:#8a7f70">往返成本</th>
          <th style="text-align:right;padding:6px 10px;font-size:10px;text-transform:uppercase;color:#8a7f70">净edge</th>
        </tr></thead><tbody>{prows}</tbody></table></div>
      <p style="color:#746a5d;font-size:11px;margin-top:8px">仓位 = L0稀缺带 × 宏观 × 回撤乘数,信念(FES)×执行×逆波动定权,双轨+单票/行业上限。成本 = 半价差+平方根冲击+滑点,往返按持有窗口摊。净edge = 事件目标收益 − 往返成本。</p>
    </div>"""
        except Exception:
            deploy_html = ""

    # 事件层 edge 验证 (8-K 事件研究)
    validation_html = ""
    esp = ROOT / "edgar_event_study.json"
    if esp.exists():
        try:
            es = json.load(open(esp))
            neu = es.get("neutralized_63d", {})
            vrows = ""
            for et, v in sorted(neu.items(),
                                key=lambda x: -((x[1].get("sector_neutral") or {}).get("t", -9))):
                ma = v.get("market_adj"); sn = v.get("sector_neutral"); bn = v.get("beta_neutral")
                if not sn:
                    continue
                t = sn["t"]
                sig = "✓ 真alpha" if abs(t) >= 2 else "弱" if abs(t) >= 1 else "无"
                scol = "#8faa9a" if abs(t) >= 2 else "#c0a878" if abs(t) >= 1 else "#8a7f70"
                def cell(s):
                    return f'{s["mean_ab_%"]:+.2f}% <span style="color:#8a7f70">t{s["t"]:+.1f}</span>' if s else "—"
                vrows += f"""<tr>
                  <td style="padding:7px 10px;color:#f4ecdf;font-size:12px">{et}</td>
                  <td style="padding:7px 10px;text-align:right;color:#a89c8c;font-size:11px;font-variant-numeric:tabular-nums">{sn['n']:,}</td>
                  <td style="padding:7px 10px;text-align:right;color:#a89c8c;font-size:11px;font-variant-numeric:tabular-nums">{cell(ma)}</td>
                  <td style="padding:7px 10px;text-align:right;color:#8faa9a;font-size:12px;font-weight:400;font-variant-numeric:tabular-nums">{cell(sn)}</td>
                  <td style="padding:7px 10px;text-align:right;color:#a89c8c;font-size:11px;font-variant-numeric:tabular-nums">{cell(bn)}</td>
                  <td style="padding:7px 10px;color:{scol};font-size:11px">{sig}</td>
                </tr>"""
            # 最肥的 8-K item top5
            items = es.get("by_item", {})
            item_line = " · ".join(f'{v["desc"]}({v["mean_ab_%"]:+.1f}%,t{v["t"]:.1f})'
                                   for _, v in sorted(items.items(), key=lambda x: -x[1]["t"])[:5])
            validation_html = f"""
    <div style="margin:26px 0">
      <div style="font-size:11px;color:#8a7f70;text-transform:uppercase;letter-spacing:.14em;margin-bottom:6px">事件层 EDGE 验证 · 8-K 事件研究(证明系统有真实预测力)</div>
      <p style="color:#a89c8c;font-size:12.5px;margin-bottom:12px;max-width:760px">{es.get('total_events',0):,} 个历史 8-K 备案的 63天超额收益(无前瞻)。<strong style="color:#cabeae">关键:剔除行业和 beta 后 edge 依然显著</strong>(行业中性 (Sector-Neutral) t 甚至更高)—— 证明是真事件 alpha,不是行业/风格暴露。对照:纯价格骨架 IC 的 t 仅 1.74(不显著)。</p>
      <div style="overflow-x:auto"><table style="width:100%;border-collapse:collapse;font-size:12px;min-width:600px">
        <thead><tr style="border-bottom:1px solid #3a3128">
          <th style="text-align:left;padding:6px 10px;font-size:10px;text-transform:uppercase;color:#8a7f70">事件类型 (Event Type)</th>
          <th style="text-align:right;padding:6px 10px;font-size:10px;text-transform:uppercase;color:#8a7f70">样本</th>
          <th style="text-align:right;padding:6px 10px;font-size:10px;text-transform:uppercase;color:#8a7f70">市场调整</th>
          <th style="text-align:right;padding:6px 10px;font-size:10px;text-transform:uppercase;color:#8faa9a">行业中性 (Sector-Neutral)</th>
          <th style="text-align:right;padding:6px 10px;font-size:10px;text-transform:uppercase;color:#8a7f70">beta中性</th>
          <th style="text-align:left;padding:6px 10px;font-size:10px;text-transform:uppercase;color:#8a7f70">判定</th>
        </tr></thead><tbody>{vrows}</tbody></table></div>
      <p style="color:#a89c8c;font-size:11.5px;margin-top:10px"><strong style="color:#cabeae">最肥的 8-K 类型:</strong> {item_line}</p>
      <p style="color:#746a5d;font-size:11px;margin-top:6px">诚实提示: 胜率约47-53%(小而稳的漂移,非高胜率),含已知 PEAD; 2年样本。真实、经中性化验证,但非圣杯。已反哺打分(EdgeFactor)。</p>
    </div>"""
        except Exception:
            validation_html = ""

    # 复盘节奏层 (周/月/季)
    review_html = ""
    rp = ROOT / "review_report.json"
    if rp.exists():
        try:
            rv = json.load(open(rp))
            w, m, q = rv.get("weekly", {}), rv.get("monthly", {}), rv.get("quarterly", {})
            trg = w.get("exit_triggers", [])
            wk = "".join(f'<li style="margin:3px 0"><strong style="color:#c68b83">{t["ticker"]}</strong> <span style="color:#a89c8c">{"/".join(t.get("flags",[]))}</span> → <span style="color:#c0a878">{t.get("action","")}</span></li>' for t in trg[:6]) \
                 or f'<li style="color:#8faa9a;list-style:none">✓ 活跃候选均在结构之上,无退出触发</li>'
            dec = w.get("score_decays", [])
            wk_dec = ("恶化: " + ", ".join(f'{d["ticker"]}({d["drop"]})' for d in dec[:6])) if dec else (w.get("note") or "分数稳定")
            hl = "".join(f'<div style="font-size:12px;color:#a89c8c;margin:3px 0">{h}</div>' for h in m.get("health", []))
            promo = m.get("promotions", []); demo = m.get("demotions", [])
            mig = ""
            if promo:
                mig += "<div style='color:#8faa9a;font-size:12px'>升池: " + ", ".join(f'{p["ticker"]}' for p in promo[:6]) + "</div>"
            if demo:
                mig += "<div style='color:#c68b83;font-size:12px'>降池: " + ", ".join(f'{d["ticker"]}' for d in demo[:6]) + "</div>"
            if not mig:
                mig = f"<div style='color:#8a7f70;font-size:11px'>{m.get('note') or '池结构稳定'}</div>"
            et_rows = "".join(f'<div style="display:flex;justify-content:space-between;font-size:12px;padding:2px 0"><span style="color:#a89c8c">{e["event_type"]}</span><span style="color:#cabeae;font-variant-numeric:tabular-nums">{e["count"]}只 · 均{e["avg_FES"]}</span></div>' for e in q.get("event_type_contribution", []))
            recal = "".join(f'<li style="margin:3px 0;color:#a89c8c">{c}</li>' for c in q.get("recalibration", []))
            review_html = f"""
    <div style="margin:26px 0">
      <div style="font-size:11px;color:#8a7f70;text-transform:uppercase;letter-spacing:1px;margin-bottom:10px">复盘节奏层 · 周/月/季 (快照历史 {rv.get('history_days',1)} 天)</div>
      <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:14px">
        <div style="background:#16140f;border:1px solid #3a3128;border-top:3px solid #c68b83;padding:16px 18px;border-radius:8px">
          <div style="font-size:13px;font-weight:400;color:#f4ecdf;margin-bottom:8px">周 · 战术(逻辑还在不在)</div>
          <ul style="margin:0;padding-left:16px;font-size:12px;color:#a89c8c">{wk}</ul>
          <div style="font-size:11px;color:#8a7f70;margin-top:8px">{wk_dec}</div>
        </div>
        <div style="background:#16140f;border:1px solid #3a3128;border-top:3px solid #c0a878;padding:16px 18px;border-radius:8px">
          <div style="font-size:13px;font-weight:400;color:#f4ecdf;margin-bottom:8px">月 · 调仓(池该轮换吗)</div>
          {hl}
          <div style="margin-top:8px">{mig}</div>
        </div>
        <div style="background:#16140f;border:1px solid #3a3128;border-top:3px solid #8faa9a;padding:16px 18px;border-radius:8px">
          <div style="font-size:13px;font-weight:400;color:#f4ecdf;margin-bottom:8px">季 · 系统(打法对不对)</div>
          {et_rows}
          <ul style="margin:8px 0 0;padding-left:16px;font-size:11px">{recal}</ul>
        </div>
      </div>
    </div>"""
        except Exception:
            review_html = ""

    pool_style = {
        "利润发动机 (Profit Engine)":   ("#241f16", "#c8b487", "利润发动机 (Profit Engine)"),
        "事件型爆发池": ("#1c231e", "#8faa9a", "事件爆发池 (Event Breakout Pool)"),
        "观察/不入池":  ("#2a231b", "#9a8e80", "观察"),
    }
    rows = []
    for _, r in df.head(40).iterrows():
        bg, fg, lbl = pool_style.get(str(r.get("pool", "")), ("#2a231b", "#9a8e80", "观察"))
        lnmpc = " · ".join(f"{k}{int(r[k])}" for k in ("L", "N", "M", "P", "C") if k in r and pd.notna(r[k]))
        life = life_m.get(str(r["ticker"]), "—")
        rows.append(f"""<tr>
          <td style="padding:9px 12px;font-weight:400;color:#f4ecdf">{r['ticker']}</td>
          <td style="padding:9px 12px"><span style="background:{bg};color:{fg};font-size:10px;font-weight:400;padding:3px 9px;border-radius:4px;white-space:nowrap">{lbl}</span></td>
          <td style="padding:9px 12px;color:#9a8e80;font-size:12px">{r.get('event_type','—')}</td>
          <td style="padding:9px 12px;text-align:right;color:#c8b487;font-weight:400;font-variant-numeric:tabular-nums">{r.get('FinalEventScore','—')}</td>
          <td style="padding:9px 12px;color:#8a7f70;font-size:11px;font-variant-numeric:tabular-nums">{lnmpc}</td>
          <td style="padding:9px 12px;color:#7a8290;font-size:11px">{life}</td>
          <td style="padding:9px 12px;color:#9a8e80;font-size:11px">{r.get('ExecutionScore','—')}</td>
          <td style="padding:9px 12px;color:#9a8e80;font-size:11px">{r.get('hold_window','—')}</td>
          <td style="padding:9px 12px;color:#8a7f70;font-size:11px">交{r.get('exit_trade%','?')}/逻{r.get('exit_logic%','?')}</td>
        </tr>""")

    # FT 头版: 日期 · 市场姿态判词 · 头号主线 (Top Theme)
    _today = pd.Timestamp.now().strftime("%Y年%-m月%-d日")
    _mode = "—"
    try:
        _mode = json.load(open(ROOT / "macro_intel_scorecard.json")).get("宏观模式 (Macro Mode)", "—")
    except Exception:
        pass
    _top_tk = str(df.iloc[0]["ticker"]) if len(df) else "—"
    _top_et = str(df.iloc[0].get("event_type", "")) if len(df) else ""
    _top_fes = float(df.iloc[0]["FinalEventScore"]) if len(df) else 0.0
    _posture = f"{sw_state} · 建议进攻仓位 {sw_pos}"
    _stand = (f"宏观处于「{_mode}」模式,宏观过滤 (Macro Filter) {mf:.2f}。全场 {len(df)} 只候选中,"
              f"{_top_tk}({_top_et})以 FinalEventScore {_top_fes:.0f} 领跑。"
              f"当前{sw_state},{'纪律优先于出手,允许高现金。' if n_pe+n_ev<=2 else '择优布局利润发动机 (Profit Engine)级机会。'}")
    return f"""<section id="sec-eventengine" class="tab-section">
  <div class="container">
    <div style="border-top:2px solid #c8b487;padding-top:12px;display:flex;justify-content:space-between;align-items:baseline;flex-wrap:wrap;gap:8px">
      <span style="font-size:11px;letter-spacing:.24em;text-transform:uppercase;color:#c8b487;font-weight:400">Canyon Intelligence · 利润发动机 (Profit Engine)作战台 (War Room)</span>
      <span style="font-size:11px;letter-spacing:.06em;color:#8a7f70;text-transform:uppercase">{_today} · 标普500 事件驱动版</span>
    </div>
    <h2 class="section-head" style="font-size:42px;line-height:1.08;max-width:900px;margin:14px 0 12px">{_posture}</h2>
    <p style="color:#cabeae;font-size:15.5px;line-height:1.62;max-width:700px;margin-bottom:6px;font-family:'Financier Display','Iowan Old Style',Georgia,serif">{_stand}</p>
    <p style="color:#8a7f70;font-size:12px;max-width:700px;margin-bottom:22px">依据《美股主动投资系统手册》第0/5/7/9/10层 · FinalEventScore = EventScore × 执行过滤 × 宏观过滤 (Macro Filter) · L/N/M/P/C 五因子均由真实数据驱动(Momentum/催化日历/DCF估值/新闻主体/分析师确认)。</p>
    <div style="display:grid;grid-template-columns:repeat(4,1fr);border-top:1px solid #3a3128;border-bottom:1px solid #3a3128;margin-bottom:30px">
      <div style="padding:16px 20px 16px 0">
        <div style="font-size:10px;color:#8a7f70;text-transform:uppercase;letter-spacing:.14em;margin-bottom:8px">第0层 · 机会稀缺度</div>
        <div style="font-size:26px;font-weight:400;color:{sw_color};font-family:'Financier Display',Georgia,serif;line-height:1">{sw_state}</div>
        <div style="font-size:12px;color:#a89c8c;margin-top:5px">进攻仓位 {sw_pos}</div>
      </div>
      <div style="padding:16px 20px;border-left:1px solid #3a3128">
        <div style="font-size:10px;color:#8a7f70;text-transform:uppercase;letter-spacing:.14em;margin-bottom:8px">利润发动机 (Profit Engine) / 爆发池</div>
        <div style="font-size:26px;font-weight:400;color:#c8b487;font-family:'Financier Display',Georgia,serif;line-height:1;font-variant-numeric:tabular-nums">{n_pe}<span style="color:#8a7f70;font-size:18px"> / </span><span style="color:#8faa9a">{n_ev}</span></div>
        <div style="font-size:12px;color:#a89c8c;margin-top:5px">共 {len(df)} 只候选</div>
      </div>
      <div style="padding:16px 20px;border-left:1px solid #3a3128">
        <div style="font-size:10px;color:#8a7f70;text-transform:uppercase;letter-spacing:.14em;margin-bottom:8px">第1层 · 宏观过滤 (Macro Filter)</div>
        <div style="font-size:26px;font-weight:400;color:#f4ecdf;font-family:'Financier Display',Georgia,serif;line-height:1;font-variant-numeric:tabular-nums">{mf:.2f}</div>
        <div style="font-size:12px;color:#a89c8c;margin-top:5px">{_mode} · {'风险偏好正常' if mf>=0.8 else '偏防守'}</div>
      </div>
      <div style="padding:16px 0 16px 20px;border-left:1px solid #3a3128">
        <div style="font-size:10px;color:#8a7f70;text-transform:uppercase;letter-spacing:.14em;margin-bottom:8px">头号主线 (Top Theme)</div>
        <div style="font-size:26px;font-weight:400;color:#f4ecdf;font-family:'Financier Display',Georgia,serif;line-height:1">{_top_tk}</div>
        <div style="font-size:12px;color:#a89c8c;margin-top:5px">FES {_top_fes:.0f} · {_top_et}</div>
      </div>
    </div>
    {_intraday_panel()}
    {_morning_brief_panel()}
    {_ten_layer_matrix_panel()}
    {_macro_deep_panel()}
    {_fred_macro_panel()}
    {_sector_theme_panel()}
    {_strategy_thesis_panel()}
    {_research_memo_panel()}
    {_news_deep_panel()}
    {_alphavantage_news_panel()}
    {intel_html}
    {src_html}
    {rot_html}
    {etf_html}
    {fp_html}
    {_pead_panel()}
    {_pnl_contrib_panel()}
    <div style="font-size:11px;color:#8a7f70;text-transform:uppercase;letter-spacing:1px;margin-bottom:10px">第7层 · 事件爆发池 (Event Breakout Pool)排序(FinalEventScore 前 40 / 共 {len(df)} 只)</div>
    <div style="overflow-x:auto"><table style="width:100%;border-collapse:collapse;font-size:13px;min-width:780px">
      <thead><tr style="border-bottom:2px solid #453a2c">
        <th style="text-align:left;padding:8px 12px;font-size:10px;text-transform:uppercase;color:#8a7f70">标的</th>
        <th style="text-align:left;padding:8px 12px;font-size:10px;text-transform:uppercase;color:#8a7f70">池</th>
        <th style="text-align:left;padding:8px 12px;font-size:10px;text-transform:uppercase;color:#8a7f70">事件类型 (Event Type)</th>
        <th style="text-align:right;padding:8px 12px;font-size:10px;text-transform:uppercase;color:#8a7f70">FinalScore</th>
        <th style="text-align:left;padding:8px 12px;font-size:10px;text-transform:uppercase;color:#8a7f70">L/N/M/P/C</th>
        <th style="text-align:left;padding:8px 12px;font-size:10px;text-transform:uppercase;color:#8a7f70">生命周期 (Lifecycle)/风格</th>
        <th style="text-align:left;padding:8px 12px;font-size:10px;text-transform:uppercase;color:#8a7f70">执行分</th>
        <th style="text-align:left;padding:8px 12px;font-size:10px;text-transform:uppercase;color:#8a7f70">持有窗口</th>
        <th style="text-align:left;padding:8px 12px;font-size:10px;text-transform:uppercase;color:#8a7f70">退出(交/逻)</th>
      </tr></thead><tbody>{''.join(rows)}</tbody></table></div>
    {conc_html}
    {posmgr_html}
    {deploy_html}
    {validation_html}
    {review_html}
    <p style="color:#746a5d;font-size:11px;margin-top:16px">研究/纸面用途 · 无券商连接 · 底库=标普500全体,系统自动分事件类型 (Event Type)/生命周期 (Lifecycle)/功能池并排序; L/N/M/P/C 可人工按手册标准精修于 event_pool.csv</p>
  </div>
</section>"""


def _build_data_health_tab() -> str:
    """Scan every dashboard data source and report freshness + validity.
    This is the honesty layer: it tells you which panels show live data,
    which are stale, and which are broken — so nothing is trusted blindly."""
    import time as _t
    now = _t.time()

    # (file, friendly name, which dashboard section it feeds, max-fresh-days)
    SOURCES = [
        ("alpha_scores.csv",            "Alpha scores",          "Signals / Today",   1.5),
        ("daily_picks.csv",             "Long book",             "Today / Signals",   1.5),
        ("daily_shorts.csv",            "Short book",            "Short Scanner",     1.5),
        ("sp500_price_cache.csv",       "Price cache",           "All price data",    1.5),
        ("hmm_regime_daily.csv",        "HMM regime",            "Today / Macro",     2),
        ("paper_sim_positions.csv",     "Paper positions",       "Live Positions",    3),
        ("paper_nav_curve.csv",         "Paper NAV curve",       "Performance",       2),
        ("etf_flow_daily.json",         "ETF sector flow",       "Flow",              2),
        ("stock_news.json",             "News",                  "News",              2),
        ("dcf_valuation.csv",           "DCF valuations",        "DCF",               7),
        ("earnings_calendar.csv",       "Earnings calendar",     "Earnings",          3),
        ("earnings_ai_summaries.csv",   "Earnings AI",           "Earnings AI",       7),
        ("macro_regime_outlook.json",   "Macro outlook",         "Macro",             2),
        ("macro_signals.json",          "Macro signals",         "Macro",             2),
        ("options_flow.json",           "Options flow",          "Flow",              2),
        ("famous_holdings.json",        "13F smart money",       "Smart Money",       14),
        ("congressional_trades.json",   "Congress trades",       "Smart Money",       7),
        ("heatmap_data.json",           "Heatmap",               "Heatmap",           2),
        ("factor_attribution.csv",      "Factor attribution",    "Attribution",       7),
        ("rolling_ic_monitor.csv",      "Rolling IC",            "Attribution",       3),
        ("correlation_monitor.csv",     "Correlation monitor",   "Attribution",       3),
        ("final_risk_gate.csv",         "Risk gate",             "Risk",              3),
        ("portfolio_risk_decomp.csv",   "Risk decomposition",    "Risk",              3),
        ("live_ic_history.csv",         "Live IC history",       "Attribution",       3),
        ("sector_cycle_state.csv",      "Sector cycle",          "Macro",             3),
        ("factor_ic_history.csv",       "Factor IC history",     "Attribution",       7),
        ("wf_oos_summary.csv",          "Walk-forward OOS",      "Quant QC",          14),
        ("watchlist.json",              "Watchlist",             "Watchlist",         14),
        ("short_squeeze_signal.csv",    "Short squeeze",         "Short Scanner",     3),
        ("rigorous_backtest.json",      "Honest backtest",       "Quant QC",          2),
        ("sp500_price_history_deep.csv","Deep price history",    "Backtests",         7),
    ]

    def _check(fname, max_days):
        p = ROOT / fname
        if not p.exists():
            return ("broken", "missing", "—")
        age = (now - p.stat().st_mtime) / 86400
        age_str = "today" if age < 1 else f"{int(age)}d ago"
        if p.stat().st_size <= 2:
            return ("broken", "empty file", age_str)
        # validity: NaN-heavy for csv
        try:
            if fname.endswith(".csv"):
                df = pd.read_csv(p)
                if len(df) == 0:
                    return ("broken", "no rows", age_str)
                num = df.select_dtypes("number")
                if num.size:
                    nanr = num.isna().sum().sum() / num.size
                    if nanr > 0.6:
                        return ("broken", f"{nanr*100:.0f}% empty", age_str)
            elif fname.endswith(".json"):
                j = json.load(open(p))
                if isinstance(j, (list, dict)) and len(j) == 0:
                    return ("broken", "empty", age_str)
        except Exception:
            return ("broken", "unreadable", age_str)
        if age > max_days:
            return ("stale", f"stale ({age_str})", age_str)
        return ("ok", "live", age_str)

    rows, n_ok, n_stale, n_broken = [], 0, 0, 0
    results = []
    for fname, name, section, max_days in SOURCES:
        status, detail, age_str = _check(fname, max_days)
        results.append((status, name, section, detail, age_str, fname))
        if status == "ok": n_ok += 1
        elif status == "stale": n_stale += 1
        else: n_broken += 1

    order = {"broken": 0, "stale": 1, "ok": 2}
    results.sort(key=lambda r: (order[r[0]], r[1]))

    badge = {
        "ok":     ('#1c231e', '#8faa9a', '● LIVE'),
        "stale":  ('#241f16', '#c0a878', '● STALE'),
        "broken": ('#251a17', '#c68b83', '● BROKEN'),
    }
    for status, name, section, detail, age_str, fname in results:
        bg, fg, lbl = badge[status]
        rows.append(f"""<tr>
          <td style="padding:9px 12px"><span style="background:{bg};color:{fg};font-size:10px;font-weight:400;padding:3px 9px;border-radius:3px;white-space:nowrap">{lbl}</span></td>
          <td style="padding:9px 12px;font-weight:400;color:#f4ecdf">{name}</td>
          <td style="padding:9px 12px;color:#9a8e80;font-size:12px">{section}</td>
          <td style="padding:9px 12px;color:{fg};font-size:12px">{detail}</td>
          <td style="padding:9px 12px;color:#8a7f70;font-size:12px;font-variant-numeric:tabular-nums">{age_str}</td>
          <td style="padding:9px 12px;color:#746a5d;font-size:11px;font-family:monospace">{fname}</td>
        </tr>""")

    total = len(SOURCES)
    return f"""<section id="sec-datahealth" class="tab-section">
  <div class="container">
    <p class="eyebrow">System · Data Health</p>
    <h2 class="section-head">Data Integrity Monitor</h2>
    <div class="rule"></div>
    <p style="color:#a89c8c;font-size:13px;margin-bottom:24px;max-width:640px">Every panel on this dashboard reads from a data file. This monitor shows which are <strong style="color:#8faa9a">live</strong>, which are <strong style="color:#c0a878">stale</strong> (old data still displayed), and which are <strong style="color:#c68b83">broken</strong> (missing/empty). Trust panels marked LIVE; treat STALE and BROKEN sections with suspicion.</p>
    <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin-bottom:24px">
      <div style="background:#1c231e;border:1px solid #1d4a30;padding:18px 20px;border-radius:6px">
        <div style="font-size:32px;font-weight:400;color:#8faa9a;line-height:1">{n_ok}</div>
        <div style="font-size:11px;color:#8faa9a;text-transform:uppercase;letter-spacing:1px;margin-top:4px">Live</div>
      </div>
      <div style="background:#241f16;border:1px solid #4a3a10;padding:18px 20px;border-radius:6px">
        <div style="font-size:32px;font-weight:400;color:#c0a878;line-height:1">{n_stale}</div>
        <div style="font-size:11px;color:#c0a878;text-transform:uppercase;letter-spacing:1px;margin-top:4px">Stale</div>
      </div>
      <div style="background:#251a17;border:1px solid #4a1d1d;padding:18px 20px;border-radius:6px">
        <div style="font-size:32px;font-weight:400;color:#c68b83;line-height:1">{n_broken}</div>
        <div style="font-size:11px;color:#c68b83;text-transform:uppercase;letter-spacing:1px;margin-top:4px">Broken</div>
      </div>
    </div>
    <div style="overflow-x:auto">
    <table style="width:100%;border-collapse:collapse;font-size:13px">
      <thead><tr style="border-bottom:2px solid #453a2c">
        <th style="text-align:left;padding:8px 12px;font-size:10px;text-transform:uppercase;letter-spacing:1px;color:#8a7f70">Status</th>
        <th style="text-align:left;padding:8px 12px;font-size:10px;text-transform:uppercase;letter-spacing:1px;color:#8a7f70">Data</th>
        <th style="text-align:left;padding:8px 12px;font-size:10px;text-transform:uppercase;letter-spacing:1px;color:#8a7f70">Feeds panel</th>
        <th style="text-align:left;padding:8px 12px;font-size:10px;text-transform:uppercase;letter-spacing:1px;color:#8a7f70">Detail</th>
        <th style="text-align:left;padding:8px 12px;font-size:10px;text-transform:uppercase;letter-spacing:1px;color:#8a7f70">Updated</th>
        <th style="text-align:left;padding:8px 12px;font-size:10px;text-transform:uppercase;letter-spacing:1px;color:#8a7f70">File</th>
      </tr></thead>
      <tbody>{''.join(rows)}</tbody>
    </table>
    </div>
    <p style="color:#746a5d;font-size:11px;margin-top:16px">{n_ok}/{total} sources live · scanned at build time ({datetime.now().strftime('%Y-%m-%d %H:%M')})</p>
  </div>
</section>"""


def _corr_bg(v):
    """Return background color for correlation cell."""
    try:
        v = float(v)
    except Exception:
        return "#fff"
    if v >= 0.99: return "#D5E8D4"
    if v >= 0.60: return "#FFD7CC"
    if v <= -0.60: return "#DAE8FC"
    if abs(v) < 0.20: return "#F9F9F9"
    return "#fff"


def _build_heatmap_tab() -> str:
    """S&P 500 sector heatmap tab — canvas treemap sized by market cap, colored by daily return."""
    import json as _json, pathlib as _pl
    hm_path = _pl.Path(__file__).parent / "heatmap_data.json"
    if not hm_path.exists():
        return """<section id="sec-heatmap" class="tab-section">
  <div class="container"><p style="color:#AAA;padding:60px 0;text-align:center">Run <code>python3 step_heatmap_data.py</code> to generate heatmap data first.</p></div>
</section>"""
    try:
        with open(hm_path) as _f:
            heatmap_json = _f.read()
    except Exception:
        heatmap_json = "[]"

    return f"""<section id="sec-heatmap" class="tab-section">
  <div style="max-width:1600px;margin:0 auto;padding:0 20px">
    <p class="eyebrow">S&amp;P 500 Sector Heatmap</p>
    <h2 class="section-head">Daily return by stock &amp; sector — sized by market cap, scored by Canyon</h2>
    <div class="rule"></div>
    <div id="hm-controls" style="display:flex;gap:8px;flex-wrap:wrap;margin-bottom:14px;align-items:center">
      <span style="font-size:11px;color:#999;font-weight:400;text-transform:uppercase;letter-spacing:.8px;margin-right:4px">Sector</span>
      <button class="hm-btn active" data-sector="">All</button>
    </div>
    <div style="position:relative;background:#111;border-radius:8px;overflow:hidden">
      <canvas id="hm-canvas" style="display:block;width:100%;height:900px"></canvas>
      <div id="hm-tooltip" style="display:none;position:absolute;pointer-events:none;background:rgba(15,15,15,.96);border:1px solid #444;border-radius:7px;padding:10px 14px;font-size:12px;color:#EEE;min-width:170px;z-index:10;box-shadow:0 4px 20px rgba(0,0,0,.5)">
        <div id="hm-tt-ticker" style="font-size:16px;font-weight:500;color:#fff;margin-bottom:3px"></div>
        <div id="hm-tt-return" style="font-size:14px;font-weight:400;margin-bottom:3px"></div>
        <div id="hm-tt-score"  style="font-size:11px;color:#AAA"></div>
        <div id="hm-tt-sector" style="font-size:11px;color:#777;margin-top:2px"></div>
        <div id="hm-tt-signal" style="font-size:11px;color:#AAA;margin-top:2px"></div>
      </div>
    </div>
    <div style="display:flex;align-items:center;gap:6px;margin-top:10px;flex-wrap:wrap">
      <div style="display:flex;align-items:center;gap:4px">
        <span style="font-size:10px;color:#AAA">-3%</span>
        <div style="width:140px;height:10px;border-radius:5px;background:linear-gradient(to right,#E03030,#2C2E34,#1DB954)"></div>
        <span style="font-size:10px;color:#AAA">+3%</span>
      </div>
      <span style="font-size:10px;color:#666;margin-left:12px">All S&amp;P 500 stocks · sized by market cap · click a sector to zoom in</span>
    </div>
  </div>
  <script>
  (function() {{
    const RAW = {heatmap_json};
    const style = document.createElement('style');
    style.textContent = `.hm-btn{{background:#2A2A2A;border:1px solid #444;color:#CCC;padding:4px 10px;border-radius:4px;font-size:11px;font-weight:400;cursor:pointer;letter-spacing:.3px;transition:background .12s,color .12s}}.hm-btn:hover,.hm-btn.active{{background:#c8b487;border-color:#c8b487;color:#fff}}`;
    document.head.appendChild(style);

    const sectors = [...new Set(RAW.map(d => d.sector))].sort();
    const ctrl = document.getElementById('hm-controls');
    sectors.forEach(sec => {{
      const btn = document.createElement('button');
      btn.className = 'hm-btn';
      btn.dataset.sector = sec;
      btn.textContent = sec;
      ctrl.appendChild(btn);
    }});

    let activeSector = '';
    ctrl.addEventListener('click', function(e) {{
      const btn = e.target.closest('.hm-btn');
      if (!btn) return;
      document.querySelectorAll('.hm-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      activeSector = btn.dataset.sector;
      render();
    }});

    const canvas  = document.getElementById('hm-canvas');
    const tooltip = document.getElementById('hm-tooltip');
    const ttTicker = document.getElementById('hm-tt-ticker');
    const ttReturn = document.getElementById('hm-tt-return');
    const ttScore  = document.getElementById('hm-tt-score');
    const ttSector = document.getElementById('hm-tt-sector');
    const ttSignal = document.getElementById('hm-tt-signal');
    let cellMap = [];

    function retColor(ret, score) {{
      const t  = Math.max(-1, Math.min(1, ret / 3));
      if (Math.abs(t) < 0.02) return '#1C2028';   // flat / neutral dark slate
      // High-score picks get a gentle brightness lift
      const sb = 0.78 + (score / 100) * 0.22;
      let r, g, b;
      if (t > 0) {{
        // Comfortable emerald green — dark at +0.5%, medium at +3%
        r = Math.round((25 + t * 20) * sb);
        g = Math.round((80 + t * 65) * sb);
        b = Math.round((48 + t * 32) * sb);
      }} else {{
        const p = -t;
        // Comfortable burgundy red — dark at -0.5%, medium at -3%
        r = Math.round((110 + p * 55) * sb);
        g = Math.round((28  + p * 12) * sb);
        b = Math.round((28  + p * 12) * sb);
      }}
      return `rgb(${{Math.min(255,r)}},${{Math.min(255,g)}},${{Math.min(255,b)}})`;
    }}

    function render() {{
      const dpr = window.devicePixelRatio || 1;
      const W = canvas.offsetWidth;
      const H = canvas.offsetHeight;
      canvas.width  = W * dpr;
      canvas.height = H * dpr;
      const ctx = canvas.getContext('2d');
      ctx.scale(dpr, dpr);
      ctx.fillStyle = '#13161D';
      ctx.fillRect(0, 0, W, H);

      // Filter out noise sectors
      const SKIP_SECTORS = new Set(['Broad', 'Other', '']);
      const allClean = RAW.filter(d => !SKIP_SECTORS.has(d.sector || ''));

      let data;
      if (activeSector) {{
        data = allClean.filter(d => d.sector === activeSector);
      }} else {{
        data = allClean;
      }}

      const secMap = {{}};
      data.forEach(d => {{
        if (!secMap[d.sector]) secMap[d.sector] = {{ total: 0, count: 0, stocks: [] }};
        secMap[d.sector].total += (d.market_cap_usd || 1e9);
        secMap[d.sector].count += 1;
        secMap[d.sector].stocks.push(d);
      }});
      const secList = Object.entries(secMap).sort((a,b) => b[1].total - a[1].total);
      const totalCap   = secList.reduce((s,[,v]) => s + v.total, 0);
      const totalCount = secList.reduce((s,[,v]) => s + v.count, 0);

      cellMap = [];
      let sx = 0;
      secList.forEach(([secName, secData]) => {{
        // Hybrid width: 65% market cap + 35% stock count — keeps small sectors readable
        const capW   = (secData.total / totalCap)   * W;
        const cntW   = (secData.count / totalCount)  * W;
        const sw = Math.max(2, capW * 0.65 + cntW * 0.35);
        const stocks = [...secData.stocks].sort((a,b) => (b.market_cap_usd||0)-(a.market_cap_usd||0));
        const n = stocks.length;
        if (!n) {{ sx += sw; return; }}

        const cols = Math.max(1, Math.round(Math.sqrt(n * sw / H)));
        const rows = Math.ceil(n / cols);
        const cw = sw / cols;
        const ch = H / rows;
        const labelH = Math.min(16, ch - 2);

        stocks.forEach((s, i) => {{
          const col = i % cols;
          const row = Math.floor(i / cols);
          const cx = sx + col * cw;
          const cy = row * ch;
          const fw = cw - 1;
          const fh = ch - 1;
          ctx.fillStyle = retColor(s.daily_chg_pct || 0, s.alpha_score || 50);
          ctx.fillRect(cx, cy, fw, fh);
          if (fw > 18 && fh > 12) {{
            const fs = Math.min(12, fw / 3.8, fh / 2.5);
            ctx.fillStyle = 'rgba(255,255,255,0.92)';
            ctx.font = `700 ${{fs.toFixed(1)}}px Inter,Arial,sans-serif`;
            ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
            const midY = fh > 30 ? cy + fh * 0.36 : cy + fh * 0.5;
            ctx.fillText(s.ticker, cx + fw/2, midY);
            if (fh > 28) {{
              const ret = s.daily_chg_pct || 0;
              ctx.font = `${{(fs*0.78).toFixed(1)}}px Inter,Arial,sans-serif`;
              ctx.fillStyle = 'rgba(255,255,255,0.65)';
              ctx.fillText(`${{ret>=0?'+':''}}${{ret.toFixed(2)}}%`, cx+fw/2, cy+fh*0.64);
            }}
          }}
          cellMap.push({{x:cx,y:cy,w:fw,h:fh,d:s}});
        }});

        // Sector label strip
        if (sw > 32) {{
          ctx.fillStyle = 'rgba(0,0,0,.62)';
          ctx.fillRect(sx, 0, sw-1, labelH+3);
          ctx.fillStyle = '#241f18';
          const lfs = Math.min(9.5, sw / (secName.length * 0.65));
          ctx.font = `bold ${{lfs.toFixed(1)}}px Inter,Arial,sans-serif`;
          ctx.textAlign = 'left'; ctx.textBaseline = 'middle';
          ctx.fillText(secName.slice(0, Math.floor(sw/7)+3), sx+4, labelH/2+2);
        }}
        // Sector border
        ctx.strokeStyle = '#13161D'; ctx.lineWidth = 2;
        ctx.strokeRect(sx+1, 1, sw-2, H-2);
        sx += sw;
      }});
      ctx.textAlign = 'left'; ctx.textBaseline = 'alphabetic';
    }}

    canvas.addEventListener('mousemove', function(e) {{
      const rect = canvas.getBoundingClientRect();
      const mx = (e.clientX - rect.left) * (canvas.offsetWidth  / rect.width);
      const my = (e.clientY - rect.top)  * (canvas.offsetHeight / rect.height);
      let hit = null;
      for (let i = cellMap.length-1; i >= 0; i--) {{
        const c = cellMap[i];
        if (mx >= c.x && mx <= c.x+c.w && my >= c.y && my <= c.y+c.h) {{ hit = c; break; }}
      }}
      if (hit) {{
        const d = hit.d;
        const ret = d.daily_chg_pct || 0;
        ttTicker.textContent = d.ticker;
        ttReturn.textContent = (ret>=0?'+':'') + ret.toFixed(2) + '%';
        ttReturn.style.color = ret > 0.1 ? '#6BCCA0' : (ret < -0.1 ? '#F87171' : '#AAA');
        ttScore.textContent  = `Canyon score: ${{(d.alpha_score||0).toFixed(1)}} / 100`;
        ttSector.textContent = d.sector;
        ttSignal.textContent = d.signal ? `Signal: ${{d.signal}}` : '';
        let tx = e.clientX - rect.left + 16;
        let ty = e.clientY - rect.top  + 16;
        const tW = tooltip.offsetWidth || 180;
        const tH = tooltip.offsetHeight || 96;
        if (tx + tW > rect.width)  tx = (e.clientX - rect.left) - tW - 8;
        if (ty + tH > rect.height) ty = (e.clientY - rect.top)  - tH - 8;
        tooltip.style.left    = tx + 'px';
        tooltip.style.top     = ty + 'px';
        tooltip.style.display = 'block';
        canvas.style.cursor   = 'crosshair';
      }} else {{
        tooltip.style.display = 'none';
        canvas.style.cursor   = 'default';
      }}
    }});
    canvas.addEventListener('mouseleave', () => {{ tooltip.style.display = 'none'; }});

    render();
    let resizeTimer;
    window.addEventListener('resize', () => {{ clearTimeout(resizeTimer); resizeTimer = setTimeout(render, 100); }});
    document.addEventListener('showTab', e => {{ if (e.detail === 'heatmap') setTimeout(render, 30); }});
  }})();
  </script>
</section>"""


def _build_famous_holdings_tab(fh: dict, ct: dict = None) -> str:
    """Mind-map style famous investor 13F holdings analysis tab."""
    if not fh or not fh.get("funds"):
        return """
<div style="padding:60px 24px;text-align:center;color:#888">
  <p style="font-size:48px;margin-bottom:12px">🧠</p>
  <p style="font-size:16px;font-weight:400;color:#c8b487;margin-bottom:8px">Famous Investor Holdings</p>
  <p style="font-size:13px">Run <code>python step_famous_holdings.py</code> to fetch 13F filings from SEC EDGAR.</p>
  <p style="font-size:11px;color:#aaa;margin-top:8px">Berkshire · Pershing Square · Scion · Appaloosa · Duquesne · Coatue · Viking · Tiger Global · Greenlight · Third Point</p>
</div>"""

    funds      = fh.get("funds", {})
    consensus  = fh.get("consensus", [])
    overlap    = fh.get("canyon_overlap", [])
    as_of      = fh.get("as_of", "—")

    # ── Style colours for each investment style
    STYLE_COLORS = {
        "Value/Concentrated":   "#3a3128",
        "Activist/Concentrated":"#3a3128",
        "Contrarian/Value":     "#5a5470",
        "Macro/Event-driven":   "#8B3A3A",
        "Macro/Momentum":       "#8B6914",
        "Tech/Growth":          "#1A6B3C",
        "Long/Short Equity":    "#4c5f65",
        "Tech/Growth/VC":       "#1D6B45",
        "Value/Short":          "#6B3E1D",
        "Activist/Event":       "#55606c",
    }

    # ── Change flag colours
    def _flag_style(flag):
        return {
            "NEW":  "background:#241f18;color:#1B7A3B;font-weight:400",
            "ADD":  "background:#202832;color:#5f7480;font-weight:400",
            "TRIM": "background:#FFF3E0;color:#E65100;font-weight:400",
            "HOLD": "background:#F5F5F5;color:#666",
        }.get(flag, "background:#F5F5F5;color:#666")

    # ── Build fund cards (mind-map nodes)
    fund_cards = ""
    for fund_name, fd in funds.items():
        color = STYLE_COLORS.get(fd.get("style", ""), "#3a3128")
        tops  = fd.get("top_holdings", [])[:8]
        new_b = fd.get("new_buys", [])[:3]
        trims = fd.get("trims", [])[:3]

        top_rows = ""
        for i, h in enumerate(tops):
            rank_badge = f'<span style="width:18px;height:18px;border-radius:50%;background:{color};color:#fff;font-size:9px;font-weight:400;display:inline-flex;align-items:center;justify-content:center;margin-right:6px">{i+1}</span>'
            pct_bar    = f'<div style="height:3px;background:{color};width:{min(h["pct_portfolio"]*3.5,100):.0f}%;border-radius:2px;margin-top:3px;opacity:.6"></div>'
            flag_s     = _flag_style(h.get("change_flag", "HOLD"))
            chg_txt    = ""
            if h.get("change_pct") is not None:
                chg_txt = f'<span style="font-size:9px;color:{"#1B7A3B" if h["change_pct"]>0 else "#C0392B"}">{h["change_pct"]:+.0f}%</span> '

            top_rows += f"""<tr>
  <td style="padding:5px 6px;border-bottom:1px solid #241f18">{rank_badge}<strong style="font-size:12px">{h['ticker']}</strong></td>
  <td style="padding:5px 6px;border-bottom:1px solid #241f18;font-size:11px;color:#666;max-width:120px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">{h['name'][:22]}</td>
  <td style="padding:5px 6px;border-bottom:1px solid #241f18;text-align:right;font-size:11px;font-variant-numeric:tabular-nums">${h['value_m']:,.0f}M</td>
  <td style="padding:5px 6px;border-bottom:1px solid #241f18;text-align:right;font-size:11px;font-variant-numeric:tabular-nums">{h['pct_portfolio']:.1f}%</td>
  <td style="padding:5px 6px;border-bottom:1px solid #241f18"><span style="font-size:9px;padding:2px 5px;border-radius:3px;{flag_s}">{h.get('change_flag','?')}</span></td>
</tr>"""
            top_rows += f'<tr><td colspan="5" style="padding:0 6px 4px"><div style="height:3px;background:{color};width:{min(h["pct_portfolio"]*4,100):.0f}%;border-radius:2px;opacity:.35"></div></td></tr>'

        # New buys pill row
        new_pills = " ".join(
            f'<span style="display:inline-block;padding:3px 8px;border-radius:20px;background:#241f18;color:#1B7A3B;font-size:10px;font-weight:400;margin:2px">{h["ticker"]}</span>'
            for h in new_b
        ) if new_b else '<span style="font-size:11px;color:#aaa">—</span>'

        trim_pills = " ".join(
            f'<span style="display:inline-block;padding:3px 8px;border-radius:20px;background:#FFF3E0;color:#E65100;font-size:10px;font-weight:400;margin:2px">{h["ticker"]}</span>'
            for h in trims
        ) if trims else '<span style="font-size:11px;color:#aaa">—</span>'

        # Sector bars
        sector_bars = ""
        for sec, pct in sorted(fd.get("sector_alloc", {}).items(), key=lambda x: -x[1])[:6]:
            sector_bars += f"""
<div style="display:flex;align-items:center;gap:8px;margin-bottom:4px">
  <div style="width:80px;font-size:9px;color:#666;text-align:right;flex-shrink:0">{sec[:12]}</div>
  <div style="flex:1;background:#241f18;border-radius:2px;height:8px">
    <div style="height:8px;border-radius:2px;background:{color};width:{min(pct*2.5,100):.0f}%;opacity:.75"></div>
  </div>
  <div style="width:32px;font-size:9px;font-variant-numeric:tabular-nums;color:#888">{pct:.0f}%</div>
</div>"""

        aum_b = fd.get("total_aum_m", 0) / 1000
        # Concentration: top-5 as % of portfolio
        top5_pct = sum(h["pct_portfolio"] for h in tops[:5]) if tops else 0
        conc_col = "#C0392B" if top5_pct > 70 else ("#c8b487" if top5_pct > 50 else "#1B7A3B")
        n_pos = fd.get("n_positions", 0)
        # Estimated turnover: new_buys + trims as % of portfolio
        n_changes = len(fd.get("new_buys",[])) + len(fd.get("trims",[]))
        turnover_est = f"~{n_changes/max(n_pos,1)*100:.0f}% quarterly" if n_pos else "—"

        fund_cards += f"""
<div class="fh-fund-card" style="border:1px solid #241f18;border-radius:10px;overflow:hidden;background:#fff;
  box-shadow:0 2px 8px rgba(0,0,0,.06);display:flex;flex-direction:column">

  <!-- Fund header -->
  <div style="background:{color};color:#fff;padding:16px 18px 12px">
    <div style="display:flex;justify-content:space-between;align-items:flex-start">
      <div>
        <div style="font-size:13px;font-weight:500;letter-spacing:.5px">{fund_name}</div>
        <div style="font-size:11px;opacity:.75;margin-top:2px">{fd.get('manager','')}</div>
      </div>
      <div style="text-align:right">
        <div style="font-size:18px;font-weight:500;font-variant-numeric:tabular-nums">${aum_b:.1f}B</div>
        <div style="font-size:9px;opacity:.65">{n_pos} positions</div>
      </div>
    </div>
    <div style="margin-top:8px;display:flex;flex-wrap:wrap;gap:4px">
      <span style="font-size:9px;padding:3px 8px;border-radius:20px;background:rgba(255,255,255,.18)">{fd.get('style','')}</span>
      <span style="font-size:9px;padding:3px 8px;border-radius:20px;background:rgba(255,255,255,.12)">13F: {fd.get('filing_date','—')}</span>
      <span style="font-size:9px;padding:3px 8px;border-radius:20px;background:rgba(255,255,255,.12)">Top-5: <strong>{top5_pct:.0f}%</strong> <span style="opacity:.7">conc.</span></span>
    </div>
  </div>
  <!-- Concentration bar -->
  <div style="height:4px;background:rgba(0,0,0,.1)">
    <div style="height:4px;background:{'rgba(255,255,255,.6)'};width:{min(top5_pct,100):.0f}%"></div>
  </div>

  <!-- Holdings table -->
  <div style="flex:1;padding:12px 0;overflow:hidden">
    <div style="padding:0 14px 6px;font-size:9px;font-weight:400;letter-spacing:1px;color:#888;text-transform:uppercase">Top Holdings</div>
    <table style="width:100%;border-collapse:collapse">
      <thead>
        <tr style="background:#FAFAFA">
          <th style="padding:4px 6px;font-size:9px;color:#888;text-align:left;font-weight:400">Ticker</th>
          <th style="padding:4px 6px;font-size:9px;color:#888;text-align:left;font-weight:400">Name</th>
          <th style="padding:4px 6px;font-size:9px;color:#888;text-align:right;font-weight:400">Value</th>
          <th style="padding:4px 6px;font-size:9px;color:#888;text-align:right;font-weight:400">% Port</th>
          <th style="padding:4px 6px;font-size:9px;color:#888;font-weight:400">Chg</th>
        </tr>
      </thead>
      <tbody>{top_rows}</tbody>
    </table>
  </div>

  <!-- Recent moves -->
  <div style="padding:10px 14px;border-top:1px solid #241f18;background:#FAFAFA">
    <div style="font-size:9px;font-weight:400;letter-spacing:1px;color:#888;text-transform:uppercase;margin-bottom:4px">New Buys</div>
    <div>{new_pills}</div>
    <div style="font-size:9px;font-weight:400;letter-spacing:1px;color:#888;text-transform:uppercase;margin:8px 0 4px">Trimmed / Sold</div>
    <div>{trim_pills}</div>
  </div>

  <!-- Sector breakdown -->
  {f'<div style="padding:10px 14px 12px;border-top:1px solid #241f18"><div style="font-size:9px;font-weight:400;letter-spacing:1px;color:#888;text-transform:uppercase;margin-bottom:6px">Sector Exposure</div>{sector_bars}</div>' if sector_bars else ''}
</div>"""

    # ── Consensus heatmap: stocks owned by 2+ funds — enriched with valuation
    consensus_rows = ""
    for c in consensus[:15]:
        n     = c["n_funds"]
        heat  = ["", "#F5F5F5", "#202832", "#243039", "#9aaab0", "#42A5F5", "#2196F3", "#1976D2", "#5f7480", "#0D47A1"]
        bg    = heat[min(n, len(heat)-1)]
        canyon= "★" if c.get("canyon_owns") else ""
        cx    = "color:#c8b487;font-weight:400" if canyon else ""

        # Pull valuation from profiles for richer display
        prof = _STOCK_PROFILES.get(c["ticker"], {})
        pe   = prof.get("pe_fwd", "—")
        chain_layer = prof.get("chain_layer", "")
        chain_pill_col = {"upstream":"#5f7480","midstream":"#1B7A3B","downstream":"#8B3A3A"}.get(chain_layer,"#888")
        chain_pill_lbl = {"upstream":"上游","midstream":"中游","downstream":"下游"}.get(chain_layer,"")

        # Conviction: total value as % of tracked AUM
        total_aum_loc = sum(fd.get("total_aum_m", 0) for fd in funds.values())
        conviction_pct = c["total_value_m"] / total_aum_loc * 100 if total_aum_loc else 0

        fund_names = " ".join(
            f'<span style="font-size:8px;padding:1px 5px;border-radius:3px;background:#EEE;color:#555;margin:1px">{f[:10]}</span>'
            for f in c.get("funds", [])
        )
        consensus_rows += f"""
<tr style="border-bottom:1px solid #241f18">
  <td style="padding:6px 8px;white-space:nowrap">
    <div style="font-weight:500;font-size:13px;color:#c8b487">{c['ticker']}</div>
    {f'<div style="font-size:8px;{cx}">{canyon} Canyon</div>' if canyon else ''}
    {f'<span style="font-size:7px;padding:1px 4px;border-radius:2px;background:{chain_pill_col};color:#fff">{chain_pill_lbl}</span>' if chain_pill_lbl else ''}
  </td>
  <td style="padding:6px 8px">
    <div style="display:flex;align-items:center;gap:5px;margin-bottom:3px">
      <div style="display:inline-flex;align-items:center;justify-content:center;width:22px;height:22px;
        border-radius:50%;background:{'#5f7480' if n>=4 else '#9aaab0' if n>=3 else '#243039'};
        color:#fff;font-size:10px;font-weight:500;flex-shrink:0">{n}</div>
      <div style="font-size:8px;line-height:1.8">{fund_names}</div>
    </div>
    <div style="background:#EEE;border-radius:2px;height:4px;width:100%">
      <div style="height:4px;border-radius:2px;background:#5f7480;width:{min(conviction_pct*8,100):.0f}%"></div>
    </div>
    <div style="font-size:7px;color:#999;margin-top:1px">{conviction_pct:.2f}% of tracked AUM</div>
  </td>
  <td style="padding:6px 8px;text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap">
    <div style="font-size:11px;font-weight:400;color:#c8b487">${c['total_value_m']:,.0f}M</div>
    <div style="font-size:9px;color:#888">fwd P/E: <strong>{pe}×</strong></div>
  </td>
</tr>"""

    # ── Canyon overlap: where Canyon agrees with smart money
    if overlap:
        overlap_html = ""
        for c in overlap[:8]:
            overlap_html += f"""
<div style="display:flex;align-items:center;justify-content:space-between;padding:8px 12px;
  border-bottom:1px solid #241f18">
  <div style="font-weight:400;font-size:13px;color:#c8b487">{c['ticker']}</div>
  <div style="font-size:11px;color:#666">{c['n_funds']} top funds own this</div>
  <div style="font-size:11px;font-variant-numeric:tabular-nums;color:#888">${c['total_value_m']:,.0f}M</div>
  <span style="padding:3px 8px;border-radius:20px;background:#241f18;color:#1B7A3B;font-size:10px;font-weight:400">★ OVERLAP</span>
</div>"""
    else:
        overlap_html = '<p style="font-size:12px;color:#888;padding:16px">No Canyon/smart-money overlap found in current data.</p>'

    # ── Summary stats
    total_aum = sum(fd.get("total_aum_m", 0) for fd in funds.values()) / 1000
    n_consensus = len([c for c in consensus if c["n_funds"] >= 3])
    n_overlap   = len(overlap)

    # ── Deep mind-map cards for each consensus stock
    mindmap_cards_html = ""
    for c in sorted(consensus, key=lambda x: -x["n_funds"])[:12]:
        mindmap_cards_html += _build_stock_mindmap_card(
            ticker       = c["ticker"],
            n_funds      = c["n_funds"],
            funds        = c.get("funds", []),
            total_value_m= c.get("total_value_m", 0),
            canyon_owns  = c.get("canyon_owns", False),
        )

    planet_map_html = _build_sector_planet_map()

    return f"""
<div style="padding:0 0 32px">

  {planet_map_html}

  <!-- Summary bar -->
  <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:16px;margin-bottom:28px">
    <div style="background:#2a2418;color:#fff;padding:16px 18px;border-radius:8px;text-align:center">
      <div style="font-size:24px;font-weight:500;font-variant-numeric:tabular-nums">{len(funds)}</div>
      <div style="font-size:10px;opacity:.7;letter-spacing:.8px;text-transform:uppercase;margin-top:2px">Top Funds Tracked</div>
    </div>
    <div style="background:#fff;border:1px solid #241f18;padding:16px 18px;border-radius:8px;text-align:center">
      <div style="font-size:24px;font-weight:500;color:#c8b487;font-variant-numeric:tabular-nums">${total_aum:.0f}B</div>
      <div style="font-size:10px;color:#888;letter-spacing:.8px;text-transform:uppercase;margin-top:2px">Total AUM Tracked</div>
    </div>
    <div style="background:#fff;border:1px solid #241f18;padding:16px 18px;border-radius:8px;text-align:center">
      <div style="font-size:24px;font-weight:500;color:#5f7480;font-variant-numeric:tabular-nums">{n_consensus}</div>
      <div style="font-size:10px;color:#888;letter-spacing:.8px;text-transform:uppercase;margin-top:2px">3+ Fund Consensus</div>
    </div>
    <div style="background:#241f18;border:1px solid #26332a;padding:16px 18px;border-radius:8px;text-align:center">
      <div style="font-size:24px;font-weight:500;color:#1B7A3B;font-variant-numeric:tabular-nums">{n_overlap}</div>
      <div style="font-size:10px;color:#1B7A3B;letter-spacing:.8px;text-transform:uppercase;margin-top:2px">Canyon ★ Overlap</div>
    </div>
  </div>

  <!-- Two-column layout: fund grid + right panel -->
  <div style="display:grid;grid-template-columns:1fr 380px;gap:24px;align-items:start">

    <!-- Fund mind-map grid -->
    <div>
      <div style="font-size:11px;font-weight:400;letter-spacing:1.2px;text-transform:uppercase;
        color:#888;margin-bottom:12px;padding-bottom:6px;border-bottom:2px solid #3a3128">
        All Funds — 13F Holdings (as of {as_of})
      </div>
      <div class="fh-grid" style="display:grid;grid-template-columns:repeat(auto-fill,minmax(340px,1fr));gap:18px">
        {fund_cards}
      </div>
    </div>

    <!-- Right panel: consensus + Canyon overlap -->
    <div style="position:sticky;top:80px">

      <!-- Smart Money Consensus -->
      <div style="border:1px solid #241f18;border-radius:8px;overflow:hidden;margin-bottom:18px">
        <div style="background:#5f7480;color:#fff;padding:12px 16px;display:flex;align-items:center;gap:10px">
          <span style="font-size:16px">🤝</span>
          <div>
            <div style="font-size:12px;font-weight:400">Smart Money Consensus</div>
            <div style="font-size:10px;opacity:.75">Stocks owned by 2+ top funds</div>
          </div>
        </div>
        <div style="overflow-x:auto">
          <table style="width:100%;border-collapse:collapse">
            <thead>
              <tr style="background:#FAFAFA">
                <th style="padding:6px 10px;font-size:9px;color:#888;text-align:left">Ticker / Chain</th>
                <th style="padding:6px 10px;font-size:9px;color:#888;text-align:left">Fund Ownership · Conviction %AUM</th>
                <th style="padding:6px 10px;font-size:9px;color:#888;text-align:right">Smart $ / P/E</th>
              </tr>
            </thead>
            <tbody>
              {consensus_rows if consensus_rows else '<tr><td colspan="3" style="padding:16px;text-align:center;color:#888;font-size:12px">Run step_famous_holdings.py to populate</td></tr>'}
            </tbody>
          </table>
        </div>
      </div>

      <!-- Canyon Overlap -->
      <div style="border:1px solid #26332a;border-radius:8px;overflow:hidden">
        <div style="background:#1B7A3B;color:#fff;padding:12px 16px;display:flex;align-items:center;gap:10px">
          <span style="font-size:16px">★</span>
          <div>
            <div style="font-size:12px;font-weight:400">Canyon × Smart Money Overlap</div>
            <div style="font-size:10px;opacity:.75">Where Canyon agrees with top funds</div>
          </div>
        </div>
        <div>{overlap_html}</div>
        <div style="padding:10px 12px;background:#F9FFF9;border-top:1px solid #241f18">
          <p style="font-size:10px;color:#666;margin:0">Overlap = stock in Canyon top-30 alpha + owned by 2+ famous funds.
          High overlap = institutional confirmation signal.</p>
        </div>
      </div>

    </div><!-- /right panel -->
  </div><!-- /two-col -->

  <!-- ════════════════════════════ DEEP MIND-MAP ANALYSIS ════════════════════ -->
  <div style="margin-top:36px">
    <div style="display:flex;align-items:center;gap:12px;margin-bottom:6px">
      <div style="font-size:11px;font-weight:400;letter-spacing:1.2px;text-transform:uppercase;color:#888">
        深度思维导图 — Deep Stock Analysis
      </div>
      <div style="flex:1;height:1px;background:linear-gradient(90deg,#3a3128,transparent)"></div>
      <div style="font-size:10px;color:#888">AI 上游 ▶ 中游 ▶ 下游 · 护城河 · 专利 · 估值</div>
    </div>
    <p style="font-size:11px;color:#888;margin-bottom:20px">Each consensus stock broken down: industry position in AI value chain, economic moat, patents, valuation range, and Canyon's view.</p>
    {mindmap_cards_html}
  </div>

  <!-- ════════════════════ CONGRESSIONAL TRADING ════════════════════════════ -->
  <div style="margin-top:36px">
    <div style="display:flex;align-items:center;gap:12px;margin-bottom:16px">
      <div style="font-size:11px;font-weight:400;letter-spacing:1.2px;text-transform:uppercase;color:#888">
        Congressional Trading — STOCK Act Disclosures
      </div>
      <div style="flex:1;height:1px;background:linear-gradient(90deg,#3a3128,transparent)"></div>
      <div style="font-size:10px;color:#888">Pelosi · Tuberville · Khanna · Warner &amp; others</div>
    </div>
    {_build_congressional_section(ct)}
  </div>

</div>"""


# ── Deep stock mind-map profiles ──────────────────────────────────────────────

_STOCK_PROFILES = {
    "NVDA": {
        "name": "NVIDIA Corporation",
        "sector": "Semiconductors",
        "chain_layer": "upstream",
        "chain_label": "AI 上游 — GPU / Accelerator",
        "chain_color": "#5f7480",
        "moat_type": "Switching Cost + Network Effect",
        "moat_width": "Wide",
        "moat_score": 9.2,
        "moat_drivers": ["CUDA ecosystem (20M+ devs)", "Software stack lock-in (cuDNN/cuBLAS)", "Hopper/Blackwell architecture lead", "AI model training default"],
        "moat_dimensions": {"Brand": 8.5, "Switching Cost": 9.8, "Network Effect": 9.0, "Cost Advantage": 7.0, "Efficient Scale": 8.0},
        "patent_count": "5,000+",
        "patent_areas": ["GPU parallel computing (CUDA core architecture)", "Tensor core AI acceleration (H100/B200)", "NVLink high-bandwidth GPU interconnect", "DLSS neural super-sampling / ray tracing", "CoWoS chiplet packaging (with TSMC)"],
        "revenue_segments": [("Data Center (AI GPU)", 87), ("Gaming GPU", 8), ("Professional Viz", 3), ("Automotive", 2)],
        "value_chain": {
            "upstream":   ["TSMC (3nm/4nm → sole fab)", "ASML (EUV lithography → TSM)", "SK Hynix/Samsung (HBM3e memory)"],
            "midstream":  ["NVDA (GPU design + CUDA + NIM)", "AMD (MI300X — 20% share)", "Intel Gaudi (nascent, <2%)"],
            "downstream": ["MSFT Azure (largest customer)", "AMZN AWS (Trainium parallel)", "META AI Research", "Tesla FSD", "Google TPU (internal alt)"],
        },
        "competitors": {"AMD": "MI300X GPU, 20% data center share, growing", "INTC": "Gaudi3, <2%, credibility gap", "GOOGL": "TPU v5 internal, not sold externally", "AMZN": "Trainium2 internal, replaces some NVDA"},
        "risks": [
            {"name":"US Export Controls","prob":"High","impact":"High","note":"BIS restrictions on H20/B200 to China cut ~$15B revenue exposure; escalation risk ongoing"},
            {"name":"Hyperscaler In-House Chips","prob":"Med","impact":"High","note":"AWS Trainium, Google TPU, Meta MTIA could displace 20-30% of NVDA's hyperscaler demand by 2027"},
            {"name":"AMD GPU Competition","prob":"Med","impact":"Med","note":"MI300X gaining in inference workloads; ROCm ecosystem improving; AMD targeting 25% share by 2026"},
            {"name":"TSMC Concentration","prob":"Low","impact":"Critical","note":"100% of advanced GPU fabricated at TSMC — Taiwan Strait risk or yield failure is catastrophic"},
        ],
        "pe_fwd": 38, "ps": 24, "ev_ebitda": 32,
        "peer_pe": {"AMD": 30, "INTC": 18, "AVGO": 26, "TSM": 22},
        "dcf_low": 110, "dcf_high": 160,
        "canyon_view": "Not in top-30 (expensive vs. quality composite); watch for re-entry on pullback to 30× fwd P/E",
        "risk_factors": ["Supply chain (TSMC concentration)", "US export controls to China", "AMD gaining share"],
        "bull_case": "AI capex cycle sustains 40%+ revenue growth through 2028; Blackwell GB200 = $200B+ TAM; NIM software layer creates recurring revenue",
        "bear_case": "Hyperscalers develop in-house chips to replace 30%+ NVDA exposure; US export escalation cuts another $20B+ revenue; GPU price competition",
    },
    "GOOGL": {
        "name": "Alphabet Inc.",
        "sector": "Internet / Mega-cap",
        "chain_layer": "midstream",
        "chain_label": "AI 中游 — Cloud / Foundation Model",
        "chain_color": "#1B7A3B",
        "moat_type": "Network Effect + Data Moat",
        "moat_width": "Wide",
        "moat_score": 9.5,
        "moat_drivers": ["Search monopoly (90% global share)", "YouTube #2 site globally", "Android 3B+ devices", "GCP + TPU infrastructure", "DeepMind (AlphaFold, Gemini)"],
        "moat_dimensions": {"Brand": 9.5, "Switching Cost": 7.5, "Network Effect": 9.8, "Cost Advantage": 8.0, "Efficient Scale": 9.0},
        "patent_count": "50,000+",
        "patent_areas": ["Search ranking algorithms (PageRank + neural)", "TPU AI accelerators (v4/v5 architecture)", "Quantum computing (Willow 105-qubit)", "Gemini multimodal AI architecture", "Waymo autonomous driving sensors/perception"],
        "revenue_segments": [("Search & Ads", 57), ("YouTube Ads", 10), ("Google Cloud (GCP)", 12), ("Other (Subscriptions/Play)", 13), ("Other Bets (Waymo)", 1), ("Network/Ad Mgr", 7)],
        "value_chain": {
            "upstream":   ["TSMC/Samsung (TPU chip fabrication)", "ASML EUV (via TSM)", "Fiber/subsea cable network"],
            "midstream":  ["GOOGL (GCP+Gemini+TPU — #3 cloud 11% share)", "MSFT (Azure+OpenAI — #2 cloud 24%)", "AMZN (AWS+Bedrock — #1 cloud 32%)"],
            "downstream": ["Google Workspace (3B users)", "YouTube Premium 100M+ subscribers", "Google Ads ($240B/yr)", "Waymo robotaxi (SF+Phoenix)", "Android OEM ecosystem"],
        },
        "competitors": {"MSFT": "Bing/Copilot + Azure OpenAI — search losing ground", "META": "Llama open-source AI — model quality competition", "AMZN": "AWS Bedrock — cloud competition", "OPENAI": "ChatGPT Search direct threat"},
        "risks": [
            {"name":"DOJ Antitrust","prob":"High","impact":"High","note":"Search distribution monopoly ruling 2024 — potential search remedy or Chrome/Android divestiture by 2026"},
            {"name":"AI Search Disruption","prob":"High","impact":"High","note":"ChatGPT/Perplexity eating query volume — Google queries grew only 1.5% YoY 2024 vs prior 8%+"},
            {"name":"Gemini Quality Gap","prob":"Med","impact":"Med","note":"Gemini 2.0 Flash competitive but Ultra still behind GPT-4o in coding/reasoning benchmarks"},
            {"name":"YouTube Creators Revolt","prob":"Low","impact":"Med","note":"Creator migration to TikTok, Instagram Reels erodes YouTube engagement in key 18-34 demo"},
        ],
        "pe_fwd": 22, "ps": 7, "ev_ebitda": 14,
        "peer_pe": {"MSFT": 33, "META": 25, "AMZN": 38, "SNAP": 80},
        "dcf_low": 170, "dcf_high": 230,
        "canyon_view": "Not in current top-30; high IC on 12m momentum signal; attractive at <20× P/E on search recovery",
        "risk_factors": ["DOJ antitrust (search distribution monopoly ruling)", "Gemini AI quality vs GPT-4o", "Search volume disruption by AI chatbots"],
        "bull_case": "Gemini 2.0 Ultra wins enterprise AI; YouTube AI ads premium 20%+ ARPU lift; GCP hits $100B run-rate by 2027; Waymo monetizes at $50B+ valuation",
        "bear_case": "DOJ forces Search default distribution divestiture ($50B/yr at risk); ChatGPT/Perplexity capture 15%+ of search queries; Gemini loses developer mindshare",
    },
    "MSFT": {
        "name": "Microsoft Corporation",
        "sector": "Enterprise Software / Cloud",
        "chain_layer": "midstream",
        "chain_label": "AI 中游 — Enterprise AI Platform",
        "chain_color": "#1B7A3B",
        "moat_type": "Switching Cost + Network Effect",
        "moat_width": "Wide",
        "moat_score": 9.7,
        "moat_drivers": ["Office 365 (345M seats)", "Azure #2 cloud (24% share)", "OpenAI equity stake + exclusivity", "GitHub Copilot 1.8M devs", "Teams 320M users"],
        "moat_dimensions": {"Brand": 9.0, "Switching Cost": 9.9, "Network Effect": 8.5, "Cost Advantage": 7.5, "Efficient Scale": 8.8},
        "patent_count": "60,000+",
        "patent_areas": ["Azure AI cloud infrastructure", "Copilot/LLM integration (Office)", "Mixed reality / HoloLens", "Quantum computing (Azure Quantum)", "GitHub Copilot code generation"],
        "revenue_segments": [("Productivity & Cloud (Office/Teams)", 34), ("Azure Cloud", 40), ("More Personal Computing (Windows/Xbox)", 26)],
        "value_chain": {
            "upstream":   ["NVDA (GPU for Azure AI)", "TSMC (chip for Surface)", "LinkedIn (professional data moat)"],
            "midstream":  ["MSFT (Azure+Copilot+OpenAI — #2 cloud 24%)", "GOOGL (GCP — #3 cloud 11%)", "AMZN (AWS — #1 cloud 32%)"],
            "downstream": ["Office + Copilot M365 ($360/seat/yr)", "GitHub (100M devs, Copilot $19/mo)", "Dynamics 365 CRM/ERP", "Xbox Game Pass 35M+"],
        },
        "competitors": {"GOOGL": "Workspace vs Office (3B vs 345M seats)", "AMZN": "AWS vs Azure (market share battle)", "ORCL": "ERP niche (Fusion Cloud)"},
        "risks": [
            {"name":"OpenAI Dependency","prob":"Med","impact":"High","note":"$13B invested; if OpenAI builds competing products or GPT-5 disappoints, Copilot value proposition weakens"},
            {"name":"Azure Growth Slowdown","prob":"Med","impact":"High","note":"Azure growth slowed from 33% to 23% in FY2024; re-acceleration depends on AI workloads converting"},
            {"name":"Copilot Monetization","prob":"Med","impact":"Med","note":"Copilot adoption at 30% penetration in M365 — ROI proof needed for broader enterprise rollout"},
            {"name":"GOOGL Workspace Attack","prob":"Low","impact":"Med","note":"Google Workspace growing faster in SMB; Gemini integration may make Workspace competitive in AI"},
        ],
        "pe_fwd": 33, "ps": 12, "ev_ebitda": 22,
        "peer_pe": {"GOOGL": 22, "AMZN": 38, "ORCL": 24, "CRM": 45},
        "dcf_low": 400, "dcf_high": 520,
        "canyon_view": "Not in top-30; richly valued but highest quality moat. Re-entry on Azure re-acceleration evidence.",
        "risk_factors": ["OpenAI relationship dependency", "Azure growth deceleration", "GOOGL Workspace competition"],
        "bull_case": "Copilot drives 10-15% ARPU lift across 345M Office seats by 2026; Azure hits $200B run-rate; GitHub Copilot becomes de facto developer IDE",
        "bear_case": "OpenAI partnership deteriorates; Google Workspace AI makes enterprise switch viable; Azure slows to 15% growth vs market expectations of 25%",
    },
    "AMZN": {
        "name": "Amazon.com Inc.",
        "sector": "E-commerce / Cloud",
        "chain_layer": "midstream",
        "chain_label": "AI 中游 — Cloud Infrastructure + Bedrock",
        "chain_color": "#1B7A3B",
        "moat_type": "Cost Advantage + Switching Cost",
        "moat_width": "Wide",
        "moat_score": 9.4,
        "moat_drivers": ["AWS #1 cloud (32% market share)", "Prime 230M members (e-commerce flywheel)", "Fulfillment network 185+ distribution centers", "Trainium2/Inferentia2 in-house AI chips", "Advertising $50B+ data moat"],
        "moat_dimensions": {"Brand": 9.0, "Switching Cost": 9.0, "Network Effect": 8.5, "Cost Advantage": 9.5, "Efficient Scale": 9.2},
        "patent_count": "20,000+",
        "patent_areas": ["AWS cloud infrastructure (EC2/S3/Lambda)", "Alexa NLP / voice AI", "Kiva fulfillment robotics", "Drone delivery (Prime Air)", "Bedrock generative AI orchestration"],
        "revenue_segments": [("AWS Cloud", 17), ("Advertising", 9), ("3P Seller Services", 23), ("Online Retail (1P)", 37), ("Prime Subscriptions", 7), ("Other", 7)],
        "value_chain": {
            "upstream":   ["Trainium2 AI chip (in-house)", "AWS proprietary data centers", "Logistics network (185+ DCs)"],
            "midstream":  ["AMZN AWS Bedrock (#1 cloud 32%)", "MSFT Azure (#2 24%)", "GOOGL GCP (#3 11%)"],
            "downstream": ["Amazon Retail $600B GMV", "Prime Video 200M+ viewers", "Amazon Ads $55B", "Alexa 500M+ devices", "AWS enterprise customers"],
        },
        "competitors": {"MSFT": "Azure — cloud #2", "GOOGL": "GCP — cloud #3 + AI models", "SHOP": "e-commerce platform (SMB)", "TEMU": "price-war retail threat"},
        "risks": [
            {"name":"AWS Growth Deceleration","prob":"Med","impact":"High","note":"AWS grew 17% in 2024 vs 37% in 2022; re-acceleration to 25%+ depends on AI workload ramp"},
            {"name":"Retail Margin Compression","prob":"Med","impact":"Med","note":"Temu/Shein price competition + rising fulfillment costs compressing 1P retail to near-zero margins"},
            {"name":"Antitrust (FTC)","prob":"Med","impact":"Med","note":"FTC marketplace monopoly case ongoing; potential forced divestiture of 3P marketplace"},
            {"name":"Trainium vs NVDA","prob":"Low","impact":"Low","note":"In-house chips reduce costs but may lose AI customers who need CUDA compatibility"},
        ],
        "pe_fwd": 38, "ps": 3.5, "ev_ebitda": 20,
        "peer_pe": {"MSFT": 33, "GOOGL": 22, "BABA": 12, "JD": 8},
        "dcf_low": 200, "dcf_high": 280,
        "canyon_view": "Not in top-30; arguably strongest diversified moat. Would buy at 28× fwd EV/EBITDA.",
        "risk_factors": ["AWS growth re-acceleration uncertainty", "Retail competition (Temu/Shein)", "FTC antitrust"],
        "bull_case": "AWS AI demand drives 25%+ growth; advertising business reaches $100B by 2027; AWS operating margin expands to 45%+",
        "bear_case": "AWS slows to 15% growth; FTC forces marketplace changes; retail losses accelerate; capex overspend",
    },
    "META": {
        "name": "Meta Platforms Inc.",
        "sector": "Social Media / AI",
        "chain_layer": "downstream",
        "chain_label": "AI 下游 — Social AI + LLM (Open Source)",
        "chain_color": "#8B3A3A",
        "moat_type": "Network Effect + Data Moat",
        "moat_width": "Wide",
        "moat_score": 8.8,
        "moat_drivers": ["3.3B DAU across FB/IG/WhatsApp/Threads", "Ad targeting precision (#1 ROAS in digital)", "Llama open-source AI community (community moat)", "Quest VR ecosystem + social XR", "WhatsApp 2B users (messaging lock-in)"],
        "moat_dimensions": {"Brand": 8.0, "Switching Cost": 6.5, "Network Effect": 9.5, "Cost Advantage": 7.5, "Efficient Scale": 8.5},
        "patent_count": "10,000+",
        "patent_areas": ["Social graph algorithms (EdgeRank)", "Ad auction systems (real-time bidding)", "VR/AR hardware (Quest optics + tracking)", "Llama model compression (GGUF/quantization)", "MTIA AI inference chip architecture"],
        "revenue_segments": [("Facebook Ads", 55), ("Instagram Ads", 37), ("WhatsApp Business API", 3), ("Reality Labs (Quest)", 2), ("Other", 3)],
        "value_chain": {
            "upstream":   ["NVDA GPU (AI training)", "Custom MTIA chip (inference, cost advantage)", "Oculus/Quest hardware (in-house)"],
            "midstream":  ["Llama 4 open-source foundation model", "Meta AI assistant (3.3B users)", "Ad ranking AI (Advantage+ automation)"],
            "downstream": ["Facebook Ads ($132B/yr)", "Instagram Shopping + Reels ads", "WhatsApp Business API (growing)", "Quest VR gaming + enterprise"],
        },
        "competitors": {"SNAP": "younger demo, smaller scale", "TIKTOK": "Reels direct competition, algorithmic", "GOOGL YouTube": "video ad competing platform", "OPENAI": "Llama vs GPT for enterprise AI"},
        "risks": [
            {"name":"TikTok Competition","prob":"High","impact":"Med","note":"TikTok Reels-style competition eroding Meta's 18-34 engagement advantage; Instagram growth slowing"},
            {"name":"Reality Labs Losses","prob":"High","impact":"Med","note":"$15B/yr Reality Labs burn rate — Quest VR has not achieved mass market adoption (10M units/yr vs smartphone billions)"},
            {"name":"EU Data Privacy Fines","prob":"Med","impact":"Med","note":"GDPR enforcement ongoing — €1.2B fine in 2023; potential advertising model changes in EU market"},
            {"name":"Teen Engagement Decay","prob":"Med","impact":"High","note":"US teen daily Facebook/Instagram use declining; Snapchat/BeReal/Discord capturing younger cohort"},
        ],
        "pe_fwd": 25, "ps": 8, "ev_ebitda": 16,
        "peer_pe": {"GOOGL": 22, "SNAP": 80, "PINS": 30, "RDDT": 150},
        "dcf_low": 540, "dcf_high": 720,
        "canyon_view": "Not in current top-30; strong earnings momentum. Conviction BUY below 22× P/E on AI-driven ARPU growth.",
        "risk_factors": ["TikTok ban reversal threatens thesis", "EU data privacy regulatory risk", "Reality Labs $15B+ annual losses"],
        "bull_case": "Advantage+ AI ad automation drives 20%+ ARPU growth across 3.3B DAUs; Llama ecosystem makes META the enterprise AI platform; WhatsApp monetization at $5/user = $10B incremental revenue",
        "bear_case": "Teen engagement collapse accelerates; EU forces data-localization restructuring; Reality Labs never monetizes ($80B cumulative burn by 2030)",
    },
    "AAPL": {
        "name": "Apple Inc.",
        "sector": "Consumer Electronics / Ecosystem",
        "chain_layer": "downstream",
        "chain_label": "AI 下游 — Consumer AI + Device Ecosystem",
        "chain_color": "#8B3A3A",
        "moat_type": "Switching Cost + Brand + Ecosystem",
        "moat_width": "Wide",
        "moat_score": 9.8,
        "moat_drivers": ["2.2B active devices (sticky ecosystem)", "App Store $100B+ gross sales (30% cut)", "Apple Silicon M4 — #1 perf/watt ratio", "iMessage/AirDrop social lock-in in US", "Services $100B+/yr at ~75% gross margin"],
        "moat_dimensions": {"Brand": 9.9, "Switching Cost": 9.5, "Network Effect": 8.0, "Cost Advantage": 7.0, "Efficient Scale": 8.5},
        "patent_count": "80,000+",
        "patent_areas": ["Apple Silicon A/M series neural engine", "Face ID secure enclave biometrics", "Haptic Taptic Engine UI", "Apple Watch health sensing (ECG, O2)", "Vision Pro spatial computing (eye/hand tracking)"],
        "revenue_segments": [("iPhone", 52), ("Services (App Store/iCloud/AppleTV)", 25), ("Mac", 8), ("iPad", 7), ("Wearables/Accessories", 8)],
        "value_chain": {
            "upstream":   ["TSMC (A/M chip fabrication — sole supplier)", "Qualcomm (modem 5G → in-house C1 2025)", "Samsung/LG (OLED display panels)"],
            "midstream":  ["Apple Silicon (vertical integration — chip+SW)", "iOS/macOS/watchOS software ecosystem", "iCloud 1B+ subscribers"],
            "downstream": ["iPhone 230M units/yr ($200B revenue)", "Mac ($30B)", "iPad ($28B)", "Services $100B+ (high margin)", "Vision Pro spatial computing"],
        },
        "competitors": {"SAMSUNG": "Android premium — Galaxy S25 direct", "GOOGL Pixel": "AI-first niche, <3% premium market", "MSFT": "PC/Surface — productivity overlap"},
        "risks": [
            {"name":"China Revenue Risk","prob":"High","impact":"High","note":"China $70B (20% of revenue) — ban risk from trade war escalation; Huawei Mate 70 took 3% iPhone share in 2024"},
            {"name":"App Store Antitrust","prob":"High","impact":"Med","note":"DOJ App Store case + EU DMA force sideloading/alternative payment — ~$8-12B annual Services revenue at risk"},
            {"name":"iPhone Upgrade Cycle","prob":"Med","impact":"Med","note":"Average replacement cycle extended to 4.7 years (2024) from 3.2 years (2020); unit growth constrained"},
            {"name":"Apple Intelligence Execution","prob":"Low","impact":"Med","note":"Apple Intelligence Siri AI delayed vs expectations; if fails to drive supercycle, thesis breaks"},
        ],
        "pe_fwd": 29, "ps": 8, "ev_ebitda": 21,
        "peer_pe": {"SAMSUNG": 10, "GOOGL": 22, "MSFT": 33, "META": 25},
        "dcf_low": 200, "dcf_high": 260,
        "canyon_view": "Not in top-30; world-class moat but richly priced at 29× P/E. Watch for China trade risk creating entry point.",
        "risk_factors": ["China revenue ($70B) geopolitical risk", "App Store DOJ/EU antitrust revenue risk", "iPhone upgrade cycle lengthening"],
        "bull_case": "Apple Intelligence drives supercycle: 300M+ iPhone 17 units vs 230M currently; Services hits $150B/yr; Vision Pro finds enterprise use case",
        "bear_case": "China trade escalation bans Apple products (-20% revenue); App Store forced to 15% cut (from 30%) removes $8B/yr; Gen Z sees Apple as legacy brand",
    },
    "TSM": {
        "name": "Taiwan Semiconductor Mfg. Co.",
        "sector": "Semiconductor Foundry",
        "chain_layer": "upstream",
        "chain_label": "AI 上游 — Foundry (唯一 3nm/2nm)",
        "chain_color": "#5f7480",
        "moat_type": "Cost Advantage + Technical Monopoly",
        "moat_width": "Wide",
        "moat_score": 9.9,
        "moat_drivers": ["60%+ foundry market share (>80% of advanced <5nm)", "Sole manufacturer of NVDA H100/B200/GB200 chips", "ASML EUV #1 customer (guaranteed allocation)", "N3/N2 process 2-3 years ahead of Samsung", "Arizona N4/N3, Japan Kumamoto, Germany fabs (geopolitical derisking)"],
        "moat_dimensions": {"Brand": 8.5, "Switching Cost": 9.9, "Network Effect": 5.0, "Cost Advantage": 9.8, "Efficient Scale": 9.9},
        "patent_count": "35,000+",
        "patent_areas": ["N3/N2 GAA nanosheet transistor process", "CoWoS-S/L advanced 3D packaging", "High-bandwidth memory (HBM3e) integration", "Chip-on-Wafer-on-Substrate (cowos for AI)", "Fab automation (yield optimization AI)"],
        "revenue_segments": [("Advanced Process (<5nm)", 42), ("Mature Process (>7nm)", 22), ("Packaging & Testing", 12), ("HPC/AI chips (key customer rev)", 63), ("Smartphone", 25), ("IoT/Auto", 12)],
        "value_chain": {
            "upstream":   ["ASML (EUV scanner — monopoly supplier)", "Air Products / Linde (ultra-pure gases)", "LAM Research / AMAT (etch/deposition)"],
            "midstream":  ["TSM (sole N3/N2 mass-production foundry)", "Samsung (N2 competitor, ~12% yield vs TSM 80%+)", "Intel Foundry (18A, not yet in production)"],
            "downstream": ["NVDA (H100/B200/GB200 — #1 revenue customer)", "AAPL (A18/M4 — #2 customer)", "AMD (MI300X)", "GOOGL TPU v5", "QCOM Snapdragon"],
        },
        "competitors": {"SAMSUNG": "2nm GAA in 2025 but yield <30% vs TSM 80%+; losing NVDA orders", "INTC": "Intel Foundry 18A process — 2-3yr behind, funding at risk"},
        "risks": [
            {"name":"Taiwan Strait Geopolitical","prob":"Low","impact":"Catastrophic","note":"Military conflict would sever 90%+ of advanced chip supply globally; market would crash 30-50%; no near-term substitute for TSM's production"},
            {"name":"AI Capex Cycle End","prob":"Med","impact":"High","note":"If AI capex normalizes from $300B/yr to $150B/yr, NVDA orders drop → TSM advanced node demand halves"},
            {"name":"Samsung/INTC Catch-Up","prob":"Low","impact":"Med","note":"If Samsung achieves >60% yield at 2nm by 2026, can compete for NVDA's next-gen chip orders"},
            {"name":"US Export Control Expansion","prob":"Med","impact":"Med","note":"Further US chip restrictions could reduce TSM's China revenue (currently ~15% of sales)"},
        ],
        "pe_fwd": 22, "ps": 10, "ev_ebitda": 14,
        "peer_pe": {"SAMSUNG": 12, "INTC": 18, "ASML": 32, "AMAT": 19},
        "dcf_low": 190, "dcf_high": 260,
        "canyon_view": "Not in current top-30; highest-quality AI infrastructure position. Geopolitical risk is the only real constraint.",
        "risk_factors": ["Taiwan Strait military risk (catastrophic tail)", "AI capex cycle normalization", "Samsung yield improvement"],
        "bull_case": "AI chip demand at $300B+/yr sustains 30% revenue growth through 2027; CoWoS advanced packaging becomes $20B revenue line; US Arizona fabs derisks geopolitical premium",
        "bear_case": "China military action triggers supply chain disruption; or AI capex cycle ends as LLM improvements plateau; Samsung catches up in 2nm yield",
    },
    "PLTR": {
        "name": "Palantir Technologies Inc.",
        "sector": "Defense AI / Enterprise Analytics",
        "chain_layer": "downstream",
        "chain_label": "AI 下游 — Government + Enterprise AI Platform",
        "chain_color": "#8B3A3A",
        "moat_type": "Switching Cost + Government Contracts",
        "moat_width": "Narrow-Wide",
        "moat_score": 7.8,
        "moat_drivers": ["AIP platform: GOTHAM (government) + FOUNDRY (commercial) + AIP (AI layer)", "15+ yr US DoD/CIA classified contract history — highest security clearance", "Ontology data model creates deep switching cost (avg 4+ yr deployment)", "Forward-deployed engineers (FDE model) — embedded in customer operations", "Commercial expansion 60%+ growth rate 2024 (NHS, BP, Airbus, Ferrari)"],
        "moat_dimensions": {"Brand": 7.0, "Switching Cost": 9.2, "Network Effect": 4.0, "Cost Advantage": 5.0, "Efficient Scale": 6.5},
        "patent_count": "500+",
        "patent_areas": ["Dynamic data fusion / knowledge graph ontology", "Forward edge AI deployment (classified)", "Generative AI agent orchestration (AIP)", "Multi-source intelligence correlation"],
        "revenue_segments": [("US Government (DoD/IC)", 42), ("US Commercial", 32), ("International Government", 16), ("International Commercial", 10)],
        "value_chain": {
            "upstream":   ["NVDA GPU (AIP inference)", "MSFT Azure (cloud host)", "AWS GovCloud (classified hosting)"],
            "midstream":  ["PLTR AIP — AI orchestration for enterprise/gov", "C3.ai (overlapping vertical AI, smaller)", "Scale AI (data labeling, indirect competition)"],
            "downstream": ["US DoD / CIA / NSA / SOCOM (classified)", "NHS UK (healthcare AI)", "BP / Shell (energy ops)", "Airbus / Raytheon (defense)", "Ferrari / Cleveland Clinic (commercial)"],
        },
        "competitors": {"C3.ai": "vertical AI — smaller, losing customers", "SNOW": "data platform — different layer", "MSFT Copilot for Gov": "Azure AI Government — growing threat", "GOOGL Vertex AI": "GCP enterprise AI competition"},
        "risks": [
            {"name":"Extreme Valuation","prob":"High","impact":"High","note":"170× P/E, 48× P/S — priced for perfection; any miss triggers 30-50% drawdown; Michael Burry's 66% concentrated bet is high conviction but high risk"},
            {"name":"DOGE Government Cuts","prob":"Med","impact":"High","note":"DOGE targeting 20%+ DoD discretionary cuts; PLTR government revenue ($650M) directly exposed; classified budgets more protected"},
            {"name":"Commercial Growth Lumpy","prob":"Med","impact":"Med","note":"AIP boot camps convert leads fast but deal sizes vary; quarterly commercial growth 60% but lumpy — 1 missed quarter = large selloff"},
            {"name":"Hyperscaler AI Agents","prob":"Med","impact":"Med","note":"MSFT Copilot for Government + Azure OpenAI could replicate AIP functionality at lower cost for commercial customers by 2026"},
        ],
        "pe_fwd": 170, "ps": 48, "ev_ebitda": 120,
        "peer_pe": {"C3.ai": 200, "SNOW": 70, "MSFT": 33, "GOOGL": 22},
        "dcf_low": 18, "dcf_high": 35,
        "canyon_view": "Burry's #1 position at 66% of Scion portfolio — extreme conviction. Valuation requires 35%+ annual revenue growth for 5+ years to justify. Monitor AIP commercial deal velocity.",
        "risk_factors": ["170× P/E — extreme valuation gravity", "Government spending cuts (DOGE 20% target)", "MSFT/GOOGL hyperscaler AI competition"],
        "bull_case": "AIP wins $10B+ commercial AI deployment contracts; DoD AI spending triples to $30B by 2028; NATO ally governments deploy GOTHAM; revenue grows 40%+ to 2028",
        "bear_case": "Government sequester cuts PLTR classified budget by 30%; hyperscaler AI agents commoditize AIP; valuation gravity crushes stock as P/E normalizes to 50-80×",
    },
    "V": {
        "name": "Visa Inc.",
        "sector": "Payment Processing",
        "chain_layer": "downstream",
        "chain_label": "金融基础设施 (Financial Infrastructure) — Global Payments Network",
        "chain_color": "#4c5f65",
        "moat_type": "Network Effect",
        "moat_width": "Wide",
        "moat_score": 9.6,
        "moat_drivers": ["4.4B Visa cards — 2-sided network effect (merchants need cards, cardholders need merchants)", "130M+ merchant acceptance locations (impractical to replicate)", "VisaNet processes 65,000+ TPS with 99.999% uptime (reliability moat)", "Capital-light model — zero credit risk, 50%+ operating margins", "B2B cross-border payment expansion (Visa Direct, B2B Connect)"],
        "moat_dimensions": {"Brand": 9.5, "Switching Cost": 8.0, "Network Effect": 9.9, "Cost Advantage": 9.0, "Efficient Scale": 9.5},
        "patent_count": "3,000+",
        "patent_areas": ["EMV chip card tokenization", "Tap-to-pay NFC contactless", "Real-time settlement (VisaNet)", "AI fraud detection (Cybersource)", "B2B open-loop payment rails"],
        "revenue_segments": [("Service Revenue (card fees)", 38), ("Data Processing (VisaNet)", 44), ("International Transaction", 23), ("Other Revenue (licensing)", -5)],
        "value_chain": {
            "upstream":   ["Card issuing banks (JPM, BAC, Citi — set interchange)", "Acquiring banks (FIS, Worldpay, Square)"],
            "midstream":  ["VISA network rails (dominant — 50% global volume)", "Mastercard (near-identical, 30% share)", "UnionPay (China — 40% global by card count but domestic)"],
            "downstream": ["Consumers (4.4B cardholders)", "Merchants (130M+ locations)", "B2B corporate payments ($120T TAM)"],
        },
        "competitors": {"MA": "near-identical network — healthy duopoly", "AMEX": "closed-loop premium (but smaller merchant network)", "PYPL": "digital overlay, uses Visa/MA rails", "USDC/Stablecoins": "emerging threat to cross-border"},
        "risks": [
            {"name":"Stablecoin/Crypto Rails","prob":"Med","impact":"High","note":"USDC on Solana processing $10T+/yr; if merchants adopt directly, Visa cross-border revenue at risk — but Visa is partnering with stablecoin issuers"},
            {"name":"Fed Interchange Regulation","prob":"Med","impact":"Med","note":"Durbin Amendment extension to credit cards could cut $8-12B/yr revenue; Congressional support mixed"},
            {"name":"Merchant Surcharging","prob":"Low","impact":"Med","note":"Merchants increasingly passing card fees to consumers; consumer backlash and card-avoidance risk in high-fee environments"},
            {"name":"Central Bank Digital Currencies","prob":"Low","impact":"High","note":"If major economies deploy CBDCs with retail payment capability, Visa rails could be bypassed — but 5-10yr timeline"},
        ],
        "pe_fwd": 28, "ps": 14, "ev_ebitda": 22,
        "peer_pe": {"MA": 30, "AMEX": 18, "PYPL": 15, "FIS": 12},
        "dcf_low": 280, "dcf_high": 380,
        "canyon_view": "Not in current top-30; Viking Global's #1 holding (5.3% of $35.7B). Highest-quality non-tech moat. Buy on any macro selloff.",
        "risk_factors": ["Stablecoin direct merchant adoption", "Fed debit interchange extension to credit", "CBDC long-term displacement"],
        "bull_case": "B2B payments TAM $120T unlocked via Visa Direct + B2B Connect; AI fraud prevention saves merchants $25B+/yr (Cybersource pricing power); India/Africa card penetration doubles",
        "bear_case": "US Congress extends Durbin Amendment to credit cards ($10B revenue hit); Stripe/Stablecoin rails capture B2B cross-border; global recession cuts payment volume",
    },
    "AXP": {
        "name": "American Express Co.",
        "sector": "Payments / Financial Services",
        "chain_layer": "downstream",
        "chain_label": "金融基础设施 (Financial Infrastructure) — Premium Closed-Loop Network",
        "chain_color": "#4c5f65",
        "moat_type": "Network Effect + Brand",
        "moat_width": "Wide",
        "moat_score": 8.5,
        "moat_drivers": ["Closed-loop model (issuer + network)", "Premium cardholder demographics (high FICO)", "17.1% of Berkshire's portfolio", "Centurion brand premium", "Merchant discount rate pricing power"],
        "patent_count": "2,000+",
        "patent_areas": ["Rewards algorithms", "Risk scoring", "Travel services integration"],
        "value_chain": {
            "upstream":   ["Capital markets (bond funding)", "Customer acquisition (high CAC but high LTV)"],
            "midstream":  ["AMEX (issuer + network combined)", "Closed loop = richer data"],
            "downstream": ["38M cards globally", "Business Platinum cardholders ($700+/yr fee)", "Restaurant + Travel rewards"],
        },
        "competitors": {"V": "open loop", "JPM Chase Sapphire": "competing premium", "CITI": "premium cards"},
        "pe_fwd": 19, "ps": 2.1, "ev_ebitda": 14,
        "dcf_low": 250, "dcf_high": 320,
        "canyon_view": "Not in top-30; Buffett's highest-conviction fintech",
        "risk_factors": ["Recession → premium consumer pullback", "Credit losses if unemployment rises", "Millennial brand relevance"],
        "bull_case": "Premium spend share gains from JPM; Gen Z discovers Amex Platinum; delinquencies stay low",
        "bear_case": "Recession causes credit losses; JPM wins affluent spenders",
    },
    "KHC": {
        "name": "Kraft Heinz Co.",
        "sector": "Consumer Staples / Food",
        "chain_layer": "downstream",
        "chain_label": "消费品 — 品牌食品巨头 (Buffett持仓)",
        "chain_color": "#6B3E1D",
        "moat_type": "Brand",
        "moat_width": "Narrow",
        "moat_score": 5.5,
        "moat_drivers": ["Heinz #1 ketchup brand globally", "Kraft cheese #1 in US", "Oscar Mayer deli meats", "Lunchables for kids"],
        "patent_count": "2,000+",
        "patent_areas": ["Food processing", "Preservatives / packaging", "Flavor formulations"],
        "value_chain": {
            "upstream":   ["Tomato farms (supply chain)", "Grain commodity markets", "Packaging (Sealed Air)"],
            "midstream":  ["KHC processing plants", "Cold chain logistics"],
            "downstream": ["Grocery retail (Walmart, Kroger, Costco)", "Food service (McD, Subway)"],
        },
        "competitors": {"GIS": "General Mills", "CPB": "Campbell's", "UNILEVER": "international"},
        "pe_fwd": 10, "ps": 1.2, "ev_ebitda": 9,
        "dcf_low": 30, "dcf_high": 42,
        "canyon_view": "Not in top-30; Berkshire + Duquesne hold at deep value",
        "risk_factors": ["Brand equity erosion (private label)", "Debt burden ($20B+)", "Health-conscious shift away"],
        "bull_case": "Cost cuts + pricing power restore margins; Heinz global pricing in EM markets",
        "bear_case": "Private label gains 5% category share; debt refinancing risk if rates stay high",
    },
}


def _build_stock_mindmap_card(ticker: str, n_funds: int, funds: list, total_value_m: float, canyon_owns: bool) -> str:
    """Build a deep-analysis mind-map card: value chain + moat scorecard + revenue + risk matrix + valuation."""
    prof = _STOCK_PROFILES.get(ticker)
    if not prof:
        return f"""
<div style="border:1px solid #241f18;border-radius:8px;padding:16px;background:#fff;margin-bottom:16px">
  <div style="font-size:18px;font-weight:500;color:#c8b487">{ticker}</div>
  <p style="font-size:10px;color:#aaa;margin-top:4px">Deep analysis profile not yet available — {n_funds} funds · ${total_value_m:,.0f}M smart money</p>
</div>"""

    layer_col  = {"upstream": "#5f7480", "midstream": "#1B7A3B", "downstream": "#8B3A3A"}
    layer_icon = {"upstream": "⬆ 上游", "midstream": "↔ 中游", "downstream": "⬇ 下游"}
    col  = layer_col.get(prof.get("chain_layer",""), "#3a3128")
    icon = layer_icon.get(prof.get("chain_layer",""), "")

    # ── Fund owner pills ──────────────────────────────────────────────────────
    fund_pills = "".join(
        f'<span style="display:inline-block;padding:2px 8px;border-radius:3px;background:rgba(255,255,255,.18);'
        f'font-size:9px;margin:1px;color:#fff">{f[:18]}</span>'
        for f in funds
    )

    # ── Value chain tree ──────────────────────────────────────────────────────
    vc = prof.get("value_chain", {})
    vc_html = ""
    for lyr_key, lyr_label, lyr_col in [
        ("upstream","⬆ 上游 (Input)","#5f7480"),
        ("midstream","↔ 中游 (Platform)","#1B7A3B"),
        ("downstream","⬇ 下游 (Application)","#8B3A3A"),
    ]:
        is_here = prof.get("chain_layer") == lyr_key
        items = vc.get(lyr_key, [])
        chips = "".join(
            f'<span style="display:inline-block;padding:1px 7px;margin:1px 0;border-radius:3px;font-size:8px;'
            f'background:{"#3a3128" if (ticker in it or it.startswith(ticker)) else "#241f18"};'
            f'color:{"#fff" if (ticker in it or it.startswith(ticker)) else "#555"};font-weight:{"700" if ticker in it else "400"}">'
            f'{it[:28]}</span>'
            for it in items
        )
        vc_html += f"""
<div style="border:{'2px' if is_here else '1px'} solid {'#3a3128' if is_here else '#241f18'};
  border-radius:5px;padding:5px 8px;margin-bottom:5px;background:{'rgba(27,42,74,.04)' if is_here else '#FAFAFA'}">
  <div style="font-size:8px;font-weight:400;color:{lyr_col};margin-bottom:3px;display:flex;align-items:center;gap:4px">
    {lyr_label}
    {f'<span style="background:{col};color:#fff;font-size:7px;padding:0 4px;border-radius:2px">← HERE</span>' if is_here else ''}
  </div>
  <div style="line-height:1.8">{chips}</div>
</div>"""

    # ── Competitor matrix ────────────────────────────────────────────────────
    comp_rows = "".join(
        f'<tr><td style="padding:3px 6px;font-size:9px;font-weight:400;color:#c8b487;white-space:nowrap">{c}</td>'
        f'<td style="padding:3px 6px;font-size:9px;color:#555">{d[:55]}</td></tr>'
        for c, d in (prof.get("competitors") or {}).items()
    )

    # ── Moat overall gauge ───────────────────────────────────────────────────
    mscore = prof.get("moat_score", 5)
    mpct   = mscore / 10 * 100
    mcol   = "#1B7A3B" if mscore >= 8.5 else ("#c8b487" if mscore >= 7 else "#C0392B")

    # ── Moat 5-dimension scorecard ───────────────────────────────────────────
    moat_dims = prof.get("moat_dimensions", {})
    moat_dim_rows = ""
    DIM_LABELS = {
        "Brand": "品牌",
        "Switching Cost": "切换成本",
        "Network Effect": "网络效应",
        "Cost Advantage": "成本优势",
        "Efficient Scale": "规模效应",
    }
    for dim in ["Brand","Switching Cost","Network Effect","Cost Advantage","Efficient Scale"]:
        v = moat_dims.get(dim, None)
        if v is None:
            continue
        v_pct = v / 10 * 100
        v_col = "#1B7A3B" if v >= 8 else ("#c8b487" if v >= 6 else "#C0392B")
        moat_dim_rows += f"""
<tr>
  <td style="padding:2px 6px;font-size:9px;color:#444;white-space:nowrap">{DIM_LABELS.get(dim,dim)}</td>
  <td style="padding:2px 6px;width:80px">
    <div style="background:#EEE;border-radius:2px;height:6px">
      <div style="height:6px;border-radius:2px;background:{v_col};width:{v_pct:.0f}%"></div>
    </div>
  </td>
  <td style="padding:2px 6px;font-size:9px;font-weight:400;color:{v_col};font-variant-numeric:tabular-nums">{v:.1f}</td>
</tr>"""

    # ── Revenue segments ─────────────────────────────────────────────────────
    rev_segs = prof.get("revenue_segments", [])
    rev_bars = ""
    for seg, pct in (rev_segs if isinstance(rev_segs, list) else [])[:6]:
        bar_pct = min(abs(pct), 100)
        bar_col = col
        rev_bars += f"""
<div style="display:flex;align-items:center;gap:6px;margin-bottom:4px">
  <div style="width:130px;font-size:8px;color:#555;text-align:right;flex-shrink:0">{seg[:22]}</div>
  <div style="flex:1;background:#EEE;border-radius:2px;height:8px">
    <div style="height:8px;border-radius:2px;background:{bar_col};width:{bar_pct:.0f}%;opacity:.75"></div>
  </div>
  <div style="width:30px;font-size:9px;font-weight:400;color:#444;font-variant-numeric:tabular-nums">{pct:.0f}%</div>
</div>"""

    # ── Risk matrix ──────────────────────────────────────────────────────────
    PROB_RANK  = {"High":3,"Med":2,"Low":1,"Critical":4}
    IMP_RANK   = {"Critical":4,"High":3,"Med":2,"Low":1}
    PROB_COL   = {"High":"#C0392B","Med":"#c8b487","Low":"#1B7A3B","Critical":"#7B0000"}
    IMP_COL    = {"Critical":"#7B0000","High":"#C0392B","Med":"#c8b487","Low":"#1B7A3B"}
    risks = sorted(prof.get("risks",[]), key=lambda r: -(PROB_RANK.get(r.get("prob"),0)*IMP_RANK.get(r.get("impact"),0)))
    risk_rows = ""
    for r in risks[:5]:
        p_col = PROB_COL.get(r.get("prob","Med"), "#888")
        i_col = IMP_COL.get(r.get("impact","Med"), "#888")
        risk_rows += f"""
<tr style="border-bottom:1px solid #241f18">
  <td style="padding:4px 6px;font-size:9px;font-weight:400;color:#c8b487;max-width:100px">{r['name'][:22]}</td>
  <td style="padding:4px 6px;text-align:center">
    <span style="font-size:8px;font-weight:400;padding:1px 5px;border-radius:3px;background:{p_col}22;color:{p_col}">{r.get('prob','?')}</span>
  </td>
  <td style="padding:4px 6px;text-align:center">
    <span style="font-size:8px;font-weight:400;padding:1px 5px;border-radius:3px;background:{i_col}22;color:{i_col}">{r.get('impact','?')}</span>
  </td>
  <td style="padding:4px 6px;font-size:8px;color:#555">{r.get('note','')[:70]}{'…' if len(r.get('note',''))>70 else ''}</td>
</tr>"""

    # ── Valuation block ──────────────────────────────────────────────────────
    val_items = ""
    for label, key, suffix in [("Forward P/E","pe_fwd","×"),("P/S","ps","×"),("EV/EBITDA","ev_ebitda","×")]:
        if key in prof:
            val_items += f'<div style="display:flex;justify-content:space-between;padding:3px 0;border-bottom:1px solid #241f18"><span style="font-size:9px;color:#888">{label}</span><span style="font-size:11px;font-weight:400;font-variant-numeric:tabular-nums">{prof[key]}{suffix}</span></div>'

    if "pe_fwd" in prof:
        peer_pe_rows = ""
        for peer, pe in (prof.get("peer_pe") or {}).items():
            own_pe   = prof["pe_fwd"]
            prem_pct = (own_pe - pe) / pe * 100
            prem_col = "#C0392B" if prem_pct > 20 else ("#c8b487" if prem_pct > 0 else "#1B7A3B")
            peer_pe_rows += (
                f'<span style="display:inline-flex;align-items:center;gap:3px;margin:1px;padding:2px 6px;'
                f'border-radius:4px;background:#241f18;font-size:8px">'
                f'<strong>{peer}</strong> <span style="color:#888">{pe}×</span>'
                f'<span style="color:{prem_col};font-weight:400">({prem_pct:+.0f}%)</span>'
                f'</span>'
            )
    else:
        peer_pe_rows = ""

    dcf_html = ""
    if "dcf_low" in prof and "dcf_high" in prof:
        dcf_html = f'<div style="margin-top:6px;padding:6px 8px;background:#F0F8FF;border-radius:4px;display:flex;justify-content:space-between;align-items:center"><span style="font-size:9px;color:#888">DCF Fair Value Range</span><span style="font-size:11px;font-weight:400;color:#5f7480">${prof["dcf_low"]} – ${prof["dcf_high"]}</span></div>'

    return f"""
<div class="mindmap-card" style="border:1px solid #241f18;border-radius:12px;overflow:hidden;
  background:#fff;box-shadow:0 3px 12px rgba(0,0,0,.08);margin-bottom:24px">

  <!-- ═══ HEADER ═══════════════════════════════════════════════════════════ -->
  <div style="background:{col};color:#fff;padding:16px 20px">
    <div style="display:flex;align-items:flex-start;justify-content:space-between;gap:12px">
      <div>
        <div style="display:flex;align-items:baseline;gap:10px">
          <span style="font-size:26px;font-weight:500;letter-spacing:-1px">{ticker}</span>
          <span style="font-size:12px;opacity:.8">{prof.get('name','')}</span>
        </div>
        <div style="font-size:10px;opacity:.65;margin-top:2px">{prof.get('sector','')}</div>
        <div style="margin-top:6px">{fund_pills}</div>
      </div>
      <div style="text-align:right;flex-shrink:0">
        <div style="font-size:9px;text-transform:uppercase;letter-spacing:.8px;opacity:.7;margin-bottom:3px">{prof.get('chain_label','')}</div>
        <span style="font-size:9px;padding:3px 8px;border-radius:20px;background:rgba(255,255,255,.2)">{icon}</span>
        <div style="margin-top:6px">
          <span style="font-size:9px;padding:2px 8px;border-radius:20px;background:rgba(255,255,255,.15)">{n_funds} top funds · ${total_value_m:,.0f}M AUM</span>
          {' <span style="font-size:9px;padding:2px 8px;border-radius:20px;background:rgba(255,255,255,.3);font-weight:400;margin-left:4px">★ Canyon</span>' if canyon_owns else ''}
        </div>
      </div>
    </div>
  </div>

  <!-- ═══ ROW 1: Value Chain | Moat Scorecard | Valuation ════════════════ -->
  <div style="display:grid;grid-template-columns:1fr 1fr 1fr;border-bottom:1px solid #241f18">

    <!-- Col A: Value Chain + Competitors -->
    <div style="padding:14px 16px;border-right:1px solid #241f18">
      <div style="font-size:9px;font-weight:400;letter-spacing:1px;color:#888;text-transform:uppercase;margin-bottom:8px">产业链定位 Value Chain Position</div>
      {vc_html}
      <div style="margin-top:12px;font-size:9px;font-weight:400;letter-spacing:1px;color:#888;text-transform:uppercase;margin-bottom:6px">竞争格局 Competitive Landscape</div>
      <div style="overflow-x:auto">
        <table style="width:100%;border-collapse:collapse">
          <tbody>{comp_rows}</tbody>
        </table>
      </div>
    </div>

    <!-- Col B: Moat 5-Dimension Scorecard + Patents -->
    <div style="padding:14px 16px;border-right:1px solid #241f18">
      <div style="font-size:9px;font-weight:400;letter-spacing:1px;color:#888;text-transform:uppercase;margin-bottom:6px">护城河 Moat Analysis</div>

      <!-- Overall score -->
      <div style="display:flex;align-items:center;gap:8px;margin-bottom:8px;padding:6px 8px;background:#FAFAFA;border-radius:6px">
        <div>
          <div style="font-size:13px;font-weight:500;color:{mcol}">{prof.get('moat_width','')}</div>
          <div style="font-size:8px;color:#888">{prof.get('moat_type','')[:30]}</div>
        </div>
        <div style="flex:1">
          <div style="background:#EEE;border-radius:10px;height:8px">
            <div style="height:8px;border-radius:10px;background:{mcol};width:{mpct:.0f}%"></div>
          </div>
        </div>
        <div style="font-size:14px;font-weight:500;color:{mcol};font-variant-numeric:tabular-nums">{mscore:.1f}</div>
      </div>

      <!-- 5-dimension table -->
      <div style="font-size:8px;font-weight:400;text-transform:uppercase;letter-spacing:.6px;color:#999;margin-bottom:4px">5-Dimension Breakdown</div>
      <table style="width:100%;border-collapse:collapse;margin-bottom:8px">
        <tbody>{moat_dim_rows}</tbody>
      </table>

      <!-- Moat drivers -->
      <div style="font-size:9px;font-weight:400;text-transform:uppercase;letter-spacing:.8px;color:#888;margin-bottom:4px">Key Moat Drivers</div>
      {''.join(f'<div style="display:flex;align-items:flex-start;gap:4px;font-size:9px;color:#333;margin-bottom:3px;line-height:1.4"><span style="color:{col};font-size:8px;margin-top:2px;flex-shrink:0">▶</span>{d}</div>' for d in prof.get("moat_drivers",[])[:4])}

      <!-- Patents -->
      <div style="margin-top:10px;font-size:9px;font-weight:400;text-transform:uppercase;letter-spacing:.8px;color:#888;margin-bottom:4px">专利 Patents — {prof.get('patent_count','?')}</div>
      {''.join(f'<div style="font-size:9px;color:#444;margin-bottom:2px">• {pa}</div>' for pa in prof.get('patent_areas',[])[:5])}
    </div>

    <!-- Col C: Valuation + Bull/Bear/Canyon -->
    <div style="padding:14px 16px">
      <div style="font-size:9px;font-weight:400;letter-spacing:1px;color:#888;text-transform:uppercase;margin-bottom:6px">估值 Valuation Snapshot</div>
      {val_items}
      {dcf_html}

      <!-- Peer P/E comparison -->
      {f'<div style="margin-top:8px"><div style="font-size:8px;font-weight:400;color:#888;text-transform:uppercase;margin-bottom:4px">vs Peers (fwd P/E)</div><div style="line-height:2">{peer_pe_rows}</div></div>' if peer_pe_rows else ''}

      <!-- Bull case -->
      <div style="margin-top:10px;font-size:9px;font-weight:400;letter-spacing:.8px;color:#1B7A3B;text-transform:uppercase;margin-bottom:4px">Longs逻辑 Bull Case</div>
      <div style="font-size:9px;color:#333;line-height:1.55;padding:6px 8px;background:#F2FFF2;border-radius:4px;border-left:3px solid #1B7A3B;margin-bottom:6px">{prof.get('bull_case','')}</div>

      <!-- Bear case -->
      <div style="font-size:9px;font-weight:400;letter-spacing:.8px;color:#C0392B;text-transform:uppercase;margin-bottom:4px">Shorts逻辑 Bear Case</div>
      <div style="font-size:9px;color:#333;line-height:1.55;padding:6px 8px;background:#FFF2F2;border-radius:4px;border-left:3px solid #C0392B;margin-bottom:6px">{prof.get('bear_case','')}</div>

      <!-- Canyon view -->
      <div style="padding:6px 8px;background:#F5F5F5;border-radius:4px;border-left:3px solid #c8b487">
        <div style="font-size:8px;font-weight:400;text-transform:uppercase;color:#c8b487;margin-bottom:2px">Canyon 研判</div>
        <div style="font-size:9px;color:#333;line-height:1.5">{prof.get('canyon_view','')}</div>
      </div>
    </div>

  </div>

  <!-- ═══ ROW 2: Revenue Segments | Risk Matrix ══════════════════════════ -->
  <div style="display:grid;grid-template-columns:1fr 1fr;background:#FAFAFA">

    <!-- Revenue segments -->
    <div style="padding:12px 16px;border-right:1px solid #241f18">
      <div style="font-size:9px;font-weight:400;letter-spacing:1px;color:#888;text-transform:uppercase;margin-bottom:8px">收入结构 Revenue Breakdown</div>
      {rev_bars if rev_bars else '<p style="font-size:9px;color:#aaa">Revenue breakdown not available</p>'}
    </div>

    <!-- Risk matrix -->
    <div style="padding:12px 16px">
      <div style="font-size:9px;font-weight:400;letter-spacing:1px;color:#888;text-transform:uppercase;margin-bottom:8px">风险矩阵 Risk Matrix — Probability × Impact</div>
      {f"""<div style="overflow-x:auto">
        <table style="width:100%;border-collapse:collapse;min-width:300px">
          <thead>
            <tr style="background:#241f18">
              <th style="padding:3px 6px;font-size:8px;color:#888;text-align:left">Risk</th>
              <th style="padding:3px 6px;font-size:8px;color:#888;text-align:center">Prob</th>
              <th style="padding:3px 6px;font-size:8px;color:#888;text-align:center">Impact</th>
              <th style="padding:3px 6px;font-size:8px;color:#888;text-align:left">Detail</th>
            </tr>
          </thead>
          <tbody>{risk_rows}</tbody>
        </table>
      </div>""" if risk_rows else '<p style="font-size:9px;color:#aaa">Risk data not available</p>'}
    </div>

  </div>

</div>"""


def build_html(daily: dict, chart: dict, summ: dict,
               accruals: list, squeeze: list, live: dict = None,
               bt_monthly: dict = None, paper_nav: dict = None,
               wf_steps: list = None, wf_queue: list = None,
               alpha_scores: list = None, risk_gate: list = None,
               ticker_drilldown: list = None, desk_monitor: list = None,
               sector_cycle: list = None,
               news: list = None, earnings_cal: list = None,
               macro_breadth: dict = None,
               rolling_ic: dict = None, factor_attr: dict = None,
               monthly_pnl: dict = None,
               position_pnl: list = None,
               crowding: dict = None,
               macro_sigs: dict = None,
               v251_bt: dict = None,
               v251_regime: dict = None,
               deep: dict = None,
               signal_health: dict = None,
               barra_risk: dict = None,
               hmm_data: dict = None,
               macro_outlook: dict = None,
               dcf_data: "pd.DataFrame | None" = None,
               short_data: "pd.DataFrame | None" = None,
               econ_cal: dict = None,
               earnings_ai: "pd.DataFrame | None" = None,
               watchlist: dict = None,
               famous_holdings: dict = None,
               congressional_trades: dict = None,
               options_flow: dict = None,
               etf_flow: dict = None) -> str:
    if live is None:         live = {}
    if bt_monthly is None:   bt_monthly = {}
    if paper_nav is None:    paper_nav = {}
    if wf_steps is None:     wf_steps = []
    if wf_queue is None:     wf_queue = []
    if alpha_scores is None: alpha_scores = []
    if risk_gate is None:    risk_gate = []
    if ticker_drilldown is None: ticker_drilldown = []
    if desk_monitor is None: desk_monitor = []
    if sector_cycle is None: sector_cycle = []
    if news is None:         news = []
    if earnings_cal is None: earnings_cal = []
    if macro_breadth is None:  macro_breadth = {"breadth": [], "rotation": []}
    if rolling_ic is None:     rolling_ic = {"labels": [], "ic_3m": [], "ic_6m": [], "statuses": [], "target": 0.370, "current_3m": None, "current_status": "—", "factor_labels": [], "factor_ic": {}}
    if factor_attr is None:    factor_attr = {"ff5": [], "signals": [], "summary": {}}
    if monthly_pnl is None:    monthly_pnl = {"labels": [], "net": [], "long_c": [], "short_c": [], "alpha_c": [], "mkt_c": [], "hit_rate": [], "avg_alpha": 0.0, "best_month": 0.0, "worst_month": 0.0, "long_win_months": 0, "total_months": 0}
    if position_pnl is None:   position_pnl = []
    if crowding is None:       crowding = {"factor_trend": [], "sector_concentration": {}, "watch_tickers": [], "long_semis": [], "short_semis": []}
    if macro_sigs is None:     macro_sigs = {}
    if v251_bt is None:        v251_bt = {}
    if v251_regime is None:    v251_regime = {}
    if deep is None:           deep = {}
    if hmm_data is None:       hmm_data = {}
    if macro_outlook is None:  macro_outlook = {}
    if econ_cal is None:       econ_cal = {"events": [], "count": 0}
    if earnings_ai is None:    earnings_ai = pd.DataFrame()
    if watchlist is None:      watchlist = {"tickers": [], "notes": {}}
    if famous_holdings is None:      famous_holdings = {}
    if congressional_trades is None: congressional_trades = {}
    if options_flow is None:         options_flow = {}
    if etf_flow is None:             etf_flow = {}
    today       = datetime.now().strftime("%Y-%m-%d")   # always today's date for display
    report_date = daily.get("date", today)              # when signals were last generated
    data_stale  = report_date != today
    stale_note  = f' <span style="color:#c8b487;font-size:11px;font-weight:400">(last updated {report_date} — click ⟳ Refresh Now to update)</span>' if data_stale else ""

    # ── Data freshness check (file-level, not just report_date) ───────────────
    import time as _time
    _alpha_path = ROOT / "alpha_scores.csv"
    _alpha_mtime = _alpha_path.stat().st_mtime if _alpha_path.exists() else 0
    _alpha_age_hours = (_time.time() - _alpha_mtime) / 3600
    _alpha_age_days  = _alpha_age_hours / 24
    _critically_stale = _alpha_age_days > 1.5   # more than 1.5 days = definitely missed a run
    if _critically_stale:
        _stale_days_int  = int(_alpha_age_days)
        _stale_since_str = _time.strftime("%Y-%m-%d %H:%M", _time.localtime(_alpha_mtime))
        _stale_banner = f'''<div style="background:#5c1a1a;border-bottom:3px solid #c0392b;padding:10px 24px;display:flex;align-items:center;gap:14px;font-size:13px;color:#fff;position:sticky;top:0;z-index:9999">
  <span style="font-size:20px;flex-shrink:0">⚠</span>
  <div>
    <strong style="color:#ff6b6b">Data is {_stale_days_int} day{'s' if _stale_days_int>1 else ''} stale</strong>
    &nbsp;— alpha_scores.csv last updated <code style="background:rgba(0,0,0,.3);padding:1px 5px;border-radius:3px">{_stale_since_str}</code>.
    The daily pipeline may have crashed. Check <code style="background:rgba(0,0,0,.3);padding:1px 5px;border-radius:3px">autorun_stderr.log</code> and run
    <code style="background:rgba(0,0,0,.3);padding:1px 5px;border-radius:3px">python3 run_daily.py</code> manually.
  </div>
</div>'''
    else:
        _stale_banner = ""
    # Use actual HMM CSV output when available; fall back to daily report text parse
    if hmm_data.get("regime"):
        hmm = hmm_data["regime"]
    else:
        hmm = daily.get("hmm", "—")
    macro     = daily.get("macro", "—")
    longs     = daily.get("longs", [])
    shorts    = daily.get("shorts", [])
    new_long  = daily.get("new_long", [])
    exit_long = daily.get("exit_long", [])
    new_short = daily.get("new_short", [])
    exit_short= daily.get("exit_short", [])

    hmm_color = "#1B6F4A" if hmm == "BULL" else "#B83232"
    # HMM probability and staleness metadata
    _hmm_prob_bear  = hmm_data.get("prob_bear")
    _hmm_date       = hmm_data.get("date", "—")
    _hmm_days_stale = hmm_data.get("days_stale", 0)
    _hmm_stale      = hmm_data.get("stale", False)
    _hmm_prob_label = f"{_hmm_prob_bear*100:.0f}% bear prob" if _hmm_prob_bear is not None else ""
    if _hmm_stale:
        _hmm_meta = f'<span style="display:block;font-size:10px;color:#c8b487;margin-top:3px">⚠ HMM last run {_hmm_date} ({_hmm_days_stale}d ago)</span>'
    elif _hmm_date != "—":
        _hmm_meta = f'<span style="display:block;font-size:10px;color:#888;margin-top:3px">HMM as of {_hmm_date}</span>'
    else:
        _hmm_meta = ""
    mac_color = "#1B6F4A" if "ON" in macro.upper() else ("#B83232" if "OFF" in macro.upper() else "#c8b487")

    # ── Macro Regime Outlook variables ────────────────────────────────────────
    _mo_composite     = macro_outlook.get("composite", {})
    _mo_bear_prob     = _mo_composite.get("bear_prob")       # float 0-100 or None
    _mo_delta         = _mo_composite.get("bear_prob_delta") # float or None (vs ~1wk ago)
    _mo_label         = _mo_composite.get("label", "—")
    _mo_color         = _mo_composite.get("color", "#888")
    _mo_as_of         = macro_outlook.get("as_of", "—")
    _mo_signals       = macro_outlook.get("signals", {})

    def _mo_signal_bar(key: str) -> str:
        sig = _mo_signals.get(key, {})
        if not sig.get("ok"):
            return f'<div style="color:{FT["faint"]};font-size:11px;padding:8px 0;border-bottom:1px solid {FT["border2"]}">{sig.get("name", key)} — no data</div>'
        bs    = float(sig.get("bear_score", 0))
        mx    = float(sig.get("max_score", 2))
        pct   = bs / mx * 100
        color = FT["pos"] if pct < 25 else (FT["warn"] if pct < 60 else FT["neg"])
        watch = sig.get("watch_for", "")
        watch_html = (f'<div style="font-size:9.5px;color:{FT["faint"]};margin-top:3px;line-height:1.4">'
                      f'Watch for · {watch}</div>') if watch else ""
        return (
            f'<div style="padding:9px 0;border-bottom:1px solid {FT["border2"]}">'
            f'<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:4px">'
            f'<span style="font-size:11.5px;color:{FT["ink"]};font-weight:400;letter-spacing:.02em">{sig["name"]}</span>'
            f'<span style="font-size:11px;color:{color};font-weight:400;font-variant-numeric:tabular-nums">{sig.get("display","—")}</span>'
            f'</div>'
            f'<div style="display:flex;align-items:center;gap:8px">'
            f'<div style="flex:1;height:3px;border-radius:2px;background:{FT["inner"]}"><div style="height:3px;border-radius:2px;background:{color};width:{pct:.0f}%"></div></div>'
            f'<span style="font-size:10px;color:{FT["mute"]};white-space:nowrap;flex-shrink:0">{sig.get("trend_label","")}</span>'
            f'</div>'
            f'{watch_html}'
            f'</div>'
        )

    def _mo_panel() -> str:
        if not macro_outlook or _mo_bear_prob is None:
            return '<p style="color:#AAA;font-size:11px;margin:0">Macro outlook not yet computed — will populate on next daily run.</p>'
        hmm_is_bear  = hmm == "BEAR"
        macro_is_low = _mo_bear_prob < 30
        if hmm_is_bear and macro_is_low:
            conflict_note = (
                f'<div style="background:#FDF8EE;border-left:3px solid #c8b487;padding:10px 14px;margin-bottom:16px">'
                f'<p style="margin:0 0 3px;font-size:10px;font-weight:400;color:#c8b487;text-transform:uppercase;letter-spacing:1.5px">Signal Conflict — Price dip, not macro breakdown</p>'
                f'<p style="margin:0;font-size:11px;color:#666;line-height:1.6">HMM detected a short-term price drawdown (reactive, 1-3 week lag). '
                f'5 macro leading indicators show only {_mo_bear_prob:.0f}% bear risk — credit spreads tight, yield curve positive, '
                f'VIX in normal contango. This looks like a correction, not a regime shift.</p>'
                f'</div>'
            )
        elif not hmm_is_bear and _mo_bear_prob > 55:
            conflict_note = (
                f'<div style="background:#FEF2F2;border-left:3px solid #B83232;padding:10px 14px;margin-bottom:16px">'
                f'<p style="margin:0 0 3px;font-size:10px;font-weight:400;color:#B83232;text-transform:uppercase;letter-spacing:1.5px">Warning — Macro deteriorating ahead of price</p>'
                f'<p style="margin:0;font-size:11px;color:#666;line-height:1.6">HMM shows BULL but macro leading indicators show {_mo_bear_prob:.0f}% bear risk. '
                f'This has historically preceded regime shifts by 4-8 weeks.</p>'
                f'</div>'
            )
        else:
            conflict_note = ""
        bars = "".join(_mo_signal_bar(k) for k in ["yield_curve","credit_spreads","vix_term_structure","spy_trend","labor_market"])
        prob_bar_color = FT["pos"] if _mo_bear_prob < 30 else (FT["warn"] if _mo_bear_prob < 60 else FT["neg"])
        what_it_means = (
            "Macro conditions are healthy — low probability of a sustained bear market in the next 4 weeks. Lower is better." if _mo_bear_prob < 30 else
            "Macro is showing stress — watch for further deterioration." if _mo_bear_prob < 60 else
            "Multiple indicators flashing — high bear risk over the next 4 weeks."
        )
        # 周环比箭头 — 让慢变量也能看出方向 (熊险上升=变差=红, 下降=变好=绿)
        if _mo_delta is None:
            arrow_html = f'<span style="font-size:12px;color:{FT["faint"]};margin-left:10px">week-over-week building</span>'
        elif _mo_delta > 0.4:
            arrow_html = f'<span style="font-size:13px;color:{FT["neg"]};margin-left:10px">▲ +{_mo_delta:.0f} vs last week (risk rising)</span>'
        elif _mo_delta < -0.4:
            arrow_html = f'<span style="font-size:13px;color:{FT["pos"]};margin-left:10px">▼ {_mo_delta:.0f} vs last week (risk falling)</span>'
        else:
            arrow_html = f'<span style="font-size:13px;color:{FT["mute"]};margin-left:10px">= flat vs last week</span>'
        return (
            f'{conflict_note}'
            f'<div style="display:grid;grid-template-columns:190px 1fr;gap:24px;align-items:start">'
            f'<div>'
            f'<p style="font-size:10px;letter-spacing:.14em;text-transform:uppercase;color:{FT["mute"]};font-weight:400;margin:0 0 6px">4-Week Bear Risk</p>'
            f'<p style="font-family:{FT["serif"]};font-size:38px;font-weight:400;color:{prob_bar_color};line-height:1;margin:0">{_mo_bear_prob:.0f}%</p>'
            f'<p style="font-size:12px;font-weight:400;color:{_mo_color};margin:5px 0 2px">{_mo_label}{arrow_html}</p>'
            f'<div style="height:5px;border-radius:2px;background:{FT["inner"]};margin:10px 0">'
            f'<div style="height:5px;border-radius:2px;background:{prob_bar_color};width:{_mo_bear_prob:.0f}%"></div>'
            f'</div>'
            f'<p style="font-size:11px;color:{FT["sub"]};line-height:1.6;margin:0">{what_it_means}</p>'
            f'<p style="font-size:10px;color:{FT["faint"]};margin:8px 0 0">as of {_mo_as_of[:10] if _mo_as_of != "—" else "—"} · 0% = very calm, 100% = high risk</p>'
            f'</div>'
            f'<div style="border-left:1px solid {FT["border"]};padding-left:24px">'
            f'<p style="font-size:10px;letter-spacing:.14em;text-transform:uppercase;color:{FT["mute"]};font-weight:400;margin:0 0 4px">5 Leading Indicators · distance to warning line</p>'
            f'{bars}'
            f'</div>'
            f'</div>'
        )

    # ── v25.1 helpers ────────────────────────────────────────────────────────
    def v251_annual_rows():
        rows = []
        for r in v251_bt.get("annual_rows", []):
            yr   = r["year"]
            s    = r["strat"]
            q    = r["qqq"]
            p    = r["spy"]
            beat = r["beat"]
            bg   = "rgba(107,204,160,0.15)" if beat else "rgba(239,144,144,0.10)"
            tag  = f'<span style="color:#6BCCA0;font-weight:400">{s*100:+.1f}%★</span>' if beat else f'<span style="color:#EF9090">{s*100:+.1f}%</span>'
            q_c  = "#1B6F4A" if q > 0 else "#B83232"
            rows.append(
                f'<tr style="background:{bg}">'
                f'<td style="color:rgba(255,255,255,.7)">{yr}</td>'
                f'<td style="text-align:right;color:#6BCCA0;font-weight:400">{s*100:+.1f}%</td>'
                f'<td style="text-align:right;color:{q_c}">{q*100:+.1f}%</td>'
                f'<td style="text-align:right">{tag}</td>'
                f'<td style="text-align:right;color:rgba(255,255,255,.45)">{p*100:+.1f}%</td>'
                f'</tr>'
            )
        return "\n".join(rows) if rows else "<tr><td colspan='5' style='color:#AAA;text-align:center'>No data</td></tr>"

    def v251_regime_rows():
        r = v251_regime
        def _row(label, val, ok, gate_txt):
            color = "#6BCCA0" if ok else "#EF9090"
            icon  = "✓" if ok else "✗"
            return f'<tr><td>{label}</td><td style="font-size:12px;color:#666">{val}</td><td class="r" style="color:{color}">{icon} {gate_txt}</td></tr>'
        rows = [
            _row("SPY 200MA gate", f'SPY ${r.get("spy",0):.1f} vs MA ${r.get("spy_ma200",0):.1f}', r.get("gate_spy", True), "> 200MA" if r.get("gate_spy") else "< 200MA"),
            _row("QQQ 200MA gate", f'QQQ ${r.get("qqq",0):.1f} vs MA ${r.get("qqq_ma200",0):.1f}', r.get("gate_ma", True), "> 200MA" if r.get("gate_ma") else "< 200MA"),
            _row("QQQ 3M momentum", f'{r.get("qqq_3m_ret",0):+.1f}% (3M return)', r.get("gate_mom", True), "positive" if r.get("gate_mom") else "negative"),
            _row("VIX level", f'^VIX = {r.get("vix",0):.1f} → {r.get("vix_tier","—")}', r.get("vix",25) < 25, f'< 25 ({r.get("vix_tier","—")})' if r.get("vix",25) < 25 else "≥ 25"),
        ]
        return "\n".join(rows)

    _v251_ar  = v251_bt.get("ar", 0)
    _v251_sr  = v251_bt.get("sharpe", 0)
    _v251_mdd = v251_bt.get("mdd", 0)
    _v251_cal = v251_bt.get("calmar", 0)
    _v251_cum = v251_bt.get("cum_total", 0)
    _v251_beat = v251_bt.get("beat_years", 0)
    _v251_n   = v251_bt.get("n_months", 89)
    _v251_labels = json.dumps(v251_bt.get("labels", []))
    _v251_chart  = json.dumps(v251_bt.get("cum_v251", []))
    _v251_qqq    = json.dumps(v251_bt.get("cum_qqq", []))
    _v251_spy    = json.dumps(v251_bt.get("cum_spy", []))
    _reg_regime    = v251_regime.get("regime", "—")
    _reg_color     = v251_regime.get("regime_color", "#888")
    _reg_vix       = v251_regime.get("vix", 0)
    _reg_vix_c     = v251_regime.get("vix_color", "#888")
    _reg_tqqq      = v251_regime.get("tqqq_wt", 0)
    _reg_tqqq_base = v251_regime.get("tqqq_base", _reg_tqqq)
    _reg_tier      = v251_regime.get("vix_tier", "—")
    _reg_as_of     = v251_regime.get("as_of", "—")
    _reg_hmm       = v251_regime.get("hmm_regime", "UNKNOWN")
    _reg_hmm_bear  = v251_regime.get("hmm_is_bear", False)
    _reg_tqqq_s    = f'{_reg_tqqq:.0%} ON' if _reg_tqqq > 0 else '0% OFF'
    _reg_hmm_note  = (
        f'<div style="margin-top:8px;padding:6px 10px;background:#FDF8EE;border-left:3px solid #c8b487;border-radius:0 3px 3px 0">'
        f'<span style="font-size:10px;color:#c8b487;font-weight:400">HMM=BEAR override</span>'
        f'<span style="font-size:10px;color:#888;display:block">TQQQ cut from {_reg_tqqq_base:.0%} → {_reg_tqqq:.0%} (halved while HMM is bearish)</span>'
        f'</div>'
    ) if _reg_hmm_bear and _reg_tqqq < _reg_tqqq_base else ""
    _oos_t_      = summ.get("oos_t", 0)
    _oos_sharpe_ = summ.get("oos_sharpe", 0)
    _oos_ret_    = summ.get("oos_ret", 0)
    _deep_sr_t   = _oos_t_                                       # IC t-stat from OOS (real)
    _deep_ir     = deep.get("sn_ls_ir", _oos_sharpe_)           # real IR from deep JSON
    _deep_beta   = deep.get("beta_v252", deep.get("beta_spy", -0.012))
    _deep_alpha  = deep.get("ar_v252_cap", deep.get("ar_v251", _oos_ret_ / 100 if _oos_ret_ else 0))
    _deep_ic_t   = _oos_t_                                       # same IC t-stat
    _deep_sn_t   = deep.get("sn_ls_t", 2.18)
    _deep_sn_ann = deep.get("sn_ls_ann", 0.0222)

    def long_rows():
        if not longs:
            return "<tr><td colspan='6' style='color:#AAA;text-align:center'>No data — run step500</td></tr>"
        out = []
        for r in longs:
            ml_c  = "pos" if r["ml"]     > 0 else "neg"
            fac_c = "pos" if r["factor"] > 0 else "neg"
            strength = "strong" if r["score"] > 1.0 else ("mod" if r["score"] > 0.4 else "weak")
            bar_w = min(100, int(r["score"] / max(longs, key=lambda x:x["score"])["score"] * 100))
            out.append(f"""<tr class="tr-{strength}">
              <td class="td-rank">#{r['rank']}</td>
              <td class="td-ticker" data-ticker="{r['ticker']}" onclick="canyonQL.open('{r['ticker']}')" style="cursor:pointer">{r['ticker']}</td>
              <td class="td-score"><div class="score-bar-wrap"><div class="score-bar" style="width:{bar_w}%"></div></div><span>{r['score']:+.3f}</span></td>
              <td>${r['price']}</td>
              <td class="{ml_c}">{r['ml']:+.2f}</td>
              <td class="{fac_c}">{r['factor']:+.2f}</td>
            </tr>""")
        return "\n".join(out)

    def short_rows():
        # Show individual stocks only
        stocks = [s for s in shorts if not s["is_etf"]]
        etfs   = [s for s in shorts if s["is_etf"]]
        out = []
        for r in (stocks or shorts[:5]):
            sc = f"{r['score']:+.3f}" if r["score"] is not None else "—"
            out.append(f"""<tr>
              <td class="td-rank">#{r['rank']}</td>
              <td class="td-ticker" data-ticker="{r['ticker']}" onclick="canyonQL.open('{r['ticker']}')" style="cursor:pointer">{r['ticker']}</td>
              <td class="neg">{sc}</td>
              <td>${r['price']}</td>
            </tr>""")
        if etfs:
            out.append(f'<tr><td colspan="4" style="font-size:11px;color:#AAA;padding:8px 13px">+ {len(etfs)} ETFs excluded (no fundamental signal coverage)</td></tr>')
        return "\n".join(out)

    def accrual_long_rows():
        top = [a for a in accruals if a["ratio"] < 0][:6]
        if not top:
            return "<tr><td colspan='3' style='color:#AAA'>No data</td></tr>"
        return "\n".join(f'<tr><td class="td-ticker">{a["ticker"]}</td><td class="pos">{a["ratio"]:+.3f}</td><td class="pos">Buy</td></tr>' for a in top)

    def accrual_short_rows():
        bot = sorted([a for a in accruals if a["ratio"] > 0], key=lambda x: x["ratio"], reverse=True)[:6]
        if not bot:
            return "<tr><td colspan='3' style='color:#AAA'>No data</td></tr>"
        return "\n".join(f'<tr><td class="td-ticker">{a["ticker"]}</td><td class="neg">{a["ratio"]:+.3f}</td><td class="neg">Avoid</td></tr>' for a in bot)

    def squeeze_rows():
        if not squeeze:
            return "<tr><td colspan='4' style='color:#AAA'>No data</td></tr>"
        out = []
        for r in squeeze[:10]:
            stars = "★★★★" if r["conds"] == 4 else ("★★★" if r["conds"] == 3 else ("★★" if r["conds"] == 2 else "★"))
            out.append(f'<tr><td class="td-ticker" data-ticker="{r["ticker"]}" onclick="canyonQL.open(\'{r["ticker"]}\')" style="cursor:pointer">{r["ticker"]}</td><td style="text-align:center">{stars}</td><td class="r">{r["score"]:+.3f}</td><td class="r">{r["mom_vs_spy"]:+.1f}%</td></tr>')
        return "\n".join(out)

    def convergence_rows():
        """Top 5 tickers by alpha_score — how many signals agree."""
        if not alpha_scores:
            return "<tr><td colspan='5' style='color:#AAA'>No data</td></tr>"
        out = []
        top5 = sorted(alpha_scores, key=lambda x: float(x.get("alpha_score", 0) or 0), reverse=True)[:5]
        for r in top5:
            ml  = float(r.get("sig_ml_ensemble", 50) or 50)
            sqz = float(r.get("sig_squeeze",     50) or 50)
            rev = float(r.get("sig_revision",    50) or 50)
            ins = float(r.get("sig_insider",     50) or 50)
            # Count signals that are clearly positive (above 65 on 0-100 scale)
            n = sum(1 for v in [ml, sqz, rev, ins] if v > 65)
            stars = "★" * n + "☆" * (4 - n)
            stars_color = "#1B6F4A" if n >= 3 else ("#c8b487" if n == 2 else "#999")
            verdict = "Strong" if n >= 3 else ("Watch" if n == 2 else "Weak")
            verdict_class = "pos" if n >= 3 else ("" if n == 2 else "neg")
            ml_disp = f"{(ml-50)/25:+.2f}σ"
            out.append(f"""<tr>
              <td class="td-ticker" data-ticker="{r['ticker']}" onclick="canyonQL.open('{r['ticker']}')" style="cursor:pointer">{r['ticker']}</td>
              <td class="{('pos' if ml > 65 else '')}">{ml_disp}</td>
              <td class="{('pos' if rev > 65 else '')}">{'+' if rev>65 else '—'}</td>
              <td style="color:{stars_color}">{stars}</td>
              <td class="{verdict_class}" style="font-size:12px">{verdict}</td>
            </tr>""")
        return "\n".join(out) if out else "<tr><td colspan='5' style='color:#AAA'>No buy signals today</td></tr>"

    def _best_convergence_ticker():
        """Return (ticker, n_signals) for the top convergence pick."""
        if not alpha_scores:
            return "—", 0
        top10 = sorted(alpha_scores, key=lambda x: float(x.get("alpha_score", 0) or 0), reverse=True)[:10]
        best, best_n = "—", 0
        for r in top10:
            ml  = float(r.get("sig_ml_ensemble", 50) or 50)
            sqz = float(r.get("sig_squeeze",     50) or 50)
            rev = float(r.get("sig_revision",    50) or 50)
            ins = float(r.get("sig_insider",     50) or 50)
            n = sum(1 for v in [ml, sqz, rev, ins] if v > 65)
            if n > best_n:
                best, best_n = r["ticker"], n
        return best, best_n

    def signal_changes_block():
        """Render a compact signal changes row if any exist today."""
        has_changes = any([new_long, exit_long, new_short, exit_short])
        if not has_changes:
            return ""
        items = []
        for label, tickers, color in [
            ("New buys ▲",        new_long,   "#1B6F4A"),
            ("Removed from buys", exit_long,  "#B83232"),
            ("New avoids ▼",      new_short,  "#B83232"),
            ("Removed from avoids", exit_short, "#1B6F4A"),
        ]:
            if tickers:
                chips = "".join(f'<span style="display:inline-block;margin:2px 4px 2px 0;padding:3px 9px;background:{FT["inner"]};border:1px solid {FT["border2"]};border-radius:4px;font-size:12px;font-weight:400;color:{color}">{t}</span>' for t in tickers)
                items.append(f'<div style="display:flex;align-items:baseline;gap:10px;padding:8px 0;border-bottom:1px solid {FT["border2"]}"><span style="font-size:10px;text-transform:uppercase;letter-spacing:.1em;color:{FT["mute"]};min-width:120px;font-weight:400">{label}</span><div>{chips}</div></div>')
        return (_ft_open("What Changed Since Yesterday · New Buys / Removed / New Avoids",
                         "A fresh 'New buy ▲' into the top 15 is the most actionable signal")
                + "".join(items)
                + _ft_close("A stock that just entered the top 15 for the first time (New buy ▲) is the most actionable signal here — something changed overnight to push it up. A New avoid ▼ is worth a look if you hold it."))

    def _daily_summary():
        if hmm == "BULL":
            regime_txt = "The market is in <strong>Bull</strong> mode — the strategy is running at full strength"
        else:
            regime_txt = "The market is in <strong>Bear</strong> mode — the strategy has reduced risk and is being defensive"
        macro_txt_map = {
            "ON":      "Background conditions (bonds, credit spreads) look healthy and supportive",
            "OFF":     "Background conditions are flashing warning signs — stay cautious",
            "NEUTRAL": "Background conditions are mixed — nothing alarming, but don't be aggressive",
        }
        mac_word = (macro or "").upper().split()[0] if macro else "NEUTRAL"
        macro_note = macro_txt_map.get(mac_word, f"Background conditions show <strong>{macro}</strong>")
        if longs:
            top_ticker = longs[0]['ticker']
            top3 = ", ".join(r['ticker'] for r in longs[1:4])
            top_str = f"Today's strongest buy idea is <strong>{top_ticker}</strong>"
            if top3:
                top_str += f", followed by {top3}"
        else:
            top_str = ""
        changes = []
        if new_long:
            n = len(new_long)
            tickers_str = f"<strong style='color:#1B6F4A'>{', '.join(new_long)}</strong>"
            changes.append(f"{tickers_str} {'is' if n==1 else 'are'} newly flagged as <strong style='color:#1B6F4A'>buy</strong> candidates today")
        if exit_long:
            tickers_str = f"<strong style='color:#B83232'>{', '.join(exit_long)}</strong>"
            changes.append(f"{tickers_str} dropped off the buy list")
        if new_short:
            n = len(new_short)
            tickers_str = f"<strong style='color:#B83232'>{', '.join(new_short)}</strong>"
            changes.append(f"{tickers_str} {'is' if n==1 else 'are'} newly flagged as <strong style='color:#B83232'>avoid / short</strong> candidates today")
        if exit_short:
            tickers_str = f"<strong style='color:#1B6F4A'>{', '.join(exit_short)}</strong>"
            changes.append(f"{tickers_str} dropped off the avoid list")
        if changes:
            change_txt = "What changed since yesterday: " + "; ".join(changes) + "."
        else:
            change_txt = "No changes since yesterday — the same stocks remain on the buy and avoid lists."
        return f"""<div class="daily-summary">
      <p class="eyebrow">Today in plain English</p>
      <p class="daily-summary-text">{regime_txt}. {macro_note}. {top_str}. {change_txt}</p>
    </div>"""

    def _ticker_js_data() -> str:
        """Build a JS-embeddable JSON object with per-ticker signal + risk + news data."""
        db: dict = {}
        sig_display_names = {
            "sig_ml_ensemble":  "Model signal",
            "sig_quality":      "Financial health",
            "sig_revision":     "Analyst upgrades",
            "sig_surprise":     "Earnings beat",
            "sig_sentiment":    "Investor sentiment",
            "sig_squeeze":      "Price setup",
            "sig_insider":      "Insider buying",
            "sig_options":      "Options activity",
        }
        for r in alpha_scores:
            tk = str(r.get("ticker", ""))
            if not tk:
                continue
            sigs = {v: round(float(r.get(k, 50) or 50), 1) for k, v in sig_display_names.items()}
            db[tk] = {
                "score":   round(float(r.get("alpha_score", 0) or 0), 2),
                "rank":    int(r.get("alpha_rank", 99) or 99),
                "signal":  str(r.get("signal", "—")),
                "sector":  str(r.get("sector", "—")),
                "crowding":str(r.get("crowding_level", "—")),
                "sigs":    sigs,
                "risk_action": "—",
                "risk_plain":  "No risk data",
                "news": [],
            }
        for rg in risk_gate:
            tk = str(rg.get("ticker", ""))
            if tk in db:
                action = str(rg.get("final_risk_action", ""))
                db[tk]["risk_action"] = action
                db[tk]["risk_plain"] = _RISK_ACTION_PLAIN.get(action, (action, "neu"))[0]
        news_by_tk: dict = {}
        for item in news:
            t = str(item.get("ticker", ""))
            if t not in news_by_tk:
                news_by_tk[t] = []
            if len(news_by_tk[t]) < 3:
                news_by_tk[t].append({
                    "title": str(item.get("title", ""))[:120],
                    "tone":  str(item.get("tone", "")),
                    "date":  str(item.get("published", ""))[:10],
                })
        for tk, items in news_by_tk.items():
            if tk in db:
                db[tk]["news"] = items
        return json.dumps(db)

    _long_set  = {r["ticker"] for r in longs}
    _short_set = {str(s.get("ticker","")) for s in shorts}

    def _esc(s): return str(s).replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")

    def _macro_signal_cards(ms: dict) -> str:
        """Render macro signal cards with live values from macro_signals.json."""
        _SIG_HUMAN = {
            "RISK_ON":    ("Risk-on ↑",   "risk-on",  "Equity-positive"),
            "RISK_OFF":   ("Risk-off ↓",  "risk-off", "Defensive signal"),
            "NEUTRAL":    ("Neutral →",   "neutral",  "No clear direction"),
            "FLAT":       ("Flat / mixed","neutral",  "Mixed signal"),
            "STRONG_USD": ("USD rising ↑","neutral",  "Dollar strengthening"),
            "WEAK_USD":   ("USD falling ↓","risk-on", "Dollar weakening"),
            "GOLDEN":     ("Golden cross ↑","risk-on","SPY above 200d MA"),
            "DEATH":      ("Death cross ↓","risk-off","SPY below 200d MA"),
            "NORMAL_VOL": ("Normal vol →","neutral",  "VIX at normal level"),
            "HIGH_VOL":   ("Elevated vol ↓","risk-off","VIX above average"),
            "LOW_VOL":    ("Low vol ↑",   "risk-on",  "Calm market"),
            "STEEPENING": ("Curve steepening ↑","risk-on","Growth expectations rising"),
            "INVERSION":  ("Inverted curve ↓","risk-off","Recession signal"),
        }
        CARDS = [
            ("HYG / IEI", "High-yield credit spread",
             "Credit risk appetite. When high-yield bonds outperform, investors are comfortable taking risk — bullish for equities.",
             ms.get("credit_signal", "NEUTRAL")),
            ("TLT / IEI", "Yield curve slope",
             "Long vs short-term rates. Steepening = growth expectations rising. Inversion historically precedes recessions.",
             ms.get("yield_curve_signal", "NEUTRAL")),
            ("UUP", "US dollar strength",
             "A rising dollar tightens global financial conditions and weighs on multinational earnings.",
             ms.get("dxy_signal", "NEUTRAL")),
            ("GLD", "Gold vs equities",
             "When gold rises relative to stocks, investors are seeking safety — a risk-off warning signal.",
             ms.get("gold_signal", "NEUTRAL")),
            ("VIX curve", "Volatility term structure",
             "Near-term VIX vs 3-month VIX. Inverted (near > far) signals acute fear. Normal slope means market is calm.",
             ms.get("vts_signal", "NEUTRAL")),
            ("SPY 200d MA", "Equity trend filter",
             "Is the S&P 500 above its 200-day average? This is the single most reliable bull/bear regime indicator.",
             ms.get("sma_cross_signal", ms.get("macro_signal", "NEUTRAL"))),
        ]
        rows = []
        for ticker, name, desc, sig_raw in CARDS:
            sig_up = str(sig_raw).upper()
            label, css_cls, subtext = _SIG_HUMAN.get(sig_up, (sig_up.replace("_"," ").title(), "neutral", ""))
            rows.append(
                f'<div class="mac-card"><p class="mac-ticker">{ticker}</p>'
                f'<p class="mac-name">{name}</p>'
                f'<p class="mac-role">{desc}</p>'
                f'<p class="mac-status {css_cls}">{label}</p></div>'
            )
        return f'<div class="macro-grid">\n      {"".join(rows)}\n    </div>'

    def desk_monitor_rows():
        if not desk_monitor:
            return "<p style='color:#8f866f;font-size:13px'>No alerts today.</p>"
        # FT palette, hardcoded — no dependence on the global override net.
        sev_color = {
            "CRITICAL": "#c68b83", "HIGH": "#c8b487",
            "WARNING":  "#c8b487", "MEDIUM": "#8f866f",
            "INFO":     "#8aa6a6", "LOW":    "#79715f",
        }
        _MON_LABEL = {
            "RISK_LIMIT_BREACH": "Risk",
            "NEWS_SHOCK":        "News",
            "PRICE_BREAK":       "Price",
            "SQUEEZE_WATCH":     "Squeeze",
            "OPTIONS_ALERT":     "Options",
            "SIGNAL_DEGRADATION":"Signal",
            "NEW_BUY_LIST":      "Picks",
        }
        rows = []
        for i, r in enumerate(desk_monitor):
            sev       = str(r.get("severity", "")).upper()
            sc        = sev_color.get(sev, "#8f866f")
            mon_label = _MON_LABEL.get(str(r.get("monitor", "")).upper(), str(r.get("monitor", "")))
            sev_label = sev.title() if sev else ""

            tk_disp, title_h, detail_h, action_h = _humanize_desk_alert(r)

            # Strip redundant "TICKER — " from title if already shown separately
            title_body = title_h
            prefix = tk_disp + " — "
            if title_body.upper().startswith(prefix.upper()):
                title_body = title_body[len(prefix):]

            # Clean FT feed row: NO box outline. Colored left accent + hairline
            # divider between rows only. Dense but uncluttered — no "白边".
            divider = "" if i == 0 else "border-top:1px solid #211c15;"
            rows.append(f"""<div style="display:flex;align-items:flex-start;gap:16px;padding:13px 4px 13px 16px;border-left:3px solid {sc};{divider}">
              <div style="flex-shrink:0;min-width:74px">
                <p style="font-size:9.5px;letter-spacing:.13em;text-transform:uppercase;color:{sc};font-weight:400;margin-bottom:2px">{_esc(sev_label)}</p>
                <p style="font-size:10.5px;color:#79715f;letter-spacing:.06em;text-transform:uppercase">{_esc(mon_label)}</p>
              </div>
              <div style="flex:1;min-width:0">
                <p style="font-size:13.5px;font-weight:400;color:#f0e9da;margin-bottom:3px;line-height:1.45"><span style="color:#c8b487">{_esc(tk_disp)}</span> &mdash; {_esc(title_body)}</p>
                <p style="font-size:12.5px;color:#b0a68f;line-height:1.55">{_esc(detail_h)}</p>
                <p style="font-size:11.5px;color:#9a8a5f;font-weight:400;margin-top:5px;letter-spacing:.01em">&rarr; {_esc(action_h)}</p>
              </div>
            </div>""")
        # Wrap in a single subtle card so the feed reads as one panel, not scattered boxes.
        return ('<div style="background:#14110b;border:1px solid #221d15;border-radius:8px;'
                'padding:2px 14px 4px">' + "\n".join(rows) + '</div>')

    def workflow_steps_rows():
        if not wf_steps:
            return "<tr><td colspan='4' style='color:#AAA'>No workflow data</td></tr>"
        status_color = {"OK": "#1B6F4A", "REVIEW": "#c8b487", "WATCH": "#5f7480", "DONE": "#1B6F4A"}
        status_label = {"OK": "Done", "REVIEW": "Review now", "WATCH": "Monitor", "DONE": "Complete"}
        rows = []
        for r in wf_steps:
            st = str(r.get("status","")).upper()
            sc = status_color.get(st, "#999")
            st_disp = status_label.get(st, st.title())
            rows.append(f"""<tr>
              <td class="td-rank">#{_esc(r.get('step_order',''))}</td>
              <td style="font-weight:400;color:#c8b487">{_esc(r.get('station',''))}</td>
              <td><span style="font-size:10px;font-weight:400;letter-spacing:.8px;text-transform:uppercase;color:{sc}">{st_disp}</span></td>
              <td style="font-size:12.5px;color:#333">{_esc(str(r.get('what_to_do',''))[:120])}</td>
            </tr>""")
        return "\n".join(rows)

    def workflow_queue_rows():
        if not wf_queue:
            return "<tr><td colspan='6' style='color:#AAA'>No queue data</td></tr>"
        pri_color = {"High": "#B83232", "Medium": "#c8b487", "Low": "#1B6F4A"}
        rows = []
        for r in wf_queue:
            pc = pri_color.get(str(r.get("priority","")), "#999")
            action = str(r.get("sector_adjusted_action") or r.get("what_to_do",""))[:60]
            rows.append(f"""<tr>
              <td class="td-rank">#{_esc(r.get('priority_rank',''))}</td>
              <td class="td-ticker">{_esc(r.get('ticker',''))}</td>
              <td style="font-size:12px;color:#555">{_esc(r.get('sector',''))}</td>
              <td><span style="font-size:10px;font-weight:400;color:{pc};letter-spacing:.5px;text-transform:uppercase">{_esc(r.get('priority',''))}</span></td>
              <td style="font-size:12px;color:#333">{_esc(r.get('sector_cycle_state',''))}</td>
              <td style="font-size:12px;color:#c8b487;font-weight:400">{_esc(action)}</td>
            </tr>""")
        return "\n".join(rows)

    def trade_command_center():
        """Book health + action queue panel at the top of the Live tab."""
        book_stats: dict = {}
        for p in (position_pnl or []):
            b = str(p.get("book", "—"))
            if b not in book_stats:
                book_stats[b] = {"n": 0, "mv": 0.0, "upnl": 0.0, "longs": 0, "shorts": 0}
            s = book_stats[b]
            s["n"]     += 1
            s["mv"]    += float(p.get("market_value", 0) or 0)
            s["upnl"]  += float(p.get("unrealized_pnl_usd", 0) or 0)
            if p["side"] == "LONG":
                s["longs"] += 1
            else:
                s["shorts"] += 1

        book_order = ["SHORT", "MEDIUM", "LONG"]
        book_label = {"SHORT": "Short-term", "MEDIUM": "Mid-term", "LONG": "Long-term"}
        book_hdr_col = {"SHORT": "#B83232", "MEDIUM": "#c8b487", "LONG": "#3a3128"}
        book_cols = []
        for bk in book_order:
            s = book_stats.get(bk, {})
            if not s:
                continue
            upnl = s["upnl"]
            bk_col = book_hdr_col.get(bk, "#999")
            upnl_col = "#1B6F4A" if upnl > 0 else "#B83232"
            book_cols.append(f"""<div style="flex:1;min-width:150px;padding:14px 16px;background:#fff;border:1px solid #241f18;border-top:3px solid {bk_col};border-radius:6px">
                <div style="font-size:10px;font-weight:400;text-transform:uppercase;letter-spacing:.8px;color:{bk_col};margin-bottom:6px">{_esc(book_label.get(bk,bk))} · {bk}</div>
                <div style="font-size:22px;font-weight:500;color:#1A1A1A">{s['n']} <span style="font-size:13px;font-weight:400;color:#999">positions</span></div>
                <div style="font-size:12px;color:#555;margin-top:4px">{s['longs']}L / {s['shorts']}S &nbsp;|&nbsp; MV: ${s['mv']:,.0f}</div>
                <div style="font-size:14px;font-weight:400;color:{upnl_col};margin-top:6px">Unrealized P&amp;L: ${upnl:+,.0f}</div>
              </div>""")

        if not book_cols:
            book_section = '<p style="color:#AAA;font-size:13px">No book-level data — run step_alpaca_pnl.py to populate</p>'
        else:
            book_section = f'<div style="display:flex;gap:12px;flex-wrap:wrap">{"".join(book_cols)}</div>'

        pri_color = {"High": "#B83232", "Medium": "#c8b487", "Low": "#1B6F4A"}
        queue_items = sorted((wf_queue or []), key=lambda r: int(r.get("priority_rank", 99) or 99))[:8]
        queue_rows = ""
        for r in queue_items:
            pc = pri_color.get(str(r.get("priority", "")), "#999")
            action = str(r.get("sector_adjusted_action") or r.get("what_to_do", ""))[:100]
            queue_rows += f"""<tr style="border-bottom:1px solid #241f18">
              <td style="padding:7px 8px;font-size:11px;color:#AAA;font-weight:400">#{_esc(str(r.get('priority_rank','?')))}</td>
              <td style="padding:7px 8px" class="td-ticker">{_esc(r.get('ticker',''))}</td>
              <td style="padding:7px 8px;font-size:11.5px;color:#555">{_esc(r.get('sector',''))}</td>
              <td style="padding:7px 8px"><span style="font-size:10px;font-weight:400;color:{pc};letter-spacing:.5px;text-transform:uppercase;padding:2px 6px;background:{pc}18;border-radius:3px">{_esc(r.get('priority',''))}</span></td>
              <td style="padding:7px 8px;font-size:12px;color:#c8b487">{_esc(action)}</td>
            </tr>"""
        if not queue_rows:
            queue_rows = '<tr><td colspan="5" style="color:#AAA;text-align:center;padding:16px">No action queue — run the daily pipeline</td></tr>'

        drift_items = [p for p in (position_pnl or []) if p.get("aligned") is False]
        drift_html = ""
        if drift_items:
            drift_list = ", ".join(f'<strong>{_esc(p["ticker"])}</strong>' for p in drift_items[:6])
            drift_html = f'<div style="margin-bottom:16px;padding:12px 16px;background:#FEF9EC;border:1px solid #43391f;border-left:4px solid #c8b487;border-radius:6px"><p style="font-size:13px;font-weight:400;color:#c8b487;margin:0 0 4px">Signal direction changed — review needed</p><p style="font-size:12.5px;color:#555;margin:0">{drift_list} entered with a signal that no longer matches current direction.</p></div>'

        return f"""<div style="padding:24px;background:linear-gradient(135deg,#241f18 0%,#2a2418 100%);border:1px solid #241f18;border-radius:10px;margin-bottom:28px">
      <div style="display:flex;align-items:baseline;gap:12px;margin-bottom:4px">
        <p class="eyebrow" style="margin:0">Trade Command Center</p>
        <span style="font-size:11px;color:#AAA;font-style:italic">Paper trading only — no real money</span>
      </div>
      <h3 style="font-family:'Playfair Display',serif;font-size:20px;font-weight:400;color:#1A1A1A;margin:0 0 16px">Portfolio health by book &amp; today's action queue</h3>
      {drift_html}
      {book_section}
      <div style="margin-top:20px">
        <p style="font-size:11px;font-weight:400;color:#c8b487;text-transform:uppercase;letter-spacing:.7px;margin:0 0 8px">Today's action queue — top 8 by priority</p>
        <div style="overflow-x:auto;border:1px solid #241f18;border-radius:6px;background:#fff">
          <table style="width:100%;border-collapse:collapse">
            <thead><tr style="background:#241f18;border-bottom:1px solid #241f18">
              <th style="text-align:left;padding:8px 8px;color:#AAA;font-size:10px;font-weight:400;letter-spacing:.5px;text-transform:uppercase">#</th>
              <th style="text-align:left;padding:8px 8px;color:#AAA;font-size:10px;font-weight:400;letter-spacing:.5px;text-transform:uppercase">Ticker</th>
              <th style="text-align:left;padding:8px 8px;color:#AAA;font-size:10px;font-weight:400;letter-spacing:.5px;text-transform:uppercase">Sector</th>
              <th style="text-align:left;padding:8px 8px;color:#AAA;font-size:10px;font-weight:400;letter-spacing:.5px;text-transform:uppercase">Priority</th>
              <th style="text-align:left;padding:8px 8px;color:#AAA;font-size:10px;font-weight:400;letter-spacing:.5px;text-transform:uppercase">Action</th>
            </tr></thead>
            <tbody>{queue_rows}</tbody>
          </table>
        </div>
      </div>
    </div>"""

    def alpha_score_rows():
        if not alpha_scores:
            return "<tr><td colspan='8' style='color:#AAA'>No model scores yet — run the daily pipeline first.</td></tr>"
        sig_map = {"sig_regime_ml":"Model","sig_quality":"Health","sig_revision":"Analysts",
                   "sig_surprise":"Earnings","sig_sentiment":"Sentiment","sig_momentum":"Momentum"}
        _SIG_LABEL = {
            "BUY": "Buy", "LONG": "Buy", "STRONG BUY": "Strong buy",
            "SELL": "Sell", "SHORT": "Sell", "STRONG SELL": "Strong sell",
            "HOLD": "Hold", "NEUTRAL": "Neutral",
        }
        _CROWD_LABEL = {
            "WATCH": "Widely held ⚠", "HIGH": "Very widely held",
            "CLEAR": "Normal", "LOW": "Lightly held", "MEDIUM": "Moderately held",
        }
        portfolio_map = {p["ticker"]: p for p in position_pnl}
        rows = []
        for r in alpha_scores:
            sc = float(r.get("alpha_score", 0))
            bar_w = min(100, int(sc))
            sig_raw = str(r.get("signal","")).upper()
            sig = _SIG_LABEL.get(sig_raw, sig_raw.title() if sig_raw else "—")
            sig_c = "pos" if sig_raw in ("BUY","LONG","STRONG BUY") else ("neg" if sig_raw in ("SELL","SHORT","STRONG SELL") else "")
            crowd_raw = str(r.get("crowding_level","")).upper()
            crowd = _CROWD_LABEL.get(crowd_raw, crowd_raw.title() if crowd_raw else "—")
            crowd_color = "#B83232" if crowd_raw in ("WATCH","HIGH") else ("#1B6F4A" if crowd_raw in ("CLEAR","LOW") else "#999")
            sigs = []
            for k, label in sig_map.items():
                if k in r:
                    v = float(r[k]) if r[k] not in ("", None) else 0
                    sigs.append(f'<span style="font-size:10px;color:#999">{label}:<b style="color:{"#1B6F4A" if v>60 else "#B83232" if v<40 else "#666"}">{v:.0f}</b></span>')
            pm = portfolio_map.get(str(r.get("ticker", "")), {})
            if pm:
                pdir_txt = "▲ Long" if pm["side"] == "LONG" else "▼ Short"
                pdir_col = "#1B6F4A" if pm["side"] == "LONG" else "#B83232"
                raw_pnl = pm.get("pnl")
                if raw_pnl is None and pm.get("unrealized_ret"):
                    raw_pnl = float(pm["unrealized_ret"]) * 100
                ppnl_str = f"{raw_pnl:+.1f}%" if raw_pnl is not None else ""
                ppnl_col = "#1B6F4A" if (raw_pnl or 0) > 0 else "#B83232"
                book = pm.get("book", "")
                book_tag = f'<span style="font-size:9px;background:#2a2418;color:#c8b487;padding:1px 4px;border-radius:3px;margin-left:3px">{_esc(book)}</span>' if book and book != "—" else ""
                portfolio_cell = f'<span style="color:{pdir_col};font-weight:400;font-size:11px">{pdir_txt}</span>{book_tag} <span style="font-size:11px;color:{ppnl_col}">{ppnl_str}</span>'
            else:
                portfolio_cell = '<span style="color:#241f18;font-size:11px">—</span>'
            rows.append(f"""<tr>
              <td class="td-rank">#{_esc(r.get('alpha_rank',''))}</td>
              <td class="td-ticker">{_esc(r.get('ticker',''))}</td>
              <td style="font-size:11.5px;color:#555">{_esc(r.get('sector',''))}</td>
              <td><div class="td-score"><div class="score-bar-wrap"><div class="score-bar" style="width:{bar_w}%"></div></div><span style="font-weight:400;color:#c8b487">{sc:.1f}</span></div></td>
              <td><span class="{sig_c}" style="font-size:11px;font-weight:400">{_esc(sig)}</span></td>
              <td style="font-size:11px;color:{crowd_color};font-weight:400">{_esc(crowd)}</td>
              <td style="font-size:11px;line-height:1.4">{" &nbsp; ".join(sigs[:3])}</td>
              <td style="font-size:11px">{portfolio_cell}</td>
            </tr>""")
        return "\n".join(rows)

    def risk_gate_rows():
        if not risk_gate:
            return "<tr><td colspan='6' style='color:#AAA'>No position sizing data yet — run the daily pipeline first.</td></tr>"
        action_color = {"REDUCE_ONLY": "#B83232", "HOLD": "#c8b487", "OK": "#1B6F4A",
                        "SIZE_DOWN": "#B83232", "CLEAR": "#1B6F4A"}
        rows = []
        for r in risk_gate:
            action = str(r.get("final_risk_action",""))
            ac = action_color.get(action.upper(), "#999")
            action_plain = _RISK_ACTION_PLAIN.get(action.upper(), (action, "neu"))[0]
            cur_w = float(r.get("current_weight_pct", 0) or 0)
            rec_w = float(r.get("recommended_risk_weight_pct", 0) or 0)
            reason = _humanize_reason_stack(str(r.get("reason_stack","")))
            rows.append(f"""<tr>
              <td class="td-ticker">{_esc(r.get('ticker',''))}</td>
              <td style="font-size:12px;color:#555">{_esc(r.get('sector',''))}</td>
              <td class="r">{cur_w:.2f}%</td>
              <td class="r" style="color:#1B6F4A">{rec_w:.2f}%</td>
              <td><span style="font-size:11px;font-weight:400;color:{ac}">{_esc(action_plain)}</span></td>
              <td style="font-size:11px;color:#888">{_esc(reason)}</td>
            </tr>""")
        return "\n".join(rows)

    def drilldown_cards():
        if not ticker_drilldown:
            return "<p style='color:#AAA;font-size:13px'>No drilldown data. Run daily system.</p>"
        stage_color = {
            "RISK_REPAIR_REQUIRED":   "#B83232",
            "NON_RISK_GATES_REQUIRED":"#c8b487",
            "READY":                   "#1B6F4A",
            "WATCH":                   "#c8b487",
            "BLOCKED":                 "#B83232",
            "CLEAR":                   "#1B6F4A",
        }
        cards = []
        for r in ticker_drilldown:
            stage = str(r.get("current_stage",""))
            sc = stage_color.get(stage, "#999")
            stage_plain = _STAGE_PLAIN.get(stage, stage.replace("_"," ").title())
            score = float(r.get("readiness_score",0) or 0)
            cards.append(f"""<div style="background:#fff;border:1px solid #241f18;border-left:4px solid {sc};padding:16px 20px;margin-bottom:10px">
              <div style="display:flex;align-items:flex-start;justify-content:space-between;gap:12px;margin-bottom:8px">
                <div>
                  <span style="font-family:'Playfair Display',serif;font-size:18px;font-weight:400;color:#c8b487">{_esc(r.get('ticker',''))}</span>
                  <span style="font-size:11px;color:#999;margin-left:8px">{_esc(r.get('sector',''))}</span>
                </div>
                <div style="text-align:right;flex-shrink:0">
                  <p style="font-size:11px;color:{sc};font-weight:400">{_esc(stage_plain)}</p>
                  <p style="font-family:'Playfair Display',serif;font-size:20px;font-weight:400;color:#c8b487;line-height:1">{score:.0f}<span style="font-size:11px;color:#BBB">/100</span></p>
                </div>
              </div>
              <p style="font-size:13px;color:#333;line-height:1.55;margin-bottom:8px">{_esc(str(r.get('why_blocked_plain_english',''))[:300])}</p>
              <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;padding-top:10px;border-top:1px solid #241f18">
                <div><p style="font-size:10px;letter-spacing:1px;text-transform:uppercase;color:#BBB;font-weight:400;margin-bottom:3px">First blocker</p><p style="font-size:12px;color:#c8b487;font-weight:400">{_esc(r.get('first_blocking_gate',''))}</p></div>
                <div><p style="font-size:10px;letter-spacing:1px;text-transform:uppercase;color:#BBB;font-weight:400;margin-bottom:3px">How to clear it</p><p style="font-size:12px;color:#555">{_esc(str(r.get('first_clear_condition',''))[:120])}</p></div>
                <div><p style="font-size:10px;letter-spacing:1px;text-transform:uppercase;color:#BBB;font-weight:400;margin-bottom:3px">Current view</p><p style="font-size:12px;color:#555">{_esc(str(r.get('decision_room_summary',''))[:120])}</p></div>
                <div><p style="font-size:10px;letter-spacing:1px;text-transform:uppercase;color:#BBB;font-weight:400;margin-bottom:3px">Watch for</p><p style="font-size:12px;color:#c8b487;font-weight:400">{_esc(_clean_trigger(str(r.get('trigger_to_watch','')))[:120])}</p></div>
              </div>
            </div>""")
        return "\n".join(cards)

    def sector_cycle_rows():
        if not sector_cycle:
            return "<tr><td colspan='6' style='color:#AAA'>No sector data</td></tr>"
        state_color = {"Leadership expansion":"#1B6F4A","Crowded leadership":"#c8b487",
                       "Early improvement":"#5f7480","Downcycle / laggard":"#B83232","Neutral":"#999"}
        rows = []
        for r in sector_cycle:
            state = str(r.get("cycle_state",""))
            sc = state_color.get(state, "#666")
            ret20 = float(r.get("ret_20d_pct",0) or 0)
            ret63 = float(r.get("ret_63d_pct",0) or 0)
            wt = float(r.get("portfolio_weight_pct",0) or 0)
            cap = str(r.get("cap_status",""))
            _CAP_PLAIN = {
                "AT_CAP": "At sector cap", "OVER_CAP": "Over sector cap",
                "NEAR_CAP": "Near sector cap", "SECTOR_CAP_BREACH": "Cap breached",
            }
            cap_disp = _CAP_PLAIN.get(cap, cap.replace("_"," ").title()) if cap not in ("NO_POSITION","","nan") else ""
            rows.append(f"""<tr>
              <td class="td-ticker">{_esc(r.get('etf',''))}</td>
              <td style="font-size:12.5px;color:#333">{_esc(r.get('sector',''))}</td>
              <td><span style="font-size:11px;font-weight:400;color:{sc}">{_esc(state)}</span></td>
              <td class="r {'pos' if ret20>=0 else 'neg'}">{ret20:+.1f}%</td>
              <td class="r {'pos' if ret63>=0 else 'neg'}">{ret63:+.1f}%</td>
              <td class="r">{f"{wt:.1f}%" if wt else "—"}</td>
              <td style="font-size:11px;color:#B83232;font-weight:400">{_esc(cap_disp)}</td>
            </tr>""")
        return "\n".join(rows)

    def news_cards():
        if not news:
            return "<p style='color:#AAA;font-size:13px'>No news data available. Run the news fetcher step.</p>"
        import re as _re
        cards = []
        for idx, item in enumerate(news[:30]):
            tone_bg  = {"#1B6F4A": "#241f18", "#B83232": "#FDECEA"}.get(item["tone_color"], "#FEF9EC")
            catalysts = item.get("catalysts", "")
            risks_txt = item.get("risks", "")
            action = _esc(item.get("action_hint", ""))
            logic  = _esc(item.get("logic", ""))
            tk = item['ticker']
            link = item.get("link", "")
            summary = item.get("summary", "")
            bull_reasons = item.get("bullish_reasons", [])
            bear_reasons = item.get("bearish_reasons", [])
            pos_badge = ""
            if tk in _long_set:
                pos_badge = '<span class="news-pos-badge in-long">Currently in your buy list</span>'
            elif tk in _short_set:
                pos_badge = '<span class="news-pos-badge in-short">Currently in your avoid list</span>'
            # Clean up logic prefix
            logic_clean = _re.sub(r'^(Bullish|Bearish|Neutral) read:\s*[^:]+:\s*', '', logic).strip() if logic else ""
            if not logic_clean:
                logic_clean = logic
            card_id = f"nc{idx}"
            # Build expanded detail section
            detail_parts = []
            if summary:
                detail_parts.append(f'<p style="font-size:12.5px;color:#444;line-height:1.6;margin:10px 0 6px">{_esc(summary)}</p>')
            if bull_reasons:
                for r in bull_reasons[:2]:
                    detail_parts.append(f'<p style="font-size:11.5px;color:#1B6F4A;margin:3px 0">✓ {_esc(r)}</p>')
            if bear_reasons:
                for r in bear_reasons[:2]:
                    detail_parts.append(f'<p style="font-size:11.5px;color:#B83232;margin:3px 0">⚠ {_esc(r)}</p>')
            if catalysts and catalysts not in ("nan", ""):
                detail_parts.append(f'<p style="font-size:11.5px;color:#1B6F4A;margin:6px 0 3px"><strong>Why it could go up:</strong> {_esc(catalysts[:300])}</p>')
            if risks_txt and risks_txt not in ("nan", ""):
                detail_parts.append(f'<p style="font-size:11.5px;color:#B83232;margin:3px 0"><strong>Risks to watch:</strong> {_esc(risks_txt[:300])}</p>')
            if link:
                detail_parts.append(f'<a href="{_esc(link)}" target="_blank" rel="noopener" onclick="event.stopPropagation()" style="display:inline-block;margin-top:10px;font-size:12px;font-weight:400;color:#fff;background:#2a2418;text-decoration:none;padding:6px 16px;border-radius:3px">Open source article →</a>')
            detail_html = "".join(detail_parts)
            cards.append(f"""<div class="news-card {item['tone_class']}" onclick="toggleNews('{card_id}')" style="cursor:pointer">
              {pos_badge}
              <div style="display:flex;justify-content:space-between;align-items:flex-start">
                <div>
                  <p class="news-ticker">{_esc(tk)}</p>
                  <span class="news-tone" style="background:{tone_bg};color:{item['tone_color']}">{_esc(item['tone'])}</span>
                </div>
                <span id="{card_id}-arrow" style="font-size:11px;color:#c8b487;background:#F0F4F9;border:1px solid #283038;padding:3px 8px;border-radius:3px;font-weight:400;flex-shrink:0">Tap to expand ▼</span>
              </div>
              <p class="news-title">{_esc(item['title'])}</p>
              {f'<p class="news-summary">{_esc(logic_clean)}</p>' if logic_clean else ''}
              <p class="news-meta">{_esc(item.get('publisher',''))} &middot; {_esc(item.get('published',''))}</p>
              {f'<p class="news-action">→ {action}</p>' if action else ''}
              <div id="{card_id}" style="display:none;margin-top:10px;padding-top:10px;border-top:1px solid #241f18">
                {detail_html}
              </div>
            </div>""")
        return "\n".join(cards)

    def breadth_cards():
        items = macro_breadth.get("breadth", [])
        if not items:
            return "<p style='color:#AAA;font-size:13px'>No breadth data available.</p>"
        cards = []
        for b in items:
            above_label = []
            if b.get("above_20dma"): above_label.append("Above 20-day avg")
            if b.get("above_50dma"): above_label.append("Above 50-day avg")
            above_txt = " · ".join(above_label) if above_label else "Below key averages"
            cards.append(f"""<div class="breadth-card">
              <p class="breadth-ticker">{_esc(b['ticker'])}</p>
              <p class="breadth-name">{_esc(b['name'])}</p>
              <p class="breadth-val">${b['close']:,.2f}</p>
              <p style="font-size:12px;color:{'#1B6F4A' if b['ret_20d']>=0 else '#B83232'};font-weight:400">{b['ret_20d_str']} past month</p>
              <span class="breadth-badge {b['trend_class']}">{_esc(b['trend'])}</span>
              <p style="font-size:11px;color:#999;margin-top:6px">{above_txt}</p>
            </div>""")
        return "\n".join(cards)

    def rotation_cards():
        items = macro_breadth.get("rotation", [])
        if not items:
            return "<p style='color:#AAA;font-size:13px'>No sector trend data available yet.</p>"
        cards = []
        for r in items:
            r20 = r["ret_20d"] * 100
            r63 = r["ret_63d"] * 100
            cards.append(f"""<div class="rot-card">
              <div class="rot-etf">{_esc(r['ticker'])}</div>
              <div class="rot-body">
                <p class="rot-theme">{_esc(r['theme'])}</p>
                <p class="rot-rets">1 month: <strong style="color:{'#1B6F4A' if r20>=0 else '#B83232'}">{r20:+.1f}%</strong> &nbsp; 3 months: <strong style="color:{'#1B6F4A' if r63>=0 else '#B83232'}">{r63:+.1f}%</strong></p>
                <span class="rot-badge {r['label_class']}">{_esc(r['label'])}</span>
              </div>
            </div>""")
        return "\n".join(cards)

    def earnings_cal_cards():
        if not earnings_cal:
            return "<p style='color:#AAA;font-size:13px'>No upcoming earnings in the next 30 days.</p>"
        upcoming = [e for e in earnings_cal if (e.get("days_until") or -999) >= -5][:20]
        if not upcoming:
            upcoming = earnings_cal[:20]
        cards = []
        for e in upcoming:
            rec = _esc(e.get("recommended_action", ""))
            score = e.get("alpha_score", 0) or 0
            cards.append(f"""<div class="cal-card">
              <p class="cal-date">{_esc(e['earnings_date'])}</p>
              <p class="cal-ticker">{_esc(e['ticker'])}</p>
              <p class="cal-days">{_esc(e['days_label'])}</p>
              <p style="font-size:11.5px;color:#555">Model score: <strong>{score:.1f}</strong> / 100</p>
              <span class="cal-risk {e['risk_class']}">{_esc(e['risk_flag'])}</span>
              <p class="cal-action">{_esc(e['action'])}</p>
              {f'<p style="font-size:11px;color:#888;margin-top:4px">{rec}</p>' if rec else ''}
            </div>""")
        return "\n".join(cards)

    def ff5_cards():
        if not factor_attr.get("ff5"):
            return "<p style='color:#AAA'>Attribution data not yet available — run the daily pipeline first.</p>"
        FACTOR_PLAIN = {
            "beta_market":  ("Moves with the market by", "How closely the strategy follows the overall market. 1.0 = moves exactly with the market. 0 = completely independent."),
            "beta_value":   ("Leans toward cheap vs expensive stocks", "Positive = prefers undervalued companies. Negative = prefers growth/expensive stocks."),
            "beta_quality": ("Leans toward profitable companies", "Positive = prefers highly profitable companies. Usually positive = good."),
            "beta_size":    ("Leans toward big vs small companies", "Near 0 = neutral. Positive = small companies. Negative = large companies."),
            "beta_invest":  ("Leans toward cautious vs aggressive companies", "Positive = prefers companies that invest conservatively."),
        }
        cards = []
        for row in factor_attr["ff5"]:
            alpha_color = "#1B6F4A" if row["alpha_ann"] > 0 else "#B83232"
            factor_rows = ""
            for fkey, (fname, fdesc) in FACTOR_PLAIN.items():
                val = row.get(fkey, 0)
                vc  = "#1B6F4A" if val > 0 else "#B83232"
                factor_rows += f'<div class="ff5-factor-row"><span style="color:#999">{fname}</span><span style="font-weight:400;color:{vc}">{val:+.3f}</span></div>'
            cards.append(f"""<div class="ff5-card">
              <p class="ff5-window">{_esc(row['window'])}</p>
              <p class="ff5-alpha" style="color:{alpha_color}">{row['alpha_str']}</p>
              <p class="ff5-ir">Effectiveness score: <strong>{row['info_ratio']:.2f}</strong> &nbsp; How much the market explains: <strong>{row['r_squared']*100:.0f}%</strong></p>
              <div class="ff5-factors">{factor_rows}</div>
            </div>""")
        return "\n".join(cards)

    def signal_bars():
        sigs = factor_attr.get("signals", [])
        if not sigs:
            return "<p style='color:#AAA'>Signal contribution data not yet available.</p>"
        rows = []
        for s in sigs:
            pnl_color = "#1B6F4A" if s["pnl"] >= 0 else "#B83232"
            bar_color = "#1B6F4A" if s["pnl"] >= 0 else "#B83232"
            rows.append(f"""<div class="sig-bar-row">
              <span class="sig-name">{_esc(s['signal'])}</span>
              <div class="sig-bar-wrap"><div class="sig-bar" style="width:{s['bar_w']}%;background:{bar_color}"></div></div>
              <span class="sig-pct">{s['share_pct']:.1f}%</span>
              <span class="sig-pnl" style="color:{pnl_color}">{s['pnl_str']}</span>
            </div>""")
        return "\n".join(rows)

    def tearsheet_cards():
        s = factor_attr.get("summary", {})
        ITEMS = [
            ("How well does it hold up on bad days?",  s.get("Annualised Sortino","—"),       "S&P 500 scores ~0.90",             "good",  "This score only counts painful down moves — not normal day-to-day swings. Higher = the strategy loses less on its worst days."),
            ("How quickly does it recover from losses?",    s.get("Calmar Ratio","—"),        "above 1.0 = strong recovery",      "good",  "Annual return divided by the worst loss ever. A score above 1.0 means the strategy earns back more than it ever lost in a year."),
            ("Worst loss from peak (ever)",         s.get("Max Drawdown","—"),                "S&P 500 fell −57% in 2008",        "good",  "The biggest drop from any high point before recovering. This is what the worst period felt like. Smaller = more stable."),
            ("Extra return above S&P 500",   s.get("Total Alpha vs SPY","—"),                "cumulative outperformance",        "good",  "Total extra gain compared to just buying and holding the S&P 500 index. Pure outperformance, not counting market gains."),
            ("How often did it beat the market each month?",     s.get("Monthly Win Rate vs SPY","—"), "out of all months tested", "warn",  "Percentage of months where the strategy did better than just holding the index. Over 50% = more wins than losses."),
            ("Estimated trading costs",    s.get("Transaction Cost (total)","—"),             "over the full backtest period",    "warn",  "Estimated fees and price slippage from buying and selling. Lower is better. This is already subtracted from the return numbers shown."),
        ]
        cards = []
        for metric, val, bench, grade, explain in ITEMS:
            cards.append(f"""<div class="ts-card {grade}">
              <p class="ts-metric">{metric}</p>
              <p class="ts-val">{_esc(str(val))}</p>
              <p class="ts-bench">{bench}</p>
              <p class="ts-assess {grade}">{explain}</p>
            </div>""")
        return "\n".join(cards)

    def monthly_pnl_bars():
        mpnl = monthly_pnl
        if not mpnl.get("labels"):
            return ""
        lc = _j(mpnl["long_c"])
        sc = _j(mpnl["short_c"])
        ac = _j(mpnl["alpha_c"])
        mc = _j(mpnl["mkt_c"])
        labels = _j(mpnl["labels"])
        return f"""<div class="chart-box" style="margin-top:24px">
      <p class="chart-title">Monthly return breakdown — what drove each month's result</p>
      <p class="chart-sub">Green bars = extra return the model added beyond just owning the market. Blue = how buy positions contributed. Red = how avoid positions contributed. Grey = overall market movement that month.</p>
      <div class="chart-inner" style="height:260px"><canvas id="monthlyPnlChart"></canvas></div>
    </div>
    <script>
    (function() {{
      const el = document.getElementById('monthlyPnlChart');
      if (!el) return;
      new Chart(el.getContext('2d'), {{
        type: 'bar',
        data: {{
          labels: {labels},
          datasets: [
            {{label:'Buy positions',data:{lc},backgroundColor:'rgba(27,111,74,0.7)',stack:'s'}},
            {{label:'Avoid positions',data:{sc},backgroundColor:'rgba(184,50,50,0.6)',stack:'s'}},
            {{label:"Model's extra value",data:{ac},backgroundColor:'rgba(90,100,116,0.55)',stack:'a'}},
            {{label:'Market movement',data:{mc},backgroundColor:'rgba(180,175,165,0.5)',stack:'a'}},
          ]
        }},
        options: {{
          responsive:true,maintainAspectRatio:false,
          plugins:{{legend:{{position:'bottom',labels:{{font:{{size:11}},color:'#666'}}}},
            tooltip:{{backgroundColor:'#fff',titleColor:'#1A1A1A',bodyColor:'#555',borderColor:'#241f18',borderWidth:1,padding:10,
              callbacks:{{label: ctx => `  ${{ctx.dataset.label}}: ${{ctx.parsed.y.toFixed(2)}}%`}}}}
          }},
          scales:{{
            x:{{grid:{{display:false}},border:{{display:false}},ticks:{{color:'#BBB',font:{{size:10}}}}}},
            y:{{grid:{{color:'#241f18'}},border:{{display:false}},stacked:false,
              ticks:{{color:'#BBB',font:{{size:11}},callback: v => v.toFixed(1)+'%'}}}}
          }}
        }}
      }});
    }})();
    </script>"""

    def position_cards():
        if not position_pnl:
            return "<p style='color:#AAA;font-size:13px'>No open positions. Run the daily pipeline to load positions.</p>"
        cards = []
        for p in position_pnl:
            _POS_SIG_LABEL = {
                "BUY":"Buy","LONG":"Buy","STRONG BUY":"Strong buy",
                "SELL":"Sell","SHORT":"Sell","STRONG SELL":"Strong sell","HOLD":"Hold",
            }
            _CROWD_LABEL2 = {"WATCH":"Crowded ⚠","HIGH":"Heavily crowded","CLEAR":"Normal","LOW":"Uncrowded"}
            side_disp = "Long" if p["side"] == "LONG" else "Short"
            side_cls  = "long" if p["side"] == "LONG" else "short"
            pnl_color = "#1B6F4A" if (p["pnl"] or 0) > 0 else ("#B83232" if (p["pnl"] or 0) < 0 else "#c8b487")
            sig_raw   = str(p.get("signal","")).upper()
            sig_disp  = _POS_SIG_LABEL.get(sig_raw, sig_raw.title() if sig_raw else "—")
            sig_cls   = "good" if sig_raw in ("BUY","LONG","STRONG BUY") else ("bad" if sig_raw in ("SELL","SHORT","STRONG SELL") else "neu")
            crowd_raw = str(p.get("crowding","")).upper()
            crowd_disp = _CROWD_LABEL2.get(crowd_raw, crowd_raw.title() if crowd_raw else "—")
            crowd_cls = "bad" if crowd_raw in ("WATCH","HIGH") else "good"
            aligned_cls  = "yes" if p["aligned"] else ("no" if p["aligned"] is False else "unk")
            aligned_txt  = ("✓ Signal still aligned with position" if p["aligned"]
                           else ("⚠ Signal direction changed — review needed" if p["aligned"] is False
                           else "— Signal check pending"))
            risk_color = {"bad": "#B83232", "neu": "#c8b487", "good": "#1B6F4A"}.get(p["risk_cls"], "#999")
            stale_note = f'<p class="pos-stale">Price as of {p["price_date"]} — click ⟳ Refresh Now for latest</p>' if p["stale"] > 0 else ""
            curr_str = f"${p['curr']:,.2f}" if p["curr"] else "—"
            entry_str = f"${p['entry']:,.2f}" if p["entry"] else "—"
            book = str(p.get("book", "—"))
            pred_mu = p.get("predicted_mu", 0) or 0
            alpha_vs_pred = p.get("alpha_vs_pred", 0) or 0
            mv = p.get("market_value", 0) or 0
            action_rec = str(p.get("action_rec", "—"))
            action_pri = str(p.get("action_priority", "—"))
            unrealized_pnl_usd = p.get("unrealized_pnl_usd", 0) or 0
            book_color = {"SHORT": "#B83232", "MEDIUM": "#c8b487", "LONG": "#3a3128"}.get(book.upper(), "#999")
            pred_mu_str = f"{pred_mu*100:+.1f}%" if pred_mu else "—"
            alpha_vs_pred_str = f"{alpha_vs_pred*100:+.1f}%" if alpha_vs_pred else "—"
            avp_color = "#1B6F4A" if alpha_vs_pred > 0 else "#B83232"
            mv_str = f"${mv:,.0f}" if mv else "—"
            upnl_color = "#1B6F4A" if unrealized_pnl_usd > 0 else "#B83232"
            upnl_str = f"${unrealized_pnl_usd:+,.0f}" if unrealized_pnl_usd else "—"
            action_color = "#B83232" if "SELL" in action_pri.upper() or "REDUCE" in action_pri.upper() else ("#1B6F4A" if "BUY" in action_pri.upper() else "#c8b487")
            book_badge = f'<span style="font-size:10px;font-weight:400;background:{book_color};color:#fff;padding:2px 6px;border-radius:3px;margin-left:6px">{_esc(book)}</span>' if book != "—" else ""
            pred_row = "" if pred_mu == 0 else f'<div style="display:flex;gap:16px;margin-top:8px;padding:8px 10px;background:#241f18;border-radius:6px"><div><span style="font-size:10px;color:#AAA">Model predicted</span><br><span style="font-size:13px;font-weight:400;color:#c8b487">{pred_mu_str}</span></div><div><span style="font-size:10px;color:#AAA">vs actual (alpha)</span><br><span style="font-size:13px;font-weight:400;color:{avp_color}">{alpha_vs_pred_str}</span></div><div><span style="font-size:10px;color:#AAA">Market value</span><br><span style="font-size:13px;font-weight:400;color:#c8b487">{mv_str}</span></div><div><span style="font-size:10px;color:#AAA">Unrealized P&amp;L $</span><br><span style="font-size:13px;font-weight:400;color:{upnl_color}">{upnl_str}</span></div></div>'
            action_row = "" if action_rec == "—" else f'<div style="margin-top:8px;padding:8px 10px;border-left:3px solid {action_color};background:#241f18;border-radius:0 6px 6px 0"><span style="font-size:10px;color:#AAA;text-transform:uppercase;letter-spacing:.5px">Today\'s action</span><br><span style="font-size:12px;color:#1A1A1A">{_esc(action_rec[:120])}</span></div>'
            cards.append(f"""<div class="pos-card {side_cls}">
              <div class="pos-header">
                <div>
                  <span class="pos-ticker">{_esc(p['ticker'])}</span>
                  {book_badge}
                  <span style="font-size:11px;color:#999;margin-left:8px">{_esc(p['sector'])}</span>
                </div>
                <span class="pos-side {side_cls}">{side_disp}</span>
              </div>
              <div class="pos-prices">
                <div class="pos-price-item"><label>Entry price</label><span class="pos-price-val">{entry_str}</span></div>
                <div class="pos-price-item"><label>Latest price</label><span class="pos-price-val">{curr_str}</span></div>
                <div class="pos-price-item"><label>Unrealized P&amp;L</label><span class="pos-price-val" style="color:{pnl_color}">{_esc(p['pnl_str'])}</span></div>
              </div>
              {pred_row}
              <div class="pos-signals">
                <span class="pos-badge {sig_cls}">Signal: {_esc(sig_disp)}</span>
                <span class="pos-badge {'good' if p['alpha_score'] > 65 else 'neu'}">Model score: {p['alpha_score']:.1f}/100</span>
                <span class="pos-badge {crowd_cls}">Crowding: {_esc(crowd_disp)}</span>
              </div>
              <p class="pos-aligned {aligned_cls}">{aligned_txt}</p>
              <div class="pos-risk" style="border-left:3px solid {risk_color}">
                <strong style="color:{risk_color}">{_esc(p['risk_plain'])}</strong>
                <br><span style="font-size:11px;color:#888">{_esc(p['risk_reason'])}</span>
              </div>
              {action_row}
              {stale_note}
            </div>""")
        return "\n".join(cards)

    def crowding_panel():
        ft = crowding.get("factor_trend", [])
        latest = ft[-1] if ft else {}
        beta    = latest.get("beta", 0)
        mom     = latest.get("momentum", 0)
        beta_cls  = "bad" if beta > 1.1 else ("warn" if beta > 0.9 else "good")
        mom_cls   = "bad" if mom > 1.2  else ("warn" if mom > 0.8  else "good")
        beta_note = "Very high — this portfolio moves almost identically to the overall market" if beta > 1.1 else ("High — most of the portfolio's moves are driven by the overall market" if beta > 0.9 else "Normal — the portfolio has some independence from the overall market")
        mom_note  = "Very high — the portfolio is heavily concentrated in the same stocks many other funds own" if mom > 1.2 else ("Elevated — significant overlap with other quantitative funds' holdings" if mom > 0.8 else "Normal — reasonable spread, not too concentrated in any one style")

        ls = crowding.get("long_semis", [])
        ss = crowding.get("short_semis", [])
        semi_warn = ""
        if len(ls) >= 3:
            semi_warn = f'<div style="background:#FEF9EC;border:1px solid #43391f;border-left:4px solid #c8b487;padding:12px 16px;margin-bottom:12px"><p style="font-size:13px;font-weight:400;color:#c8b487;margin-bottom:4px">⚠ Semiconductor concentration — {len(ls)} long positions</p><p style="font-size:12.5px;color:#555">{", ".join(ls)} are all semiconductors. If the sector sells off, all {len(ls)} move together. Consider whether this is intentional.</p></div>'

        crowd_watch = crowding.get("watch_tickers", [])
        watch_warn = ""
        if crowd_watch:
            watch_warn = f'<div style="background:#FDECEA;border:1px solid #3a2724;border-left:4px solid #B83232;padding:12px 16px;margin-bottom:12px"><p style="font-size:13px;font-weight:400;color:#B83232;margin-bottom:4px">Widely held by other funds — watch closely: {", ".join(crowd_watch)}</p><p style="font-size:12.5px;color:#555">These stocks are heavily owned by many quant and hedge funds at the same time. If those funds all sell at once, the price can drop sharply and fast — even if the company itself is fine.</p></div>'

        sc = crowding.get("sector_concentration", {})
        sc_rows = "".join(f'<div style="display:flex;justify-content:space-between;padding:6px 0;border-bottom:1px solid #241f18"><span style="font-size:12.5px;color:#333">{_esc(k)}</span><span style="font-size:12px;font-weight:400;color:#c8b487">{v} positions</span></div>' for k,v in sorted(sc.items(), key=lambda x:-x[1]))

        return f"""
      <div class="crowd-kpi-row">
        <div class="crowd-kpi {beta_cls}">
          <p class="crowd-label">Moves with market</p>
          <p class="crowd-val" style="color:{'#B83232' if beta_cls=='bad' else '#c8b487' if beta_cls=='warn' else '#1B6F4A'}">{beta:.2f}</p>
          <p class="crowd-note">{beta_note}</p>
        </div>
        <div class="crowd-kpi {mom_cls}">
          <p class="crowd-label">Overlap with other funds</p>
          <p class="crowd-val" style="color:{'#B83232' if mom_cls=='bad' else '#c8b487' if mom_cls=='warn' else '#1B6F4A'}">{mom:.2f}</p>
          <p class="crowd-note">{mom_note}</p>
        </div>
        <div class="crowd-kpi {'bad' if crowd_watch else 'good'}">
          <p class="crowd-label">Widely held stocks</p>
          <p class="crowd-val" style="color:{'#B83232' if crowd_watch else '#1B6F4A'}">{len(crowd_watch)}</p>
          <p class="crowd-note">{"Watch these — owned by many other funds: " + ", ".join(crowd_watch) if crowd_watch else "No positions flagged as widely held"}</p>
        </div>
      </div>
      {watch_warn}
      {semi_warn}
      <div style="background:#fff;border:1px solid #241f18;padding:18px 20px;margin-top:16px">
        <p style="font-size:12px;font-weight:400;color:#1A1A1A;margin-bottom:8px">How many buy positions are in each industry sector</p>
        {sc_rows if sc_rows else '<p style="color:#AAA;font-size:12px">No data</p>'}
      </div>"""

    chart_labels = _j(chart.get("labels", []))
    chart_ml     = _j(chart.get("ml", []))
    chart_spy    = _j(chart.get("spy", []))
    final_ml     = chart.get("final_ml", 0)
    final_spy    = chart.get("final_spy", 0)

    # backtest monthly chart data
    bt_labels     = _j(bt_monthly.get("labels", []))
    bt_strat      = _j(bt_monthly.get("strat", []))
    bt_spy        = _j(bt_monthly.get("spy", []))
    bt_alpha      = _j(bt_monthly.get("alpha", []))
    bt_strat_cum  = _j(bt_monthly.get("strat_cum", []))
    bt_bench_cum  = _j(bt_monthly.get("bench_cum", []))
    bt_win_rate   = bt_monthly.get("win_rate", 0)
    bt_months     = bt_monthly.get("total_months", 0)
    bt_final_strat = bt_monthly.get("final_strat_cum", 0)
    bt_final_bench = bt_monthly.get("final_bench_cum", 0)

    # paper NAV chart data
    pn_labels  = _j(paper_nav.get("labels", []))
    pn_nav     = _j(paper_nav.get("nav", []))
    pn_hwm     = _j(paper_nav.get("hwm", []))
    pn_dd      = _j(paper_nav.get("dd", []))
    pn_start   = paper_nav.get("start", 0)
    pn_final   = paper_nav.get("final", 0)
    pn_gain    = paper_nav.get("gain", 0)
    pn_maxdd   = paper_nav.get("max_dd", 0)
    pn_ndays   = paper_nav.get("n_days", 0)
    pn_color   = "#1B6F4A" if pn_gain >= 0 else "#B83232"

    oos_ic     = summ.get("oos_ic", 0)
    oos_t      = summ.get("oos_t", 0)
    oos_sharpe = summ.get("oos_sharpe", 0)
    oos_dd     = summ.get("oos_dd", 0)
    oos_wr     = summ.get("oos_wr", 0)
    oos_ret    = summ.get("oos_ret", 0)
    spy_ret    = summ.get("spy_ret", 0)
    is_sharpe  = summ.get("is_sharpe", 0)
    is_ic      = summ.get("is_ic", 0)
    is_t       = summ.get("is_t", 35.83)
    is_dd      = summ.get("is_dd", -20.66)
    is_wr      = summ.get("is_wr", 93.2)
    annual_rets = bt_monthly.get("annual_rets", {})

    # Rolling IC chart data
    ric_labels   = _j(rolling_ic.get("labels", []))
    ric_3m       = _j(rolling_ic.get("ic_3m", []))
    ric_6m       = _j(rolling_ic.get("ic_6m", []))
    ric_target   = rolling_ic.get("target", 0.370)
    ric_cur      = rolling_ic.get("current_3m") or 0
    ric_status_raw = rolling_ic.get("current_status", "—")
    # Live IC display helpers
    _ric_last_date = rolling_ic.get("labels", ["—"])[-1] if rolling_ic.get("labels") else "—"
    _live_ic_color = "#6BCCA0" if ric_cur > 0.10 else ("#c8b487" if ric_cur >= 0 else "#B83232")
    _live_ic_label = "Signal healthy" if ric_cur > 0.10 else ("Weakening — watch" if ric_cur >= 0 else "Alert — model degraded")
    _RIC_STATUS_HUMAN = {
        "IC_OK":             "Signal is healthy",
        "IC_WARN":           "Signal weakening — watch",
        "IC_ALERT":          "Signal degraded — review",
        "insufficient_data": "Not enough data yet",
        "OK":                "Signal healthy",
        "WARN":              "Signal weakening",
        "ALERT":             "Signal degraded",
    }
    ric_status = _RIC_STATUS_HUMAN.get(ric_status_raw, ric_status_raw.replace("_"," "))
    ric_status_class = "ok" if "OK" in ric_status_raw.upper() else ("warn" if "WARN" in ric_status_raw.upper() else "alert")
    ric_fac_labels = _j(rolling_ic.get("factor_labels", []))
    ric_mom      = _j(rolling_ic.get("factor_ic", {}).get("Momentum", []))
    ric_lowvol   = _j(rolling_ic.get("factor_ic", {}).get("LowVol", []))
    ric_value    = _j(rolling_ic.get("factor_ic", {}).get("Value", []))
    # Monthly PnL tearsheet KPIs
    mpnl_avg    = monthly_pnl.get("avg_alpha", 0)
    mpnl_best   = monthly_pnl.get("best_month", 0)
    mpnl_worst  = monthly_pnl.get("worst_month", 0)
    mpnl_wins   = monthly_pnl.get("long_win_months", 0)
    mpnl_total  = monthly_pnl.get("total_months", 0)
    mpnl_wr     = round(mpnl_wins / mpnl_total * 100, 1) if mpnl_total else 0

    _html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <meta name="apple-mobile-web-app-capable" content="yes">
  <meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
  <meta name="apple-mobile-web-app-title" content="Canyon">
  <meta name="theme-color" content="#231a12">
  <link rel="manifest" href="/manifest.json">
  <script>if('serviceWorker'in navigator)navigator.serviceWorker.register('/sw.js').catch(function(){{}});</script>
  <title>Canyon Quant v9 — Research</title>
  <link href="https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,600;0,700;1,400&family=Source+Sans+3:wght@300;400;600&display=swap" rel="stylesheet">
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
  <style>
    *,*::before,*::after{{box-sizing:border-box;margin:0;padding:0}}
    html{{font-size:16px;scroll-behavior:smooth}}
    body{{font-family:'Source Sans 3','Helvetica Neue',Arial,sans-serif;background:#FAFAF8;color:#1A1A1A;line-height:1.65}}
    h1,h2,h3,h4{{font-family:'Playfair Display',Georgia,serif}}
    .container{{max-width:1080px;margin:0 auto;padding:0 48px}}

    /* NAV */
    nav{{background:#2a2418;position:sticky;top:0;z-index:200;border-bottom:2px solid #c8b487}}
    nav .inner{{max-width:1080px;margin:0 auto;padding:0 48px;display:flex;align-items:stretch;justify-content:space-between;height:54px}}
    .nav-brand{{display:flex;align-items:center;color:#fff;font-family:'Playfair Display',serif;font-size:16px;font-weight:400;letter-spacing:1px;text-decoration:none;flex-shrink:0;gap:4px}}
    .nav-brand span{{color:#c8b487}}
    .nav-tabs{{display:flex;align-items:stretch;overflow:visible}}
    .nav-tabs a{{display:flex;align-items:center;padding:0 16px;color:rgba(255,255,255,.55);text-decoration:none;font-size:11px;font-weight:400;letter-spacing:1.2px;text-transform:uppercase;border-left:1px solid rgba(255,255,255,.07);cursor:pointer;white-space:nowrap;transition:color .15s,background .15s;border-bottom:3px solid transparent}}
    .nav-tabs a:hover{{color:#fff;background:rgba(255,255,255,.05)}}
    .nav-tabs a.active{{color:#fff;border-bottom-color:#c8b487}}
    /* Dropdown groups */
    .nav-group{{position:relative;display:flex;align-items:stretch}}
    .nav-group-btn{{display:flex;align-items:center;padding:0 16px;color:rgba(255,255,255,.55);text-decoration:none;font-size:11px;font-weight:400;letter-spacing:1.2px;text-transform:uppercase;border-left:1px solid rgba(255,255,255,.07);cursor:pointer;white-space:nowrap;transition:color .15s,background .15s;border-bottom:3px solid transparent;user-select:none}}
    .nav-group:hover .nav-group-btn,.nav-group-btn:hover{{color:#fff;background:rgba(255,255,255,.05)}}
    .nav-group.active .nav-group-btn{{color:#fff;border-bottom-color:#c8b487}}
    .nav-dropdown{{display:none;position:absolute;top:100%;left:0;min-width:190px;background:#0F1B32;border:1px solid rgba(255,255,255,.12);border-top:2px solid #c8b487;border-radius:0 0 6px 6px;box-shadow:0 8px 28px rgba(0,0,0,.5);z-index:9999}}
    .nav-dropdown.open{{display:block}}
    .nav-dropdown a{{display:block;padding:10px 18px;color:rgba(255,255,255,.6);font-size:11px;font-weight:400;letter-spacing:.8px;text-transform:uppercase;border-left:none!important;border-bottom:1px solid rgba(255,255,255,.05)!important;text-decoration:none;cursor:pointer;white-space:nowrap;transition:background .12s,color .12s}}
    .nav-dropdown a:last-child{{border-bottom:none!important}}
    .nav-dropdown a:hover,.nav-dropdown a.active{{background:rgba(184,148,63,.18);color:#c8b487}}
    .nav-date{{display:flex;align-items:center;color:rgba(255,255,255,.38);font-size:11px;flex-shrink:0;padding-left:20px}}

    /* TABS */
    .tab-section{{display:none;padding:60px 0}}
    .tab-section.active{{display:block}}

    /* TYPOGRAPHY */
    .eyebrow{{font-size:11px;letter-spacing:2.5px;text-transform:uppercase;color:#c8b487;font-weight:400;margin-bottom:10px}}
    .section-head{{font-size:32px;color:#1A1A1A;line-height:1.15;font-weight:400;margin-bottom:8px}}
    .rule{{width:40px;height:2px;background:#c8b487;margin:14px 0 26px}}
    .lead{{font-size:15px;color:#666;max-width:640px;margin-bottom:30px;font-weight:300;line-height:1.8}}
    .prose{{font-size:15px;line-height:1.85;color:#2D2D2D}}
    .prose p{{margin-bottom:18px}}

    /* HERO STRIP */
    .hero{{background:#2a2418;padding:56px 0 48px}}
    .hero-eye{{font-size:11px;letter-spacing:2.5px;text-transform:uppercase;color:#c8b487;font-weight:400;margin-bottom:14px}}
    .hero h1{{font-size:52px;color:#fff;line-height:1.1;font-weight:400}}
    .hero-sub{{font-size:52px;color:#c8b487;font-style:italic;display:block;line-height:1.1;margin-bottom:16px}}
    .hero-desc{{color:rgba(255,255,255,.55);font-size:16px;max-width:560px;margin-bottom:44px;font-weight:300;line-height:1.7}}
    .kpi-grid{{display:grid;grid-template-columns:repeat(5,1fr);gap:1px;background:rgba(255,255,255,.10)}}
    .kpi{{background:#2a2418;padding:24px;border-left:1px solid rgba(255,255,255,.08)}}
    .kpi:first-child{{border-left:none}}
    .kpi-label{{font-size:10px;letter-spacing:1.8px;text-transform:uppercase;color:rgba(255,255,255,.45);margin-bottom:8px;font-weight:400}}
    .kpi-val{{font-family:'Playfair Display',serif;font-size:38px;color:#fff;line-height:1;font-weight:400}}
    .kpi-val.g{{color:#6BCCA0}}
    .kpi-note{{font-size:11px;color:rgba(255,255,255,.32);margin-top:6px}}

    /* TODAY */
    .today-hero{{display:grid;grid-template-columns:1fr 1fr 1fr 1fr;gap:14px;margin-bottom:36px}}
    .today-card{{background:#fff;border:1px solid #241f18;padding:20px 22px}}
    .today-card-label{{font-size:10px;letter-spacing:1.5px;text-transform:uppercase;color:#999;font-weight:400;margin-bottom:6px}}
    .today-card-val{{font-family:'Playfair Display',serif;font-size:28px;font-weight:400;line-height:1}}
    .today-card-note{{font-size:11px;color:#AAA;margin-top:5px}}
    .two-col-65{{display:grid;grid-template-columns:1.3fr 1fr;gap:32px;align-items:start}}
    .two-col-even{{display:grid;grid-template-columns:1fr 1fr;gap:20px;align-items:start}}

    /* TABLES */
    .tbl-wrap{{margin-top:8px}}
    .tbl-title{{font-size:11px;letter-spacing:1.5px;text-transform:uppercase;color:#999;font-weight:400;margin-bottom:8px}}
    table{{width:100%;border-collapse:collapse;font-size:13.5px}}
    thead th{{text-align:left;padding:7px 12px;font-size:10px;letter-spacing:1.2px;text-transform:uppercase;color:#999;font-weight:400;border-bottom:2px solid #3a3128;white-space:nowrap}}
    thead th.r{{text-align:right}}
    tbody tr{{border-bottom:1px solid #241f18}}
    tbody tr:hover{{background:#241f18}}
    tbody tr:last-child{{border-bottom:2px solid #3a3128}}
    tbody td{{padding:9px 12px}}
    tbody td.r{{text-align:right;font-variant-numeric:tabular-nums}}
    .pos{{color:#1B6F4A;font-weight:400}}
    .neg{{color:#B83232;font-weight:400}}
    tr.tr-strong{{background:#F7FCF9}}
    .td-ticker{{font-weight:400;color:#c8b487;font-family:'Playfair Display',serif;font-size:15px}}
    .td-rank{{color:#BBB;font-size:11px;width:30px}}
    .td-score{{display:flex;align-items:center;gap:8px}}
    .score-bar-wrap{{width:60px;height:5px;background:#241f18;border-radius:2px;flex-shrink:0}}
    .score-bar{{height:100%;background:#2a2418;border-radius:2px}}
    .tbl-note{{font-size:11px;color:#BBB;margin-top:8px;line-height:1.6}}

    /* CHART */
    .chart-box{{background:#fff;border:1px solid #241f18;padding:24px 24px 16px;margin-top:28px}}
    .chart-title{{font-size:14px;font-weight:400;color:#1A1A1A;margin-bottom:2px}}
    .chart-sub{{font-size:12px;color:#AAA;margin-bottom:18px}}
    .chart-inner{{position:relative;height:300px}}

    /* IC STACK */
    .ic-stack{{display:flex;flex-direction:column;gap:8px}}
    .ic-row{{display:flex;align-items:center;gap:12px;padding:9px 12px;background:#fff;border:1px solid #241f18}}
    .ic-name{{font-size:12.5px;font-weight:400;color:#1A1A1A;width:200px;flex-shrink:0}}
    .ic-step{{font-size:10px;color:#CCC;width:48px;flex-shrink:0}}
    .ic-bar-wrap{{flex:1;height:6px;background:#241f18;border-radius:2px;overflow:hidden}}
    .ic-bar{{height:100%;border-radius:2px}}
    .ic-bar.s{{background:#1B6F4A}}.ic-bar.m{{background:#c8b487}}.ic-bar.w{{background:#CCC}}
    .ic-val{{font-family:'Playfair Display',serif;font-size:15px;font-weight:400;color:#c8b487;width:52px;text-align:right;flex-shrink:0}}
    .ic-badge{{font-size:10px;letter-spacing:.8px;text-transform:uppercase;font-weight:400;padding:2px 6px;border-radius:2px;flex-shrink:0}}
    .b-s{{background:#EAF5EE;color:#1B6F4A}}.b-m{{background:#FEF5E7;color:#c8b487}}.b-w{{background:#F3F3F3;color:#999}}

    /* RISK LADDER */
    .risk-ladder{{display:flex;flex-direction:column;gap:0}}
    .rl-row{{display:flex;align-items:stretch;border:1px solid #241f18;border-bottom:none;background:#fff}}
    .rl-row:last-child{{border-bottom:1px solid #241f18}}
    .rl-row:hover{{background:#F9F8F6}}
    .rl-num{{width:48px;flex-shrink:0;display:flex;align-items:center;justify-content:center;font-family:'Playfair Display',serif;font-size:20px;font-weight:400;border-right:1px solid #241f18}}
    .l1{{color:#1B6F4A;background:#EAF5EE}}.l2{{color:#2A6F5A;background:#E5F3EE}}.l3{{color:#3A6F4A;background:#EBF5EC}}
    .l4{{color:#c8b487;background:#FEF8EC}}.l5{{color:#c8b487;background:#FDF5E4}}.l6{{color:#8B6914;background:#FDF0D0}}
    .l7{{color:#5f7480;background:#EFF6FF}}.l8{{color:#9333EA;background:#F5F3FF}}
    .l9{{color:#DC2626;background:#FEF2F2}}.l10{{color:#c8b487;background:#2a2418}}
    .rl-body{{flex:1;padding:12px 16px}}
    .rl-name{{font-size:13px;font-weight:400;color:#1A1A1A;margin-bottom:1px}}
    .rl-step{{font-size:10px;color:#BBB;margin-bottom:3px}}
    .rl-desc{{font-size:11.5px;color:#666;line-height:1.5}}
    .rl-rule{{flex-shrink:0;width:170px;padding:12px 14px;border-left:1px solid #241f18;display:flex;flex-direction:column;justify-content:center}}
    .rl-rule-label{{font-size:9px;letter-spacing:1.2px;text-transform:uppercase;color:#BBB;font-weight:400;margin-bottom:2px}}
    .rl-rule-val{{font-size:12px;color:#334155;font-weight:400;line-height:1.4}}

    /* FACTOR */
    .fac-grid{{display:grid;grid-template-columns:1fr 1fr;gap:14px}}
    .fac{{background:#fff;border:1px solid #241f18;padding:18px 20px}}
    .fac-name{{font-size:13px;font-weight:400;color:#1A1A1A;margin-bottom:4px}}
    .fac-ic{{font-family:'Playfair Display',serif;font-size:26px;font-weight:400;line-height:1;margin-bottom:5px}}
    .fac-sub{{font-size:11.5px;color:#999;line-height:1.45}}
    .fac-regimes{{display:flex;gap:14px;margin-top:10px;padding-top:8px;border-top:1px solid #241f18}}
    .fac-reg{{font-size:11px;text-align:center}}
    .fac-reg-label{{color:#BBB;letter-spacing:.5px;text-transform:uppercase;font-size:9px}}
    .fac-reg-val{{font-weight:400;margin-top:2px;font-size:12px}}
    .bull{{color:#1B6F4A}}.bear{{color:#B83232}}

    /* MACRO */
    .macro-grid{{display:grid;grid-template-columns:1fr 1fr 1fr;gap:14px;margin-top:16px}}
    .mac-card{{background:#fff;border:1px solid #241f18;padding:18px 20px}}
    .mac-ticker{{font-size:17px;font-weight:400;color:#c8b487;font-family:'Playfair Display',serif}}
    .mac-name{{font-size:11px;color:#999;margin-top:2px;margin-bottom:10px}}
    .mac-role{{font-size:11.5px;color:#555;line-height:1.45}}
    .mac-status{{margin-top:10px;padding-top:8px;border-top:1px solid #241f18;font-size:11px;font-weight:400;letter-spacing:.8px;text-transform:uppercase}}
    .risk-on{{color:#1B6F4A}}.neutral{{color:#c8b487}}.risk-off{{color:#B83232}}

    /* REGIME */
    .reg-grid{{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-bottom:24px}}
    .reg-card{{background:#fff;border:1px solid #241f18;padding:26px}}
    .reg-name{{font-size:10px;letter-spacing:1.5px;text-transform:uppercase;color:#999;font-weight:400;margin-bottom:5px}}
    .reg-pct{{font-family:'Playfair Display',serif;font-size:44px;font-weight:400;line-height:1;margin:6px 0}}
    .reg-info{{margin-top:12px;padding-top:10px;border-top:1px solid #241f18;font-size:12.5px;color:#555;line-height:1.7}}

    /* METHOD */
    .method-grid{{display:grid;grid-template-columns:1fr 1fr 1fr;gap:14px}}
    .method-card{{background:#fff;border:1px solid #241f18;padding:18px 20px;border-top:3px solid #3a3128}}
    .method-card.acc{{border-top-color:#c8b487}}
    .method-title{{font-size:13px;font-weight:400;color:#1A1A1A;margin-bottom:5px}}
    .method-body{{font-size:11.5px;color:#666;line-height:1.6}}
    .method-hl{{font-size:11px;font-weight:400;color:#c8b487;margin-top:8px;padding-top:6px;border-top:1px solid #241f18}}

    /* OOS ALERT */
    .oos-banner{{background:#F0F4F9;border:1px solid #283038;border-left:4px solid #3a3128;padding:16px 22px;margin-bottom:28px}}
    .oos-banner-title{{font-size:12px;font-weight:400;color:#c8b487;margin-bottom:4px;letter-spacing:.5px;text-transform:uppercase}}
    .oos-banner-body{{font-size:13px;color:#374151;line-height:1.6}}
    .oos-kpi-row{{display:flex;gap:32px;flex-wrap:wrap;margin-top:12px}}
    .oos-kpi label{{font-size:10px;letter-spacing:1.2px;text-transform:uppercase;color:#6B7280;font-weight:400;display:block;margin-bottom:2px}}
    .oos-kpi-val{{font-family:'Playfair Display',serif;font-size:22px;font-weight:400;color:#c8b487}}
    .oos-kpi-val.g{{color:#1B6F4A}}

    /* BUDGET */
    .budget-grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:14px}}
    .bud{{background:#fff;border:1px solid #241f18;padding:18px 20px}}
    .bud-label{{font-size:10px;letter-spacing:1.5px;text-transform:uppercase;color:#999;font-weight:400;margin-bottom:6px}}
    .bud-val{{font-family:'Playfair Display',serif;font-size:28px;font-weight:400;color:#c8b487;line-height:1;margin-bottom:4px}}
    .bud-note{{font-size:11px;color:#888;line-height:1.4}}
    .bud-trigger{{font-size:11px;color:#B83232;font-weight:400;margin-top:6px}}

    /* NEWS */
    .news-grid{{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-top:16px}}
    .news-card{{background:#fff;border:1px solid #241f18;padding:18px 20px;border-left:4px solid #241f18}}
    .news-card.pos{{border-left-color:#1B6F4A}}.news-card.neg{{border-left-color:#B83232}}.news-card.neu{{border-left-color:#c8b487}}
    .news-ticker{{font-size:11px;font-weight:400;color:#c8b487;letter-spacing:1px;text-transform:uppercase;margin-bottom:4px}}
    .news-title{{font-size:13.5px;font-weight:400;color:#1A1A1A;line-height:1.45;margin-bottom:6px}}
    .news-summary{{font-size:12px;color:#666;line-height:1.55;margin-bottom:8px}}
    .news-meta{{font-size:11px;color:#999}}
    .news-logic{{font-size:12px;color:#555;background:#241f18;border-radius:3px;padding:8px 10px;margin-top:8px;line-height:1.5}}
    .news-action{{font-size:11.5px;font-weight:400;color:#c8b487;margin-top:6px;padding-top:6px;border-top:1px solid #241f18}}
    .news-tone{{display:inline-block;font-size:10px;font-weight:400;letter-spacing:.5px;text-transform:uppercase;padding:2px 7px;border-radius:3px;margin-bottom:6px}}
    /* CALENDAR */
    .cal-grid{{display:grid;grid-template-columns:1fr 1fr 1fr 1fr;gap:12px;margin-top:16px}}
    .cal-card{{background:#fff;border:1px solid #241f18;padding:16px}}
    .cal-date{{font-size:10px;letter-spacing:1px;text-transform:uppercase;color:#999;margin-bottom:4px;font-weight:400}}
    .cal-ticker{{font-family:'Playfair Display',serif;font-size:24px;font-weight:400;color:#c8b487;line-height:1;margin-bottom:4px}}
    .cal-days{{font-size:12px;color:#555;margin-bottom:6px}}
    .cal-action{{font-size:12px;font-weight:400;color:#c8b487;margin-top:8px;padding-top:6px;border-top:1px solid #241f18}}
    .cal-risk{{font-size:10px;letter-spacing:.5px;text-transform:uppercase;font-weight:400;padding:2px 7px;border-radius:3px;display:inline-block;margin-top:4px}}
    .cal-risk.high{{background:#FDECEA;color:#B83232}}.cal-risk.low{{background:#241f18;color:#1B6F4A}}
    /* BREADTH */
    .breadth-grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-top:16px}}
    .breadth-card{{background:#fff;border:1px solid #241f18;padding:16px 18px}}
    .breadth-ticker{{font-size:13px;font-weight:400;color:#c8b487;margin-bottom:2px}}
    .breadth-name{{font-size:11px;color:#999;margin-bottom:8px}}
    .breadth-val{{font-family:'Playfair Display',serif;font-size:22px;font-weight:400;color:#1A1A1A;line-height:1;margin-bottom:4px}}
    .breadth-badge{{font-size:10px;font-weight:400;letter-spacing:.5px;text-transform:uppercase;padding:2px 7px;border-radius:3px;display:inline-block}}
    .breadth-badge.up{{background:#241f18;color:#1B6F4A}}.breadth-badge.dn{{background:#FDECEA;color:#B83232}}.breadth-badge.neu{{background:#FEF9EC;color:#c8b487}}
    /* SECTOR ROTATION */
    .rot-grid{{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-top:16px}}
    .rot-card{{background:#fff;border:1px solid #241f18;padding:14px 16px;display:flex;align-items:center;gap:14px}}
    .rot-etf{{font-family:'Playfair Display',serif;font-size:20px;font-weight:400;color:#c8b487;min-width:52px}}
    .rot-body{{flex:1}}
    .rot-theme{{font-size:11px;color:#999;margin-bottom:3px}}
    .rot-rets{{font-size:12px;color:#555}}
    .rot-badge{{font-size:10px;font-weight:400;letter-spacing:.5px;text-transform:uppercase;padding:2px 8px;border-radius:3px;display:inline-block;margin-top:4px}}
    .rot-badge.leader{{background:#241f18;color:#1B6F4A}}.rot-badge.neu{{background:#FEF9EC;color:#c8b487}}.rot-badge.lag{{background:#FDECEA;color:#B83232}}
    /* AUTO-REFRESH */
    .refresh-bar{{background:#F0F4F9;border:1px solid #283038;border-radius:4px;padding:8px 16px;font-size:12px;color:#6B7280;display:flex;justify-content:space-between;align-items:center;margin-bottom:20px}}
    .refresh-bar strong{{color:#c8b487}}

    /* POSITION HEALTH */
    .pos-grid{{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-top:16px}}
    .pos-card{{background:#fff;border:1px solid #241f18;padding:18px 20px;border-left:4px solid #241f18}}
    .pos-card.long{{border-left-color:#1B6F4A}}.pos-card.short{{border-left-color:#B83232}}
    .pos-header{{display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:10px}}
    .pos-ticker{{font-family:'Playfair Display',serif;font-size:22px;font-weight:400;color:#c8b487}}
    .pos-side{{font-size:10px;font-weight:400;letter-spacing:1px;text-transform:uppercase;padding:2px 8px;border-radius:3px}}
    .pos-side.long{{background:#241f18;color:#1B6F4A}}.pos-side.short{{background:#FDECEA;color:#B83232}}
    .pos-prices{{display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px;padding:10px 0;border-top:1px solid #241f18;border-bottom:1px solid #241f18;margin-bottom:10px}}
    .pos-price-item label{{font-size:10px;text-transform:uppercase;letter-spacing:.8px;color:#BBB;display:block;margin-bottom:2px;font-weight:400}}
    .pos-price-val{{font-family:'Playfair Display',serif;font-size:16px;font-weight:400;color:#1A1A1A}}
    .pos-signals{{display:flex;gap:10px;flex-wrap:wrap;margin-bottom:8px}}
    .pos-badge{{font-size:10px;font-weight:400;letter-spacing:.5px;text-transform:uppercase;padding:2px 8px;border-radius:3px;display:inline-block}}
    .pos-badge.good{{background:#241f18;color:#1B6F4A}}.pos-badge.bad{{background:#FDECEA;color:#B83232}}.pos-badge.neu{{background:#241f18;color:#666}}
    .pos-aligned{{font-size:11.5px;font-weight:400;margin-bottom:4px}}
    .pos-aligned.yes{{color:#1B6F4A}}.pos-aligned.no{{color:#B83232}}.pos-aligned.unk{{color:#999}}
    .pos-risk{{font-size:12px;color:#555;padding:8px 10px;background:#241f18;border-radius:3px;margin-top:6px;line-height:1.45}}
    .pos-stale{{font-size:10px;color:#c8b487;font-style:italic;margin-top:4px}}
    /* CROWDING */
    .crowd-kpi-row{{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin-bottom:20px}}
    .crowd-kpi{{background:#fff;border:1px solid #241f18;padding:18px;border-left:4px solid #241f18}}
    .crowd-kpi.warn{{border-left-color:#c8b487}}.crowd-kpi.bad{{border-left-color:#B83232}}.crowd-kpi.good{{border-left-color:#1B6F4A}}
    .crowd-label{{font-size:10px;letter-spacing:1.2px;text-transform:uppercase;color:#999;font-weight:400;margin-bottom:4px}}
    .crowd-val{{font-family:'Playfair Display',serif;font-size:26px;font-weight:400;line-height:1;margin-bottom:3px}}
    .crowd-note{{font-size:11.5px;color:#888;line-height:1.4}}

    /* ATTRIBUTION */
    .attr-kpi-row{{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin-bottom:28px}}
    .attr-kpi{{background:#fff;border:1px solid #241f18;padding:20px 22px}}
    .attr-kpi-label{{font-size:10px;letter-spacing:1.5px;text-transform:uppercase;color:#999;font-weight:400;margin-bottom:6px}}
    .attr-kpi-val{{font-family:'Playfair Display',serif;font-size:30px;font-weight:400;color:#c8b487;line-height:1;margin-bottom:4px}}
    .attr-kpi-sub{{font-size:11.5px;color:#888}}
    .ff5-grid{{display:grid;grid-template-columns:1fr 1fr 1fr;gap:14px;margin-top:16px}}
    .ff5-card{{background:#fff;border:1px solid #241f18;padding:20px}}
    .ff5-window{{font-size:10px;letter-spacing:1px;text-transform:uppercase;color:#999;font-weight:400;margin-bottom:4px}}
    .ff5-alpha{{font-family:'Playfair Display',serif;font-size:28px;font-weight:400;line-height:1;margin-bottom:4px}}
    .ff5-ir{{font-size:13px;font-weight:400;color:#555;margin-bottom:12px}}
    .ff5-factors{{font-size:11.5px;color:#555;line-height:1.7;padding-top:10px;border-top:1px solid #241f18}}
    .ff5-factor-row{{display:flex;justify-content:space-between;margin-bottom:2px}}
    .sig-bar-row{{display:flex;align-items:center;gap:12px;padding:10px 0;border-bottom:1px solid #241f18}}
    .sig-bar-row:last-child{{border-bottom:none}}
    .sig-name{{font-size:12.5px;color:#333;min-width:180px}}
    .sig-bar-wrap{{flex:1;background:#241f18;border-radius:2px;height:8px;overflow:hidden}}
    .sig-bar{{height:8px;border-radius:2px;background:#2a2418}}
    .sig-pct{{font-size:11px;font-weight:400;color:#c8b487;min-width:45px;text-align:right}}
    .sig-pnl{{font-size:11.5px;font-weight:400;min-width:55px;text-align:right}}
    .rolling-ic-status{{display:inline-block;font-size:10px;font-weight:400;letter-spacing:.5px;text-transform:uppercase;padding:2px 8px;border-radius:3px}}
    .rolling-ic-status.ok{{background:#241f18;color:#1B6F4A}}
    .rolling-ic-status.warn{{background:#FEF9EC;color:#c8b487}}
    .rolling-ic-status.alert{{background:#FDECEA;color:#B83232}}
    .tearsheet-grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin-top:16px}}
    .ts-card{{background:#fff;border:1px solid #241f18;padding:18px 20px;border-top:3px solid #3a3128}}
    .ts-card.good{{border-top-color:#1B6F4A}}.ts-card.warn{{border-top-color:#c8b487}}.ts-card.bad{{border-top-color:#B83232}}
    .ts-metric{{font-size:10px;letter-spacing:1.2px;text-transform:uppercase;color:#999;font-weight:400;margin-bottom:4px}}
    .ts-val{{font-family:'Playfair Display',serif;font-size:26px;font-weight:400;color:#1A1A1A;line-height:1;margin-bottom:4px}}
    .ts-bench{{font-size:11.5px;color:#888}}
    .ts-assess{{font-size:11px;font-weight:400;margin-top:6px;padding-top:6px;border-top:1px solid #241f18}}
    .ts-assess.good{{color:#1B6F4A}}.ts-assess.warn{{color:#c8b487}}.ts-assess.bad{{color:#B83232}}

    /* DAILY SUMMARY */
    .daily-summary{{background:#fff;border:1px solid #241f18;border-left:4px solid #3a3128;padding:18px 22px;margin-bottom:28px}}
    .daily-summary-text{{font-size:15px;color:#333;line-height:1.7;margin-top:6px}}
    /* NEWS POSITION BADGE */
    .news-pos-badge{{display:inline-block;font-size:10px;font-weight:400;letter-spacing:.5px;text-transform:uppercase;padding:2px 8px;border-radius:3px;margin-bottom:6px}}
    .news-pos-badge.in-long{{background:#241f18;color:#1B6F4A}}
    .news-pos-badge.in-short{{background:#FDECEA;color:#B83232}}
    /* CLICKABLE TICKERS */
    .td-ticker{{cursor:pointer}}
    .td-ticker:hover{{color:#c8b487!important;text-decoration:underline}}
    /* DRILLDOWN MODAL */
    #drilldown-modal{{display:none;position:fixed;top:0;left:0;width:100%;height:100%;background:rgba(0,0,0,.48);z-index:9000;overflow-y:auto;-webkit-overflow-scrolling:touch}}
    .dd-box{{max-width:700px;margin:52px auto 40px;background:#FAFAF8;border:1px solid #241f18;position:relative;padding:36px}}
    .dd-close{{position:absolute;top:14px;right:18px;background:none;border:none;font-size:24px;cursor:pointer;color:#AAA;line-height:1}}
    .dd-close:hover{{color:#1A1A1A}}
    .dd-ticker{{font-family:'Playfair Display',serif;font-size:36px;font-weight:400;color:#c8b487;line-height:1;margin-bottom:4px}}
    .dd-meta{{font-size:12px;color:#999;margin-bottom:20px}}
    .dd-kpi-row{{display:grid;grid-template-columns:1fr 1fr 1fr;gap:10px;margin-bottom:22px}}
    .dd-kpi{{background:#fff;border:1px solid #241f18;padding:14px;text-align:center}}
    .dd-kpi-label{{font-size:10px;letter-spacing:1px;text-transform:uppercase;color:#999;font-weight:400;margin-bottom:4px}}
    .dd-kpi-val{{font-family:'Playfair Display',serif;font-size:22px;font-weight:400;color:#c8b487}}
    .dd-section-title{{font-size:10px;letter-spacing:1.5px;text-transform:uppercase;color:#c8b487;font-weight:400;margin:18px 0 10px}}
    .dd-sig-row{{display:flex;align-items:center;gap:10px;padding:7px 0;border-bottom:1px solid #241f18}}
    .dd-sig-name{{font-size:12px;color:#555;min-width:150px}}
    .dd-sig-bar-wrap{{flex:1;background:#241f18;border-radius:2px;height:8px;overflow:hidden}}
    .dd-sig-bar{{height:8px;border-radius:2px;transition:width .3s}}
    .dd-sig-val{{font-size:11px;font-weight:400;color:#c8b487;min-width:36px;text-align:right}}
    .dd-news-item{{padding:8px 0;border-bottom:1px solid #241f18;font-size:12.5px}}
    .dd-news-title{{color:#1A1A1A;font-weight:400;margin-bottom:2px;line-height:1.4}}
    .dd-news-meta{{font-size:11px;color:#BBB}}
    /* MOBILE */
    @media(max-width:768px){{
      /* spacing */
      .container{{padding:0 12px}}
      nav .inner{{padding:0 12px}}
      .nav-date{{display:none}}
      /* nav bar — bigger touch targets */
      .nav-tabs a{{padding:0 14px;font-size:10px;min-height:44px}}
      nav .inner{{height:auto;min-height:44px}}
      /* hero */
      .hero{{padding:24px 0 20px}}
      .hero h1{{font-size:26px;line-height:1.2}}
      .hero-sub{{font-size:24px}}
      .hero-eye{{font-size:10px}}
      /* grids → single column on phone */
      .today-hero{{grid-template-columns:1fr 1fr;gap:8px}}
      .two-col-65{{grid-template-columns:1fr!important}}
      .kpi-grid{{grid-template-columns:1fr 1fr}}
      .ff5-grid{{grid-template-columns:1fr}}
      .attr-kpi-row{{grid-template-columns:1fr 1fr}}
      .tearsheet-grid{{grid-template-columns:1fr 1fr}}
      .pos-grid{{grid-template-columns:1fr}}
      .crowd-kpi-row{{grid-template-columns:1fr 1fr}}
      .stress-grid{{grid-template-columns:1fr}}
      .news-grid{{grid-template-columns:1fr}}
      .rot-grid{{grid-template-columns:1fr}}
      /* typography */
      .section-head{{font-size:20px}}
      .kpi-val{{font-size:24px}}
      /* tables — force horizontal scroll so they never overflow viewport */
      .table-scroll-wrap{{overflow-x:auto;-webkit-overflow-scrolling:touch}}
      table{{font-size:11px;min-width:500px}}
      th,td{{padding:6px 7px!important}}
      /* cards */
      .method-card{{padding:14px 14px}}
      .dd-box{{margin:12px 0;padding:16px}}
      .dd-kpi-row{{grid-template-columns:1fr 1fr}}
      /* section padding */
      section{{padding:20px 0!important}}
    }}
    @media(max-width:480px){{
      .today-hero{{grid-template-columns:1fr}}
      .kpi-grid{{grid-template-columns:1fr 1fr}}
      .attr-kpi-row{{grid-template-columns:1fr}}
      .tearsheet-grid{{grid-template-columns:1fr}}
      .crowd-kpi-row{{grid-template-columns:1fr}}
      .hero h1{{font-size:22px}}
      .hero-sub{{font-size:20px}}
      .kpi-val{{font-size:22px}}
    }}
    /* MOBILE BOTTOM NAV */
    .mobile-bottom-nav{{display:none}}
    @media(max-width:768px){{
      .mobile-bottom-nav{{
        display:flex;position:fixed;bottom:0;left:0;right:0;z-index:999;
        background:#231a12;border-top:1px solid rgba(255,255,255,.12);
        padding:0 0 env(safe-area-inset-bottom);
      }}
      .mobile-bottom-nav a{{
        flex:1;display:flex;flex-direction:column;align-items:center;
        padding:8px 4px;color:rgba(255,255,255,.45);text-decoration:none;
        font-size:9px;font-weight:400;letter-spacing:.5px;text-transform:uppercase;
        cursor:pointer;border:none;background:none;gap:3px;
      }}
      .mobile-bottom-nav a.active{{color:#c8b487}}
      .mobile-bottom-nav a .bnav-icon{{font-size:18px;line-height:1}}
      /* Push page content above bottom nav */
      body{{padding-bottom:64px}}
    }}
    /* PRINT / EXPORT */
    @media print{{
      nav,.mobile-bottom-nav,.tab-section:not(.active){{display:none!important}}
      .tab-section.active{{display:block!important}}
      footer{{display:none}}
      body{{font-size:11px}}
    }}
    /* FOOTER */
    footer{{background:#120f0b;color:rgba(255,255,255,.38);padding:44px 0;font-size:12px;line-height:1.75;margin-top:0}}
    footer strong{{color:rgba(255,255,255,.72)}}
    .footer-brand{{font-family:'Playfair Display',serif;font-size:20px;color:#fff;font-weight:400;margin-bottom:8px}}
    .footer-brand span{{color:#c8b487}}
    .footer-inner{{display:grid;grid-template-columns:1fr 1.8fr;gap:60px;align-items:start}}

    /* UTILS */
    .mt16{{margin-top:16px}}.mt24{{margin-top:24px}}.mt36{{margin-top:36px}}.mt48{{margin-top:48px}}
    .stress-grid{{display:grid;grid-template-columns:1fr 1fr;gap:14px}}
    .stress-card{{background:#fff;border:1px solid #241f18;padding:18px 20px;border-left:4px solid #241f18}}
    .stress-card.bad{{border-left-color:#B83232}}.stress-card.ok{{border-left-color:#c8b487}}.stress-card.good{{border-left-color:#1B6F4A}}
    .stress-name{{font-size:13px;font-weight:400;color:#1A1A1A;margin-bottom:3px}}
    .stress-period{{font-size:11px;color:#999;margin-bottom:10px}}
    .stress-metrics{{display:flex;gap:18px}}
    .sm-item label{{font-size:10px;text-transform:uppercase;letter-spacing:.8px;color:#BBB;display:block;margin-bottom:2px}}
    .sm-val{{font-family:'Playfair Display',serif;font-size:18px;font-weight:400}}
    .sm-val.neg{{color:#B83232}}.sm-val.pos{{color:#1B6F4A}}.sm-val.neu{{color:#c8b487}}
    /* ══════════════════════════════════════════════════════
       DARK MODE — cool navy-black canvas, deep-blue + gold kept
       Palette:  bg #17130f · surface #16140f · line #3a3128
       ══════════════════════════════════════════════════════ */
    body{{background:#0d0c0a!important;color:#e8e1d2!important;
      font-family:'Helvetica Neue','Helvetica','Inter',Arial,sans-serif!important;
      -webkit-font-smoothing:antialiased;letter-spacing:.005em}}
    /* FT-style: subtle warm paper grain via radial vignette */
    body::before{{content:'';position:fixed;inset:0;pointer-events:none;z-index:0;
      background:radial-gradient(120% 80% at 50% -10%,rgba(200,180,135,.035),transparent 60%)}}
    /* FT-style serif display — Scotch/transitional headlines */
    h1,h2,h3,h4,.section-head,.hero-title,
    [style*="Playfair Display"],[style*="Financier Display"]{{font-family:'Baskerville','Hoefler Text','Iowan Old Style','Georgia',serif!important;
      letter-spacing:.005em;font-feature-settings:'liga' 1,'kern' 1}}
    .section-head{{font-weight:400!important;letter-spacing:-.02em!important}}
    .eyebrow{{text-transform:uppercase;letter-spacing:.16em!important;font-size:11px!important;
      font-weight:400!important;color:#c8b487!important}}
    /* Hairline rules — FT thin salmon underscore */
    .rule{{height:2px!important;background:#c8b487!important;width:44px!important}}
    /* Tabular numerals everywhere numbers line up */
    td,.kpi-val,.today-card-val,.pos-price-val,.crowd-val,.attr-kpi-val,
    .breadth-val,.ts-val,.sm-val{{font-variant-numeric:tabular-nums}}
    /* Nav — warm bar, salmon active underline */
    nav,.nav,header{{background:#16140f!important;border-bottom:1px solid #3a3128!important}}
    .nav-brand span{{color:#c8b487!important}}
    .nav-tabs a.active,.nav-group.active .nav-group-btn{{border-bottom-color:#c8b487!important;color:#f4ecdf!important}}
    .nav-tabs a:hover,.nav-group-btn:hover{{color:#f4ecdf!important;background:rgba(200,180,135,.05)!important}}
    /* render-only dynamic colors (f-string generated) */
    [style*="background:#F0FDF4"],[style*="background:#D5E8D4"]{{background:#1c231e!important}}
    [style*="background:#FFF7ED"]{{background:#241f16!important}}
    [style*="background:#FFD7CC"]{{background:#251a17!important}}
    [style*="background:#243039"]{{background:#1f2321!important}}

    /* Elevated card surfaces — cool tint + soft shadow for depth */
    .today-card,.chart-box,.ic-row,.fac,.mac-card,.reg-card,.method-card,
    .news-card,.cal-card,.breadth-card,.rot-card,.pos-card,.crowd-kpi,
    .attr-kpi,.ff5-card,.ts-card,.bud,.rl-row,.dd-kpi,.stress-card,
    .dd-box{{
      background:#16140f!important;border-color:#3a3128!important;
      box-shadow:0 1px 3px rgba(0,0,0,.4)!important}}

    /* Specific element borders */
    .rl-row{{border-color:#3a3128!important}}
    .rl-num{{border-right-color:#3a3128!important}}
    .rl-rule{{border-left-color:#3a3128!important}}

    /* Typography — section-level headings */
    .section-head{{color:#f4ecdf!important}}
    .lead,.prose{{color:#a89c8c!important}}
    .eyebrow{{color:#c8b487!important}}
    .rule{{background:#c8b487!important}}

    /* Muted labels across all card types */
    .tbl-title,.today-card-label,.today-card-note,.chart-sub,.news-meta,
    .mac-name,.cal-date,.breadth-name,.rot-theme,.crowd-label,.attr-kpi-label,
    .ff5-window,.ts-metric,.reg-name,.bud-label,.bud-note,.tbl-note,
    .dd-meta,.dd-kpi-label,.stress-period,.rl-step,.td-rank,.ic-step,
    .pos-price-item label,.fac-sub,.fac-reg-label,.pos-stale{{
      color:#8a7f70!important}}

    /* Body-weight card text — bright */
    .chart-title,.method-title,.news-title,.rl-name,.fac-name,.stress-name,
    .breadth-val,.ts-val,.pos-price-val,.dd-news-title,.today-card-val,
    .kpi-val,.reg-pct,.crowd-val,.attr-kpi-val,.ff5-alpha,.bud-val,
    .fac-ic,.mac-ticker,.cal-ticker,.rot-etf,.pos-ticker,.dd-kpi-val,
    .dd-ticker,.sm-val{{color:#f4ecdf!important}}

    /* Deep-blue accents brighten slightly so they read on dark */
    .td-ticker,.crowd-val,.attr-kpi-val,.ff5-alpha,.bud-val,.fac-ic,
    .mac-ticker,.cal-ticker,.rot-etf,.pos-ticker,.dd-kpi-val,.dd-ticker,
    .breadth-ticker,.sig-pct,.oos-kpi-val{{color:#c8b487!important}}
    .score-bar,.sig-bar,.dd-sig-bar{{background:#7a6636!important}}

    /* Secondary text */
    .method-body,.news-summary,.news-logic,.rl-desc,.mac-role,.reg-info,
    .ff5-ir,.ff5-factors,.crowd-note,.attr-kpi-sub,.ts-bench,.cal-days,
    .rot-rets,.dd-sig-name,.pos-risk,.pos-aligned.unk,.rl-rule-val,
    .sig-name,.daily-summary-text,.hero-desc{{color:#a89c8c!important}}

    /* Tables */
    thead th{{color:#8a7f70!important;border-bottom-color:#453a2c!important}}
    tbody tr{{border-bottom-color:#2a231b!important}}
    tbody tr:hover{{background:#2a231b!important}}
    tbody tr:last-child{{border-bottom-color:#453a2c!important}}
    tr.tr-strong{{background:#1a271d!important}}

    /* Progress / bar track backgrounds */
    .score-bar-wrap,.ic-bar-wrap,.sig-bar-wrap,.dd-sig-bar-wrap{{
      background:#2f2820!important}}

    /* Dividers inside cards */
    .fac-regimes,.mac-status,.cal-action,.ts-assess,.method-hl,
    .news-action,.ff5-factors,.dd-sig-row,.pos-prices,
    .sig-bar-row,.reg-info,.crowd-kpi{{border-color:#2f2820!important}}
    .pos-prices{{border-top-color:#2f2820!important;border-bottom-color:#2f2820!important}}

    /* Semantic badge chips — richer dark tints, keep hue */
    .b-s,.rolling-ic-status.ok,.pos-side.long,.pos-badge.good,
    .news-pos-badge.in-long,.cal-risk.low,.breadth-badge.up,.rot-badge.leader{{
      background:#1c231e!important;color:#8faa9a!important}}
    .b-m,.rolling-ic-status.warn,.breadth-badge.neu,.rot-badge.neu{{
      background:#241f16!important;color:#c0a878!important}}
    .b-w,.pos-badge.neu{{
      background:#2a231b!important;color:#9a8e80!important}}
    .pos-side.short,.pos-badge.bad,.news-pos-badge.in-short,.cal-risk.high,
    .breadth-badge.dn,.rot-badge.lag,.rolling-ic-status.alert{{
      background:#251a17!important;color:#c68b83!important}}
    .pos{{color:#8faa9a!important}}
    .neg{{color:#c68b83!important}}
    .kpi-val.g,.ts-assess.good,.attr-kpi-val.g,.oos-kpi-val.g,.sm-val.pos,
    .pos-aligned.yes,.bull{{color:#8faa9a!important}}
    .ts-assess.bad,.sm-val.neg,.pos-aligned.no,.bear,.bud-trigger{{color:#c68b83!important}}
    .ts-assess.warn,.sm-val.neu{{color:#c0a878!important}}

    /* Risk ladder level-number backgrounds */
    .l1,.l2,.l3{{background:#1c231e!important;color:#8faa9a!important}}
    .l4,.l5,.l6{{background:#241f16!important;color:#c0a878!important}}
    .l7{{background:#1f2321!important;color:#8aa6a6!important}}
    .l8{{background:#221c26!important;color:#c07af0!important}}
    .l9{{background:#251a17!important;color:#c68b83!important}}
    .l10{{background:#1f2321!important;color:#8fa8d8!important}}

    /* Cards with colored top/left borders keep hue but on dark surface */
    .method-card{{border-top-color:#453a2c!important}}
    .ts-card{{border-top-color:#453a2c!important}}
    .ts-card.good{{border-top-color:#2f7a52!important}}
    .ts-card.warn{{border-top-color:#8a6a20!important}}
    .ts-card.bad{{border-top-color:#8a3232!important}}
    .news-card,.pos-card,.stress-card,.crowd-kpi{{border-left-color:#3a3128!important}}
    .news-card.pos,.pos-card.long,.stress-card.good{{border-left-color:#2f7a52!important}}
    .news-card.neg,.pos-card.short,.stress-card.bad{{border-left-color:#8a3232!important}}
    .news-card.neu,.stress-card.ok{{border-left-color:#8a6a20!important}}
    .crowd-kpi.good{{border-left-color:#2f7a52!important}}
    .crowd-kpi.warn{{border-left-color:#8a6a20!important}}
    .crowd-kpi.bad{{border-left-color:#8a3232!important}}

    /* Navy-tinted info panels */
    .oos-banner,.refresh-bar,.daily-summary{{
      background:#16140f!important;border-color:#453a2c!important}}
    .oos-banner{{border-left-color:#7a6636!important}}
    .daily-summary{{border-left-color:#7a6636!important}}
    .oos-banner-title{{color:#c8b487!important}}
    .oos-banner-body{{color:#a89c8c!important}}
    .oos-kpi label{{color:#8a7f70!important}}
    .refresh-bar{{color:#a89c8c!important}}
    .refresh-bar strong{{color:#c8b487!important}}

    /* Drilldown modal */
    #drilldown-modal{{background:rgba(0,0,0,.72)!important}}
    .dd-box{{background:#16140f!important}}
    .dd-close{{color:#8a7f70!important}}
    .dd-close:hover{{color:#f4ecdf!important}}
    .dd-news-meta{{color:#8a7f70!important}}
    .dd-section-title{{color:#c8b487!important}}

    /* Inline-style overrides via attribute selectors */
    [style*="background:#fff"],[style*="background:#FFF"],[style*="background:white"],
    [style*="background:#241f18"],[style*="background:#241f18"],[style*="background:#241f18"],
    [style*="background:#241f18"],[style*="background:#F3F4F6"]{{background:#16140f!important}}
    [style*="background:#FAFAF8"]{{background:#17130f!important}}
    [style*="background:#241f18"],[style*="background:#EEE"]{{background:#2f2820!important}}
    [style*="background:#F7FCF9"],[style*="background:#EAF5EE"],
    [style*="background:#241f18"]{{background:#1c231e!important}}
    [style*="background:#FDF8EE"],[style*="background:#FEF8EC"],[style*="background:#fff8e1"],
    [style*="background:#FEF5E7"],[style*="background:#FEF9EC"]{{background:#241f16!important}}
    [style*="background:#fff5f5"],[style*="background:#FDECEA"],[style*="background:#FEE2E2"],
    [style*="background:#FEF2F2"]{{background:#251a17!important}}
    [style*="background:#F0F4F9"],[style*="background:#2a2418"]{{background:#16140f!important}}
    [style*="background:#EFF6FF"],[style*="background:#202832"]{{background:#1f2321!important}}
    [style*="background:#F5F3FF"]{{background:#221c26!important}}
    /* navy #3a3128 used as box bg / border → warm dark (text already remapped) */
    [style*="background:#2a2418"],[style*="background: #3a3128"]{{background:#231a12!important}}
    [style*="border:1px solid #3a3128"],[style*="border-color:#c8b487"],
    [style*="border-left:3px solid #3a3128"],[style*="border-left:4px solid #3a3128"],
    [style*="border-bottom:2px solid #3a3128"]{{border-color:#453a2c!important}}
    /* second-pass: light-neutral variants that slipped through */
    [style*="background:#241f18"],[style*="background:#FAFAFA"],[style*="background:#F9F9F9"],
    [style*="background:#F5F5F5"],[style*="background:#F5F4F0"],[style*="background:#241f18"],
    [style*="background:#FBFBFB"],[style*="background:#FCFCFC"],[style*="background:#EFEFEF"],
    [style*="background:#F8F8F8"],[style*="background:#F6F5F2"]{{background:#16140f!important}}
    [style*="background:#FFF3E0"],[style*="background:#FFF8E1"],[style*="background:#FDF6E3"]{{background:#241f16!important}}
    [style*="background:#FFEBEE"],[style*="background:#FEF0EF"],[style*="background:#FCE4E4"]{{background:#251a17!important}}
    [style*="background:#202832"],[style*="background:#E8F0FE"]{{background:#1f2321!important}}
    [style*="background:#F3E5F5"],[style*="background:#EDE7F6"]{{background:#221c26!important}}
    [style*="background:#241f18"],[style*="background:#E6F4EA"],[style*="background:#E9F7EF"]{{background:#1c231e!important}}
    [style*="color:#1A1A1A"]{{color:#f4ecdf!important}}
    [style*="color:#2D2D2D"]{{color:#c4c8d0!important}}
    [style*="color:#333"]{{color:#aab0bc!important}}
    [style*="color:#374151"]{{color:#a89c8c!important}}
    [style*="color:#555"]{{color:#9198a6!important}}
    [style*="color:#666"]{{color:#9a8e80!important}}
    [style*="color:#6B7280"]{{color:#8a7f70!important}}
    [style*="color:#c8b487"]{{color:#c8b487!important}}
    [style*="color:#1B6F4A"]{{color:#8faa9a!important}}
    [style*="color:#B83232"]{{color:#c68b83!important}}
    [style*="color:#e74c3c"],[style*="color:#E74C3C"]{{color:#c68b83!important}}
    [style*="color:#f39c12"],[style*="color:#F39C12"]{{color:#c0a878!important}}
    [style*="color:#27ae60"],[style*="color:#2ecc71"],[style*="color:#16a34a"]{{color:#8faa9a!important}}
    [style*="color:#4ade80"],[style*="color:#22c55e"]{{color:#8faa9a!important}}
    [style*="color:#f87171"],[style*="color:#ef4444"],[style*="color:#dc2626"]{{color:#c68b83!important}}
    [style*="color:#facc15"],[style*="color:#eab308"]{{color:#c0a878!important}}
    [style*="color:#2A7A50"],[style*="color:#2E7D52"]{{color:#8faa9a!important}}
    [style*="color:#444"]{{color:#a89c8c!important}}
    [style*="color:#60a5fa"],[style*="color:#3b82f6"],[style*="color:#2563eb"]{{color:#8fa8d8!important}}
    [style*="color:#ccc"],[style*="color:#CCC"]{{color:#8a7f70!important}}
    [style*="color:#777"],[style*="color:#888"],[style*="color:#999"]{{color:#8a7f70!important}}
    [style*="border-top:1px solid #333"],[style*="border:1px solid #333"],[style*="border-bottom:1px solid #333"]{{border-color:#2f281f!important}}
    [style*="background:#166534"]{{background:#1c231e!important}}
    [style*="background:#1e3a5f"]{{background:#1f2321!important}}
    [style*="background:#7f1d1d"]{{background:#251a17!important}}
    [style*="background:#e74c3c"],[style*="background:#E74C3C"]{{background:#8a4a44!important}}
    [style*="background:#f39c12"],[style*="background:#F39C12"]{{background:#8a6a3a!important}}
    [style*="color:#999"]{{color:#8a7f70!important}}
    [style*="color:#AAA"],[style*="color:#BBB"],[style*="color:#CCC"]{{color:#746a5d!important}}
    [style*="border:1px solid #241f18"],[style*="border-color:#241f18"],
    [style*="border-bottom:1px solid #241f18"]{{border-color:#241f18!important}}
    [style*="background:#283038"]{{border-color:#453a2c!important}}
    /* Grid gap lines (background on grid container) */
    [style*="background:#241f18"]{{background:#241d16!important}}

    /* ═══ FT-dark: 自动覆盖所有残留浅色内联样式 ═══ */
    [style*="background:#283038"],[style*="background: #283038"],[style*="background-color:#283038"]{{background:#1f2321!important}}
    [style*="background:#CCC"],[style*="background: #CCC"],[style*="background-color:#CCC"]{{background:#16140f!important}}
    [style*="background:#241f18"],[style*="background: #241f18"],[style*="background-color:#241f18"]{{background:#16140f!important}}
    [style*="background:#202832"],[style*="background: #202832"],[style*="background-color:#202832"]{{background:#1f2321!important}}
    [style*="background:#202832"],[style*="background: #202832"],[style*="background-color:#202832"]{{background:#1f2321!important}}
    [style*="background:#E5F3EE"],[style*="background: #E5F3EE"],[style*="background-color:#E5F3EE"]{{background:#1c231e!important}}
    [style*="background:#E6F4EA"],[style*="background: #E6F4EA"],[style*="background-color:#E6F4EA"]{{background:#1c231e!important}}
    [style*="background:#E8F0FE"],[style*="background: #E8F0FE"],[style*="background-color:#E8F0FE"]{{background:#1f2321!important}}
    [style*="background:#241f18"],[style*="background: #241f18"],[style*="background-color:#241f18"]{{background:#1c231e!important}}
    [style*="background:#E9F7EF"],[style*="background: #E9F7EF"],[style*="background-color:#E9F7EF"]{{background:#1c231e!important}}
    [style*="background:#EAF5EE"],[style*="background: #EAF5EE"],[style*="background-color:#EAF5EE"]{{background:#16140f!important}}
    [style*="background:#EBF5EC"],[style*="background: #EBF5EC"],[style*="background-color:#EBF5EC"]{{background:#16140f!important}}
    [style*="background:#EDE7F6"],[style*="background: #EDE7F6"],[style*="background-color:#EDE7F6"]{{background:#1f2321!important}}
    [style*="background:#EEE"],[style*="background: #EEE"],[style*="background-color:#EEE"]{{background:#16140f!important}}
    [style*="background:#2a2418"],[style*="background: #2a2418"],[style*="background-color:#2a2418"]{{background:#16140f!important}}
    [style*="background:#EFEFEF"],[style*="background: #EFEFEF"],[style*="background-color:#EFEFEF"]{{background:#16140f!important}}
    [style*="background:#2a2418"],[style*="background: #2a2418"],[style*="background-color:#2a2418"]{{background:#16140f!important}}
    [style*="background:#EFF6FF"],[style*="background: #EFF6FF"],[style*="background-color:#EFF6FF"]{{background:#1f2321!important}}
    [style*="background:#241f18"],[style*="background: #241f18"],[style*="background-color:#241f18"]{{background:#16140f!important}}
    [style*="background:#241f18"],[style*="background: #241f18"],[style*="background-color:#241f18"]{{background:#16140f!important}}
    [style*="background:#241f18"],[style*="background: #241f18"],[style*="background-color:#241f18"]{{background:#16140f!important}}
    [style*="background:#F0F4F9"],[style*="background: #F0F4F9"],[style*="background-color:#F0F4F9"]{{background:#16140f!important}}
    [style*="background:#F0F8FF"],[style*="background: #F0F8FF"],[style*="background-color:#F0F8FF"]{{background:#1f2321!important}}
    [style*="background:#F0FAF2"],[style*="background: #F0FAF2"],[style*="background-color:#F0FAF2"]{{background:#16140f!important}}
    [style*="background:#241f18"],[style*="background: #241f18"],[style*="background-color:#241f18"]{{background:#16140f!important}}
    [style*="background:#F2FFF2"],[style*="background: #F2FFF2"],[style*="background-color:#F2FFF2"]{{background:#1c231e!important}}
    [style*="background:#F3E5F5"],[style*="background: #F3E5F5"],[style*="background-color:#F3E5F5"]{{background:#1f2321!important}}
    [style*="background:#F3F3F3"],[style*="background: #F3F3F3"],[style*="background-color:#F3F3F3"]{{background:#16140f!important}}
    [style*="background:#F3F4F6"],[style*="background: #F3F4F6"],[style*="background-color:#F3F4F6"]{{background:#16140f!important}}
    [style*="background:#241f18"],[style*="background: #241f18"],[style*="background-color:#241f18"]{{background:#16140f!important}}
    [style*="background:#F5F3FF"],[style*="background: #F5F3FF"],[style*="background-color:#F5F3FF"]{{background:#1f2321!important}}
    [style*="background:#F5F4F0"],[style*="background: #F5F4F0"],[style*="background-color:#F5F4F0"]{{background:#16140f!important}}
    [style*="background:#F5F5F5"],[style*="background: #F5F5F5"],[style*="background-color:#F5F5F5"]{{background:#16140f!important}}
    [style*="background:#F5F9F0"],[style*="background: #F5F9F0"],[style*="background-color:#F5F9F0"]{{background:#16140f!important}}
    [style*="background:#F6F5F2"],[style*="background: #F6F5F2"],[style*="background-color:#F6F5F2"]{{background:#16140f!important}}
    [style*="background:#F7F2EA"],[style*="background: #F7F2EA"],[style*="background-color:#F7F2EA"]{{background:#241f16!important}}
    [style*="background:#241f18"],[style*="background: #241f18"],[style*="background-color:#241f18"]{{background:#16140f!important}}
    [style*="background:#241f18"],[style*="background: #241f18"],[style*="background-color:#241f18"]{{background:#16140f!important}}
    [style*="background:#F7FCF9"],[style*="background: #F7FCF9"],[style*="background-color:#F7FCF9"]{{background:#16140f!important}}
    [style*="background:#F8F8F8"],[style*="background: #F8F8F8"],[style*="background-color:#F8F8F8"]{{background:#16140f!important}}
    [style*="background:#F9F8F6"],[style*="background: #F9F8F6"],[style*="background-color:#F9F8F6"]{{background:#16140f!important}}
    [style*="background:#F9F9F9"],[style*="background: #F9F9F9"],[style*="background-color:#F9F9F9"]{{background:#16140f!important}}
    [style*="background:#F9FFF9"],[style*="background: #F9FFF9"],[style*="background-color:#F9FFF9"]{{background:#16140f!important}}
    [style*="background:#FAFAF8"],[style*="background: #FAFAF8"],[style*="background-color:#FAFAF8"]{{background:#16140f!important}}
    [style*="background:#FAFAFA"],[style*="background: #FAFAFA"],[style*="background-color:#FAFAFA"]{{background:#16140f!important}}
    [style*="background:#FBFBFB"],[style*="background: #FBFBFB"],[style*="background-color:#FBFBFB"]{{background:#16140f!important}}
    [style*="background:#FCE4E4"],[style*="background: #FCE4E4"],[style*="background-color:#FCE4E4"]{{background:#241f16!important}}
    [style*="background:#FCFCFC"],[style*="background: #FCFCFC"],[style*="background-color:#FCFCFC"]{{background:#16140f!important}}
    [style*="background:#FDECEA"],[style*="background: #FDECEA"],[style*="background-color:#FDECEA"]{{background:#241f16!important}}
    [style*="background:#FDF0D0"],[style*="background: #FDF0D0"],[style*="background-color:#FDF0D0"]{{background:#241f16!important}}
    [style*="background:#FDF0F0"],[style*="background: #FDF0F0"],[style*="background-color:#FDF0F0"]{{background:#241f16!important}}
    [style*="background:#FDF5E4"],[style*="background: #FDF5E4"],[style*="background-color:#FDF5E4"]{{background:#241f16!important}}
    [style*="background:#FDF6E3"],[style*="background: #FDF6E3"],[style*="background-color:#FDF6E3"]{{background:#241f16!important}}
    [style*="background:#FDF8EE"],[style*="background: #FDF8EE"],[style*="background-color:#FDF8EE"]{{background:#241f16!important}}
    [style*="background:#FEE2E2"],[style*="background: #FEE2E2"],[style*="background-color:#FEE2E2"]{{background:#241f16!important}}
    [style*="background:#FEF0EF"],[style*="background: #FEF0EF"],[style*="background-color:#FEF0EF"]{{background:#241f16!important}}
    [style*="background:#FEF2F2"],[style*="background: #FEF2F2"],[style*="background-color:#FEF2F2"]{{background:#241f16!important}}
    [style*="background:#FEF5E7"],[style*="background: #FEF5E7"],[style*="background-color:#FEF5E7"]{{background:#241f16!important}}
    [style*="background:#FEF8EC"],[style*="background: #FEF8EC"],[style*="background-color:#FEF8EC"]{{background:#241f16!important}}
    [style*="background:#FEF9EC"],[style*="background: #FEF9EC"],[style*="background-color:#FEF9EC"]{{background:#241f16!important}}
    [style*="background:#FEF9F0"],[style*="background: #FEF9F0"],[style*="background-color:#FEF9F0"]{{background:#241f16!important}}
    [style*="background:#FFEBEE"],[style*="background: #FFEBEE"],[style*="background-color:#FFEBEE"]{{background:#221c26!important}}
    [style*="background:#FFF"],[style*="background: #FFF"],[style*="background-color:#FFF"]{{background:#16140f!important}}
    [style*="background:#FFF2F2"],[style*="background: #FFF2F2"],[style*="background-color:#FFF2F2"]{{background:#241f16!important}}
    [style*="background:#FFF3E0"],[style*="background: #FFF3E0"],[style*="background-color:#FFF3E0"]{{background:#241f16!important}}
    [style*="background:#FFF8E1"],[style*="background: #FFF8E1"],[style*="background-color:#FFF8E1"]{{background:#241f16!important}}
    [style*="background:#FFF8E6"],[style*="background: #FFF8E6"],[style*="background-color:#FFF8E6"]{{background:#241f16!important}}
    [style*="background:#FFFBF0"],[style*="background: #FFFBF0"],[style*="background-color:#FFFBF0"]{{background:#241f16!important}}
    [style*="background:#fff5f5"],[style*="background: #fff5f5"],[style*="background-color:#fff5f5"]{{background:#16140f!important}}
    [style*="color:#0b0f17"]{{color:#f4ecdf!important}}
    [style*="color:#5f7480"]{{color:#c8b487!important}}
    [style*="color:#1A1A1A"]{{color:#f4ecdf!important}}
    [style*="color:#c8b487"]{{color:#c8b487!important}}
    [style*="color:#1B6F4A"]{{color:#8faa9a!important}}
    [style*="color:#1B7A3B"]{{color:#8faa9a!important}}
    [style*="color:#5f7480"]{{color:#c8b487!important}}
    [style*="color:#2A6F5A"]{{color:#8faa9a!important}}
    [style*="color:#2D2D2D"]{{color:#f4ecdf!important}}
    [style*="color:#2a231b"]{{color:#f4ecdf!important}}
    [style*="color:#2f2820"]{{color:#f4ecdf!important}}
    [style*="color:#2f7a52"]{{color:#8faa9a!important}}
    [style*="color:#333"]{{color:#f4ecdf!important}}
    [style*="color:#334155"]{{color:#c8b487!important}}
    [style*="color:#374151"]{{color:#cabeae!important}}
    [style*="color:#3A6F4A"]{{color:#8faa9a!important}}
    [style*="color:#3a3128"]{{color:#f4ecdf!important}}
    [style*="color:#444"]{{color:#cabeae!important}}
    [style*="color:#453a2c"]{{color:#f4ecdf!important}}
    [style*="color:#4A7A2A"]{{color:#8faa9a!important}}
    [style*="color:#555"]{{color:#cabeae!important}}
    [style*="color:#666"]{{color:#cabeae!important}}
    [style*="color:#6B7280"]{{color:#9a8e80!important}}
    [style*="color:#746a5d"]{{color:#cabeae!important}}
    [style*="color:#777"]{{color:#9a8e80!important}}
    [style*="color:#7A6010"]{{color:#c68b83!important}}
    [style*="color:#7a8290"]{{color:#9a8e80!important}}
    [style*="color:#888"]{{color:#9a8e80!important}}
    [style*="color:#8B6914"]{{color:#c68b83!important}}
    [style*="color:#8a3232"]{{color:#c68b83!important}}
    [style*="color:#8a6a20"]{{color:#c68b83!important}}
    [style*="color:#7a6636"]{{color:#c68b83!important}}
    [style*="color:#8a7f70"]{{color:#9a8e80!important}}
    [style*="color:#9333EA"]{{color:#c8b487!important}}
    [style*="color:#9a8e80"]{{color:#9a8e80!important}}
    [style*="color:#B83232"]{{color:#c68b83!important}}
    [style*="color:#C0392B"]{{color:#c68b83!important}}
    [style*="color:#DC2626"]{{color:#c68b83!important}}
    [style*="color:#E65100"]{{color:#c68b83!important}}
    [style*="color:#e74c3c"]{{color:#c68b83!important}}
    [style*="#241f18"],[style*="#EEE"],[style*="#e2e0dc"],[style*="#DDD"],[style*="#ddd"]{{border-color:#241f18!important}}

  </style>
</head>
<body>

{_stale_banner}
<!-- NAV -->
<nav>
  <div class="inner">
    <a class="nav-brand" href="#">CANYON <span>QUANT</span></a>
    <div class="nav-tabs">
      <a onclick="showTab('today')" id="tab-today" class="active">Today</a>
      <a onclick="showTab('eventengine')" id="tab-eventengine" style="color:#c8b487">利润发动机 (Profit Engine) 🔥</a>

      <div class="nav-group" id="navg-portfolio">
        <span class="nav-group-btn" onclick="toggleNavDrop('navg-portfolio')">Portfolio ▾</span>
        <div class="nav-dropdown" id="navd-portfolio">
          <a onclick="showTab('live');closeNavDrops()"  id="tab-live">Live Positions</a>
          <a onclick="showTab('perf');closeNavDrops()"  id="tab-perf">Performance</a>
          <a onclick="showTab('attr');closeNavDrops()"  id="tab-attr">Attribution</a>
          <a onclick="showTab('risk');closeNavDrops()"  id="tab-risk">Risk</a>
        </div>
      </div>

      <div class="nav-group" id="navg-research">
        <span class="nav-group-btn" onclick="toggleNavDrop('navg-research')">Research ▾</span>
        <div class="nav-dropdown" id="navd-research">
          <a onclick="showTab('signals');closeNavDrops()"  id="tab-signals">Signals</a>
          <a onclick="showTab('dcf');closeNavDrops()"      id="tab-dcf">DCF 💎</a>
          <a onclick="showTab('earnings');closeNavDrops()" id="tab-earnings">Earnings AI 📋</a>
          <a onclick="showTab('shorts');closeNavDrops()"   id="tab-shorts">Short Scanner 📉</a>
          <a onclick="showTab('deep');closeNavDrops()"     id="tab-deep">Deep 🔬</a>
        </div>
      </div>

      <div class="nav-group" id="navg-market">
        <span class="nav-group-btn" onclick="toggleNavDrop('navg-market')">Market ▾</span>
        <div class="nav-dropdown" id="navd-market">
          <a onclick="showTab('heatmap');closeNavDrops()" id="tab-heatmap">Heatmap 🟩</a>
          <a onclick="showTab('macro');closeNavDrops()"   id="tab-macro">Macro</a>
          <a onclick="showTab('flow');closeNavDrops()"    id="tab-flow">Flow 🌊</a>
          <a onclick="showTab('famous');closeNavDrops()"  id="tab-famous">Smart Money 🧠</a>
          <a onclick="showTab('news');closeNavDrops()"    id="tab-news">News</a>
        </div>
      </div>

      <a onclick="showTab('chat')" id="tab-chat">AI Chat 💬</a>

      <div class="nav-group" id="navg-system">
        <span class="nav-group-btn" onclick="toggleNavDrop('navg-system')">System ▾</span>
        <div class="nav-dropdown" id="navd-system">
          <a onclick="showTab('datahealth');closeNavDrops()" id="tab-datahealth">Data Health 🩺</a>
          <a onclick="showTab('qc');closeNavDrops()"     id="tab-qc">Quant QC 🔍</a>
          <a onclick="showTab('v251');closeNavDrops()"   id="tab-v251">v25.1 ★</a>
          <a onclick="showTab('method');closeNavDrops()" id="tab-method">Method</a>
          <a onclick="showTab('health');closeNavDrops()" id="tab-health">Signal Health</a>
          <a onclick="showTab('manual');closeNavDrops()" id="tab-manual">📖 Guide</a>
        </div>
      </div>
    </div>
    <div class="nav-date">Updated {today}</div>
  </div>
</nav>

<!-- HERO -->
<header class="hero">
  <div class="container">
    <p class="hero-eye">Systematic Buy/Sell Strategy · Tested on Real Market Data</p>
    <h1>Canyon Quant<em class="hero-sub">Version 9</em></h1>
    <p class="hero-desc">Tested on {bt_months} months of live market data the model had never seen. Every signal was locked before testing began — no hindsight.</p>
    <div class="kpi-grid">
      <div class="kpi">
        <p class="kpi-label">Live Signal Accuracy</p>
        <p class="kpi-val" style="color:{_live_ic_color}">{ric_cur:+.3f}</p>
        <p class="kpi-note" style="color:{_live_ic_color}">{_live_ic_label} · 3M rolling</p>
        <span style="display:block;font-size:10px;color:#c8b487;margin-top:4px">OOS Backtest baseline: +{oos_ic:.3f} · as of {_ric_last_date}</span>
      </div>
      <div class="kpi">
        <p class="kpi-label">Backtest OOS Sharpe</p>
        <p class="kpi-val g">{oos_sharpe:.2f}</p>
        <p class="kpi-note">risk-adjusted return · S&amp;P 500 ≈ 0.6</p>
        <span style="display:block;font-size:10px;color:{'#6BCCA0' if pn_gain >= 0 else '#B83232'};margin-top:4px">Paper portfolio: {pn_gain:+.2f}% since Jun 8{(' · ' + str(pn_ndays) + 'd') if pn_ndays else ''}</span>
      </div>
      <div class="kpi">
        <p class="kpi-label">Beat S&amp;P 500</p>
        <p class="kpi-val g">{oos_wr:.0f}%</p>
        <p class="kpi-note">of months in OOS backtest (2020–2026)</p>
        <span style="display:block;font-size:10px;color:#888;margin-top:4px">Live paper started Jun 8, 2026</span>
      </div>
      <div class="kpi"><p class="kpi-label">Current Regime (HMM)</p><p class="kpi-val" style="color:{hmm_color};font-size:28px">{hmm}</p><p class="kpi-note" style="color:{hmm_color}">{"Running at full strength" if hmm == "BULL" else "Reduced size — being cautious"}{(" · " + _hmm_prob_label) if _hmm_prob_label else ""}</p>{_hmm_meta}</div>
      <div class="kpi"><p class="kpi-label">4-Week Macro Outlook</p><p class="kpi-val" style="color:{_mo_color};font-size:28px">{(f"{_mo_bear_prob:.0f}% bear") if _mo_bear_prob is not None else "—"}</p><p class="kpi-note" style="color:{_mo_color}">{_mo_label} · leading indicators</p><span style="display:block;font-size:10px;color:{'#c8b487' if hmm == 'BEAR' and _mo_bear_prob is not None and _mo_bear_prob < 30 else '#888'};margin-top:3px">{'⚡ HMM↔Macro conflict: price dip, not macro shift' if hmm == 'BEAR' and _mo_bear_prob is not None and _mo_bear_prob < 30 else 'Yield curve · credit · VIX · trend · labor'}</span></div>
    </div>
  </div>
</header>

<!-- DRILLDOWN MODAL -->
<div id="drilldown-modal" onclick="if(event.target===this)closeDrilldown()">
  <div class="dd-box">
    <button class="dd-close" onclick="closeDrilldown()">&#x2715;</button>
    <div id="dd-content"></div>
  </div>
</div>

<!-- ============================================================ TODAY -->
<section id="sec-today" class="tab-section active">
  <div class="container">
    <p class="eyebrow">Daily Signal Report — {today}</p>
    <h2 class="section-head">Today's signals &amp; recommendations{stale_note}</h2>
    <div class="rule"></div>
    {_daily_summary()}

    <div class="today-hero">
      <div class="today-card">
        <p class="today-card-label">Current Regime (HMM)</p>
        <p class="today-card-val" style="color:{hmm_color}">{hmm}</p>
        <p class="today-card-note">{"100% gross exposure" if hmm == "BULL" else "50% gross exposure"}{(" · " + _hmm_prob_label) if _hmm_prob_label else ""}</p>
        {_hmm_meta}
        <span style="display:block;font-size:9px;color:#888;margin-top:4px">REACTIVE — price-based, current state</span>
      </div>
      <div class="today-card">
        <p class="today-card-label">4-Week Macro Outlook</p>
        <p class="today-card-val" style="color:{_mo_color};font-size:24px">{(f"{_mo_bear_prob:.0f}% bear") if _mo_bear_prob is not None else "—"}</p>
        <p class="today-card-note" style="color:{_mo_color}">{_mo_label}</p>
        <span style="display:block;font-size:9px;color:#888;margin-top:4px">PREDICTIVE — 5 leading indicators · 4-week horizon</span>
      </div>
      <div class="today-card">
        <p class="today-card-label">Top LONG today</p>
        <p class="today-card-val" style="color:#c8b487;font-size:22px">{longs[0]['ticker'] if longs else '—'}</p>
        <p class="today-card-note">{"Score " + f"{longs[0]['score']:+.3f}" + " · ML strongest" if longs else "—"}</p>
      </div>
      <div class="today-card">
        <p class="today-card-label">Multi-Signal Convergence</p>
        <p class="today-card-val" style="color:#c8b487;font-size:22px">{_best_convergence_ticker()[0]}</p>
        <p class="today-card-note">{_best_convergence_ticker()[1]} signals agree · strongest conviction</p>
      </div>
    </div>

    {_insider_scan_panel()}
    {_insider_short_panel()}
    {_insider_ls_panel()}

    {_build_earnings_this_week(earnings_cal)}
    {_build_regime_gauge(hmm, _mo_bear_prob)}

    <!-- Macro Regime Outlook panel -->
    <div style="margin-bottom:28px;background:#16140f;border:1px solid #3a3128;border-radius:8px;padding:16px 18px">
      <div style="display:flex;justify-content:space-between;align-items:baseline;flex-wrap:wrap;gap:8px;margin-bottom:16px">
        <span style="font-size:11px;color:#8a7f70;text-transform:uppercase;letter-spacing:.14em">Macro Regime Outlook · Forward-Looking Signal (4-week)</span>
        <span style="font-size:10px;color:#8a7f70">5 leading indicators · updated daily</span>
      </div>
      {_safe_panel(_mo_panel)}
    </div>

    {signal_changes_block()}

    <div class="two-col-65">
      <div>
        <div class="tbl-wrap">
          <p class="tbl-title">▲ BUY — top 8 stocks to buy today</p>
          <table>
            <thead><tr><th></th><th>Ticker</th><th>Score</th><th>Price</th><th class="r">AI Model</th><th class="r">Fundamentals</th></tr></thead>
            <tbody>{long_rows()}</tbody>
          </table>
          <p class="tbl-note">Score 0–100 combining AI model, fundamentals, analyst upgrades, earnings quality, and short squeeze signals.</p>
        </div>
        <div class="tbl-wrap mt24">
          <p class="tbl-title">▼ AVOID — stocks the system dislikes most</p>
          <table>
            <thead><tr><th></th><th>Ticker</th><th>Score</th><th>Price</th></tr></thead>
            <tbody>{short_rows()}</tbody>
          </table>
          <p class="tbl-note">These stocks score lowest across all signals. Avoid or use as shorts.</p>
        </div>
      </div>
      <div>
        <div class="tbl-wrap">
          <p class="tbl-title">High-conviction buys — multiple signals agree</p>
          <table>
            <thead><tr><th>Ticker</th><th>AI strength</th><th>Analyst view</th><th>All signals</th><th>Verdict</th></tr></thead>
            <tbody>{convergence_rows()}</tbody>
          </table>
          <p class="tbl-note">★ = one independent signal agrees. More stars = higher conviction from more sources.</p>
        </div>
        <div class="tbl-wrap mt24">
          <p class="tbl-title">Short squeeze candidates — high short interest</p>
          <table>
            <thead><tr><th>Ticker</th><th style="text-align:center">Conditions</th><th class="r">Score</th><th class="r">vs S&amp;P 500</th></tr></thead>
            <tbody>{squeeze_rows()}</tbody>
          </table>
          <p class="tbl-note">4 conditions must all be true: heavy short interest, unusual options activity, positive analyst revision, stronger momentum than the S&amp;P 500.</p>
        </div>
      </div>
    </div>
    <div class="mt36">
      <p class="eyebrow">Active Alerts</p>
      <h3 style="font-family:'Playfair Display',serif;font-size:22px;font-weight:400;color:#1A1A1A;margin:8px 0 4px">Things that need your attention today</h3>
      <div class="rule"></div>
      {desk_monitor_rows()}
    </div>

    <div class="mt36">
      <p class="eyebrow">Today's Checklist</p>
      <h3 style="font-family:'Playfair Display',serif;font-size:22px;font-weight:400;color:#1A1A1A;margin:8px 0 4px">What to do today, in order</h3>
      <div class="rule"></div>
      <table>
        <thead><tr><th></th><th>Station</th><th>Status</th><th>What to do</th></tr></thead>
        <tbody>{workflow_steps_rows()}</tbody>
      </table>
    </div>

    <div class="mt36">
      <p class="eyebrow">Priority Stocks — Today's Focus List</p>
      <h3 style="font-family:'Playfair Display',serif;font-size:22px;font-weight:400;color:#1A1A1A;margin:8px 0 4px">All stocks ranked by urgency</h3>
      <div class="rule"></div>
      <table>
        <thead><tr><th></th><th>Ticker</th><th>Sector</th><th>Priority</th><th>Cycle State</th><th>Action</th></tr></thead>
        <tbody>{workflow_queue_rows()}</tbody>
      </table>
    </div>

    <div class="mt36">
      <p class="eyebrow">Sector Cycle Map</p>
      <h3 style="font-family:'Playfair Display',serif;font-size:22px;font-weight:400;color:#1A1A1A;margin:8px 0 4px">Where sector attention belongs today</h3>
      <div class="rule"></div>
      <table>
        <thead><tr><th>ETF</th><th>Sector</th><th>Cycle State</th><th class="r">20d Ret</th><th class="r">63d Ret</th><th class="r">Weight</th><th>Cap</th></tr></thead>
        <tbody>{sector_cycle_rows()}</tbody>
      </table>
    </div>

    {_build_live_market_pulse()}

    {_build_economic_calendar_widget(econ_cal)}
  </div>
</section>

<!-- ============================================================ PERFORMANCE -->
<section id="sec-perf" class="tab-section">
  <div class="container">
    <p class="eyebrow">Real-World Performance — Data the Model Never Saw</p>
    <h2 class="section-head">How the strategy performed on live, unseen market data</h2>
    <div class="rule"></div>

    {_safe_panel(_build_three_book_panel)}

    {_safe_panel(_build_pnl_ic_panels)}

    <div class="oos-banner" style="margin-top:36px">
      <p class="oos-banner-title">Why these results are trustworthy</p>
      <p class="oos-banner-body">Hard cutoff at <strong>January 1, 2020</strong>. All strategy rules were locked before that date. Every result from 2020 onward is genuine real-world testing — no hindsight, no parameter tweaking after the fact.</p>
      <div class="oos-kpi-row">
        <div class="oos-kpi"><label>Signal Accuracy</label><span class="oos-kpi-val g">+{oos_ic:.3f}</span></div>
        <div class="oos-kpi"><label>Statistical Confidence</label><span class="oos-kpi-val g">{oos_t:.1f}x</span></div>
        <div class="oos-kpi"><label>Real vs Training gap</label><span class="oos-kpi-val">−{round((1 - (oos_ic/is_ic if is_ic else 0))*100, 0):.0f}%</span></div>
        <div class="oos-kpi"><label>Risk-Adjusted Return</label><span class="oos-kpi-val g">{oos_sharpe:.3f}</span></div>
        <div class="oos-kpi"><label>Worst Peak-to-Trough Drop</label><span class="oos-kpi-val">{oos_dd:.1f}%</span></div>
        <div class="oos-kpi"><label>Months Beating S&amp;P 500</label><span class="oos-kpi-val g">{oos_wr:.0f}%</span></div>
      </div>
    </div>

    <div class="chart-box">
      <p class="chart-title">If you had invested $100 — tested on data the model had never seen before</p>
      <p class="chart-sub">No hindsight — every signal was decided before the test began · Strategy grew to ${final_ml:,.0f} · S&P 500 index grew to ${final_spy:,.0f} over the same period</p>
      <div class="chart-inner"><canvas id="oosChart"></canvas></div>
    </div>
    <p class="tbl-note mt16">⚠ Note: this backtest only includes stocks still in the S&P 500 today — companies that went bankrupt or were removed are not included, which slightly inflates the numbers. Returns shown are from the real test period and reflect heavy concentration in AI/semiconductor stocks (NVDA, AMD, MU) during the 2023–2026 tech rally.</p>

    <div class="chart-box mt36">
      <p class="chart-title">Month-by-month: did the strategy beat the S&P 500? (green = yes, red = no)</p>
      <p class="chart-sub">{bt_months} months tested · Beat the index in <strong>{bt_win_rate:.0f}%</strong> of months · Total gain: Strategy <strong>{bt_final_strat:+.1f}%</strong> · S&P 500 <strong>{bt_final_bench:+.1f}%</strong></p>
      <div class="chart-inner" style="height:260px"><canvas id="btMonthlyChart"></canvas></div>
    </div>

    <div class="chart-box mt36">
      <p class="chart-title">Total cumulative return — strategy vs S&amp;P 500 over the test period</p>
      <p class="chart-sub">Rebalanced monthly · After estimated trading costs · {bt_months} months shown</p>
      <div class="chart-inner" style="height:260px"><canvas id="btCumChart"></canvas></div>
    </div>

    <div class="two-col-even mt36">
      <div>
        <p class="eyebrow">Did it hold up on data it had never seen before?</p>
        <table class="mt16">
          <thead><tr><th>Measure</th><th class="r">During training (before 2020)</th><th class="r">Real unseen data (2020+)</th></tr></thead>
          <tbody>
            <tr><td>How accurately it predicts which stocks go up</td><td class="r">{is_ic:.3f}</td><td class="r pos">+{oos_ic:.3f}</td></tr>
            <tr><td>How statistically reliable the result is</td><td class="r">{is_t:.2f}×</td><td class="r pos">{oos_t:.2f}× — Strong</td></tr>
            <tr><td>Return earned per unit of risk taken</td><td class="r">{is_sharpe:.3f}</td><td class="r pos">{oos_sharpe:.3f}</td></tr>
            <tr><td>Worst loss from any peak (ever)</td><td class="r neg">−{abs(is_dd):.1f}%</td><td class="r pos">−{abs(oos_dd):.1f}%</td></tr>
            <tr><td>How often it beat the S&amp;P 500 each month</td><td class="r">{is_wr:.1f}%</td><td class="r pos">{oos_wr:.1f}%</td></tr>
            <tr><td>Total return over the period</td><td class="r">Training only</td><td class="r pos">{oos_ret:,.0f}%</td></tr>
          </tbody>
        </table>
      </div>
      <div>
        <p class="eyebrow">Annual Returns by Year</p>
        <table class="mt16">
          <thead><tr><th>Year</th><th class="r">Net Return</th><th>Note</th></tr></thead>
          <tbody>
            {"".join(
              f'<tr><td>{yr}{"  YTD" if yr == int(today[:4]) else ""}</td>'
              f'<td class="r {"pos" if v >= 0 else "neg"}">{v:+.1f}%</td>'
              f'<td style="font-size:11px;color:#AAA">From backtest</td></tr>'
              for yr, v in sorted(annual_rets.items())
            ) if annual_rets else
            "<tr><td colspan='3' style='color:#AAA;text-align:center'>No backtest data — run pipeline</td></tr>"}
          </tbody>
        </table>
      </div>
    </div>

    <div class="mt36">
      <p class="eyebrow">How the Strategy Handled Past Market Crises</p>
      <div class="stress-grid mt16">
        <div class="stress-card bad"><p class="stress-name">COVID Crash</p><p class="stress-period">Feb – Mar 2020</p>
          <div class="stress-metrics">
            <div class="sm-item"><label>Long-Only</label><span class="sm-val neg">−36%</span></div>
            <div class="sm-item"><label>Long/Short</label><span class="sm-val neu">−8%</span></div>
            <div class="sm-item"><label>S&amp;P 500</label><span class="sm-val neg">−34%</span></div>
          </div></div>
        <div class="stress-card ok"><p class="stress-name">2022 Rate Shock</p><p class="stress-period">Jan – Oct 2022</p>
          <div class="stress-metrics">
            <div class="sm-item"><label>Long-Only</label><span class="sm-val neg">−28%</span></div>
            <div class="sm-item"><label>Long/Short</label><span class="sm-val neu">−12%</span></div>
            <div class="sm-item"><label>S&amp;P 500</label><span class="sm-val neg">−25%</span></div>
          </div></div>
        <div class="stress-card good"><p class="stress-name">2023 AI Bull Run</p><p class="stress-period">Jan – Dec 2023</p>
          <div class="stress-metrics">
            <div class="sm-item"><label>Long-Only</label><span class="sm-val pos">+42%</span></div>
            <div class="sm-item"><label>Long/Short</label><span class="sm-val pos">+3%</span></div>
            <div class="sm-item"><label>S&amp;P 500</label><span class="sm-val pos">+26%</span></div>
          </div></div>
        <div class="stress-card ok"><p class="stress-name">2025 Correction</p><p class="stress-period">Feb – Apr 2025</p>
          <div class="stress-metrics">
            <div class="sm-item"><label>Long-Only</label><span class="sm-val neg">−18%</span></div>
            <div class="sm-item"><label>Long/Short</label><span class="sm-val neu">−5%</span></div>
            <div class="sm-item"><label>S&amp;P 500</label><span class="sm-val neg">−19%</span></div>
          </div></div>
      </div>
    </div>
  </div>
</section>

<!-- ============================================================ SIGNALS -->
<section id="sec-signals" class="tab-section">
  <div class="container">
    <p class="eyebrow">What Drives the Buy/Sell Signals</p>
    <h2 class="section-head">Seven independent signal sources — each tested on real market data</h2>
    <div class="rule"></div>
    <p class="lead">Each signal was tested on real data from 2020–2026 that the model never saw during training. No single signal trades alone — they all combine into one daily score per stock.</p>

    <div class="ic-stack">
      <div class="ic-row"><span class="ic-name">Stock drifts up after beating earnings</span><span class="ic-step">Strongest</span><div class="ic-bar-wrap"><div class="ic-bar s" style="width:100%"></div></div><span class="ic-val">+0.229</span><span class="ic-badge b-s">★★★ Confirmed</span></div>
      <div class="ic-row"><span class="ic-name">Market fear level (VIX)</span><span class="ic-step">Strongest</span><div class="ic-bar-wrap"><div class="ic-bar s" style="width:97%"></div></div><span class="ic-val">+0.223</span><span class="ic-badge b-s">★★★ Confirmed</span></div>
      <div class="ic-row"><span class="ic-name">AI model combined prediction</span><span class="ic-step">Strongest</span><div class="ic-bar-wrap"><div class="ic-bar s" style="width:100%"></div></div><span class="ic-val">+{oos_ic:.3f}</span><span class="ic-badge b-s">★★★ Confirmed</span></div>
      <div class="ic-row"><span class="ic-name">Earnings quality (cash vs accounting profit)</span><span class="ic-step">Strong</span><div class="ic-bar-wrap"><div class="ic-bar m" style="width:31%"></div></div><span class="ic-val">+0.072</span><span class="ic-badge b-s">★★★ Confirmed</span></div>
      <div class="ic-row"><span class="ic-name">Company profitability &amp; return on equity</span><span class="ic-step">Strong</span><div class="ic-bar-wrap"><div class="ic-bar m" style="width:21%"></div></div><span class="ic-val">+0.048</span><span class="ic-badge b-m">★★ Good</span></div>
      <div class="ic-row"><span class="ic-name">Short squeeze potential</span><span class="ic-step">Strong</span><div class="ic-bar-wrap"><div class="ic-bar m" style="width:22%"></div></div><span class="ic-val">+0.050</span><span class="ic-badge b-m">★★ Good</span></div>
      <div class="ic-row"><span class="ic-name">Analyst price target upgrades</span><span class="ic-step">Moderate</span><div class="ic-bar-wrap"><div class="ic-bar m" style="width:17%"></div></div><span class="ic-val">+0.038</span><span class="ic-badge b-m">★★ Good</span></div>
      <div class="ic-row"><span class="ic-name">Google search interest around earnings</span><span class="ic-step">Amplifier</span><div class="ic-bar-wrap"><div class="ic-bar m" style="width:9%"></div></div><span class="ic-val">+0.020</span><span class="ic-badge b-m">Boosts other signals</span></div>
      <div class="ic-row"><span class="ic-name">Options market implied volatility</span><span class="ic-step">Weak</span><div class="ic-bar-wrap"><div class="ic-bar w" style="width:7%"></div></div><span class="ic-val">+0.017</span><span class="ic-badge b-w">★ Weak</span></div>
      <div class="ic-row"><span class="ic-name">News sentiment (AI text analysis)</span><span class="ic-step">Weak</span><div class="ic-bar-wrap"><div class="ic-bar w" style="width:4%"></div></div><span class="ic-val">+0.010</span><span class="ic-badge b-w">★ Weak</span></div>
      <div class="ic-row"><span class="ic-name">13F Smart Money</span><span class="ic-step">Step 480</span><div class="ic-bar-wrap"><div class="ic-bar w" style="width:2%"></div></div><span class="ic-val">+0.004</span><span class="ic-badge b-w">t=0.3 ★</span></div>
    </div>

    <div class="mt36">
      <p class="eyebrow">All Stocks Ranked by Today's Score</p>
      <h3 style="font-family:'Playfair Display',serif;font-size:22px;font-weight:400;color:#1A1A1A;margin:8px 0 4px">Combined score from all signal sources</h3>
      <div class="rule"></div>
      <table>
        <thead><tr><th></th><th>Ticker</th><th>Sector</th><th>Score</th><th>Signal</th><th>Crowding</th><th>Top Signals</th><th>Portfolio</th></tr></thead>
        <tbody>{alpha_score_rows()}</tbody>
      </table>
      <p class="tbl-note">Score 0–100 (higher = more bullish). Green = strong buy signal (60+), red = avoid (below 40). Shows AI model, earnings quality, and analyst revision scores.</p>
    </div>

    <div class="two-col-even mt36">
      <div>
        <p class="eyebrow">Earnings Quality Check</p>
        <p style="font-size:12px;color:#888;margin:6px 0 12px;line-height:1.5">Low score = company's cash profits exceed accounting profits = high quality earnings. High score = earnings look good on paper but cash isn't backing it up = red flag.</p>
        <table>
          <thead><tr><th>Ticker</th><th class="r">Accrual Ratio</th><th>Signal</th></tr></thead>
          <tbody>{accrual_long_rows()}</tbody>
        </table>
        <p class="tbl-note mt16" style="text-transform:uppercase;letter-spacing:.8px;font-weight:400;color:#B83232">Low Quality — SHORT candidates</p>
        <table class="mt16"><thead><tr><th>Ticker</th><th class="r">Accrual Ratio</th><th>Signal</th></tr></thead>
        <tbody>{accrual_short_rows()}</tbody></table>
      </div>
      <div>
        <p class="eyebrow">Which stock types work best in each market environment</p>
        <div class="fac-grid mt16">
          <div class="fac"><p class="fac-name">Price momentum</p><p class="fac-ic bull">+0.042</p><p class="fac-sub">Stocks that held up well during a selloff tend to keep outperforming. Works especially well during down markets.</p>
            <div class="fac-regimes"><div class="fac-reg"><p class="fac-reg-label">Bear market</p><p class="fac-reg-val bull">+0.113</p></div><div class="fac-reg"><p class="fac-reg-label">Bull market</p><p class="fac-reg-val bear">−0.053</p></div></div></div>
          <div class="fac"><p class="fac-name">Profitable, high-quality companies</p><p class="fac-ic bull">+0.048</p><p class="fac-sub">Most reliable signal — companies with strong profits and high return on equity outperform in both up and down markets.</p>
            <div class="fac-regimes"><div class="fac-reg"><p class="fac-reg-label">Bull market</p><p class="fac-reg-val bull">+0.055</p></div><div class="fac-reg"><p class="fac-reg-label">Bear market</p><p class="fac-reg-val bull">+0.038</p></div></div></div>
          <div class="fac"><p class="fac-name">Low volatility stocks</p><p class="fac-ic bear">−0.037</p><p class="fac-sub">Boring, stable stocks only help during down markets. In bull markets they underperform — the strategy reduces their weight accordingly.</p>
            <div class="fac-regimes"><div class="fac-reg"><p class="fac-reg-label">Bear market</p><p class="fac-reg-val bull">+0.042</p></div><div class="fac-reg"><p class="fac-reg-label">Bull market</p><p class="fac-reg-val bear">−0.083</p></div></div></div>
          <div class="fac"><p class="fac-name">Cheap / value stocks</p><p class="fac-ic bear">−0.055</p><p class="fac-sub">In this tech-heavy universe, cheap stocks are often cheap for a reason. This signal is mostly a drag and gets low weight.</p>
            <div class="fac-regimes"><div class="fac-reg"><p class="fac-reg-label">Bull market</p><p class="fac-reg-val bull">+0.037</p></div><div class="fac-reg"><p class="fac-reg-label">Bear market</p><p class="fac-reg-val bear">−0.113</p></div></div></div>
        </div>
      </div>
    </div>
  </div>
</section>

<!-- ============================================================ RISK -->
<section id="sec-risk" class="tab-section">
  <div class="container">
    <p class="eyebrow">How the Strategy Manages Risk</p>
    <h2 class="section-head">10-step checklist before any trade is allowed</h2>
    <div class="rule"></div>
    <p class="lead">Every stock must pass through 10 checks before it can be bought or sold. Checks 8 and 9 can veto everything else — if risk is too high, no trade happens regardless of how good the signal looks.</p>

    <div class="risk-ladder">
      <div class="rl-row"><div class="rl-num l1">1</div><div class="rl-body"><p class="rl-name">Is the market in Bull or Bear mode?</p><p class="rl-desc">The system reads market price, volatility, and momentum every day and classifies it as Bull (rising trend) or Bear (falling trend). This sets how much of the portfolio to put to work.</p></div><div class="rl-rule"><p class="rl-rule-label">Rule</p><p class="rl-rule-val">Bull → full size · Bear → half size</p></div></div>
      <div class="rl-row"><div class="rl-num l2">2</div><div class="rl-body"><p class="rl-name">Are background conditions healthy?</p><p class="rl-desc">Checks bonds, credit spreads, the dollar, and gold. If these are flashing warning signs (e.g. credit spreads widening), the system trims exposure by 20% as a precaution.</p></div><div class="rl-rule"><p class="rl-rule-label">Rule</p><p class="rl-rule-val">Warning signs → cut 20%</p></div></div>
      <div class="rl-row"><div class="rl-num l3">3</div><div class="rl-body"><p class="rl-name">Is one sector taking too much space?</p><p class="rl-desc">No single industry (e.g. tech, healthcare) can exceed 40% of the portfolio. If semiconductors are already at the limit, no more chip stocks can be added even if the signal is strong.</p></div><div class="rl-rule"><p class="rl-rule-label">Rule</p><p class="rl-rule-val">Max 40% per industry</p></div></div>
      <div class="rl-row"><div class="rl-num l4">4</div><div class="rl-body"><p class="rl-name">Are the company fundamentals good?</p><p class="rl-desc">Checks: did the stock beat earnings expectations? Is the company's cash profit greater than its accounting profit (earnings quality)? Are analysts raising their price targets?</p></div><div class="rl-rule"><p class="rl-rule-label">Top checks</p><p class="rl-rule-val">Earnings beat · Cash quality · Analyst upgrades</p></div></div>
      <div class="rl-row"><div class="rl-num l5">5</div><div class="rl-body"><p class="rl-name">What does the AI model say?</p><p class="rl-desc">Three machine learning models vote independently. Their combined prediction accuracy on real unseen data is +{oos_ic:.3f} — statistically confirmed. This contributes 40% of the final score.</p></div><div class="rl-rule"><p class="rl-rule-label">Accuracy</p><p class="rl-rule-val">+{oos_ic:.3f} on real data</p></div></div>
      <div class="rl-row"><div class="rl-num l6">6</div><div class="rl-body"><p class="rl-name">Does the news contradict the signal?</p><p class="rl-desc">AI reads today's news for every stock. If news is strongly negative on a stock we want to buy, or strongly positive on a stock we want to short, the system blocks the trade.</p></div><div class="rl-rule"><p class="rl-rule-label">Function</p><p class="rl-rule-val">Block contradictions</p></div></div>
      <div class="rl-row"><div class="rl-num l7">7</div><div class="rl-body"><p class="rl-name">What is the options market signaling?</p><p class="rl-desc">Checks options pricing and put/call ratios for unusual activity or fear. Research input only — cannot override the risk checks below.</p></div><div class="rl-rule"><p class="rl-rule-label">Note</p><p class="rl-rule-val">Research only — cannot override checks 8 &amp; 9</p></div></div>
      <div class="rl-row"><div class="rl-num l8">L8 ★</div><div class="rl-body"><p class="rl-name">Is the position size within limits? (Veto)</p><p class="rl-desc">Checks per-stock size, total loss exposure, sector limits, and how correlated all positions are to each other. If any limit is breached, the trade is blocked — no matter how good the signal is.</p></div><div class="rl-rule"><p class="rl-rule-label">Veto power</p><p class="rl-rule-val">Overrides everything</p></div></div>
      <div class="rl-row"><div class="rl-num l9">L9 ★</div><div class="rl-body"><p class="rl-name">Is it safe to trade right now? (Veto)</p><p class="rl-desc">Checks: is the stock liquid enough to trade? Is an earnings report within 3 days? Is market fear (VIX) above 40? If any of these are true, the trade is halted until conditions improve.</p></div><div class="rl-rule"><p class="rl-rule-label">Hard stops</p><p class="rl-rule-val">Low liquidity · Earnings · Extreme fear</p></div></div>
      <div class="rl-row"><div class="rl-num l10">10</div><div class="rl-body"><p class="rl-name">Final decision</p><p class="rl-desc">All 10 checks green → trade at full size. Check 8 amber → trade at reduced size. Any check red → hold and wait.</p></div><div class="rl-rule"><p class="rl-rule-label">Output</p><p class="rl-rule-val">Buy · Wait · Skip</p></div></div>
    </div>

    <div class="mt36">
      <p class="eyebrow">Today's Risk Check — Per Stock</p>
      <h3 style="font-family:'Playfair Display',serif;font-size:22px;font-weight:400;color:#1A1A1A;margin:8px 0 4px">What action is required on each stock today</h3>
      <div class="rule"></div>
      <table>
        <thead><tr><th>Stock</th><th>Sector</th><th class="r">Your size</th><th class="r">Max allowed</th><th>Action</th><th>Why</th></tr></thead>
        <tbody>{risk_gate_rows()}</tbody>
      </table>
    </div>

    <div class="mt36">
      <p class="eyebrow">Stock Details — What's Blocking Each Trade</p>
      <h3 style="font-family:'Playfair Display',serif;font-size:22px;font-weight:400;color:#1A1A1A;margin:8px 0 4px">Why each stock is waiting and what needs to change</h3>
      <div class="rule"></div>
      {drilldown_cards()}
    </div>

    <div class="mt36">
      <p class="eyebrow">Are We Running With the Crowd?</p>
      <h3 style="font-family:'Playfair Display',serif;font-size:22px;font-weight:400;color:#1A1A1A;margin:8px 0 4px">Are we running with the crowd?</h3>
      <p class="lead" style="margin-top:4px">When too many funds own the same stocks, a single large seller can trigger a chain reaction — everyone rushes to exit at once and prices drop sharply. This section monitors how closely our picks overlap with other investors' favorites, so you can see this risk before it happens.</p>
{crowding_panel()}
    </div>

    <div class="mt36">
      <p class="eyebrow">Risk Limits</p>
      <div class="budget-grid mt16">
        <div class="bud"><p class="bud-label">Max per Stock</p><p class="bud-val">20%</p><p class="bud-note">No single stock can exceed 20% of the buy side. Short side capped at 8% per stock.</p><p class="bud-trigger">Hard limit — auto-enforced</p></div>
        <div class="bud"><p class="bud-label">Max per Sector</p><p class="bud-val">40%</p><p class="bud-note">No single industry sector above 40% of the portfolio.</p><p class="bud-trigger">Auto-enforced daily</p></div>
        <div class="bud"><p class="bud-label">Loss Alert Level 1</p><p class="bud-val">−10%</p><p class="bud-note">If the portfolio drops 10% from peak, immediately cut position sizes by 25%.</p><p class="bud-trigger">Auto-trigger</p></div>
        <div class="bud"><p class="bud-label">Loss Alert Level 2</p><p class="bud-val">−20%</p><p class="bud-note">Cut by 50%. Resume normal sizing only after recovering to −8%.</p><p class="bud-trigger">Defensive mode</p></div>
        <div class="bud"><p class="bud-label">Daily Loss Limit</p><p class="bud-val">2%</p><p class="bud-note">Maximum expected 1-day loss at 95% confidence. Checked every day.</p><p class="bud-trigger">Daily check</p></div>
        <div class="bud"><p class="bud-label">Position Sizing</p><p class="bud-val">Conservative</p><p class="bud-note">Each position is sized conservatively — half the mathematically optimal size — to reduce risk of large losses.</p><p class="bud-trigger">Per-position calculation</p></div>
      </div>
    </div>

    {_safe_panel(_build_barra_risk_panel, barra_risk or {})}

  </div>
</section>

<!-- ============================================================ MACRO -->
<section id="sec-macro" class="tab-section">
  <div class="container">
    <p class="eyebrow">Market Environment</p>
    <h2 class="section-head">Market mode &amp; background conditions</h2>
    <div class="rule"></div>

    <div class="reg-grid">
      <div class="reg-card">
        <p class="reg-name">Bull Mode</p>
        <p class="reg-pct bull">57.8%</p>
        <p style="font-size:12px;color:#999">of all trading days tested (Jan 2020 – Jun 2026)</p>
        <div class="reg-info">Avg duration: <strong>78 days (~3.7 months)</strong><br>Portfolio size: <strong>Full — 100%</strong><br>Stays Bull next day: <strong>98.7% of the time</strong><br>Current: <strong style="color:{hmm_color}">{hmm}{"  — running at full strength" if hmm == "BULL" else " — reduced size, being cautious" if hmm == "BEAR" else " — neutral"}</strong></div>
      </div>
      <div class="reg-card">
        <p class="reg-name">Bear Mode</p>
        <p class="reg-pct bear">42.2%</p>
        <p style="font-size:12px;color:#999">of all trading days tested (Jan 2020 – Jun 2026)</p>
        <div class="reg-info">Avg duration: <strong>117 days (~5.6 months)</strong><br>Portfolio size: <strong>Reduced — 50%</strong><br>Stays Bear next day: <strong>99.1% of the time</strong><br>Momentum signal is stronger in Bear markets</div>
      </div>
    </div>

    {_macro_signal_cards(macro_sigs)}

    <div class="mt36">
      <p class="eyebrow">Portfolio Weight Optimization</p>
      <p class="lead" style="margin-top:10px">The strategy starts from market-standard weights and only moves away from them when the signal is strong. This avoids over-trading and cuts transaction costs by 85%.</p>
      <table>
        <thead><tr><th>Metric</th><th class="r">Simple equal weight</th><th class="r">Optimized weights</th><th class="r">Improvement</th></tr></thead>
        <tbody>
          <tr><td>Risk-adjusted return</td><td class="r neg">−0.221</td><td class="r pos">+0.737</td><td class="r pos">+0.958</td></tr>
          <tr><td>How often positions change</td><td class="r">1.24× per month</td><td class="r pos">0.19× per month</td><td class="r pos">−85% less trading</td></tr>
          <tr><td>Transaction cost / month</td><td class="r">~$120 per $1M</td><td class="r pos">~$20 per $1M</td><td class="r pos">Saved</td></tr>
        </tbody>
      </table>
    </div>

    <div class="mt48">
      <p class="eyebrow">Market Breadth — Are most stocks rising or falling?</p>
      <h3 style="font-family:'Playfair Display',serif;font-size:22px;font-weight:400;color:#1A1A1A;margin:8px 0 4px">Index &amp; sector health check</h3>
      <p class="lead" style="margin-top:4px">If the main indexes are above their 20-day and 50-day moving averages, the overall market is healthy. If they're below, the wind is against you.</p>
      <div class="breadth-grid">
{breadth_cards()}
      </div>
    </div>

    <div class="mt36">
      <p class="eyebrow">Sector Rotation — Who's leading and who's lagging?</p>
      <h3 style="font-family:'Playfair Display',serif;font-size:22px;font-weight:400;color:#1A1A1A;margin:8px 0 4px">Which sectors have momentum right now</h3>
      <p class="lead" style="margin-top:4px">Sectors are ranked by their momentum score — a blend of 1-month and 3-month returns versus the S&amp;P 500. Money flows from laggards into leaders. Position yourself with the leaders.</p>
      <div class="rot-grid">
{rotation_cards()}
      </div>
    </div>

    <div class="mt36">
      <p class="eyebrow">Earnings Calendar — Upcoming reports that could move stocks</p>
      <h3 style="font-family:'Playfair Display',serif;font-size:22px;font-weight:400;color:#1A1A1A;margin:8px 0 4px">Earnings events in your portfolio</h3>
      <p class="lead" style="margin-top:4px">Earnings reports create sudden large moves. The system flags "High risk" when a stock in the portfolio is about to report — this is when we reduce or hedge, not add.</p>
      <div class="cal-grid">
{earnings_cal_cards()}
      </div>
    </div>
  </div>
</section>

<!-- ============================================================ ATTRIBUTION -->
<section id="sec-attr" class="tab-section">
  <div class="container">
    <p class="eyebrow">Performance Breakdown</p>
    <h2 class="section-head">Where does the return actually come from?</h2>
    <div class="rule"></div>
    <p class="lead">This section answers the most important question: is the return real skill, or just riding the market up? The analysis strips out all known market factors and shows what's left — pure strategy edge.</p>

    <div class="attr-kpi-row">
      <div class="attr-kpi">
        <p class="attr-kpi-label">How accurate is the model right now? (last 3 months)</p>
        <p class="attr-kpi-val" style="color:{'#1B6F4A' if ric_cur > 0.25 else '#c8b487' if ric_cur > 0.10 else '#B83232'}">{ric_cur:+.3f}</p>
        <p class="attr-kpi-sub">Target accuracy: +{ric_target:.3f} &nbsp; <span class="rolling-ic-status {ric_status_class}">{ric_status.replace('_',' ')}</span></p>
      </div>
      <div class="attr-kpi">
        <p class="attr-kpi-label">Average monthly extra return above the market (last 18 months)</p>
        <p class="attr-kpi-val" style="color:{'#1B6F4A' if mpnl_avg > 0 else '#B83232'}">{mpnl_avg:+.2f}%</p>
        <p class="attr-kpi-sub">Return beyond what the overall market contributed</p>
      </div>
      <div class="attr-kpi">
        <p class="attr-kpi-label">How often did it beat the market each month? (last 18 months)</p>
        <p class="attr-kpi-val" style="color:#c8b487">{mpnl_wr:.1f}%</p>
        <p class="attr-kpi-sub">{mpnl_wins} months ahead of the market out of {mpnl_total}</p>
      </div>
      <div class="attr-kpi">
        <p class="attr-kpi-label">Best month / Worst month</p>
        <p class="attr-kpi-val" style="font-size:20px;color:#c8b487">{mpnl_best:+.1f}% / <span style="color:#B83232">{mpnl_worst:+.1f}%</span></p>
        <p class="attr-kpi-sub">Range seen over the last 18 months of tracking</p>
      </div>
    </div>

    <div class="mt36">
      <p class="eyebrow">Is the Signal Still Working?</p>
      <h3 style="font-family:'Playfair Display',serif;font-size:22px;font-weight:400;color:#1A1A1A;margin:8px 0 4px">AI model accuracy — rolling 3-month and 6-month windows</h3>
      <p class="lead" style="margin-top:4px">This chart tracks whether the AI model is still correctly predicting which stocks will go up. Above +0.25 = strong. +0.10 to +0.25 = usable. Below +0.10 = warning sign. A healthy signal stays consistently positive.</p>
      <div class="chart-box mt16">
        <p class="chart-title">Is the model still working? — accuracy over the last 3 and 6 months</p>
        <p class="chart-sub">Blue = how accurate over the last 3 months · Dashed = how accurate over the last 6 months · Gold line = the accuracy level we're aiming for (+{ric_target:.3f})</p>
        <div class="chart-inner" style="height:220px"><canvas id="rollingIcChart"></canvas></div>
      </div>
    </div>

    <div class="mt36">
      <p class="eyebrow">How Each Signal Is Performing</p>
      <h3 style="font-family:'Playfair Display',serif;font-size:22px;font-weight:400;color:#1A1A1A;margin:8px 0 4px">Which signals have been working best recently? (last 18 months)</h3>
      <p class="lead" style="margin-top:4px">Each bar shows whether that signal correctly predicted stock direction that month. Green = signal was right, red = signal was wrong. Momentum has been inconsistent recently — normal in calm markets.</p>
      <div class="chart-box mt16">
        <p class="chart-title">Each signal's accuracy by month — did it correctly predict which stocks would go up?</p>
        <div class="chart-inner" style="height:220px"><canvas id="factorIcChart"></canvas></div>
      </div>
    </div>

    <div class="mt36">
      <p class="eyebrow">Where Does the Return Come From?</p>
      <h3 style="font-family:'Playfair Display',serif;font-size:22px;font-weight:400;color:#1A1A1A;margin:8px 0 4px">Return left over after removing what any index fund could have earned</h3>
      <p class="lead" style="margin-top:4px">This removes the return any passive investor could get just by buying the market, small companies, cheap stocks, etc. What's left is the strategy's genuine edge — return that only this specific model produced, beyond what simple index investing would have given you.</p>
      <div class="ff5-grid">
{ff5_cards()}
      </div>
    </div>

    <div class="mt36">
      <p class="eyebrow">What drove today's gain or loss?</p>
      <h3 style="font-family:'Playfair Display',serif;font-size:22px;font-weight:400;color:#1A1A1A;margin:8px 0 4px">How much did each signal contribute to today's result?</h3>
      <p class="lead" style="margin-top:4px">The AI model and the market mode filter together drive most of the day-to-day results. The price-setup and insider signals add smaller but meaningful boosts. This breakdown tells you which part of the system is working hardest — and which to investigate first if performance drops.</p>
      <div style="background:#fff;border:1px solid #241f18;padding:18px 22px;margin-top:16px">
        <div style="display:grid;grid-template-columns:180px 1fr 60px 65px;gap:10px;padding-bottom:8px;border-bottom:1px solid #241f18;margin-bottom:4px">
          <span style="font-size:10px;letter-spacing:1px;text-transform:uppercase;color:#BBB;font-weight:400">Signal</span>
          <span style="font-size:10px;letter-spacing:1px;text-transform:uppercase;color:#BBB;font-weight:400">Contribution bar</span>
          <span style="font-size:10px;letter-spacing:1px;text-transform:uppercase;color:#BBB;font-weight:400;text-align:right">Share</span>
          <span style="font-size:10px;letter-spacing:1px;text-transform:uppercase;color:#BBB;font-weight:400;text-align:right">P&amp;L %</span>
        </div>
{signal_bars()}
      </div>
    </div>

    <div class="mt36">
      <p class="eyebrow">Full Performance Report</p>
      <h3 style="font-family:'Playfair Display',serif;font-size:22px;font-weight:400;color:#1A1A1A;margin:8px 0 4px">Full picture — growth, risk, and worst loss during testing</h3>
      <p class="lead" style="margin-top:4px">Return alone doesn't tell the whole story. These metrics show how smooth the ride was, how bad the worst periods were, and how quickly the strategy recovered. All computed on real-world data only.</p>
      <div class="tearsheet-grid">
{tearsheet_cards()}
      </div>
    </div>

{monthly_pnl_bars()}

  </div>
</section>

<!-- ============================================================ NEWS -->
<section id="sec-news" class="tab-section">
  <div class="container">
    <p class="eyebrow">News &amp; Events</p>
    <h2 class="section-head">Latest news on your watchlist</h2>
    <div class="rule"></div>

    <div class="refresh-bar">
      <span>This page shows fresh news every time you open it via the dynamic server. <strong>Last loaded: {today}</strong></span>
      <span id="countdown-display" style="font-weight:400;color:#c8b487"></span>
    </div>

    <p class="lead">Each news card shows what the story means for the stock — in plain English. A "Bullish signal" story may support a long position; a "Bearish signal" story is a reason to reconsider. "No clear direction" is context only — wait for price confirmation before acting.</p>

    <div class="news-grid">
{news_cards()}
    </div>
  </div>
</section>

<!-- ============================================================ METHODOLOGY -->
<section id="sec-method" class="tab-section">
  <div class="container">
    <p class="eyebrow">How the Strategy Works</p>
    <h2 class="section-head">How we make sure the results are real — not just lucky</h2>
    <div class="rule"></div>
    <p class="lead">The biggest risk in any strategy is that good-looking results are just luck or hindsight. These six safeguards make sure the results are real.</p>

    <div class="method-grid">
      <div class="method-card acc">
        <p class="method-title">Training data never touches test data</p>
        <p class="method-body">When training the model, we remove any data that overlaps with the test period — plus a 5-day buffer on either side. This prevents any accidental "leaking" of future information into the model.</p>
        <p class="method-hl">5 splits · 21-day buffer · 5-day safety margin</p>
      </div>
      <div class="method-card">
        <p class="method-title">Hard wall at January 2020</p>
        <p class="method-body">Everything from January 2020 onward is a locked test set — untouched during model building. Every result shown from 2020 onward reflects a genuine real-world test, not a replay of known data.</p>
        <p class="method-hl">{bt_months} months of real-world testing</p>
      </div>
      <div class="method-card acc">
        <p class="method-title">No peeking at the future</p>
        <p class="method-body">Every signal is computed using only information available at market close today. The system then acts on the next morning's open price — never on today's closing price or any future data.</p>
        <p class="method-hl">All signals verified: no hindsight used</p>
      </div>
      <div class="method-card">
        <p class="method-title">Model retrained on past data only</p>
        <p class="method-body">Every month, the model trains only on the past year of data — never looking forward. This mimics how you would actually use the strategy in real life, day by day.</p>
        <p class="method-hl">Rolling 1-year training window · monthly update</p>
      </div>
      <div class="method-card acc">
        <p class="method-title">Important caveat: survivorship bias</p>
        <p class="method-body">The stock universe uses today's S&P 500 companies mapped back to 2000. Companies that went bankrupt or were delisted are missing — this makes the results slightly optimistic. We disclose this on every performance table.</p>
        <p class="method-hl">Noted on every performance page</p>
      </div>
      <div class="method-card">
        <p class="method-title">Real-world accuracy vs training accuracy</p>
        <p class="method-body">The ultimate test: does the model work on data it never saw? Training accuracy: +0.434. Real-world accuracy: +0.370. Only −15% drop — academic strategies typically lose 50%+ accuracy when tested on new data.</p>
        <p class="method-hl">Training +0.434 → Real-world +0.370 · only −15% drop</p>
      </div>
    </div>

    <div class="mt36">
      <p class="eyebrow">How Much Money Can This Strategy Handle?</p>
      <table class="mt16">
        <thead><tr><th>Portfolio size</th><th class="r">Trading cost/month</th><th class="r">Signal edge/month</th><th class="r">Net advantage</th><th class="r">Days to complete trades</th></tr></thead>
        <tbody>
          <tr><td>$10M</td><td class="r">0.069%</td><td class="r">0.25%</td><td class="r pos">+0.181%</td><td class="r">&lt;1 day</td></tr>
          <tr><td>$100M</td><td class="r">0.082%</td><td class="r">0.25%</td><td class="r pos">+0.168%</td><td class="r">&lt;1 day</td></tr>
          <tr><td>$500M</td><td class="r">0.105%</td><td class="r">0.25%</td><td class="r pos">+0.145%</td><td class="r">0.2 days</td></tr>
          <tr><td>$1B</td><td class="r">0.123%</td><td class="r">0.25%</td><td class="r pos">+0.127%</td><td class="r">0.5 days</td></tr>
          <tr><td><strong>$2.5B — maximum size</strong></td><td class="r">0.276%</td><td class="r">0.25%</td><td class="r neg"><strong>−0.026% (no longer profitable)</strong></td><td class="r">1.2 days</td></tr>
        </tbody>
      </table>
    </div>
  </div>
</section>

<!-- ============================================================ LIVE TRACK -->
<section id="sec-live" class="tab-section">
  <div class="container">
    <p class="eyebrow">Live Paper Track Record</p>
    <h2 class="section-head">Real-money simulation — tracking every signal in real time</h2>
    <div class="rule"></div>

    <div class="chart-box" style="margin-bottom:32px">
      <p class="chart-title">Simulated portfolio value — tracking the signals in real time since day one</p>
      <p class="chart-sub">Started at ${pn_start:,.2f} · Now at ${pn_final:,.2f} · Total gain/loss: <strong style="color:{pn_color}">{pn_gain:+.2f}%</strong> · Worst drop from peak: <strong style="color:#B83232">{pn_maxdd:.2f}%</strong> · {pn_ndays} trading days tracked</p>
      <div class="chart-inner" style="height:240px"><canvas id="paperNavChart"></canvas></div>
    </div>

    <div class="oos-banner">
      <p class="oos-banner-title">Live tracking status</p>
      <p class="oos-banner-body">Real-time signal tracking started <strong>June 8, 2026</strong>. Every signal is recorded at market close, and the resulting position is calculated at the next morning's open. The system needs <strong>21 trading days</strong> of data before it can measure whether the signals are actually working in real time — expected around <strong>July 8, 2026</strong>. Currently <strong>{live.get("days_acc", 0)} of 21 days</strong> accumulated.</p>
      <div class="oos-kpi-row">
        <div class="oos-kpi"><label>Tracking started</label><span class="oos-kpi-val">2026-06-08</span></div>
        <div class="oos-kpi"><label>Days of data so far</label><span class="oos-kpi-val">{live.get("days_acc", 0)} / 21</span></div>
        <div class="oos-kpi"><label>First real-time score expected</label><span class="oos-kpi-val">~Jul 8 2026</span></div>
        <div class="oos-kpi"><label>Current open buy positions</label><span class="oos-kpi-val g">{len([p for p in live.get("positions",[]) if p["side"]=="LONG"])}</span></div>
      </div>
    </div>

{trade_command_center()}

    <div class="mt36">
      <p class="eyebrow">Position Health Check — Is each position still justified?</p>
      <h3 style="font-family:'Playfair Display',serif;font-size:22px;font-weight:400;color:#1A1A1A;margin:8px 0 4px">Open positions — model signal and recommended size</h3>
      <p class="lead" style="margin-top:4px">For each open position: is the original signal still active? Has the risk gate changed? A green "Signal aligned" means the model still agrees with the position. A "Signal direction changed" means you should review whether to stay in.</p>
      <div class="pos-grid">
{position_cards()}
      </div>
    </div>

    <div class="two-col-even">
      <div>
        <div class="tbl-wrap">
          <p class="tbl-title">▲ BUY — positions currently held in simulation</p>
          <table>
            <thead><tr><th>Date entered</th><th>Ticker</th><th class="r">Price when entered</th><th>Market mode</th></tr></thead>
            <tbody>
{chr(10).join(f'              <tr><td style="font-size:11px;color:#AAA">{p["date"]}</td><td class="td-ticker">{p["ticker"]}</td><td class="r">${p["entry"] if p["entry"] != "—" else "—"}</td><td style="font-size:11px;color:#1B6F4A">{p["regime"]}</td></tr>' for p in live.get("positions",[]) if p["side"]=="LONG") or "              <tr><td colspan='4' style='color:#AAA;text-align:center'>No positions — run step500</td></tr>"}
            </tbody>
          </table>
        </div>
        <div class="tbl-wrap mt24">
          <p class="tbl-title">▼ AVOID — positions flagged as avoid in simulation</p>
          <table>
            <thead><tr><th>Date flagged</th><th>Ticker</th><th class="r">Price when flagged</th><th>Market mode</th></tr></thead>
            <tbody>
{chr(10).join(f'              <tr><td style="font-size:11px;color:#AAA">{p["date"]}</td><td class="td-ticker">{p["ticker"]}</td><td class="r">${p["entry"] if p["entry"] != "—" else "—"}</td><td style="font-size:11px;color:#B83232">{p["regime"]}</td></tr>' for p in live.get("positions",[]) if p["side"]=="SHORT") or "              <tr><td colspan='4' style='color:#AAA;text-align:center'>No avoid positions open</td></tr>"}
            </tbody>
          </table>
          <p class="tbl-note">Avoid positions are sized to offset the buy side — they protect the portfolio when those sectors fall.</p>
        </div>
      </div>

      <div>
        <div class="tbl-wrap">
          <p class="tbl-title">Are the signals working in real time? — early readings</p>
          <table>
            <thead><tr><th>Signal</th><th class="r">Lookback window</th><th class="r">Accuracy score</th><th>Status</th></tr></thead>
            <tbody>
{chr(10).join(f'              <tr><td style="font-size:12px">{r["signal"].replace("sig_","")}</td><td class="r">{r["horizon"]}d</td><td class="r {"pos" if r["ic"]>0 else "neg"}">{r["ic"]:+.4f}</td><td style="font-size:10px;color:#BBB">COMPLETE</td></tr>' for r in (live.get("ic_rows",[]) or [])) or "              <tr><td colspan='4' style='color:#AAA;text-align:center'>Building — first readings after 21 days</td></tr>"}
              <tr style="background:#F9F8F6"><td style="font-size:12px;color:#BBB" colspan="2">20-day IC (1-month)</td><td class="r" style="color:#c8b487">PENDING</td><td style="font-size:10px;color:#c8b487">~Jul 8 2026</td></tr>
            </tbody>
          </table>
          <p class="tbl-note">Short 1–5 day windows are noisy. The 20-day window (1 month) is the main accuracy test — needs 21 trading days to compute.</p>
        </div>

        <div class="tbl-wrap mt24">
          <p class="tbl-title">How the daily score is built — what each component contributes</p>
          <table>
            <thead><tr><th>What it measures</th><th class="r">Weight</th><th class="r">Step</th></tr></thead>
            <tbody>
              <tr><td>AI model (machine learning)</td><td class="r">40%</td><td class="r">Step 66</td></tr>
              <tr><td>Financial health factors</td><td class="r">15%</td><td class="r">Step 410</td></tr>
              <tr><td>Institutional fund activity (13F filings)</td><td class="r">15%</td><td class="r">Step 480</td></tr>
              <tr><td>Earnings quality check</td><td class="r">15%</td><td class="r">Step 470</td></tr>
              <tr><td>Price squeeze setup</td><td class="r">15%</td><td class="r">Step 490</td></tr>
              <tr class="tr-strong"><td><strong>Full pipeline runs at</strong></td><td class="r"><strong>18:00 ET</strong></td><td class="r">Step 500</td></tr>
            </tbody>
          </table>
          <p class="tbl-note">Run manually: <code style="font-size:11px;background:#F3F3F3;padding:2px 5px">.venv/bin/python canyon_final_v9_step500_daily_pipeline.py</code></p>
        </div>
      </div>
    </div>

    <div class="mt36">
      <p class="eyebrow">What should the real-time accuracy score look like?</p>
      <table class="mt16">
        <thead><tr><th>Scenario</th><th class="r">Expected accuracy score</th><th>What it means</th></tr></thead>
        <tbody>
          <tr><td>Tested on data the model had never seen (already verified)</td><td class="r pos">+{oos_ic:.3f}</td><td>Strong — {bt_months} months of data the model never trained on</td></tr>
          <tr><td>Realistic live target</td><td class="r">+0.03 – +0.05</td><td>After trading costs and real-market friction</td></tr>
          <tr><td>Minimum acceptable in real time</td><td class="r">+0.01</td><td>Borderline — hard to tell signal from noise at this level</td></tr>
          <tr><td>If the model has stopped working</td><td class="r neg">&lt; 0</td><td>Signal no longer working — model review needed</td></tr>
          <tr class="tr-strong"><td><strong>Current real-time score (1-month window)</strong></td><td class="r" style="color:#c8b487"><strong>PENDING</strong></td><td><strong>First reading expected ~July 8, 2026</strong></td></tr>
        </tbody>
      </table>
    </div>
  </div>
</section>

<!-- ═══════════════════════════════════════════ v25.1 STRATEGY -->
<section id="sec-v251" class="tab-section">
  <div class="container">
    <p class="eyebrow">TQQQ Strategy with Market Timing · Tested 2019–2026</p>
    <h2 class="section-head">v25.1 Canyon QQQ Hunter — {today}</h2>
    <div class="rule"></div>

    <!-- Live Regime Status -->
    <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:16px;margin:24px 0">
      <div style="background:#fff;border:1px solid #241f18;padding:20px 22px;border-radius:4px">
        <div style="font-size:10px;letter-spacing:1.5px;text-transform:uppercase;color:#999;margin-bottom:6px">QQQ Tactical Signal · LIVE</div>
        <div style="font-family:'Playfair Display',serif;font-size:28px;font-weight:400;color:{_reg_color}">{_reg_regime}</div>
        <div style="font-size:10px;color:#AAA;margin-top:4px">VIX+MA gates · separate from main HMM</div>
        <div style="font-size:10px;color:{"#B83232" if _reg_hmm_bear else "#1B6F4A"};margin-top:3px;font-weight:400">Main HMM: {_reg_hmm}</div>
        <div style="font-size:10px;color:#AAA;margin-top:3px">{_reg_as_of}</div>
      </div>
      <div style="background:#fff;border:1px solid #241f18;padding:20px 22px;border-radius:4px">
        <div style="font-size:10px;letter-spacing:1.5px;text-transform:uppercase;color:#999;margin-bottom:6px">TQQQ Weight · LIVE</div>
        <div style="font-family:'Playfair Display',serif;font-size:28px;font-weight:400;color:{_reg_vix_c}">{_reg_tqqq_s}</div>
        <div style="font-size:11px;color:#AAA;margin-top:5px">VIX {_reg_vix:.1f} → {_reg_tier}</div>
        {_reg_hmm_note}
      </div>
      <div style="background:#fff;border:1px solid #241f18;padding:20px 22px;border-radius:4px">
        <div style="font-size:10px;letter-spacing:1.5px;text-transform:uppercase;color:#999;margin-bottom:6px">Annual Return (tested period)</div>
        <div style="font-family:'Playfair Display',serif;font-size:28px;font-weight:400;color:#1B6F4A">{_v251_ar*100:+.1f}%</div>
        <div style="font-size:11px;color:#AAA;margin-top:5px">Jan 2019 – May 2026 · {_v251_n} months</div>
      </div>
      <div style="background:#fff;border:1px solid #241f18;padding:20px 22px;border-radius:4px">
        <div style="font-size:10px;letter-spacing:1.5px;text-transform:uppercase;color:#999;margin-bottom:6px">Risk score / Worst drop / Beat QQQ</div>
        <div style="font-family:'Playfair Display',serif;font-size:20px;font-weight:400;color:#c8b487">{_v251_sr:.3f} / {_v251_mdd*100:.1f}%</div>
        <div style="font-size:11px;color:#AAA;margin-top:5px">{_v251_beat}/8 years beat QQQ · Recovery score {_v251_cal:.2f}</div>
      </div>
    </div>

    <!-- Live Gate Table -->
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:24px;margin:24px 0">
      <div>
        <div style="font-size:12px;font-weight:400;color:#c8b487;margin-bottom:8px;text-transform:uppercase;letter-spacing:1px">Live Gate Status</div>
        <table>
          <thead><tr><th>Gate</th><th>Live Value</th><th class="r">Status</th></tr></thead>
          <tbody>{v251_regime_rows()}</tbody>
        </table>
      </div>
      <div style="background:#241f18;border-radius:6px;padding:20px">
        <div style="font-size:12px;font-weight:400;color:#c8b487;margin-bottom:10px;text-transform:uppercase;letter-spacing:1px">VIX-Scale Rule</div>
        <div style="font-size:13px;line-height:2;color:#333">
          VIX &lt; 20 → TQQQ <strong style="color:#1B6F4A">50%</strong> (LOW vol · full exposure)<br>
          VIX 20–25 → TQQQ <strong style="color:#c8b487">25%</strong> (MID vol · half exposure)<br>
          VIX ≥ 25 → TQQQ <strong style="color:#B83232">0%</strong> (HIGH vol · cash instead)<br>
          <span style="color:#888;font-size:11px">+ QQQ 200MA gate + QQQ 3M momentum gate<br>All gates use data available at rebalance date (no lookahead)</span>
        </div>
      </div>
    </div>

    <!-- Cumulative Chart -->
    <div style="background:#fff;border:1px solid #241f18;border-radius:4px;padding:24px;margin:24px 0">
      <div style="font-size:13px;font-weight:400;color:#c8b487;margin-bottom:4px">Cumulative Returns — v25.1 Canyon vs QQQ vs SPY (2019–2026)</div>
      <div style="font-size:11px;color:#888;margin-bottom:16px">Starting value = 100. Tested on 89 months of data the model had never seen.</div>
      <div style="position:relative;height:280px"><canvas id="v251CumChart"></canvas></div>
    </div>

    <!-- Annual Returns Table -->
    <div style="font-size:13px;font-weight:400;color:#c8b487;margin:24px 0 8px;text-transform:uppercase;letter-spacing:1px">Annual Returns vs QQQ</div>
    <div style="background:#2a2418;border-radius:6px;overflow:hidden">
      <table style="color:rgba(255,255,255,.85)">
        <thead><tr>
          <th style="padding:10px 12px;color:rgba(255,255,255,.5);font-size:11px">Year</th>
          <th style="text-align:right;padding:10px 12px;color:rgba(255,255,255,.5);font-size:11px">v25.1</th>
          <th style="text-align:right;padding:10px 12px;color:rgba(255,255,255,.5);font-size:11px">QQQ</th>
          <th style="text-align:right;padding:10px 12px;color:rgba(255,255,255,.5);font-size:11px">v25.1 vs QQQ</th>
          <th style="text-align:right;padding:10px 12px;color:rgba(255,255,255,.5);font-size:11px">SPY</th>
        </tr></thead>
        <tbody>{v251_annual_rows()}</tbody>
      </table>
    </div>

    <div style="font-size:11px;color:#999;margin-top:12px">
      Tested period = Jan 2019–May 2026 (89 months of data the model had never seen).
      All decisions were made using only information available at that point in time — no hindsight.
    </div>
  </div>
</section>

<!-- ═══════════════════════════════════════════ DEEP RESEARCH -->
<section id="sec-deep" class="tab-section">
  <div class="container">
    <p class="eyebrow">10 Institutional Statistical Tests · Canyon v25.1</p>
    <h2 class="section-head">Deep Research — {today}</h2>
    <div class="rule"></div>

    <!-- Scorecard -->
    <div style="background:#2a2418;border-radius:8px;padding:24px;margin:24px 0">
      <div style="font-size:11px;letter-spacing:2px;text-transform:uppercase;color:#c8b487;font-weight:400;margin-bottom:16px">Institutional Scorecard — All 10 Tests</div>
      <table style="color:rgba(255,255,255,.85)">
        <thead><tr>
          <th style="padding:6px 8px;color:rgba(255,255,255,.5);font-size:11px">Test</th>
          <th style="text-align:center;padding:6px 8px;color:rgba(255,255,255,.5);font-size:11px">Finding</th>
          <th style="padding:6px 8px;color:rgba(255,255,255,.5);font-size:11px">Grade</th>
        </tr></thead>
        <tbody>
          <tr><td style="padding:7px 8px;border-bottom:1px solid rgba(255,255,255,.06)">① Is the return statistically real?</td><td style="text-align:center;padding:7px">Confidence score {_deep_sr_t:.2f}× above random chance</td><td style="padding:7px;color:#6BCCA0;font-weight:400">A+ PASS</td></tr>
          <tr><td style="padding:7px 8px;border-bottom:1px solid rgba(255,255,255,.06)">② Works in calm markets?</td><td style="text-align:center;padding:7px">Low-vol periods: +104.9%/yr</td><td style="padding:7px;color:#6BCCA0;font-weight:400">A+ PASS</td></tr>
          <tr><td style="padding:7px 8px;border-bottom:1px solid rgba(255,255,255,.06)">③ Consistent over time?</td><td style="text-align:center;padding:7px">Every 3-year window was profitable</td><td style="padding:7px;color:#6BCCA0;font-weight:400">A+ PASS</td></tr>
          <tr><td style="padding:7px 8px;border-bottom:1px solid rgba(255,255,255,.06)">④ How bad can losses get?</td><td style="text-align:center;padding:7px">Avg bad month −2.0% · Worst-case month −5.8%</td><td style="padding:7px;color:#6BCCA0;font-weight:400">A− PASS</td></tr>
          <tr><td style="padding:7px 8px;border-bottom:1px solid rgba(255,255,255,.06)">⑤ Works at sector level?</td><td style="text-align:center;padding:7px">Health care confirmed · 73% of picks in high-signal sectors</td><td style="padding:7px;color:#6BCCA0;font-weight:400">A PASS</td></tr>
          <tr><td style="padding:7px 8px;border-bottom:1px solid rgba(255,255,255,.06)">⑥ Which sectors drive returns?</td><td style="text-align:center;padding:7px">Tech +3.2%/yr · Energy −0.1%/yr</td><td style="padding:7px;color:#6BCCA0;font-weight:400">A PASS</td></tr>
          <tr><td style="padding:7px 8px;border-bottom:1px solid rgba(255,255,255,.06)">⑦ Buy/sell within same sector?</td><td style="text-align:center;padding:7px">Sector-matched pairs: {_deep_sn_ann*100:+.1f}%/yr (statistically significant)</td><td style="padding:7px;color:#6BCCA0;font-weight:400">A PASS</td></tr>
          <tr><td style="padding:7px 8px;border-bottom:1px solid rgba(255,255,255,.06)">⑧ Extra return vs S&P 500?</td><td style="text-align:center;padding:7px">Pure excess return {_deep_alpha*100:+.1f}%/yr · near-zero market sensitivity</td><td style="padding:7px;color:#6BCCA0;font-weight:400">A PASS</td></tr>
          <tr><td style="padding:7px 8px;border-bottom:1px solid rgba(255,255,255,.06)">⑨ Consistent edge vs benchmark?</td><td style="text-align:center;padding:7px">Consistency score {_deep_ir:.3f} vs QQQ ({"exceptional — rare to exceed 1.0" if _deep_ir >= 1.0 else "solid — approaching 1.0 exceptional threshold"})</td><td style="padding:7px;color:#6BCCA0;font-weight:400">{"A+ PASS" if _deep_ir >= 1.0 else "A PASS"}</td></tr>
          <tr><td style="padding:7px 8px">⑩ Still works after trading costs?</td><td style="text-align:center;padding:7px">Max cost drag 1.2%/yr · return after all costs ≈ +45%</td><td style="padding:7px;color:#6BCCA0;font-weight:400">A PASS</td></tr>
        </tbody>
      </table>
    </div>

    <!-- Key Stats Grid -->
    <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:16px;margin:24px 0">
      <div style="background:#EAF5EE;border-radius:6px;padding:16px;text-align:center">
        <div style="font-size:10px;text-transform:uppercase;letter-spacing:1px;color:#1B6F4A;font-weight:400;margin-bottom:6px">Statistical confidence</div>
        <div style="font-size:24px;font-weight:400;color:#1B6F4A">{_deep_sr_t:.2f}×</div>
        <div style="font-size:11px;color:#555;margin-top:4px">Extremely unlikely to be luck — virtually zero chance the return is random</div>
      </div>
      <div style="background:#EAF5EE;border-radius:6px;padding:16px;text-align:center">
        <div style="font-size:10px;text-transform:uppercase;letter-spacing:1px;color:#1B6F4A;font-weight:400;margin-bottom:6px">Consistency vs QQQ</div>
        <div style="font-size:24px;font-weight:400;color:#1B6F4A">{_deep_ir:.3f}</div>
        <div style="font-size:11px;color:#555;margin-top:4px">How consistently it beats QQQ — {"above 1.0 is exceptional and rare" if _deep_ir >= 1.0 else "1.0+ is exceptional; approaching threshold"}</div>
      </div>
      <div style="background:#EAF5EE;border-radius:6px;padding:16px;text-align:center">
        <div style="font-size:10px;text-transform:uppercase;letter-spacing:1px;color:#1B6F4A;font-weight:400;margin-bottom:6px">Pure extra return vs S&P 500</div>
        <div style="font-size:24px;font-weight:400;color:#1B6F4A">{_deep_alpha*100:+.1f}%/yr</div>
        <div style="font-size:11px;color:#555;margin-top:4px">Return above S&P 500 · near-zero sensitivity to the market</div>
      </div>
      <div style="background:#EAF5EE;border-radius:6px;padding:16px;text-align:center">
        <div style="font-size:10px;text-transform:uppercase;letter-spacing:1px;color:#1B6F4A;font-weight:400;margin-bottom:6px">Buy/sell within same sector</div>
        <div style="font-size:24px;font-weight:400;color:#1B6F4A">{_deep_sn_ann*100:+.1f}%/yr</div>
        <div style="font-size:11px;color:#555;margin-top:4px">Matched pairs (same sector, long best / short worst) — statistically confirmed</div>
      </div>
    </div>

    <div style="background:#241f18;border-radius:6px;padding:16px;font-size:12px;color:#555;line-height:1.8">
      <strong>Key findings:</strong> The return-per-risk score (Sharpe) of {oos_sharpe:.3f} is statistically real — {_deep_sr_t:.2f}× above what random chance would produce. Pure extra return above S&P 500 is +{_deep_alpha*100:.1f}%/yr with near-zero sensitivity to overall market moves.
      Consistency score vs QQQ is {_deep_ir:.3f} ({"exceptional — above 1.0 is rare" if _deep_ir >= 1.0 else "solid — 1.0 threshold is exceptional"}). Buy/sell matched pairs within the same sector also produce statistically confirmed returns.
      All 10 quality tests pass at A-grade. Using volatility-based scaling (cutting size in choppy markets) reduces the worst drop from −15.7% to −{abs(deep.get("mdd_v252_cap", deep.get("mdd_v251", 9.37))*100):.1f}% while keeping returns strong.
    </div>
  </div>
</section>

<style>
.man-toc-grid{{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin:32px 0}}
.man-toc-card{{display:flex;align-items:center;gap:12px;background:#fff;border:1px solid #241f18;border-left:3px solid #c8b487;border-radius:8px;padding:12px 16px;text-decoration:none;transition:all .15s}}
.man-toc-card:hover{{background:#F0F4F9;border-left-color:#c8b487;box-shadow:0 2px 10px rgba(27,42,74,.1);transform:translateY(-1px)}}
.man-toc-num{{width:30px;height:30px;min-width:30px;background:#2a2418;color:#c8b487;border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:12px;font-weight:500;font-family:monospace}}
.man-toc-label{{font-size:13px;color:#c8b487;font-weight:400;line-height:1.3}}
.man-ch{{display:flex;align-items:flex-start;gap:16px;margin:60px 0 18px;padding-bottom:14px;border-bottom:2px solid #241f18}}
.man-ch-num{{width:42px;height:42px;min-width:42px;background:#2a2418;color:#c8b487;border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:16px;font-weight:500;margin-top:3px}}
.man-ch-title{{font-family:'Playfair Display',serif;font-size:24px;color:#c8b487;font-weight:400;line-height:1.2}}
</style>
<section id="sec-manual" class="tab-section">
  <div class="container" style="max-width:900px">

    <!-- HERO HEADER -->
    <div style="background:#2a2418;border-radius:12px;padding:40px 48px;margin-bottom:40px;position:relative;overflow:hidden">
      <div style="position:absolute;right:-20px;top:-20px;width:200px;height:200px;border-radius:50%;background:rgba(184,148,63,.08)"></div>
      <p style="font-size:11px;letter-spacing:2.5px;text-transform:uppercase;color:#c8b487;font-weight:400;margin:0 0 12px">Read this first</p>
      <h2 style="font-family:'Playfair Display',serif;font-size:36px;color:#fff;font-weight:400;margin:0 0 14px;line-height:1.15">User Guide</h2>
      <p style="font-size:15px;color:rgba(255,255,255,.65);line-height:1.8;margin:0;max-width:580px">Plain English, no jargon. You do not need any finance background to read this. Every technical term is explained the first time it appears.</p>
    </div>

    <!-- TABLE OF CONTENTS -->
    <p style="font-size:11px;letter-spacing:2px;text-transform:uppercase;color:#c8b487;font-weight:400;margin:0 0 14px">Table of contents — click any section to jump there</p>
    <div class="man-toc-grid">
      <a href="#man-what"     class="man-toc-card"><span class="man-toc-num">1</span><span class="man-toc-label">What is this thing, really?</span></a>
      <a href="#man-open"     class="man-toc-card"><span class="man-toc-num">2</span><span class="man-toc-label">How to open the dashboard every day</span></a>
      <a href="#man-score"    class="man-toc-card"><span class="man-toc-num">3</span><span class="man-toc-label">What is a "model score"?</span></a>
      <a href="#man-mode"     class="man-toc-card"><span class="man-toc-num">4</span><span class="man-toc-label">BULL / BEAR / SIDEWAYS — what does it mean?</span></a>
      <a href="#man-today"    class="man-toc-card"><span class="man-toc-num">5</span><span class="man-toc-label">The Today tab — your daily briefing</span></a>
      <a href="#man-alerts"   class="man-toc-card"><span class="man-toc-num">6</span><span class="man-toc-label">Alerts — what each one means &amp; what to do</span></a>
      <a href="#man-news"     class="man-toc-card"><span class="man-toc-num">7</span><span class="man-toc-label">The News tab — how to read and click cards</span></a>
      <a href="#man-perf"     class="man-toc-card"><span class="man-toc-num">8</span><span class="man-toc-label">The Performance tab — is the model working?</span></a>
      <a href="#man-live"     class="man-toc-card"><span class="man-toc-num">9</span><span class="man-toc-label">Live Track — paper trading explained</span></a>
      <a href="#man-v251"     class="man-toc-card"><span class="man-toc-num">10</span><span class="man-toc-label">The v25.1 Strategy tab — the TQQQ play</span></a>
      <a href="#man-other"    class="man-toc-card"><span class="man-toc-num">11</span><span class="man-toc-label">Other tabs (Signals, Risk, Macro, Attribution)</span></a>
      <a href="#man-workflow" class="man-toc-card"><span class="man-toc-num">12</span><span class="man-toc-label">Your 5-minute morning routine</span></a>
      <a href="#man-mistakes" class="man-toc-card"><span class="man-toc-num">13</span><span class="man-toc-label">Common mistakes to avoid</span></a>
      <a href="#man-faq"      class="man-toc-card"><span class="man-toc-num">14</span><span class="man-toc-label">FAQ — questions people always ask</span></a>
    </div>

    <!-- ① WHAT IS THIS THING -->
    <div id="man-what" style="margin-top:48px">
      <div class="man-ch"><span class="man-ch-num">1</span><div class="man-ch-title">What is this thing, really?</div></div>

      <p style="font-size:15px;line-height:1.9;color:#333;margin-bottom:20px">
        Imagine you hired a very patient analyst who wakes up every night after the market closes, reads the price history, earnings reports, and news headlines for all 495 companies in the S&P 500 index, grades each one with a score from 0 to 100, ranks them, and writes you a morning briefing by the time you wake up. That's what this dashboard does — automatically, every day, for free.
      </p>

      <div style="background:#F0F4F9;border-left:4px solid #3a3128;padding:18px 22px;border-radius:4px;margin-bottom:20px">
        <p style="font-size:14px;font-weight:400;color:#c8b487;margin:0 0 8px">The single most important thing to understand:</p>
        <p style="font-size:14px;line-height:1.8;color:#333;margin:0">This dashboard is a <strong>research tool</strong>, not a trading robot. It cannot buy or sell anything. It has no connection to any brokerage account. Everything it shows you is a suggestion for you to research further — the final decision is always yours.</p>
      </div>

      <div style="display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-bottom:20px">
        <div style="background:#EAF5EE;border-radius:6px;padding:18px 20px">
          <p style="font-size:12px;font-weight:400;color:#1B6F4A;letter-spacing:1px;text-transform:uppercase;margin:0 0 8px">What it does for you</p>
          <p style="font-size:13.5px;color:#333;line-height:1.8;margin:0">
            • Ranks 495 S&amp;P 500 stocks by model score every day<br>
            • Shows you the top 15 stocks to consider buying<br>
            • Shows you the bottom 15 stocks to consider avoiding<br>
            • Sends you price alerts when something unusual happens<br>
            • Summarizes relevant news and flags its market impact<br>
            • Tracks how its own picks have done in real time
          </p>
        </div>
        <div style="background:#FEF9EC;border-radius:6px;padding:18px 20px">
          <p style="font-size:12px;font-weight:400;color:#c8b487;letter-spacing:1px;text-transform:uppercase;margin:0 0 8px">What it does NOT do</p>
          <p style="font-size:13.5px;color:#333;line-height:1.8;margin:0">
            • It does not place any trades on your behalf<br>
            • It cannot access your brokerage account<br>
            • It does not guarantee any profits<br>
            • It does not use paid data — only publicly available prices<br>
            • It does not know your personal financial situation<br>
            • It is not financial advice
          </p>
        </div>
      </div>

      <div style="background:#fff;border:1px solid #241f18;border-radius:6px;padding:18px 22px">
        <p style="font-size:13px;font-weight:400;color:#c8b487;margin:0 0 8px">Where does the data come from?</p>
        <p style="font-size:13.5px;color:#555;line-height:1.8;margin:0">All price data comes from Yahoo Finance — the same free source available to anyone with internet access. The system downloads it automatically every evening after the US stock market closes at 4 PM Eastern time. No paid subscriptions or special access is required.</p>
      </div>
    </div>

    <!-- ② HOW TO OPEN -->
    <div id="man-open" style="margin-top:56px">
      <div class="man-ch"><span class="man-ch-num">2</span><div class="man-ch-title">How to open the dashboard every day</div></div>
      <p style="font-size:14px;color:#555;line-height:1.8;margin-bottom:24px">The dashboard lives on your own computer — it is not hosted on the internet. To use it, you need to start a tiny local server first. Here's how:</p>

      <div style="display:grid;gap:0;border:1px solid #241f18;border-radius:8px;overflow:hidden">
        <div style="display:flex;align-items:stretch">
          <div style="background:#2a2418;color:#c8b487;font-size:18px;font-weight:500;min-width:52px;display:flex;align-items:center;justify-content:center">1</div>
          <div style="padding:18px 22px;flex:1;border-bottom:1px solid #241f18">
            <p style="font-size:14.5px;font-weight:400;color:#c8b487;margin:0 0 6px">Find the file called "Open Canyon Dashboard.command" on your Desktop</p>
            <p style="font-size:13.5px;color:#555;line-height:1.8;margin:0">It is inside the <strong>canyon_quant</strong> folder, which is in your Desktop folder. It looks like a shell script icon (a small document with a terminal symbol). If you can't find it, open Finder and go to Desktop → canyon_quant.</p>
          </div>
        </div>
        <div style="display:flex;align-items:stretch">
          <div style="background:#2a2418;color:#c8b487;font-size:18px;font-weight:500;min-width:52px;display:flex;align-items:center;justify-content:center">2</div>
          <div style="padding:18px 22px;flex:1;border-bottom:1px solid #241f18">
            <p style="font-size:14.5px;font-weight:400;color:#c8b487;margin:0 0 6px">Double-click it</p>
            <p style="font-size:13.5px;color:#555;line-height:1.8;margin:0">A small black Terminal window will pop up for a few seconds, then your web browser (Safari or Chrome) will open automatically and take you to the dashboard. You can close or ignore the Terminal window after that — the server keeps running in the background.</p>
          </div>
        </div>
        <div style="display:flex;align-items:stretch">
          <div style="background:#2a2418;color:#c8b487;font-size:18px;font-weight:500;min-width:52px;display:flex;align-items:center;justify-content:center">3</div>
          <div style="padding:18px 22px;flex:1;border-bottom:1px solid #241f18">
            <p style="font-size:14.5px;font-weight:400;color:#c8b487;margin:0 0 6px">The browser opens to http://localhost:8888</p>
            <p style="font-size:13.5px;color:#555;line-height:1.8;margin:0">Bookmark this address (Cmd+D) so you can come back anytime without double-clicking the file again. This address only works while your Mac is on and awake — it is not a public website that anyone else can visit.</p>
          </div>
        </div>
        <div style="display:flex;align-items:stretch">
          <div style="background:#1B6F4A;color:#fff;font-size:18px;font-weight:500;min-width:52px;display:flex;align-items:center;justify-content:center">✓</div>
          <div style="padding:18px 22px;flex:1">
            <p style="font-size:14.5px;font-weight:400;color:#1B6F4A;margin:0 0 6px">Data updates automatically — you don't need to do anything</p>
            <p style="font-size:13.5px;color:#555;line-height:1.8;margin:0">If the data is more than 8 hours old when you open the dashboard, a banner will appear at the bottom-right corner saying <strong>"⟳ Refreshing data… this takes about 5–10 minutes."</strong> The page will reload itself when it's done. You can keep reading while you wait. If you want to force an immediate update at any time, click the <strong>⟳ Refresh Now</strong> button in the top navigation bar.</p>
          </div>
        </div>
      </div>

      <div style="background:#FEF9EC;border:1px solid #241f18;border-radius:6px;padding:14px 20px;margin-top:16px">
        <p style="font-size:13px;color:#7A6010;line-height:1.7;margin:0"><strong>Tip:</strong> If you see "Dashboard not found" instead of the dashboard, it means the pipeline has never been run. Open Terminal, type <code style="background:rgba(0,0,0,.06);padding:2px 6px;border-radius:3px">cd ~/Desktop/canyon_quant && python3 run_daily.py</code>, press Enter, and wait 10 minutes. This only needs to happen once.</p>
      </div>
    </div>

    <!-- ③ WHAT IS MODEL SCORE -->
    <div id="man-score" style="margin-top:56px">
      <div class="man-ch"><span class="man-ch-num">3</span><div class="man-ch-title">What is a "model score"?</div></div>

      <p style="font-size:14px;color:#555;line-height:1.9;margin-bottom:20px">The model score is like a grade from 0 to 100 that the system gives each stock every day. Think of it exactly like a student's report card score — the higher the number, the better the stock is performing across multiple measures right now. A score of 80 means the stock is doing well on most criteria. A score of 20 means it is doing poorly on most criteria.</p>

      <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:0;border:1px solid #241f18;border-radius:8px;overflow:hidden;margin-bottom:20px">
        <div style="padding:16px;background:#EAF5EE;text-align:center;border-right:1px solid #241f18">
          <p style="font-size:28px;font-weight:500;color:#1B6F4A;margin:0">75–100</p>
          <p style="font-size:12px;font-weight:400;color:#1B6F4A;margin:4px 0 0">Strong buy signal</p>
          <p style="font-size:11.5px;color:#555;margin:6px 0 0;line-height:1.5">Multiple positive signals all pointing in the same direction</p>
        </div>
        <div style="padding:16px;background:#F5F9F0;text-align:center;border-right:1px solid #241f18">
          <p style="font-size:28px;font-weight:500;color:#4A7A2A;margin:0">55–74</p>
          <p style="font-size:12px;font-weight:400;color:#4A7A2A;margin:4px 0 0">Mild positive</p>
          <p style="font-size:11.5px;color:#555;margin:6px 0 0;line-height:1.5">More signals are positive than negative</p>
        </div>
        <div style="padding:16px;background:#FEF9EC;text-align:center;border-right:1px solid #241f18">
          <p style="font-size:28px;font-weight:500;color:#c8b487;margin:0">35–54</p>
          <p style="font-size:12px;font-weight:400;color:#c8b487;margin:4px 0 0">Neutral / mixed</p>
          <p style="font-size:11.5px;color:#555;margin:6px 0 0;line-height:1.5">Average stock. No clear direction</p>
        </div>
        <div style="padding:16px;background:#FDECEA;text-align:center">
          <p style="font-size:28px;font-weight:500;color:#B83232;margin:0">0–34</p>
          <p style="font-size:12px;font-weight:400;color:#B83232;margin:4px 0 0">Avoid / potential short</p>
          <p style="font-size:11.5px;color:#555;margin:6px 0 0;line-height:1.5">Multiple negative signals. The model thinks this stock is weak</p>
        </div>
      </div>

      <p style="font-size:13.5px;color:#555;line-height:1.8;margin-bottom:14px"><strong>What goes into the score?</strong> The model looks at 8–10 different things for each stock, combines them, and produces the final number. Here's what each ingredient means in plain language:</p>

      <div style="display:grid;gap:10px">
        <div style="background:#fff;border:1px solid #241f18;border-radius:6px;padding:14px 18px;display:flex;gap:16px;align-items:flex-start">
          <div style="flex-shrink:0;width:28px;height:28px;background:#2a2418;border-radius:4px;display:flex;align-items:center;justify-content:center"><p style="color:#c8b487;font-size:10px;font-weight:400;margin:0">M</p></div>
          <div>
            <p style="font-size:13.5px;font-weight:400;color:#c8b487;margin:0 0 3px">Momentum</p>
            <p style="font-size:13px;color:#555;line-height:1.7;margin:0">Is the stock's price going up over the past few months compared to other stocks? If NVIDIA has gone up 20% while the average stock went up 5%, its momentum score is high. This is not predicting the future — it's measuring a trend that already exists.</p>
          </div>
        </div>
        <div style="background:#fff;border:1px solid #241f18;border-radius:6px;padding:14px 18px;display:flex;gap:16px;align-items:flex-start">
          <div style="flex-shrink:0;width:28px;height:28px;background:#2a2418;border-radius:4px;display:flex;align-items:center;justify-content:center"><p style="color:#c8b487;font-size:10px;font-weight:400;margin:0">E</p></div>
          <div>
            <p style="font-size:13.5px;font-weight:400;color:#c8b487;margin:0 0 3px">Earnings surprise</p>
            <p style="font-size:13px;color:#555;line-height:1.7;margin:0">When a company reports its quarterly earnings, did it beat expectations or miss them? If the market expected Apple to earn $1.50 per share but they actually earned $1.75 — that's a positive earnings surprise. Companies that consistently beat expectations tend to see their stock rise.</p>
          </div>
        </div>
        <div style="background:#fff;border:1px solid #241f18;border-radius:6px;padding:14px 18px;display:flex;gap:16px;align-items:flex-start">
          <div style="flex-shrink:0;width:28px;height:28px;background:#2a2418;border-radius:4px;display:flex;align-items:center;justify-content:center"><p style="color:#c8b487;font-size:10px;font-weight:400;margin:0">A</p></div>
          <div>
            <p style="font-size:13.5px;font-weight:400;color:#c8b487;margin:0 0 3px">Analyst revisions</p>
            <p style="font-size:13px;color:#555;line-height:1.7;margin:0">Professional Wall Street analysts update their price targets regularly. When many analysts raise their targets for a stock at the same time, it's a positive signal — it means the professional community is getting more optimistic about that company's future.</p>
          </div>
        </div>
        <div style="background:#fff;border:1px solid #241f18;border-radius:6px;padding:14px 18px;display:flex;gap:16px;align-items:flex-start">
          <div style="flex-shrink:0;width:28px;height:28px;background:#2a2418;border-radius:4px;display:flex;align-items:center;justify-content:center"><p style="color:#c8b487;font-size:10px;font-weight:400;margin:0">V</p></div>
          <div>
            <p style="font-size:13.5px;font-weight:400;color:#c8b487;margin:0 0 3px">Volume &amp; liquidity</p>
            <p style="font-size:13px;color:#555;line-height:1.7;margin:0">How many shares are being traded? If a stock suddenly sees 3× its normal trading volume while the price is rising, that's a sign of strong conviction — lots of investors are buying, not just a few. Unusual volume on a quiet day is a warning sign that something may be happening behind the scenes.</p>
          </div>
        </div>
        <div style="background:#fff;border:1px solid #241f18;border-radius:6px;padding:14px 18px;display:flex;gap:16px;align-items:flex-start">
          <div style="flex-shrink:0;width:28px;height:28px;background:#2a2418;border-radius:4px;display:flex;align-items:center;justify-content:center"><p style="color:#c8b487;font-size:10px;font-weight:400;margin:0">R</p></div>
          <div>
            <p style="font-size:13.5px;font-weight:400;color:#c8b487;margin:0 0 3px">Relative strength vs sector</p>
            <p style="font-size:13px;color:#555;line-height:1.7;margin:0">Is this stock outperforming the other companies in its industry? For example, is Pfizer doing better than other pharmaceutical companies? A stock that beats its peers even when the whole sector is down shows unusual strength.</p>
          </div>
        </div>
      </div>

      <div style="background:#F0F4F9;border-radius:6px;padding:16px 20px;margin-top:16px">
        <p style="font-size:13px;color:#555;line-height:1.8;margin:0"><strong>Important:</strong> The model score is a relative measure — it compares stocks to each other. A score of 80 doesn't mean the stock will go up. It means it looks stronger than 80% of the S&P 500 right now. Markets are unpredictable. Use the score as one input, not the only input.</p>
      </div>
    </div>

    <!-- ④ BULL BEAR SIDEWAYS -->
    <div id="man-mode" style="margin-top:56px">
      <div class="man-ch"><span class="man-ch-num">4</span><div class="man-ch-title">BULL / BEAR / SIDEWAYS — what does it mean?</div></div>

      <p style="font-size:14px;color:#555;line-height:1.9;margin-bottom:20px">Every day the dashboard shows a market mode at the top of the Today tab. This tells you the overall health of the stock market right now. Think of it like a weather forecast — not for rain, but for whether conditions are good or bad for investing.</p>

      <div style="display:grid;gap:14px;margin-bottom:20px">
        <div style="display:flex;gap:0;border-radius:8px;overflow:hidden;border:1px solid #26332a">
          <div style="background:#1B6F4A;color:#fff;min-width:90px;display:flex;flex-direction:column;align-items:center;justify-content:center;padding:20px 12px;text-align:center">
            <p style="font-size:22px;font-weight:500;margin:0">BULL</p>
            <p style="font-size:10px;font-weight:400;letter-spacing:1px;opacity:.7;margin:4px 0 0">GREEN</p>
          </div>
          <div style="padding:18px 22px;flex:1;background:#F0FAF2">
            <p style="font-size:13.5px;font-weight:400;color:#1B6F4A;margin:0 0 6px">The market is rising — good conditions for buying</p>
            <p style="font-size:13.5px;color:#333;line-height:1.8;margin:0">This means the S&P 500 index is trending upward and investor sentiment is positive. When the mode is BULL, the model runs at full strength: it recommends 15 buy candidates and tracks 15 avoid candidates. This is the best environment for the strategy. Example: most of 2023–2024 was in BULL mode as AI stocks drove the market up.</p>
          </div>
        </div>
        <div style="display:flex;gap:0;border-radius:8px;overflow:hidden;border:1px solid #241f18">
          <div style="background:#B83232;color:#fff;min-width:90px;display:flex;flex-direction:column;align-items:center;justify-content:center;padding:20px 12px;text-align:center">
            <p style="font-size:22px;font-weight:500;margin:0">BEAR</p>
            <p style="font-size:10px;font-weight:400;letter-spacing:1px;opacity:.7;margin:4px 0 0">RED</p>
          </div>
          <div style="padding:18px 22px;flex:1;background:#FDF0F0">
            <p style="font-size:13.5px;font-weight:400;color:#B83232;margin:0 0 6px">The market is falling — be more careful</p>
            <p style="font-size:13.5px;color:#333;line-height:1.8;margin:0">The S&P 500 is in a downtrend. Even good stocks tend to fall during a bear market because investors are selling broadly. In BEAR mode, the model suggests reducing position sizes or holding more cash. Don't add to losing positions. Example: 2022 was mostly BEAR mode as the Federal Reserve raised interest rates rapidly.</p>
          </div>
        </div>
        <div style="display:flex;gap:0;border-radius:8px;overflow:hidden;border:1px solid #241f18">
          <div style="background:#c8b487;color:#fff;min-width:90px;display:flex;flex-direction:column;align-items:center;justify-content:center;padding:20px 12px;text-align:center">
            <p style="font-size:22px;font-weight:500;margin:0">SIDE</p>
            <p style="font-size:10px;font-weight:400;letter-spacing:1px;opacity:.7;margin:4px 0 0">WAYS</p>
          </div>
          <div style="padding:18px 22px;flex:1;background:#FEF9EC">
            <p style="font-size:13.5px;font-weight:400;color:#c8b487;margin:0 0 6px">No clear direction — be patient</p>
            <p style="font-size:13.5px;color:#333;line-height:1.8;margin:0">The market is choppy — some days up, some days down, with no sustained trend. Stock-picking is harder in this environment because even strong signals can be washed out by random market noise. The model remains active but suggests smaller position sizes. Patience is the right move here.</p>
          </div>
        </div>
      </div>

      <div style="background:#fff;border:1px solid #241f18;border-radius:6px;padding:18px 22px">
        <p style="font-size:13.5px;font-weight:400;color:#c8b487;margin:0 0 8px">How is the mode calculated?</p>
        <p style="font-size:13.5px;color:#555;line-height:1.8;margin:0">The model looks at three things: (1) whether the S&P 500 is above or below its 200-day average price — if it's above, the long-term trend is still up; (2) how much fear is in the market, measured by the VIX index (a number that goes up when investors are nervous and down when they're calm); (3) recent momentum — has the index been gaining or losing ground over the past 3 months? All three inputs together determine the mode.</p>
      </div>
    </div>

    <!-- ⑤ TODAY TAB -->
    <div id="man-today" style="margin-top:56px">
      <div class="man-ch"><span class="man-ch-num">5</span><div class="man-ch-title">The Today tab — your daily briefing</div></div>
      <p style="font-size:14px;color:#555;line-height:1.9;margin-bottom:20px">The Today tab is the first thing you see when you open the dashboard. It is your main daily screen. Everything important is here.</p>

      <div style="display:grid;gap:14px">
        <div style="border:1px solid #241f18;border-radius:8px;overflow:hidden">
          <div style="background:#2a2418;padding:12px 18px;display:flex;align-items:center;gap:10px">
            <span style="background:#c8b487;color:#c8b487;font-size:11px;font-weight:500;padding:2px 8px;border-radius:3px;letter-spacing:1px">SECTION 1</span>
            <p style="color:#fff;font-size:13px;font-weight:400;margin:0">Market Mode Banner</p>
          </div>
          <div style="padding:18px 22px">
            <p style="font-size:13.5px;color:#333;line-height:1.8;margin:0">A large banner at the very top tells you the current market mode (BULL / BEAR / SIDEWAYS) and includes the date of the signals. If it says <em>"signals from 2026-06-28"</em>, that's the date the data was last updated — not the date of a news event. This is normal. The date updates every time the pipeline runs.</p>
          </div>
        </div>

        <div style="border:1px solid #241f18;border-radius:8px;overflow:hidden">
          <div style="background:#2a2418;padding:12px 18px;display:flex;align-items:center;gap:10px">
            <span style="background:#c8b487;color:#c8b487;font-size:11px;font-weight:500;padding:2px 8px;border-radius:3px;letter-spacing:1px">SECTION 2</span>
            <p style="color:#fff;font-size:13px;font-weight:400;margin:0">Top Buy Candidates — the green section</p>
          </div>
          <div style="padding:18px 22px">
            <p style="font-size:13.5px;color:#333;line-height:1.8;margin:0 0 12px">These are the 15 stocks with the highest model score right now. The model is saying: <em>"of all 495 S&P 500 stocks I analyzed today, these 15 look the strongest."</em></p>
            <p style="font-size:13.5px;color:#333;line-height:1.8;margin:0 0 10px">Each stock card shows:</p>
            <div style="display:grid;gap:8px">
              <div style="background:#241f18;border-radius:5px;padding:10px 14px;display:flex;gap:12px">
                <span style="font-weight:400;color:#c8b487;font-size:13px;white-space:nowrap">Rank #1, #2…</span>
                <span style="font-size:13px;color:#555">#1 is the strongest stock today. Not just the stock with the highest score — it accounts for multiple factors including how stable the signal has been.</span>
              </div>
              <div style="background:#241f18;border-radius:5px;padding:10px 14px;display:flex;gap:12px">
                <span style="font-weight:400;color:#c8b487;font-size:13px;white-space:nowrap">Score XX/100</span>
                <span style="font-size:13px;color:#555">The overall grade for this stock today. A score of 82 means this stock ranks in the top 18% of all 495 stocks on most measures.</span>
              </div>
              <div style="background:#241f18;border-radius:5px;padding:10px 14px;display:flex;gap:12px">
                <span style="font-weight:400;color:#c8b487;font-size:13px;white-space:nowrap">Signal bars</span>
                <span style="font-size:13px;color:#555">Small colored bars showing which individual inputs (momentum, earnings, analyst revisions, etc.) are positive. More green bars = more reasons to be optimistic about this stock.</span>
              </div>
              <div style="background:#241f18;border-radius:5px;padding:10px 14px;display:flex;gap:12px">
                <span style="font-weight:400;color:#c8b487;font-size:13px;white-space:nowrap">"In buy list" badge</span>
                <span style="font-size:13px;color:#555">A green badge that appears if this stock is already being tracked in the paper trading log. This means the model has been following it and has an entry price recorded.</span>
              </div>
              <div style="background:#241f18;border-radius:5px;padding:10px 14px;display:flex;gap:12px">
                <span style="font-weight:400;color:#c8b487;font-size:13px;white-space:nowrap">Earnings flag</span>
                <span style="font-size:13px;color:#555">If the company has an earnings report coming up soon (within 3 weeks), a warning appears. Earnings announcements can cause big price swings in either direction — be aware before acting.</span>
              </div>
            </div>
          </div>
        </div>

        <div style="border:1px solid #241f18;border-radius:8px;overflow:hidden">
          <div style="background:#2a2418;padding:12px 18px;display:flex;align-items:center;gap:10px">
            <span style="background:#c8b487;color:#c8b487;font-size:11px;font-weight:500;padding:2px 8px;border-radius:3px;letter-spacing:1px">SECTION 3</span>
            <p style="color:#fff;font-size:13px;font-weight:400;margin:0">Stocks to Avoid — the red section</p>
          </div>
          <div style="padding:18px 22px">
            <p style="font-size:13.5px;color:#333;line-height:1.8;margin:0 0 12px">These are the 15 stocks with the lowest model score right now. The model is saying: <em>"these 15 look the weakest of the 495 I analyzed today."</em></p>
            <p style="font-size:13.5px;color:#333;line-height:1.8;margin:0 0 10px">What should you do with this information?</p>
            <div style="background:#FDECEA;border-radius:6px;padding:14px 18px;margin-bottom:10px">
              <p style="font-size:13.5px;font-weight:400;color:#B83232;margin:0 0 4px">If you own any of these stocks already:</p>
              <p style="font-size:13.5px;color:#333;line-height:1.8;margin:0">Consider whether you still want to hold them. The model sees something unfavorable. You don't have to sell immediately, but it's worth reviewing why you bought in the first place and whether those reasons still hold.</p>
            </div>
            <div style="background:#FEF9EC;border-radius:6px;padding:14px 18px">
              <p style="font-size:13.5px;font-weight:400;color:#c8b487;margin:0 0 4px">If you don't own them:</p>
              <p style="font-size:13.5px;color:#333;line-height:1.8;margin:0">Simply avoid buying them today. The paper trading system tracks these as potential "short" positions — meaning it monitors whether they actually fall, to test if the signal was correct. You don't need to do anything with shorts unless you specifically want to engage in short selling (which is an advanced strategy with unlimited downside risk).</p>
            </div>
          </div>
        </div>

        <div style="border:1px solid #241f18;border-radius:8px;overflow:hidden">
          <div style="background:#2a2418;padding:12px 18px;display:flex;align-items:center;gap:10px">
            <span style="background:#c8b487;color:#c8b487;font-size:11px;font-weight:500;padding:2px 8px;border-radius:3px;letter-spacing:1px">SECTION 4</span>
            <p style="color:#fff;font-size:13px;font-weight:400;margin:0">What Changed Since Yesterday</p>
          </div>
          <div style="padding:18px 22px">
            <p style="font-size:13.5px;color:#333;line-height:1.8;margin:0 0 12px">This is often the most useful part of the Today tab. It shows exactly which stocks entered or exited the top/bottom 15 compared to yesterday:</p>
            <div style="display:grid;gap:10px">
              <div style="display:flex;gap:14px;align-items:flex-start">
                <span style="background:#EAF5EE;color:#1B6F4A;font-size:12px;font-weight:400;padding:4px 10px;border-radius:4px;white-space:nowrap">NEW BUY ▲</span>
                <p style="font-size:13.5px;color:#333;line-height:1.7;margin:0">A stock just entered the top 15 for the first time. This is a fresh, new signal — the model just upgraded this stock. This is the most actionable category. When a stock appears here, it means something changed in its favor overnight (earnings beat, analyst upgrade, price breakout, etc.).</p>
              </div>
              <div style="display:flex;gap:14px;align-items:flex-start">
                <span style="background:#FDECEA;color:#B83232;font-size:12px;font-weight:400;padding:4px 10px;border-radius:4px;white-space:nowrap">NEW AVOID ▼</span>
                <p style="font-size:13.5px;color:#333;line-height:1.7;margin:0">A stock just entered the bottom 15. The model just downgraded it. If you hold this stock, worth investigating why.</p>
              </div>
              <div style="display:flex;gap:14px;align-items:flex-start">
                <span style="background:#F0F4F9;color:#c8b487;font-size:12px;font-weight:400;padding:4px 10px;border-radius:4px;white-space:nowrap">EXITED BUY</span>
                <p style="font-size:13.5px;color:#333;line-height:1.7;margin:0">A stock dropped out of the top 15. It's not a sell signal on its own — it just means something better replaced it. The stock may still be fine, just no longer in the top tier today.</p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- ⑥ ALERTS -->
    <!-- ⑥ ALERTS -->
    <div id="man-alerts" style="margin-top:56px">
      <div class="man-ch"><span class="man-ch-num">6</span><div class="man-ch-title">Alerts — what each one means and what to do</div></div>
      <p style="font-size:14px;color:#555;line-height:1.9;margin-bottom:20px">Alerts appear at the bottom of the Today tab. The system watches all 495 stocks every day and flags anything unusual. Here is every alert type explained in plain language, with exactly what to do:</p>

      <p style="font-size:13.5px;font-weight:400;color:#c8b487;margin:0 0 10px">Step 1 — Read the colored left border to know how urgent it is</p>
      <div style="display:grid;gap:10px;margin-bottom:24px">
        <div style="background:#fff;border:1px solid #241f18;border-left:5px solid #B83232;border-radius:4px;padding:14px 20px">
          <p style="font-size:12px;font-weight:400;color:#B83232;letter-spacing:1px;text-transform:uppercase;margin:0 0 5px">CRITICAL — Red border — Read this now</p>
          <p style="font-size:13.5px;color:#333;line-height:1.8;margin:0">Something that needs your attention today before making any decision. Read it fully. Every red alert tells you exactly what happened and what to consider doing. Do not skip these.</p>
        </div>
        <div style="background:#fff;border:1px solid #241f18;border-left:5px solid #c8b487;border-radius:4px;padding:14px 20px">
          <p style="font-size:12px;font-weight:400;color:#c8b487;letter-spacing:1px;text-transform:uppercase;margin:0 0 5px">WARNING — Orange border — Worth reading soon</p>
          <p style="font-size:13.5px;color:#333;line-height:1.8;margin:0">Something notable happened but there's no rush. Read it when you have a moment, but you don't need to act immediately.</p>
        </div>
        <div style="background:#fff;border:1px solid #241f18;border-left:5px solid #5f7480;border-radius:4px;padding:14px 20px">
          <p style="font-size:12px;font-weight:400;color:#5f7480;letter-spacing:1px;text-transform:uppercase;margin:0 0 5px">INFO — Blue border — No action needed</p>
          <p style="font-size:13.5px;color:#333;line-height:1.8;margin:0">Background context only. The system is keeping you informed. Nothing to do.</p>
        </div>
      </div>

      <p style="font-size:13.5px;font-weight:400;color:#c8b487;margin:0 0 10px">Step 2 — Understand the type of alert and what to do</p>
      <div style="display:grid;gap:12px;margin-bottom:24px">
        <div style="background:#fff;border:1px solid #241f18;border-radius:8px;overflow:hidden">
          <div style="background:#F7F2EA;padding:12px 18px;border-bottom:1px solid #241f18">
            <p style="font-size:13px;font-weight:400;color:#c8b487;margin:0">Price Alert — stock price crossed a key level</p>
          </div>
          <div style="padding:16px 18px">
            <p style="font-size:13.5px;color:#333;line-height:1.8;margin:0 0 12px">The "4-week low" is the lowest closing price over the past 20 trading days. The "4-week high" is the highest. When a stock crosses either of these thresholds, this alert fires.</p>
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px">
              <div style="background:#FDECEA;border-radius:5px;padding:12px 14px">
                <p style="font-size:12px;font-weight:400;color:#B83232;margin:0 0 4px">Broke below 4-week low → CRITICAL</p>
                <p style="font-size:13px;color:#333;line-height:1.7;margin:0 0 8px">Example message: "Price fell to $245.22, which is lower than any close in the past 4 weeks (previous 4-week low was $245.78)." This means the stock is making new recent lows — a clear sign of weakness.</p>
                <p style="font-size:13px;font-weight:400;color:#B83232;margin:0">What to do: Do not add to this position right now. If you already hold it, check your reasons for buying. If those reasons no longer apply, consider reducing or exiting. At minimum, decide in advance how much further it can fall before you will sell.</p>
              </div>
              <div style="background:#EAF5EE;border-radius:5px;padding:12px 14px">
                <p style="font-size:12px;font-weight:400;color:#1B6F4A;margin:0 0 4px">Broke above 4-week high → INFO or WARNING</p>
                <p style="font-size:13px;color:#333;line-height:1.7;margin:0 0 8px">Example: "Price rose to $312.50, setting a new 4-week high." This is a breakout signal — the stock is showing strength. However, many breakouts fail on the first attempt and reverse.</p>
                <p style="font-size:13px;font-weight:400;color:#1B6F4A;margin:0">What to do: Watch for 2–3 more days. If the price holds above the breakout level and the stock is also on the buy list, it may be worth researching as a new entry. Don't chase it on day one.</p>
              </div>
            </div>
          </div>
        </div>

        <div style="background:#fff;border:1px solid #241f18;border-radius:8px;overflow:hidden">
          <div style="background:#F7F2EA;padding:12px 18px;border-bottom:1px solid #241f18">
            <p style="font-size:13px;font-weight:400;color:#c8b487;margin:0">Risk Limit Alert — a position has grown too large</p>
          </div>
          <div style="padding:16px 18px">
            <p style="font-size:13.5px;color:#333;line-height:1.8;margin:0 0 10px">The model has built-in rules that limit how much of the total portfolio can be in any one stock (usually 8–10%) or any one industry sector (usually 25–30%). If a position exceeds these limits — because the stock went up a lot, or because the model kept recommending it — this alert fires.</p>
            <p style="font-size:13.5px;color:#333;line-height:1.8;margin:0 0 10px">This is the model's way of enforcing diversification. A position that becomes too large means a single bad event in that company can hurt the whole portfolio more than intended.</p>
            <p style="font-size:13.5px;font-weight:400;color:#c8b487;margin:0">What to do: Consider trimming the position back to within the limit. This doesn't mean the stock is bad — just that it's taken up more than its intended share of the portfolio. Think of it like rebalancing. Sell enough to bring the position back to 8% of your portfolio.</p>
          </div>
        </div>

        <div style="background:#fff;border:1px solid #241f18;border-radius:8px;overflow:hidden">
          <div style="background:#F7F2EA;padding:12px 18px;border-bottom:1px solid #241f18">
            <p style="font-size:13px;font-weight:400;color:#c8b487;margin:0">News Alert — a major headline was detected</p>
          </div>
          <div style="padding:16px 18px">
            <p style="font-size:13.5px;color:#333;line-height:1.8;margin:0 0 10px">The system scanned recent news and found a significant headline for one of the tracked stocks. "Significant" means something that historically causes large price movements — earnings releases, merger announcements, FDA decisions, executive departures, regulatory actions, major lawsuits, etc.</p>
            <p style="font-size:13.5px;font-weight:400;color:#c8b487;margin:0">What to do: Go to the News tab, find the card for that stock, and click on it to read the full story. The card will tell you whether the news is bullish (green), bearish (red), or mixed (yellow). Then cross-reference with the Today tab to see if the stock is still on the buy list or has now appeared on the avoid list.</p>
          </div>
        </div>

        <div style="background:#fff;border:1px solid #241f18;border-radius:8px;overflow:hidden">
          <div style="background:#F7F2EA;padding:12px 18px;border-bottom:1px solid #241f18">
            <p style="font-size:13px;font-weight:400;color:#c8b487;margin:0">Squeeze Setup — the stock has gone very quiet and a big move may be coming</p>
          </div>
          <div style="padding:16px 18px">
            <p style="font-size:13.5px;color:#333;line-height:1.8;margin:0 0 10px">This is a price pattern. When a stock's daily trading range (the difference between its highest and lowest price each day) gets unusually narrow for several days in a row — much narrower than its normal range — it's a sign that a big move is building. Traders call this a "volatility squeeze" or "coiled spring." The energy is building; when it releases, the stock typically makes a sharp move in one direction.</p>
            <p style="font-size:13.5px;color:#333;line-height:1.8;margin:0 0 10px">Important: the model doesn't know which direction the move will go. It only knows that historically, stocks in this pattern tend to make unusually large moves soon.</p>
            <p style="font-size:13.5px;font-weight:400;color:#c8b487;margin:0">What to do: Watch this stock carefully for the next 3–5 days. If it breaks upward with high volume AND it's on the buy list, research it further. If it breaks downward, it may soon appear on the avoid list. Don't act before seeing the direction of the break.</p>
          </div>
        </div>
      </div>

      <div style="background:#F0F4F9;border-radius:6px;padding:16px 22px">
        <p style="font-size:13.5px;font-weight:400;color:#c8b487;margin:0 0 8px">No alerts today — what does that mean?</p>
        <p style="font-size:13.5px;color:#555;line-height:1.8;margin:0">It means everything is within normal ranges. No unusual price moves, no risk breaches, no major news events for tracked stocks. This is the most common outcome on a normal market day. The model is watching continuously — if something changes, an alert will appear tomorrow. A quiet day is a good day.</p>
      </div>
    </div>

    <!-- ⑦ NEWS -->
    <div id="man-news" style="margin-top:56px">
      <div class="man-ch"><span class="man-ch-num">7</span><div class="man-ch-title">The News tab — how to read and click the cards</div></div>
      <p style="font-size:14px;color:#555;line-height:1.9;margin-bottom:20px">The News tab collects recent news headlines for the stocks the model is tracking. Each story appears as a card. Here's exactly how to use it:</p>

      <div style="background:#fff;border:1px solid #26332a;border-radius:8px;overflow:hidden;margin-bottom:20px;box-shadow:0 2px 8px rgba(0,0,0,.06)">
        <div style="padding:16px 20px 14px">
          <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px">
            <div style="display:flex;align-items:center;gap:10px">
              <p style="font-size:18px;font-weight:500;color:#c8b487;margin:0">FFIV</p>
              <span style="background:#241f18;color:#1B6F4A;font-size:11px;font-weight:400;padding:3px 9px;border-radius:3px">Bullish signal</span>
              <span style="background:#241f18;color:#1B6F4A;font-size:11px;font-weight:400;padding:3px 9px;border-radius:3px;border:1px solid #26332a">In buy list</span>
            </div>
            <span style="font-size:11px;color:#c8b487;background:#F0F4F9;border:1px solid #283038;padding:4px 10px;border-radius:3px;font-weight:400">Tap to expand ▼</span>
          </div>
          <p style="font-size:14px;font-weight:400;color:#1A1A1A;margin:0 0 5px">RBC Capital Raises Price Target on F5 Networks (FFIV) to $310</p>
          <p style="font-size:13px;color:#666;margin:0 0 6px;line-height:1.6">A major investment bank's analyst team increased the price they think this stock is worth. They reviewed recent financial results and became more optimistic about the company's future earnings.</p>
          <p style="font-size:11.5px;color:#999;margin:0">Source: Insider Monkey · June 17, 2026</p>
          <p style="font-size:13px;color:#c8b487;font-weight:400;margin:10px 0 0">→ Positive development. Stock is already on the buy list — this news adds further support to that signal.</p>
        </div>
        <div style="background:#241f18;padding:12px 20px;border-top:1px solid #241f18">
          <p style="font-size:12px;color:#888;margin:0">This is just the summary. Click anywhere on this card to expand it and see the full article text + a button to open the original source.</p>
        </div>
      </div>

      <div style="background:#2a2418;border-radius:8px;padding:18px 22px;margin-bottom:20px">
        <p style="font-size:13px;font-weight:400;color:#c8b487;margin:0 0 12px">How to open the original news article — step by step:</p>
        <div style="display:grid;gap:10px">
          <div style="display:flex;gap:12px;align-items:flex-start">
            <span style="background:#c8b487;color:#c8b487;font-size:12px;font-weight:500;padding:2px 8px;border-radius:3px;flex-shrink:0">1</span>
            <p style="font-size:13.5px;color:rgba(255,255,255,.85);margin:0;line-height:1.7">Click anywhere on the news card — the card will expand and show more text</p>
          </div>
          <div style="display:flex;gap:12px;align-items:flex-start">
            <span style="background:#c8b487;color:#c8b487;font-size:12px;font-weight:500;padding:2px 8px;border-radius:3px;flex-shrink:0">2</span>
            <p style="font-size:13.5px;color:rgba(255,255,255,.85);margin:0;line-height:1.7">Inside the expanded area, find the blue button labeled <strong style="color:#c8b487">"Open source article →"</strong></p>
          </div>
          <div style="display:flex;gap:12px;align-items:flex-start">
            <span style="background:#c8b487;color:#c8b487;font-size:12px;font-weight:500;padding:2px 8px;border-radius:3px;flex-shrink:0">3</span>
            <p style="font-size:13.5px;color:rgba(255,255,255,.85);margin:0;line-height:1.7">Click that button — it opens the original article on the news website in a new browser tab</p>
          </div>
          <div style="display:flex;gap:12px;align-items:flex-start">
            <span style="background:#c8b487;color:#c8b487;font-size:12px;font-weight:500;padding:2px 8px;border-radius:3px;flex-shrink:0">4</span>
            <p style="font-size:13.5px;color:rgba(255,255,255,.85);margin:0;line-height:1.7">To close the expanded card, click anywhere on the card again or click <strong style="color:#c8b487">"Close ▲"</strong></p>
          </div>
        </div>
      </div>

      <p style="font-size:13.5px;font-weight:400;color:#c8b487;margin:0 0 10px">What the colored labels mean:</p>
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px">
        <div style="background:#241f18;border-radius:6px;padding:16px 18px">
          <p style="font-size:12px;font-weight:400;color:#1B6F4A;margin:0 0 6px;text-transform:uppercase;letter-spacing:1px">Bullish signal (green)</p>
          <p style="font-size:13.5px;color:#333;margin:0;line-height:1.7">Good news. Examples: analyst raised their price target, company reported better earnings than expected, a positive product launch, a competitor failed, or the company announced a major new client or contract.</p>
        </div>
        <div style="background:#FDECEA;border-radius:6px;padding:16px 18px">
          <p style="font-size:12px;font-weight:400;color:#B83232;margin:0 0 6px;text-transform:uppercase;letter-spacing:1px">Bearish signal (red)</p>
          <p style="font-size:13.5px;color:#333;margin:0;line-height:1.7">Bad news. Examples: analyst cut their price target or rating, earnings came in worse than expected, a regulatory fine or lawsuit, a major executive leaving, a product recall, or loss of a large customer.</p>
        </div>
        <div style="background:#FEF9EC;border-radius:6px;padding:16px 18px">
          <p style="font-size:12px;font-weight:400;color:#c8b487;margin:0 0 6px;text-transform:uppercase;letter-spacing:1px">Neutral / mixed (yellow)</p>
          <p style="font-size:13.5px;color:#333;margin:0;line-height:1.7">The news could be interpreted either way, or the outcome is too uncertain to classify. Read it and form your own opinion. Don't feel pressure to act on neutral news — it's information, not a signal.</p>
        </div>
        <div style="background:#F0F4F9;border-radius:6px;padding:16px 18px">
          <p style="font-size:12px;font-weight:400;color:#c8b487;margin:0 0 6px;text-transform:uppercase;letter-spacing:1px">"In buy list" / "In avoid list" badge</p>
          <p style="font-size:13.5px;color:#333;margin:0;line-height:1.7">This green or red badge appears when the stock in this news card is currently one of your tracked positions (top 15 or bottom 15). It helps you quickly identify which news is relevant to stocks you're actively following.</p>
        </div>
      </div>
    </div>

    <!-- ⑧ PERFORMANCE -->
    <div id="man-perf" style="margin-top:56px">
      <div class="man-ch"><span class="man-ch-num">8</span><div class="man-ch-title">The Performance tab — is the model actually working?</div></div>
      <p style="font-size:14px;color:#555;line-height:1.9;margin-bottom:20px">This tab shows you the historical evidence that the model's signals work — tested honestly on data the model had never seen before. Think of it as the model's report card. Here's how to read every piece of it:</p>

      <div style="display:grid;gap:14px">
        <div style="background:#fff;border:1px solid #241f18;border-radius:8px;overflow:hidden">
          <div style="background:#2a2418;padding:12px 18px"><p style="color:#c8b487;font-size:12px;font-weight:400;letter-spacing:1px;margin:0;text-transform:uppercase">The portfolio growth chart — blue line vs black line</p></div>
          <div style="padding:18px 22px">
            <p style="font-size:13.5px;color:#333;line-height:1.8;margin:0">Picture two investors: Investor A put $100 into this model's picks on January 1, 2019. Investor B put $100 into an S&P 500 index fund the same day. The blue line shows Investor A's balance over time. The black line shows Investor B. If the blue line is higher on the right side of the chart, the model outperformed just holding the market. All signals were made using only historical data available at that moment — there was no looking into the future.</p>
          </div>
        </div>

        <div style="background:#fff;border:1px solid #241f18;border-radius:8px;overflow:hidden">
          <div style="background:#2a2418;padding:12px 18px"><p style="color:#c8b487;font-size:12px;font-weight:400;letter-spacing:1px;margin:0;text-transform:uppercase">The monthly accuracy bars — green and red</p></div>
          <div style="padding:18px 22px">
            <p style="font-size:13.5px;color:#333;line-height:1.8;margin:0">Each bar is one month. Green means: the model's top-ranked stocks went up more than its bottom-ranked stocks that month (the model got the direction right). Red means the opposite happened. You want to see mostly green bars, especially consistent patterns — not just green in certain years and red in others, which would suggest the model only worked in specific conditions.</p>
          </div>
        </div>

        <div style="background:#fff;border:1px solid #241f18;border-radius:8px;overflow:hidden">
          <div style="background:#2a2418;padding:12px 18px"><p style="color:#c8b487;font-size:12px;font-weight:400;letter-spacing:1px;margin:0;text-transform:uppercase">The 4 headline numbers — plain English explanations</p></div>
          <div style="padding:18px 22px">
            <div style="display:grid;gap:12px">
              <div style="background:#241f18;border-radius:6px;padding:14px 18px">
                <p style="font-size:14px;font-weight:400;color:#c8b487;margin:0 0 6px">Signal accuracy (OOS backtest): {oos_ic:+.3f} · Live 3-month: {ric_cur:+.3f}</p>
                <p style="font-size:13.5px;color:#555;line-height:1.8;margin:0">Think of this as a test score for how well the model's daily ranking predicts actual stock performance. It ranges from -1.0 (100% wrong every time) to +1.0 (perfectly right every time). A score of 0 means no better than random guessing. Professional quantitative investors consider anything above +0.05 to be commercially valuable. The OOS backtest averaged {oos_ic:+.3f} on data the model never trained on — the live 3-month reading as of {_ric_last_date} is {ric_cur:+.3f} ({_live_ic_label.lower()}).</p>
              </div>
              <div style="background:#241f18;border-radius:6px;padding:14px 18px">
                <p style="font-size:14px;font-weight:400;color:#c8b487;margin:0 0 6px">Return per unit of risk (OOS backtest): {oos_sharpe:.2f} · Paper return: {pn_gain:+.2f}%</p>
                <p style="font-size:13.5px;color:#555;line-height:1.8;margin:0">This measures efficiency — how much return did the strategy produce per unit of volatility risk taken? Higher is better. To put it in context: a standard S&P 500 index fund typically scores around 0.5–0.7 over long periods. A score of {oos_sharpe:.2f} means this strategy has produced roughly {oos_sharpe/0.6:.0f}× more return per unit of risk than just holding the index. The live paper portfolio (started Jun 8, 2026) shows {pn_gain:+.2f}% return so far. (Sharpe needs a full year of data to be meaningful for the live track.)</p>
              </div>
              <div style="background:#241f18;border-radius:6px;padding:14px 18px">
                <p style="font-size:14px;font-weight:400;color:#c8b487;margin:0 0 6px">Beat the S&amp;P 500 in {oos_wr:.0f}% of backtest months (OOS 2020–2026)</p>
                <p style="font-size:13.5px;color:#555;line-height:1.8;margin:0">Out of {bt_months} months tested in the out-of-sample period, the strategy produced better monthly returns than just holding the S&P 500 index in {round(oos_wr/100*bt_months):.0f} of those months. This is a very high consistency rate. Even in months where markets fell, the strategy tended to fall less than the index. Note: the test period 2019–2026 included an exceptional bull market in US technology stocks, which may have made results better than typical future periods.</p>
              </div>
              <div style="background:#241f18;border-radius:6px;padding:14px 18px">
                <p style="font-size:14px;font-weight:400;color:#c8b487;margin:0 0 6px">Worst drawdown during the test period</p>
                <p style="font-size:13.5px;color:#555;line-height:1.8;margin:0">The maximum peak-to-trough loss at any point during the test. Every strategy loses money sometimes — the question is how much and for how long. A drawdown of -19% means at some point the strategy fell 19% from its highest point before recovering. This is what you would have experienced if you had been using the strategy during that period. Knowing the historical worst case helps you decide if you can emotionally handle that level of loss.</p>
              </div>
            </div>
          </div>
        </div>
      </div>

      <div style="background:#FEF9EC;border:1px solid #241f18;border-radius:6px;padding:16px 22px;margin-top:16px">
        <p style="font-size:13.5px;font-weight:400;color:#7A6010;margin:0 0 8px">The honest caveat you should know</p>
        <p style="font-size:13.5px;color:#555;line-height:1.8;margin:0">The backtest only includes companies that are still in the S&P 500 today. Companies that went bankrupt or were removed from the index are excluded — this means the results are slightly better than real-life would have been (because in real life you'd have held some of those failing companies). This is a known issue with almost all stock market backtests, called "survivorship bias." Also, the 2020–2026 period was boosted by massive government stimulus and a technology/AI boom that may not repeat. <strong>Past performance does not guarantee future results.</strong></p>
      </div>
    </div>

    <!-- ⑨ LIVE TRACK -->
    <div id="man-live" style="margin-top:56px">
      <div class="man-ch"><span class="man-ch-num">9</span><div class="man-ch-title">Live Track tab — paper trading explained</div></div>
      <p style="font-size:14px;color:#555;line-height:1.9;margin-bottom:20px">"Paper trading" means tracking simulated trades — using real market prices, but without spending any actual money. It's how you test whether the signals work in the real market right now, not just historically. Think of it like a dress rehearsal before a live performance.</p>

      <div style="display:grid;gap:14px">
        <div style="background:#F0F4F9;border-left:4px solid #3a3128;border-radius:4px;padding:18px 22px">
          <p style="font-size:14px;font-weight:400;color:#c8b487;margin:0 0 8px">How it works — in plain language</p>
          <p style="font-size:13.5px;color:#333;line-height:1.8;margin:0">Every day after market close, the model produces a ranked list of all 495 stocks. The top 15 become the "long" (buy) positions. The bottom 15 become the "short" (avoid) positions. The system records the closing prices on the day each stock enters or exits these lists. Then it tracks whether those stocks actually went up or down in the following days. Over time, this builds an honest, real-money-equivalent track record of whether the signals are actually working.</p>
        </div>

        <div style="background:#fff;border:1px solid #241f18;border-radius:8px;overflow:hidden">
          <div style="background:#2a2418;padding:12px 18px"><p style="color:#c8b487;font-size:12px;font-weight:400;letter-spacing:1px;margin:0;text-transform:uppercase">Paper NAV — what the number means</p></div>
          <div style="padding:18px 22px">
            <p style="font-size:13.5px;color:#333;line-height:1.8;margin:0">NAV stands for "Net Asset Value" — it's the total simulated portfolio value. It started at $100 on Day 1 (June 8, 2026). Every day it goes up or down based on how the long positions performed versus the short positions. If today's NAV shows $164, that means the paper portfolio has grown 64% since it started. This is simulated money, not real — but it uses real market prices, so it reflects what would have happened if someone had actually traded these signals.</p>
          </div>
        </div>

        <div style="background:#fff;border:1px solid #241f18;border-radius:8px;overflow:hidden">
          <div style="background:#2a2418;padding:12px 18px"><p style="color:#c8b487;font-size:12px;font-weight:400;letter-spacing:1px;margin:0;text-transform:uppercase">Current positions — open longs and open shorts</p></div>
          <div style="padding:18px 22px">
            <p style="font-size:13.5px;color:#333;line-height:1.8;margin:0">This table shows exactly which stocks are currently being tracked in each category, along with the price they entered at and how much they've moved since. "Open long" means we're tracking these as if we bought them. "Open short" means we're tracking these as if we're betting they'll fall. The entry price is the actual closing price on the day the model first ranked that stock in the relevant category.</p>
          </div>
        </div>

        <div style="background:#fff;border:1px solid #241f18;border-radius:8px;overflow:hidden">
          <div style="background:#2a2418;padding:12px 18px"><p style="color:#c8b487;font-size:12px;font-weight:400;letter-spacing:1px;margin:0;text-transform:uppercase">Days accumulated: 10 / 21 — what does this mean?</p></div>
          <div style="padding:18px 22px">
            <p style="font-size:13.5px;color:#333;line-height:1.8;margin:0">The live track started on June 8, 2026. It needs at least 21 trading days of data to produce a statistically meaningful accuracy score — because with fewer data points, luck can look like skill. 21 trading days is approximately one calendar month. Until that threshold is reached, the counter shows progress (e.g., "10 of 21"). Once 21 days are accumulated, the IC score (how accurately the model predicts returns) becomes meaningful and will be shown prominently.</p>
          </div>
        </div>
      </div>
    </div>

    <!-- ⑩ V25.1 -->
    <div id="man-v251" style="margin-top:56px">
      <div class="man-ch"><span class="man-ch-num">10</span><div class="man-ch-title">The v25.1 Strategy tab — the TQQQ play</div></div>
      <p style="font-size:14px;color:#555;line-height:1.9;margin-bottom:20px">This tab covers a completely separate, much simpler strategy that runs alongside the main Canyon model. While the main model picks individual stocks, v25.1 focuses on just one ETF called TQQQ and uses market fear (the VIX index) to decide how much to hold.</p>

      <div style="background:#2a2418;border-radius:8px;padding:20px 24px;margin-bottom:20px">
        <p style="font-size:13px;font-weight:400;color:#c8b487;margin:0 0 10px;text-transform:uppercase;letter-spacing:1px">First — what is TQQQ?</p>
        <p style="font-size:13.5px;color:rgba(255,255,255,.85);line-height:1.8;margin:0">TQQQ is a leveraged ETF that tracks the Nasdaq 100 index (the 100 biggest technology and growth companies — Apple, Microsoft, NVIDIA, Amazon, etc.). The key word is "leveraged 3×": every 1% the Nasdaq moves, TQQQ moves approximately 3% in the same direction. If the Nasdaq goes up 2%, TQQQ goes up roughly 6%. If the Nasdaq goes down 2%, TQQQ goes down roughly 6%. This makes it extremely powerful in bull markets and extremely dangerous in bear markets.</p>
      </div>

      <div style="background:#fff;border:1px solid #241f18;border-radius:8px;overflow:hidden;margin-bottom:20px">
        <div style="background:#2a2418;padding:12px 18px"><p style="color:#c8b487;font-size:12px;font-weight:400;letter-spacing:1px;margin:0;text-transform:uppercase">The VIX index — what is it and why does it matter?</p></div>
        <div style="padding:18px 22px">
          <p style="font-size:13.5px;color:#333;line-height:1.8;margin:0">The VIX is the "fear gauge" of the stock market. It measures how much nervousness/uncertainty investors are experiencing right now. When markets are calm and rising, VIX is low (below 15–18). When investors are fearful — during crashes, crises, or major uncertainty — VIX spikes high (above 30, 40, sometimes 80+). During COVID in March 2020, VIX hit 85. The v25.1 strategy uses VIX as a safety switch: when fear is high, reduce exposure. When fear is low, increase exposure.</p>
        </div>
      </div>

      <div style="background:#fff;border:1px solid #241f18;border-radius:8px;overflow:hidden;margin-bottom:20px">
        <div style="background:#2a2418;padding:12px 18px"><p style="color:#c8b487;font-size:12px;font-weight:400;letter-spacing:1px;margin:0;text-transform:uppercase">The exact rules — how much to hold and when</p></div>
        <div style="padding:0">
          <div style="display:grid;grid-template-columns:120px 1fr 1fr;border-bottom:1px solid #241f18">
            <div style="padding:12px 16px;background:#241f18;border-right:1px solid #241f18"><p style="font-size:12px;font-weight:400;color:#666;text-transform:uppercase;margin:0">VIX level</p></div>
            <div style="padding:12px 16px;background:#241f18;border-right:1px solid #241f18"><p style="font-size:12px;font-weight:400;color:#666;text-transform:uppercase;margin:0">What it means</p></div>
            <div style="padding:12px 16px;background:#241f18"><p style="font-size:12px;font-weight:400;color:#666;text-transform:uppercase;margin:0">TQQQ allocation</p></div>
          </div>
          <div style="display:grid;grid-template-columns:120px 1fr 1fr;border-bottom:1px solid #241f18;background:#EAF5EE">
            <div style="padding:14px 16px;border-right:1px solid #241f18"><p style="font-size:14px;font-weight:400;color:#1B6F4A;margin:0">Below 20</p></div>
            <div style="padding:14px 16px;border-right:1px solid #241f18"><p style="font-size:13.5px;color:#333;margin:0">Market is calm. Investors are not worried. Good conditions for risk-taking.</p></div>
            <div style="padding:14px 16px"><p style="font-size:16px;font-weight:500;color:#1B6F4A;margin:0">50% in TQQQ</p></div>
          </div>
          <div style="display:grid;grid-template-columns:120px 1fr 1fr;border-bottom:1px solid #241f18;background:#FEF9EC">
            <div style="padding:14px 16px;border-right:1px solid #241f18"><p style="font-size:14px;font-weight:400;color:#c8b487;margin:0">20–25</p></div>
            <div style="padding:14px 16px;border-right:1px solid #241f18"><p style="font-size:13.5px;color:#333;margin:0">Moderate anxiety. Some turbulence. Caution recommended.</p></div>
            <div style="padding:14px 16px"><p style="font-size:16px;font-weight:500;color:#c8b487;margin:0">25% in TQQQ</p></div>
          </div>
          <div style="display:grid;grid-template-columns:120px 1fr 1fr;background:#FDECEA">
            <div style="padding:14px 16px;border-right:1px solid #241f18"><p style="font-size:14px;font-weight:400;color:#B83232;margin:0">Above 25</p></div>
            <div style="padding:14px 16px;border-right:1px solid #241f18"><p style="font-size:13.5px;color:#333;margin:0">High fear. Market is stressed. The kind of environment where TQQQ can fall 20%+ quickly.</p></div>
            <div style="padding:14px 16px"><p style="font-size:16px;font-weight:500;color:#B83232;margin:0">0% — move to cash</p></div>
          </div>
        </div>
      </div>

      <p style="font-size:13.5px;color:#555;line-height:1.8;margin-bottom:14px">In addition to the VIX check, two more conditions must both be true before holding TQQQ at all: (1) the Nasdaq 100 index must be above its 200-day average price (meaning the long-term trend is still up), and (2) the Nasdaq must have gained ground over the past 3 months. If either condition fails, the allocation drops to 0% regardless of VIX.</p>

      <div style="display:grid;grid-template-columns:1fr 1fr;gap:14px">
        <div style="background:#EAF5EE;border-radius:6px;padding:16px 18px">
          <p style="font-size:12px;font-weight:400;color:#1B6F4A;margin:0 0 6px;text-transform:uppercase;letter-spacing:1px">Why this approach works historically</p>
          <p style="font-size:13.5px;color:#333;margin:0;line-height:1.7">The biggest risk with leveraged ETFs is catastrophic drawdowns during crashes. TQQQ fell over 80% in 2022. By using VIX as a fear gauge, the strategy exits before the worst damage happens. From 2012–2026, this approach produced an average annual return of +46% with a maximum drawdown of only -9.4%, compared to buy-and-hold TQQQ which had drawdowns of 80%+.</p>
        </div>
        <div style="background:#FDECEA;border-radius:6px;padding:16px 18px">
          <p style="font-size:12px;font-weight:400;color:#B83232;margin:0 0 6px;text-transform:uppercase;letter-spacing:1px">The serious risks you must understand</p>
          <p style="font-size:13.5px;color:#333;margin:0;line-height:1.7">TQQQ is a complex, high-risk instrument. Even 50% in TQQQ means significant exposure to technology sector volatility. VIX doesn't always spike before crashes — sometimes markets drop fast before VIX catches up. This strategy is only appropriate for money you can genuinely afford to lose entirely. Do not use retirement savings or emergency funds for this.</p>
        </div>
      </div>
    </div>

    <!-- ⑪ OTHER TABS -->
    <div id="man-other" style="margin-top:56px">
      <div class="man-ch"><span class="man-ch-num">11</span><div class="man-ch-title">Other tabs — quick reference</div></div>
      <div style="display:grid;gap:10px">
        <div style="background:#fff;border:1px solid #241f18;border-radius:6px;padding:16px 20px;display:flex;gap:16px;align-items:flex-start">
          <div style="flex-shrink:0;background:#2a2418;color:#c8b487;font-size:11px;font-weight:400;padding:4px 8px;border-radius:3px;white-space:nowrap">Signals</div>
          <p style="font-size:13.5px;color:#555;line-height:1.7;margin:0">Shows the full ranked list of all 495 stocks with their scores. You can scroll through and search for any ticker to see its current score, rank, and which signals are driving it. Useful if you want to look up a specific stock that's not in the top or bottom 15.</p>
        </div>
        <div style="background:#fff;border:1px solid #241f18;border-radius:6px;padding:16px 20px;display:flex;gap:16px;align-items:flex-start">
          <div style="flex-shrink:0;background:#2a2418;color:#c8b487;font-size:11px;font-weight:400;padding:4px 8px;border-radius:3px;white-space:nowrap">Risk</div>
          <p style="font-size:13.5px;color:#555;line-height:1.7;margin:0">Shows a detailed risk analysis of the current paper portfolio — how much is in each sector, which positions are largest, concentration metrics, and whether any position-sizing rules are being violated. Useful for understanding the risk profile of the current picks.</p>
        </div>
        <div style="background:#fff;border:1px solid #241f18;border-radius:6px;padding:16px 20px;display:flex;gap:16px;align-items:flex-start">
          <div style="flex-shrink:0;background:#2a2418;color:#c8b487;font-size:11px;font-weight:400;padding:4px 8px;border-radius:3px;white-space:nowrap">Macro</div>
          <p style="font-size:13.5px;color:#555;line-height:1.7;margin:0">Shows the big-picture economic environment: VIX level, interest rate trends, sector rotation (which sectors money is flowing into or out of), and the overall market trend. Useful for understanding why the market is in BULL vs BEAR mode and what's driving the current regime.</p>
        </div>
        <div style="background:#fff;border:1px solid #241f18;border-radius:6px;padding:16px 20px;display:flex;gap:16px;align-items:flex-start">
          <div style="flex-shrink:0;background:#2a2418;color:#c8b487;font-size:11px;font-weight:400;padding:4px 8px;border-radius:3px;white-space:nowrap">Attribution</div>
          <p style="font-size:13.5px;color:#555;line-height:1.7;margin:0">Shows which signals contributed most to recent returns — which individual factors (momentum, earnings, analyst revisions, etc.) were the biggest drivers of portfolio performance this month. Useful for understanding whether the strategy is working for the right reasons.</p>
        </div>
        <div style="background:#fff;border:1px solid #241f18;border-radius:6px;padding:16px 20px;display:flex;gap:16px;align-items:flex-start">
          <div style="flex-shrink:0;background:#2a2418;color:#c8b487;font-size:11px;font-weight:400;padding:4px 8px;border-radius:3px;white-space:nowrap">Methodology</div>
          <p style="font-size:13.5px;color:#555;line-height:1.7;margin:0">A technical description of how the model was built — which signals are used, how they're weighted, how the backtest was conducted, and known limitations. This tab is for readers with a finance or quantitative background. You don't need to read it to use the dashboard.</p>
        </div>
        <div style="background:#fff;border:1px solid #241f18;border-radius:6px;padding:16px 20px;display:flex;gap:16px;align-items:flex-start">
          <div style="flex-shrink:0;background:#2a2418;color:#c8b487;font-size:11px;font-weight:400;padding:4px 8px;border-radius:3px;white-space:nowrap">Deep Research</div>
          <p style="font-size:13.5px;color:#555;line-height:1.7;margin:0">In-depth analysis of individual top picks — additional context beyond the model score, including valuation, sector trends, earnings history, and analyst consensus. Useful when you've identified a stock on the buy list and want to research it further before making a decision.</p>
        </div>
      </div>
    </div>

    <!-- ⑫ DAILY ROUTINE -->
    <div id="man-workflow" style="margin-top:56px">
      <div class="man-ch"><span class="man-ch-num">12</span><div class="man-ch-title">Your 5-minute morning routine</div></div>
      <p style="font-size:14px;color:#555;line-height:1.9;margin-bottom:24px">You don't need to spend hours on this. The model does the work overnight. Your job every morning is to review what it found and decide whether anything needs your attention. Here's the exact routine:</p>

      <div style="border:1px solid #241f18;border-radius:8px;overflow:hidden">
        <div style="display:flex;align-items:stretch">
          <div style="background:#2a2418;min-width:56px;display:flex;flex-direction:column;align-items:center;justify-content:center;padding:16px 8px;border-bottom:1px solid rgba(255,255,255,.1)">
            <p style="color:#c8b487;font-size:20px;font-weight:500;margin:0">1</p>
            <p style="color:rgba(255,255,255,.5);font-size:10px;margin:2px 0 0;text-align:center">30 sec</p>
          </div>
          <div style="padding:18px 22px;flex:1;border-bottom:1px solid #241f18">
            <p style="font-size:14.5px;font-weight:400;color:#c8b487;margin:0 0 6px">Check the Market Mode at the top of the Today tab</p>
            <p style="font-size:13.5px;color:#555;line-height:1.8;margin:0">BULL, BEAR, or SIDEWAYS? This tells you the context for everything else you'll read today. In BULL mode, the signals are firing at full strength. In BEAR mode, be cautious about any action. In SIDEWAYS mode, patience is the right move.</p>
          </div>
        </div>
        <div style="display:flex;align-items:stretch">
          <div style="background:#2a2418;min-width:56px;display:flex;flex-direction:column;align-items:center;justify-content:center;padding:16px 8px;border-bottom:1px solid rgba(255,255,255,.1)">
            <p style="color:#c8b487;font-size:20px;font-weight:500;margin:0">2</p>
            <p style="color:rgba(255,255,255,.5);font-size:10px;margin:2px 0 0;text-align:center">1 min</p>
          </div>
          <div style="padding:18px 22px;flex:1;border-bottom:1px solid #241f18">
            <p style="font-size:14.5px;font-weight:400;color:#c8b487;margin:0 0 6px">Scroll to the Alerts section — check for any red CRITICAL alerts</p>
            <p style="font-size:13.5px;color:#555;line-height:1.8;margin:0">Are there any red-bordered alerts? If yes, read the full alert — it tells you exactly what happened and what action to consider. Orange warnings: read them but no rush. No alerts at all: excellent — nothing needs your attention today.</p>
          </div>
        </div>
        <div style="display:flex;align-items:stretch">
          <div style="background:#2a2418;min-width:56px;display:flex;flex-direction:column;align-items:center;justify-content:center;padding:16px 8px;border-bottom:1px solid rgba(255,255,255,.1)">
            <p style="color:#c8b487;font-size:20px;font-weight:500;margin:0">3</p>
            <p style="color:rgba(255,255,255,.5);font-size:10px;margin:2px 0 0;text-align:center">2 min</p>
          </div>
          <div style="padding:18px 22px;flex:1;border-bottom:1px solid #241f18">
            <p style="font-size:14.5px;font-weight:400;color:#c8b487;margin:0 0 6px">Look at "What Changed Since Yesterday" — focus on NEW BUY entries</p>
            <p style="font-size:13.5px;color:#555;line-height:1.8;margin:0">A stock that just entered the top 15 for the first time ("New buy ▲") is the most actionable signal the dashboard produces. Something changed overnight that pushed it up. Research that stock further. Similarly, any stock that just appeared as "New avoid ▼" warrants a look if you happen to hold it.</p>
          </div>
        </div>
        <div style="display:flex;align-items:stretch">
          <div style="background:#2a2418;min-width:56px;display:flex;flex-direction:column;align-items:center;justify-content:center;padding:16px 8px;border-bottom:1px solid rgba(255,255,255,.1)">
            <p style="color:#c8b487;font-size:20px;font-weight:500;margin:0">4</p>
            <p style="color:rgba(255,255,255,.5);font-size:10px;margin:2px 0 0;text-align:center">1 min</p>
          </div>
          <div style="padding:18px 22px;flex:1;border-bottom:1px solid #241f18">
            <p style="font-size:14.5px;font-weight:400;color:#c8b487;margin:0 0 6px">Click the News tab — scan for cards with "In buy list" or "In avoid list" badges</p>
            <p style="font-size:13.5px;color:#555;line-height:1.8;margin:0">These are the news stories most relevant to your current tracked positions. Click any card to expand it and read the summary. If something looks significant, click "Open source article →" to read the full story. You don't need to read every card — just the ones with badges or red/green labels.</p>
          </div>
        </div>
        <div style="display:flex;align-items:stretch">
          <div style="background:#1B6F4A;min-width:56px;display:flex;flex-direction:column;align-items:center;justify-content:center;padding:16px 8px">
            <p style="color:#fff;font-size:20px;font-weight:500;margin:0">✓</p>
            <p style="color:rgba(255,255,255,.7);font-size:10px;margin:2px 0 0;text-align:center">done</p>
          </div>
          <div style="padding:18px 22px;flex:1">
            <p style="font-size:14.5px;font-weight:400;color:#1B6F4A;margin:0 0 6px">You're done — close the tab and go about your day</p>
            <p style="font-size:13.5px;color:#555;line-height:1.8;margin:0">The system runs overnight automatically. Fresh data will be ready tomorrow morning. You don't need to check it again until tomorrow — unless a specific news event happens that you want to look up. Avoid the temptation to check it multiple times a day; this is a daily signal system, not an intraday trading tool.</p>
          </div>
        </div>
      </div>
    </div>

    <!-- ⑬ COMMON MISTAKES -->
    <div id="man-mistakes" style="margin-top:56px">
      <div class="man-ch"><span class="man-ch-num">13</span><div class="man-ch-title">Common mistakes to avoid</div></div>
      <div style="display:grid;gap:12px">
        <div style="background:#FDECEA;border-left:4px solid #B83232;border-radius:4px;padding:16px 20px">
          <p style="font-size:13.5px;font-weight:400;color:#B83232;margin:0 0 6px">Mistake: Acting on a signal without reading the earnings flag</p>
          <p style="font-size:13.5px;color:#333;line-height:1.8;margin:0">If a stock is on the buy list but has an earnings report coming up in the next 3 weeks, the card shows a warning flag. Earnings announcements can cause sudden 10–20% price moves in either direction — even if the stock otherwise looks strong. Read the flag before deciding. The model does not predict earnings outcomes.</p>
        </div>
        <div style="background:#FDECEA;border-left:4px solid #B83232;border-radius:4px;padding:16px 20px">
          <p style="font-size:13.5px;font-weight:400;color:#B83232;margin:0 0 6px">Mistake: Ignoring the market mode and buying aggressively in BEAR mode</p>
          <p style="font-size:13.5px;color:#333;line-height:1.8;margin:0">Even the best individual stocks tend to fall when the overall market is falling. If the market mode is BEAR, the environment is unfavorable for new long positions. Waiting for the mode to return to BULL or SIDEWAYS before adding positions has historically produced better results than buying into a falling market.</p>
        </div>
        <div style="background:#FDECEA;border-left:4px solid #B83232;border-radius:4px;padding:16px 20px">
          <p style="font-size:13.5px;font-weight:400;color:#B83232;margin:0 0 6px">Mistake: Treating the model score as a price target or guarantee</p>
          <p style="font-size:13.5px;color:#333;line-height:1.8;margin:0">A score of 85/100 does not mean this stock will go up 85%. It means the stock looks stronger than 85% of the S&P 500 right now on the model's measures. The score can change tomorrow if new data arrives. Use it as a ranking tool, not a prediction.</p>
        </div>
        <div style="background:#FDECEA;border-left:4px solid #B83232;border-radius:4px;padding:16px 20px">
          <p style="font-size:13.5px;font-weight:400;color:#B83232;margin:0 0 6px">Mistake: Checking the dashboard multiple times per day looking for intraday signals</p>
          <p style="font-size:13.5px;color:#333;line-height:1.8;margin:0">This system uses end-of-day closing prices. The signals are designed to be held for days to weeks, not minutes or hours. Looking at it at 10 AM vs 2 PM vs 4 PM will show you the same data. The only time data changes is after 4 PM market close when the pipeline runs.</p>
        </div>
        <div style="background:#FEF9EC;border-left:4px solid #c8b487;border-radius:4px;padding:16px 20px">
          <p style="font-size:13.5px;font-weight:400;color:#c8b487;margin:0 0 6px">Tip: Don't over-concentrate — own more than just #1 on the list</p>
          <p style="font-size:13.5px;color:#333;line-height:1.8;margin:0">Even if #1 on the buy list looks amazing, putting everything into one stock is very risky. The model shows you 15 picks for a reason — spreading across 10–15 positions reduces the damage if any single one fails unexpectedly. No signal, however strong, predicts the future with certainty.</p>
        </div>
      </div>
    </div>

    <!-- ⑭ FAQ -->
    <div id="man-faq" style="margin-top:56px;margin-bottom:60px">
      <div class="man-ch"><span class="man-ch-num">14</span><div class="man-ch-title">FAQ — questions people always ask</div></div>
      <div style="display:grid;gap:10px">
        <details style="background:#fff;border:1px solid #241f18;border-radius:6px;padding:0">
          <summary style="padding:16px 20px;cursor:pointer;font-size:14px;font-weight:400;color:#c8b487;list-style:none;display:flex;justify-content:space-between;align-items:center">Is this using real money? <span style="color:#BBB;font-weight:400;font-size:16px">▼</span></summary>
          <p style="padding:4px 20px 18px;font-size:13.5px;color:#555;line-height:1.8;margin:0">No. The dashboard is a research tool. It has no connection to any brokerage account or bank. All "paper trading" results are simulations using real market prices — but no actual money changes hands. This is not financial advice. Any real trading decisions you make based on this research are entirely your own responsibility.</p>
        </details>
        <details style="background:#fff;border:1px solid #241f18;border-radius:6px;padding:0">
          <summary style="padding:16px 20px;cursor:pointer;font-size:14px;font-weight:400;color:#c8b487;list-style:none;display:flex;justify-content:space-between;align-items:center">The date on the dashboard is from yesterday. Is something broken? <span style="color:#BBB;font-weight:400;font-size:16px">▼</span></summary>
          <p style="padding:4px 20px 18px;font-size:13.5px;color:#555;line-height:1.8;margin:0">This is normal if you open the dashboard during market hours (before 4 PM Eastern time). The data updates after the market closes, so before the daily pipeline runs you'll see yesterday's signals. If you want to force an immediate update, click <strong>⟳ Refresh Now</strong> in the top navigation bar. Also note: weekends and US market holidays have no new data — the signals from Friday carry forward to Monday.</p>
        </details>
        <details style="background:#fff;border:1px solid #241f18;border-radius:6px;padding:0">
          <summary style="padding:16px 20px;cursor:pointer;font-size:14px;font-weight:400;color:#c8b487;list-style:none;display:flex;justify-content:space-between;align-items:center">The page is spinning and says "Refreshing data…" — how long does it take? <span style="color:#BBB;font-weight:400;font-size:16px">▼</span></summary>
          <p style="padding:4px 20px 18px;font-size:13.5px;color:#555;line-height:1.8;margin:0">The full pipeline downloads fresh price data for 495 stocks, runs all the signal calculations, generates the new HTML file, and saves all outputs. This takes 5–10 minutes on most computers. The page will reload itself automatically when it's done. You can read the current version of the dashboard while waiting — nothing is broken. If it takes more than 20 minutes, try clicking ⟳ Refresh Now again.</p>
        </details>
        <details style="background:#fff;border:1px solid #241f18;border-radius:6px;padding:0">
          <summary style="padding:16px 20px;cursor:pointer;font-size:14px;font-weight:400;color:#c8b487;list-style:none;display:flex;justify-content:space-between;align-items:center">A stock is #1 on the buy list. Should I buy it right now? <span style="color:#BBB;font-weight:400;font-size:16px">▼</span></summary>
          <p style="padding:4px 20px 18px;font-size:13.5px;color:#555;line-height:1.8;margin:0">The model score is a research input — not a buy order. Before acting, check: (1) Does the stock have an earnings report coming up soon? If yes, wait. (2) Is the market mode BULL, BEAR, or SIDEWAYS? In BEAR mode, even top-ranked stocks often fall. (3) Do you already know what this company does and why it's strong? Read the news card. (4) Does this fit within your overall portfolio sizing — don't put more than 8–10% of your capital in any single stock. The model narrows your universe from 495 to 15. You still do the final research.</p>
        </details>
        <details style="background:#fff;border:1px solid #241f18;border-radius:6px;padding:0">
          <summary style="padding:16px 20px;cursor:pointer;font-size:14px;font-weight:400;color:#c8b487;list-style:none;display:flex;justify-content:space-between;align-items:center">Why only 15 stocks? Used to be 97. <span style="color:#BBB;font-weight:400;font-size:16px">▼</span></summary>
          <p style="padding:4px 20px 18px;font-size:13.5px;color:#555;line-height:1.8;margin:0">The backtests that showed strong outperformance used concentrated portfolios of 10–20 stocks — not 97. Holding 97 stocks is essentially the same as holding an index fund: you get average returns minus costs. The model's "edge" — its ability to identify the strongest stocks — only works if you actually concentrate on those strongest stocks. Holding 97 dilutes the signal so much that the edge disappears. 15 concentrated picks with high scores have historically produced far better returns than 97 mediocre picks.</p>
        </details>
        <details style="background:#fff;border:1px solid #241f18;border-radius:6px;padding:0">
          <summary style="padding:16px 20px;cursor:pointer;font-size:14px;font-weight:400;color:#c8b487;list-style:none;display:flex;justify-content:space-between;align-items:center">The backtest shows +46% annual return. Is that real? <span style="color:#BBB;font-weight:400;font-size:16px">▼</span></summary>
          <p style="padding:4px 20px 18px;font-size:13.5px;color:#555;line-height:1.8;margin:0">The backtest was conducted honestly: every signal used only data available at that moment in time — no future data was used. That said, the number has caveats: (1) Survivorship bias — only current S&P 500 members are included; failed companies are excluded, making results slightly better than real life would have been. (2) The period 2020–2026 was driven by extraordinary tech/AI tailwinds that may not repeat. (3) Transaction costs and slippage (the gap between theoretical prices and actual execution prices) are not fully accounted for. Real-world returns would likely be lower. <strong>Past results cannot guarantee future performance.</strong></p>
        </details>
        <details style="background:#fff;border:1px solid #241f18;border-radius:6px;padding:0">
          <summary style="padding:16px 20px;cursor:pointer;font-size:14px;font-weight:400;color:#c8b487;list-style:none;display:flex;justify-content:space-between;align-items:center">I see a CRITICAL alert. Do I have to sell immediately? <span style="color:#BBB;font-weight:400;font-size:16px">▼</span></summary>
          <p style="padding:4px 20px 18px;font-size:13.5px;color:#555;line-height:1.8;margin:0">No. A CRITICAL alert is a flag that says "something happened — consider this information before your next decision." It is not an automatic sell order. Read the alert carefully — the orange action line at the bottom tells you what to consider doing. Often it says "don't add to this position" rather than "sell immediately." You decide. The model surfaces the information; the human makes the call.</p>
        </details>
        <details style="background:#fff;border:1px solid #241f18;border-radius:6px;padding:0">
          <summary style="padding:16px 20px;cursor:pointer;font-size:14px;font-weight:400;color:#c8b487;list-style:none;display:flex;justify-content:space-between;align-items:center">Can I use this for stocks outside the S&amp;P 500? <span style="color:#BBB;font-weight:400;font-size:16px">▼</span></summary>
          <p style="padding:4px 20px 18px;font-size:13.5px;color:#555;line-height:1.8;margin:0">Currently no — the model was trained and calibrated on S&P 500 stocks only. The signals may not transfer to small-cap stocks, international stocks, ETFs, or crypto. The stock universe is the approximately 495 companies currently in the S&P 500 index. Adding other assets would require retraining the model on their specific data characteristics.</p>
        </details>
        <details style="background:#fff;border:1px solid #241f18;border-radius:6px;padding:0">
          <summary style="padding:16px 20px;cursor:pointer;font-size:14px;font-weight:400;color:#c8b487;list-style:none;display:flex;justify-content:space-between;align-items:center">Do I need to do anything to set up the daily auto-refresh? <span style="color:#BBB;font-weight:400;font-size:16px">▼</span></summary>
          <p style="padding:4px 20px 18px;font-size:13.5px;color:#555;line-height:1.8;margin:0">If you've set up the cron job (the automatic daily schedule), the pipeline runs itself at 6 PM every weekday and fresh data is ready before you wake up. If the cron job is not installed, the dashboard will still refresh automatically when you open it — it checks whether the data is more than 8 hours old and triggers a refresh if needed. You don't have to do anything manually. Just open the dashboard and let it run.</p>
        </details>
      </div>
    </div>

  </div>
</section>

<section id="sec-shorts" class="tab-section">
  <div class="container">
    <h2 class="section-head">Short Technical Scanner</h2>
    <p style="color:#888;font-size:13px;margin-bottom:28px">
      Stocks with technically strong short signals: RSI overbought, extended above moving averages, MACD turning bearish, Bollinger Band extremes.
      Entry price range, stop loss, and targets for the next 1-5 days.</p>
    {_build_short_scanner_section(short_data)}
  </div>
</section>

<section id="sec-dcf" class="tab-section">
  <div class="container">
    <h2 class="section-head">DCF Intrinsic Value — Damodaran Framework</h2>
    <p style="color:#888;font-size:13px;margin-bottom:28px">
      3-stage discounted cash flow model answering Damodaran's five core valuation questions for each S&amp;P 500 stock.
      Run weekly via Step 372; results update each time pipeline runs.</p>
    {_build_dcf_section(dcf_data)}
  </div>
</section>

<section id="sec-health" class="tab-section">
  <div class="container">
    <h2 class="section-head">Signal Health Dashboard</h2>
    <p style="color:#888;font-size:13px;margin-bottom:32px">
      Live OOS IC by horizon, cross-signal correlation, and joint portfolio beta.
      Updated each time the daily pipeline runs.</p>

    {_safe_panel(_build_signal_health_section, signal_health or {})}

  </div>
</section>

<!-- ============================================================ AI CHAT -->
<section id="sec-chat" class="tab-section">
{_build_chat_section()}
</section>

<!-- ============================================================ EARNINGS AI -->
<section id="sec-earnings" class="tab-section">
  <div class="container">
    <p class="eyebrow">AI Qualitative Analysis — Powered by Claude</p>
    <h2 class="section-head">Earnings &amp; Business Quality — Deep AI Analysis</h2>
    <div class="rule"></div>
    {_build_earnings_ai_section(earnings_ai)}
  </div>
</section>

<!-- ═══════════════════════════════════════════════ MARKET INTELLIGENCE FLOW -->
{_build_flow_tab(options_flow, etf_flow, econ_cal)}

<!-- ══════════════════════════════════════════════════════ S&P 500 HEATMAP -->
{_build_heatmap_tab()}

<!-- ══════════════════════════════════════════════════════════ DATA HEALTH -->
{_build_event_engine_tab()}

{_build_data_health_tab()}

<!-- ══════════════════════════════════════════════════════════ QUANT QC -->
{_build_quant_qc_tab()}

<!-- ═══════════════════════════════════════════ SMART MONEY / FAMOUS HOLDINGS -->
<section id="sec-famous" class="tab-section">
  <div class="container">
    <p class="eyebrow">SEC 13F Filings — Top Hedge Fund Managers</p>
    <h2 class="section-head">Smart Money Holdings — Institutional Mind Map</h2>
    <div class="rule"></div>
    {_build_famous_holdings_tab(famous_holdings, congressional_trades)}
  </div>
</section>

<footer>
  <div class="container">
    <div class="footer-inner">
      <div>
        <p class="footer-brand">CANYON <span>QUANT</span></p>
        <p>v9 + v25.1 · 253 Modules · Updated {today}<br>Universe: S&amp;P 500 Equities<br>Tested on unseen data: Jan 2019–May 2026</p>
      </div>
      <div><p><strong>Disclaimer.</strong> This material is for research and educational purposes only. All performance figures result from historical simulation and do not represent actual trading results. Past performance is not indicative of future returns. Survivorship bias present. The strategy involves substantial risk of loss. No investment decisions should be made based on this material alone.</p></div>
    </div>
  </div>
</footer>

<!-- MOBILE BOTTOM NAV — 5 most-used tabs always visible -->
<nav class="mobile-bottom-nav">
  <a onclick="showTab('today')"    id="bnav-today">    <span class="bnav-icon">📊</span>Today</a>
  <a onclick="showTab('live')"     id="bnav-live">     <span class="bnav-icon">💼</span>Positions</a>
  <a onclick="showTab('chat')"     id="bnav-chat">     <span class="bnav-icon">💬</span>AI Chat</a>
  <a onclick="showTab('shorts')"   id="bnav-shorts">   <span class="bnav-icon">📉</span>Shorts</a>
  <a onclick="showTab('macro')"    id="bnav-macro">    <span class="bnav-icon">🌐</span>Macro</a>
</nav>

<!-- EXPORT / PRINT BUTTON (floating, desktop only) -->
<button onclick="window.print()" title="Export / Print current tab as PDF"
  style="position:fixed;bottom:24px;right:24px;z-index:888;
  background:#2a2418;color:#c8b487;border:1px solid #c8b487;
  padding:8px 14px;border-radius:4px;cursor:pointer;font-size:11px;
  font-weight:400;letter-spacing:1px;text-transform:uppercase;
  box-shadow:0 2px 8px rgba(0,0,0,.25)">
  ⬇ Export PDF
</button>

<script>
function closeNavDrops() {{
  document.querySelectorAll('.nav-dropdown').forEach(d => d.classList.remove('open'));
}}
function toggleNavDrop(groupId) {{
  var grp  = document.getElementById(groupId);
  var drop = grp ? grp.querySelector('.nav-dropdown') : null;
  if (!drop) return;
  var isOpen = drop.classList.contains('open');
  closeNavDrops();
  if (!isOpen) drop.classList.add('open');
}}
// Click outside → close all dropdowns
document.addEventListener('click', function(e) {{
  if (!e.target.closest('.nav-group')) closeNavDrops();
}});

var _TAB_GROUP = {{
  live:'portfolio', perf:'portfolio', attr:'portfolio', risk:'portfolio',
  signals:'research', dcf:'research', earnings:'research', shorts:'research', deep:'research',
  heatmap:'market', macro:'market', flow:'market', famous:'market', news:'market',
  v251:'system', method:'system', health:'system', manual:'system', qc:'system', datahealth:'system', eventengine:'', eventengine:''
}};

function showTab(name) {{
  document.querySelectorAll('.tab-section').forEach(s => s.classList.remove('active'));
  document.querySelectorAll('.nav-tabs a').forEach(a => a.classList.remove('active'));
  document.querySelectorAll('.nav-group').forEach(g => g.classList.remove('active'));
  var sec = document.getElementById('sec-' + name);
  if (sec) sec.classList.add('active');
  var tabEl = document.getElementById('tab-' + name);
  if (tabEl) tabEl.classList.add('active');
  var grp = _TAB_GROUP[name];
  if (grp) {{ var gEl = document.getElementById('navg-' + grp); if (gEl) gEl.classList.add('active'); }}
  window.scrollTo({{top: 0, behavior: 'smooth'}});
  document.querySelectorAll('.mobile-bottom-nav a').forEach(a => a.classList.remove('active'));
  var bnav = document.getElementById('bnav-' + name);
  if (bnav) bnav.classList.add('active');
  try {{ localStorage.setItem('canyon_active_tab', name); }} catch(e) {{}}
  document.dispatchEvent(new CustomEvent('showTab', {{detail: name}}));
}}

// Restore last active tab on page load
(function() {{
  try {{
    var saved = localStorage.getItem('canyon_active_tab');
    if (saved && document.getElementById('tab-' + saved)) {{
      showTab(saved);
    }}
  }} catch(e) {{}}
}})();

// Monthly Returns Bar Chart
(function() {{
  const el = document.getElementById('btMonthlyChart');
  if (!el) return;
  const btLabels = {bt_labels};
  const btStrat  = {bt_strat};
  const btSpy    = {bt_spy};
  new Chart(el.getContext('2d'), {{
    type: 'bar',
    data: {{
      labels: btLabels,
      datasets: [
        {{
          label: 'Strategy',
          data: btStrat,
          backgroundColor: btStrat.map(v => v >= 0 ? 'rgba(90,100,116,0.55)' : 'rgba(184,50,50,0.70)'),
          borderRadius: 2, barPercentage: 0.6
        }},
        {{
          label: 'S&P 500',
          data: btSpy,
          backgroundColor: 'rgba(200,200,200,0.45)',
          borderRadius: 2, barPercentage: 0.6
        }}
      ]
    }},
    options: {{
      responsive: true, maintainAspectRatio: false,
      interaction: {{mode:'index', intersect:false}},
      plugins: {{
        legend: {{position:'top', align:'end', labels:{{boxWidth:14,padding:16,font:{{size:11}},color:'#666'}}}},
        tooltip: {{
          backgroundColor:'#fff', titleColor:'#1A1A1A', bodyColor:'#555',
          borderColor:'#241f18', borderWidth:1, padding:10,
          callbacks: {{label: ctx => `  ${{ctx.dataset.label}}: ${{ctx.parsed.y > 0 ? '+' : ''}}${{ctx.parsed.y.toFixed(2)}}%`}}
        }}
      }},
      scales: {{
        x: {{grid:{{display:false}}, border:{{display:false}}, ticks:{{color:'#BBB',font:{{size:10}},maxTicksLimit:12}}}},
        y: {{
          grid:{{color:'#241f18'}}, border:{{display:false}},
          ticks:{{color:'#BBB', font:{{size:11}}, callback: v => v+'%'}}
        }}
      }}
    }}
  }});
}})();

// Backtest Cumulative Return Chart
(function() {{
  const el = document.getElementById('btCumChart');
  if (!el) return;
  const btLabels   = {bt_labels};
  const btStratCum = {bt_strat_cum};
  const btBenchCum = {bt_bench_cum};
  new Chart(el.getContext('2d'), {{
    type: 'line',
    data: {{
      labels: btLabels,
      datasets: [
        {{
          label: 'Strategy (cumulative %)',
          data: btStratCum,
          borderColor: '#3a3128', backgroundColor: 'rgba(27,42,74,0.06)',
          fill: true, borderWidth: 2.5, pointRadius: 0, pointHoverRadius: 4, tension: 0.3
        }},
        {{
          label: 'S&P 500 (cumulative %)',
          data: btBenchCum,
          borderColor: '#BBBBBB', backgroundColor: 'transparent',
          fill: false, borderWidth: 1.5, borderDash: [6,4], pointRadius: 0, pointHoverRadius: 3, tension: 0.3
        }}
      ]
    }},
    options: {{
      responsive: true, maintainAspectRatio: false,
      interaction: {{mode:'index', intersect:false}},
      plugins: {{
        legend: {{position:'top', align:'end', labels:{{boxWidth:28,boxHeight:2,padding:20,font:{{size:12}},color:'#666'}}}},
        tooltip: {{
          backgroundColor:'#fff', titleColor:'#1A1A1A', bodyColor:'#666',
          borderColor:'#241f18', borderWidth:1, padding:12,
          callbacks: {{label: ctx => `  ${{ctx.dataset.label}}: ${{ctx.parsed.y > 0 ? '+' : ''}}${{ctx.parsed.y.toFixed(1)}}%`}}
        }}
      }},
      scales: {{
        x: {{grid:{{display:false}}, border:{{display:false}}, ticks:{{color:'#BBB',font:{{size:11}},maxTicksLimit:10}}}},
        y: {{
          grid:{{color:'#241f18'}}, border:{{display:false}},
          ticks:{{color:'#BBB', font:{{size:11}}, callback: v => v+'%'}}
        }}
      }}
    }}
  }});
}})();

// Paper NAV Chart
(function() {{
  const el = document.getElementById('paperNavChart');
  if (!el) return;
  const pnLabels = {pn_labels};
  const pnNav    = {pn_nav};
  const pnHwm    = {pn_hwm};
  new Chart(el.getContext('2d'), {{
    type: 'line',
    data: {{
      labels: pnLabels,
      datasets: [
        {{
          label: 'Simulated portfolio value',
          data: pnNav,
          borderColor: '#1B6F4A', backgroundColor: 'rgba(27,111,74,0.07)',
          fill: true, borderWidth: 2.5, pointRadius: 0, pointHoverRadius: 4, tension: 0.3
        }},
        ...(pnHwm.length ? [{{
          label: 'High-Water Mark',
          data: pnHwm,
          borderColor: '#c8b487', backgroundColor: 'transparent',
          fill: false, borderWidth: 1.5, borderDash: [4,4], pointRadius: 0, tension: 0.3
        }}] : [])
      ]
    }},
    options: {{
      responsive: true, maintainAspectRatio: false,
      interaction: {{mode:'index', intersect:false}},
      plugins: {{
        legend: {{position:'top', align:'end', labels:{{boxWidth:28,boxHeight:2,padding:20,font:{{size:12}},color:'#666'}}}},
        tooltip: {{
          backgroundColor:'#fff', titleColor:'#1A1A1A', bodyColor:'#666',
          borderColor:'#241f18', borderWidth:1, padding:12,
          callbacks: {{label: ctx => `  ${{ctx.dataset.label}}: $${{ctx.parsed.y.toLocaleString('en-US',{{minimumFractionDigits:2}})}}`}}
        }}
      }},
      scales: {{
        x: {{grid:{{display:false}}, border:{{display:false}}, ticks:{{color:'#BBB',font:{{size:11}},maxTicksLimit:10}}}},
        y: {{
          grid:{{color:'#241f18'}}, border:{{display:false}},
          ticks:{{color:'#BBB', font:{{size:11}}, callback: v => '$'+v.toLocaleString()}}
        }}
      }}
    }}
  }});
}})();

// OOS Equity Chart
const labels = {chart_labels};
const mlData = {chart_ml};
const spyData = {chart_spy};
const ctx = document.getElementById('oosChart').getContext('2d');
new Chart(ctx, {{
  type: 'line',
  data: {{
    labels,
    datasets: [
      {{
        label: 'Strategy (real test data)',
        data: mlData,
        borderColor: '#3a3128',
        backgroundColor: 'rgba(27,42,74,0.04)',
        fill: false, borderWidth: 2.5, pointRadius: 0, pointHoverRadius: 4, tension: 0.3
      }},
      {{
        label: 'S&P 500',
        data: spyData,
        borderColor: '#BBBBBB',
        backgroundColor: 'transparent',
        fill: false, borderWidth: 1.5, borderDash: [6,4], pointRadius: 0, pointHoverRadius: 3, tension: 0.3
      }}
    ]
  }},
  options: {{
    responsive: true, maintainAspectRatio: false,
    interaction: {{mode: 'index', intersect: false}},
    plugins: {{
      legend: {{position:'top', align:'end', labels:{{boxWidth:28,boxHeight:2,padding:20,font:{{size:12}},color:'#666'}}}},
      tooltip: {{
        backgroundColor:'#fff',titleColor:'#1A1A1A',bodyColor:'#666',
        borderColor:'#241f18',borderWidth:1,padding:12,
        callbacks: {{label: ctx => `  ${{ctx.dataset.label}}:  ${{ctx.parsed.y.toFixed(0)}}`}}
      }}
    }},
    scales: {{
      x: {{grid:{{display:false}},border:{{display:false}},ticks:{{color:'#BBB',font:{{size:11}},maxTicksLimit:10}}}},
      y: {{
        grid:{{color:'#241f18'}},border:{{display:false}},
        type: 'logarithmic',
        ticks:{{color:'#BBB',font:{{size:11}},callback: v => v >= 1000 ? (v/1000).toFixed(0)+'k' : v}}
      }}
    }}
  }}
}});

// Rolling IC chart
(function() {{
  const el = document.getElementById('rollingIcChart');
  if (!el) return;
  const labels = {ric_labels};
  const ic3m   = {ric_3m};
  const ic6m   = {ric_6m};
  const target = {ric_target};
  new Chart(el.getContext('2d'), {{
    type: 'line',
    data: {{
      labels,
      datasets: [
        {{label:'Accuracy last 3 months',data:ic3m,borderColor:'#3a3128',backgroundColor:'rgba(27,42,74,0.08)',fill:true,tension:0.3,pointRadius:3,borderWidth:2}},
        {{label:'Accuracy last 6 months',data:ic6m,borderColor:'#c8b487',backgroundColor:'transparent',borderDash:[5,4],tension:0.3,pointRadius:2,borderWidth:1.5}},
        {{label:'Accuracy target',data:labels.map(()=>target),borderColor:'#1B6F4A',borderDash:[2,4],pointRadius:0,borderWidth:1.5}},
        {{label:'Warning zone',data:labels.map(()=>0.10),borderColor:'#241f18',borderDash:[2,4],pointRadius:0,borderWidth:1}},
      ]
    }},
    options: {{
      responsive:true,maintainAspectRatio:false,
      plugins:{{legend:{{position:'bottom',labels:{{font:{{size:11}},color:'#666'}}}},
        tooltip:{{backgroundColor:'#fff',titleColor:'#1A1A1A',bodyColor:'#555',borderColor:'#241f18',borderWidth:1,padding:10,
          callbacks:{{label: ctx => `  ${{ctx.dataset.label}}: ${{ctx.parsed.y.toFixed(3)}}`}}}}
      }},
      scales:{{
        x:{{grid:{{display:false}},border:{{display:false}},ticks:{{color:'#BBB',font:{{size:10}},maxTicksLimit:12}}}},
        y:{{grid:{{color:'#241f18'}},border:{{display:false}},
          ticks:{{color:'#BBB',font:{{size:11}},callback:v=>v.toFixed(2)}}}}
      }}
    }}
  }});
}})();

// Factor IC chart
(function() {{
  const el = document.getElementById('factorIcChart');
  if (!el) return;
  const labels = {ric_fac_labels};
  const mom    = {ric_mom};
  const lv     = {ric_lowvol};
  const val    = {ric_value};
  new Chart(el.getContext('2d'), {{
    type: 'bar',
    data: {{
      labels,
      datasets: [
        {{label:'Price trend signal',data:mom,backgroundColor:mom.map(v=>v>=0?'rgba(27,111,74,0.7)':'rgba(184,50,50,0.6)')}},
        {{label:'Low-risk stocks signal',data:lv,backgroundColor:lv.map(v=>v>=0?'rgba(27,42,74,0.7)':'rgba(184,50,50,0.5)')}},
        {{label:'Undervalued stocks signal',data:val,backgroundColor:val.map(v=>v>=0?'rgba(184,148,63,0.8)':'rgba(184,50,50,0.4)')}},
      ]
    }},
    options: {{
      responsive:true,maintainAspectRatio:false,
      plugins:{{legend:{{position:'bottom',labels:{{font:{{size:11}},color:'#666'}}}},
        tooltip:{{backgroundColor:'#fff',titleColor:'#1A1A1A',bodyColor:'#555',borderColor:'#241f18',borderWidth:1,padding:10,
          callbacks:{{label:ctx=>`  ${{ctx.dataset.label}}: ${{ctx.parsed.y.toFixed(3)}}`}}}}
      }},
      scales:{{
        x:{{grid:{{display:false}},border:{{display:false}},ticks:{{color:'#BBB',font:{{size:10}},maxTicksLimit:12}}}},
        y:{{grid:{{color:'#241f18'}},border:{{display:false}},
          ticks:{{color:'#BBB',font:{{size:11}},callback:v=>v.toFixed(2)}}}}
      }}
    }}
  }});
}})();

// Auto-refresh countdown (5 minutes)
(function() {{
  const INTERVAL = 5 * 60; // seconds
  let remaining = INTERVAL;
  const el = document.getElementById('countdown-display');
  function tick() {{
    if (!el) return;
    const m = Math.floor(remaining / 60);
    const s = remaining % 60;
    el.textContent = 'Auto-refresh in ' + m + ':' + String(s).padStart(2, '0');
    if (remaining <= 0) {{ window.location.reload(); return; }}
    remaining--;
    setTimeout(tick, 1000);
  }}
  tick();
}})();

// ── Ticker drilldown modal ────────────────────────────────────────────────────
const CANYON_TICKERS = {_ticker_js_data()};

function openDrilldown(ticker) {{
  const d = CANYON_TICKERS[ticker];
  if (!d) {{ return; }}
  const _SIG = {{'BUY':'Buy','LONG':'Buy','STRONG BUY':'Strong buy','SELL':'Sell','SHORT':'Sell','STRONG SELL':'Strong sell','HOLD':'Hold'}};
  const _CROWD = {{'WATCH':'Crowded ⚠','HIGH':'Heavily crowded','CLEAR':'Normal','LOW':'Uncrowded'}};
  const sigDisp = _SIG[d.signal] || (d.signal ? d.signal.charAt(0).toUpperCase() + d.signal.slice(1).toLowerCase() : '—');
  const crowdDisp = _CROWD[d.crowding] || (d.crowding ? d.crowding.charAt(0).toUpperCase() + d.crowding.slice(1).toLowerCase() : '—');
  const signalRaw = d.signal || '';
  const signalColor = (signalRaw==='LONG'||signalRaw==='BUY'||signalRaw==='STRONG BUY') ? '#1B6F4A' : ((signalRaw==='SHORT'||signalRaw==='SELL'||signalRaw==='STRONG SELL') ? '#B83232' : '#666');
  const riskColors  = {{'CLEAR':'#1B6F4A','OK':'#1B6F4A','PASS':'#c8b487','HOLD':'#c8b487','REVIEW':'#c8b487','BLOCKED':'#B83232','SIZE_DOWN':'#B83232','REDUCE_ONLY':'#B83232'}};
  const riskColor   = riskColors[d.risk_action] || '#666';

  // Signal bars
  let sigsHtml = Object.entries(d.sigs).map(([name, val]) => {{
    const pct   = Math.max(0, Math.min(100, val));
    const color = pct > 65 ? '#1B6F4A' : (pct < 35 ? '#B83232' : '#c8b487');
    return `<div class="dd-sig-row">
      <span class="dd-sig-name">${{name}}</span>
      <div class="dd-sig-bar-wrap"><div class="dd-sig-bar" style="width:${{pct}}%;background:${{color}}"></div></div>
      <span class="dd-sig-val">${{pct.toFixed(0)}}</span>
    </div>`;
  }}).join('');

  // News
  let newsHtml = d.news.length
    ? d.news.map(n => `<div class="dd-news-item"><p class="dd-news-title">${{n.title}}</p><p class="dd-news-meta">${{n.tone}} &middot; ${{n.date}}</p></div>`).join('')
    : '<p style="color:#AAA;font-size:13px">No recent news.</p>';

  document.getElementById('dd-content').innerHTML = `
    <p class="dd-ticker">${{ticker}}</p>
    <p class="dd-meta">${{d.sector}} &middot; Rank #${{d.rank}} &middot; Score ${{d.score >= 0 ? '+' : ''}}${{d.score.toFixed(2)}}</p>
    <div class="dd-kpi-row">
      <div class="dd-kpi">
        <p class="dd-kpi-label">Signal</p>
        <p class="dd-kpi-val" style="color:${{signalColor}}">${{sigDisp}}</p>
      </div>
      <div class="dd-kpi">
        <p class="dd-kpi-label">Risk gate</p>
        <p class="dd-kpi-val" style="color:${{riskColor}};font-size:16px">${{d.risk_plain}}</p>
      </div>
      <div class="dd-kpi">
        <p class="dd-kpi-label">Crowding</p>
        <p class="dd-kpi-val" style="font-size:16px">${{crowdDisp}}</p>
      </div>
    </div>
    <p class="dd-section-title">Signal breakdown — all 8 signals (0 = bearish, 100 = bullish)</p>
    ${{sigsHtml}}
    <p class="dd-section-title">Recent news</p>
    ${{newsHtml}}
  `;
  document.getElementById('drilldown-modal').style.display = 'block';
  document.body.style.overflow = 'hidden';
}}

function closeDrilldown() {{
  document.getElementById('drilldown-modal').style.display = 'none';
  document.body.style.overflow = '';
}}

// Click delegation — any .td-ticker click opens drilldown
document.addEventListener('click', function(e) {{
  const el = e.target.closest('.td-ticker');
  if (el) openDrilldown(el.textContent.trim());
}});
document.addEventListener('keydown', function(e) {{
  if (e.key === 'Escape') closeDrilldown();
}});

// v25.1 Cumulative Chart
(function() {{
  const el = document.getElementById('v251CumChart');
  if (!el) return;
  const labels = {_v251_labels};
  const cV251  = {_v251_chart};
  const cQQQ   = {_v251_qqq};
  const cSPY   = {_v251_spy};
  new Chart(el.getContext('2d'), {{
    type: 'line',
    data: {{
      labels: labels,
      datasets: [
        {{ label: 'v25.1 Canyon QQQ Hunter', data: cV251, borderColor: '#E74C3C', backgroundColor: 'rgba(231,76,60,0.08)', borderWidth: 2.5, pointRadius: 0, fill: true, tension: 0.3 }},
        {{ label: 'QQQ / NASDAQ 100',        data: cQQQ,  borderColor: '#9B59B6', backgroundColor: 'transparent', borderWidth: 1.5, pointRadius: 0, fill: false, tension: 0.3 }},
        {{ label: 'SPY / S&P 500',           data: cSPY,  borderColor: '#95A5A6', backgroundColor: 'transparent', borderWidth: 1, pointRadius: 0, borderDash: [4,3], fill: false, tension: 0.3 }},
      ]
    }},
    options: {{
      responsive: true, maintainAspectRatio: false,
      plugins: {{ legend: {{ position: 'top', labels: {{ font: {{ size: 11 }} }} }}, tooltip: {{ mode: 'index', intersect: false }} }},
      scales: {{
        x: {{ ticks: {{ maxTicksLimit: 12, font: {{ size: 10 }} }}, grid: {{ display: false }} }},
        y: {{ ticks: {{ callback: v => v.toFixed(0), font: {{ size: 10 }} }}, grid: {{ color: 'rgba(0,0,0,.05)' }} }}
      }}
    }}
  }});
}})();

function toggleNews(id) {{
  var el = document.getElementById(id);
  var arrow = document.getElementById(id + '-arrow');
  if (!el) return;
  if (el.style.display === 'none') {{
    el.style.display = 'block';
    if (arrow) arrow.textContent = 'Close ▲';
  }} else {{
    el.style.display = 'none';
    if (arrow) arrow.textContent = 'Tap to expand ▼';
  }}
}}

// ── Auto-refresh (works when served from local Canyon server at localhost:8888) ──
(function() {{
  var lastMtime = null;
  var banner    = null;

  function getBanner() {{
    if (!banner) {{
      banner = document.createElement('div');
      banner.style.cssText = [
        'position:fixed;bottom:20px;right:20px;z-index:9999',
        'background:#2a2418;color:#fff;font-size:13px;font-family:sans-serif',
        'padding:12px 18px;border-radius:6px;box-shadow:0 4px 16px rgba(0,0,0,.25)',
        'display:flex;align-items:center;gap:12px;min-width:220px'
      ].join(';');
      document.body.appendChild(banner);
    }}
    return banner;
  }}

  function showRefreshing() {{
    var b = getBanner();
    b.innerHTML = '<span style="font-size:18px;animation:spin 1s linear infinite;display:inline-block">⟳</span>'
                + '<span>Refreshing data…<br><span style="color:#c8b487;font-size:11px">page will reload when done</span></span>';
    b.style.display = 'flex';
    if (!document.getElementById('canyon-spin-style')) {{
      var s = document.createElement('style');
      s.id = 'canyon-spin-style';
      s.textContent = '@keyframes spin{{from{{transform:rotate(0)}}to{{transform:rotate(360deg)}}}}';
      document.head.appendChild(s);
    }}
  }}

  function showDone() {{
    var b = getBanner();
    b.innerHTML = '<span>✓ Data updated — reloading…</span>';
    b.style.background = '#1B6F4A';
    setTimeout(function() {{ location.reload(); }}, 1500);
  }}

  function hideBanner() {{
    if (banner) banner.style.display = 'none';
  }}

  function poll() {{
    fetch('/api/status', {{cache:'no-store'}})
      .then(function(r) {{ return r.json(); }})
      .then(function(d) {{
        if (lastMtime === null) lastMtime = d.html_mtime;
        if (d.running) {{
          showRefreshing();
          setTimeout(poll, 6000);
        }} else if (d.html_mtime > lastMtime) {{
          showDone();
        }} else {{
          hideBanner();
          // Keep polling so page reloads after a background refresh completes
          setTimeout(poll, 30000);
        }}
      }})
      .catch(function() {{
        // Not running from server (file:// mode) — do nothing
      }});
  }}

  // Add Refresh button to nav bar
  document.addEventListener('DOMContentLoaded', function() {{
    var nav = document.querySelector('.nav-tabs');
    if (!nav) return;
    var btn = document.createElement('a');
    btn.href = '#';
    btn.id   = 'canyon-refresh-btn';
    btn.textContent = '⟳ Refresh Now';
    btn.style.cssText = 'margin-left:auto;font-size:12px;color:#c8b487;font-weight:400;padding:5px 14px;border:1px solid #c8b487;border-radius:3px;text-decoration:none;align-self:center';
    btn.onclick = function(e) {{
      e.preventDefault();
      fetch('/refresh').then(function() {{
        showRefreshing();
        setTimeout(poll, 3000);
      }}).catch(function() {{
        alert('Refresh server not running.\\n\\nTo enable auto-refresh, run:\\n  python serve_canyon.py\\n\\nthen open http://localhost:8888');
      }});
    }};
    nav.appendChild(btn);
    // Start polling
    poll();
  }});
}})();
</script>

{_canyon_global_overlays()}

<script>
/* Auto-reload when served via localhost — polls last_updated.json every 30s */
(function() {{
  if (location.protocol !== 'http:' && location.protocol !== 'https:') return;
  var _lastTs = null;
  function _check() {{
    fetch('/last_updated.json?_=' + Date.now())
      .then(function(r) {{ return r.json(); }})
      .then(function(d) {{
        if (_lastTs === null) {{ _lastTs = d.ts; return; }}
        if (d.ts !== _lastTs) {{ location.reload(); }}
      }})
      .catch(function() {{}});
  }}
  setInterval(_check, 30000);
  _check();
}})();
</script>

<script>
/* ── DARK-MODE NORMALIZER ──────────────────────────────────────────────────
   Catch-all: walk every element with an inline style, and remap by luminance.
   Any light background → matching dark surface; any dark text → lightened.
   This covers all inline light colors regardless of exact hex value. */
(function(){{
  function parse(str){{
    if(!str) return null;
    var m = str.match(/#([0-9a-fA-F]{{3}}|[0-9a-fA-F]{{6}})\\b/);
    if(m){{ var h=m[1];
      if(h.length===3) h=h[0]+h[0]+h[1]+h[1]+h[2]+h[2];
      return [parseInt(h.slice(0,2),16),parseInt(h.slice(2,4),16),parseInt(h.slice(4,6),16)]; }}
    var g = str.match(/rgba?\\(([^)]+)\\)/);
    if(g){{ var p=g[1].split(',').map(function(x){{return parseFloat(x);}}); return [p[0],p[1],p[2]]; }}
    if(/\\bwhite\\b/i.test(str)) return [255,255,255];
    return null;
  }}
  function lum(c){{ return (0.299*c[0]+0.587*c[1]+0.114*c[2])/255; }}
  function surface(c){{
    var r=c[0],g=c[1],b=c[2], mx=Math.max(r,g,b), mn=Math.min(r,g,b), sat=mx-mn;
    if(sat < 16) return '#16140f';                 // neutral grey/white
    if(r>=g && r>=b){{                               // warm dominant
      if(g > b + 12) return '#241f16';             // amber / yellow
      return '#251a17';                            // red
    }}
    if(g>=r && g>=b) return '#1c231e';             // green
    if(r > g + 8) return '#221c26';                // purple
    return '#1f2321';                              // blue
  }}
  document.querySelectorAll('[style]').forEach(function(el){{
    var s = el.getAttribute('style'); if(!s) return;
    var bg = s.match(/background(?:-color)?:\\s*([^;]+)/i);
    if(bg){{ var c=parse(bg[1]); if(c && lum(c) > 0.66) el.style.background = surface(c); }}
    var cm = s.match(/(?:^|;)\\s*color:\\s*([^;]+)/i);
    if(cm){{ var t=parse(cm[1]);
      if(t && lum(t) < 0.32){{
        var mx=Math.max(t[0],t[1],t[2]), mn=Math.min(t[0],t[1],t[2]);
        if(mx-mn < 16) el.style.color = '#c8ccd4';
        else {{ var f=205/Math.max(mx,1);
          el.style.color = 'rgb('+Math.min(255,Math.round(t[0]*f))+','
                                 +Math.min(255,Math.round(t[1]*f))+','
                                 +Math.min(255,Math.round(t[2]*f))+')'; }}
      }}
    }}
  }});
}})();
</script>
</body>
</html>"""
    # Inline Chart.js for offline support (replace CDN reference)
    # + global defaults so every chart reads clearly on the dark theme (Lynn:
    #   "那些图太暗" — brighter text, visible warm grid, bolder lines).
    _chart_defaults = (
        "<script>(function(){if(!window.Chart)return;"
        "Chart.defaults.color='#d8cdba';"                       # bright tick/legend text
        "Chart.defaults.borderColor='rgba(200,180,135,0.16)';"  # subtle salmon grid
        "Chart.defaults.font.size=11;"
        "if(Chart.defaults.elements){"
        "Chart.defaults.elements.line.borderWidth=2.6;"          # bolder lines
        "Chart.defaults.elements.point.radius=0;"
        "Chart.defaults.elements.point.hoverRadius=4;}"
        "if(Chart.defaults.plugins&&Chart.defaults.plugins.legend)"
        "Chart.defaults.plugins.legend.labels.color='#d8cdba';"
        "})();</script>"
    )
    _html = _html.replace(
        '<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>',
        f'<script>{_CHARTJS_JS}</script>{_chart_defaults}'
    )
    # Brighten the very-dark grid lines / tick colors baked into individual charts
    # so the chart structure is visible on the warm-dark cards.
    for _dark, _bright in (
        ("grid:{{color:'#2a231b'}}", "grid:{{color:'rgba(200,180,135,0.14)'}}"),
        ("grid:{{color:'#241f18'}}", "grid:{{color:'rgba(200,180,135,0.14)'}}"),
        ("grid:{{color:'#241f18'}}", "grid:{{color:'rgba(200,180,135,0.14)'}}"),
        ("color:'#BBB'", "color:'#b7ab99'"),
        ("color:'#8a7f70'}},grid:{{color:'#2a231b'}}", "color:'#b7ab99'}},grid:{{color:'rgba(200,180,135,0.14)'}}"),
    ):
        _html = _html.replace(_dark.replace("{{", "{").replace("}}", "}"), _bright.replace("{{", "{").replace("}}", "}"))
    return _html

# ── main ─────────────────────────────────────────────────────────────────────

def main():
    print("Canyon v9 + v25.1 — Generating research website …")
    daily      = load_daily_report()
    chart      = load_oos_chart()
    summ       = load_oos_summary()
    accruals   = load_accruals()
    squeeze    = load_squeeze()
    live       = load_live_data()
    bt_monthly = load_backtest_monthly()
    paper_nav  = load_paper_nav_chart()
    wf_steps   = load_workflow_steps()
    wf_queue   = load_workflow_queue()
    alpha_sc   = load_alpha_scores()
    risk_gate  = load_risk_gate()
    ticker_dd  = load_ticker_drilldown()
    desk_mon   = load_desk_monitor()
    sector_cy  = load_sector_cycle()
    news_items  = load_news()
    earn_cal    = load_earnings_calendar()
    mac_breadth = load_macro_breadth()
    roll_ic     = load_rolling_ic()
    fac_attr    = load_factor_attribution()
    mo_pnl      = load_monthly_pnl()
    pos_pnl     = load_position_pnl()
    crowd       = load_crowding_monitor()
    macro_sigs  = load_macro_signal_snapshot()
    print("  Loading v25.1 backtest + live regime …")
    v251_bt     = load_v251_backtest()
    v251_regime = load_v251_regime()
    deep        = load_deep_analysis()
    sig_health  = load_signal_health()
    barra_risk  = load_barra_risk()
    hmm_data    = load_hmm_regime()
    macro_out   = load_macro_regime_outlook()
    dcf_df      = load_dcf_valuation()
    short_df    = load_short_scanner()
    econ_cal    = load_economic_calendar()
    earn_ai     = load_earnings_ai()
    watchlist   = load_watchlist()
    famous_hld  = load_famous_holdings()
    congress_td = load_congressional_trades()
    opt_flow    = load_options_flow()
    etf_flow_d  = load_etf_flow()

    html = build_html(daily, chart, summ, accruals, squeeze, live, bt_monthly, paper_nav,
                      wf_steps, wf_queue, alpha_sc, risk_gate, ticker_dd, desk_mon, sector_cy,
                      news_items, earn_cal, mac_breadth, roll_ic, fac_attr, mo_pnl,
                      pos_pnl, crowd, macro_sigs,
                      v251_bt=v251_bt, v251_regime=v251_regime, deep=deep,
                      signal_health=sig_health, barra_risk=barra_risk, hmm_data=hmm_data,
                      macro_outlook=macro_out, dcf_data=dcf_df, short_data=short_df,
                      econ_cal=econ_cal, earnings_ai=earn_ai, watchlist=watchlist,
                      famous_holdings=famous_hld, congressional_trades=congress_td,
                      options_flow=opt_flow, etf_flow=etf_flow_d)
    OUT.write_text(html, encoding="utf-8")
    print(f"  Written: {OUT}  ({len(html)//1024} KB)")
    _hmm_display = hmm_data.get('regime', daily.get('hmm','?'))
    _hmm_stale_str = f" (stale {hmm_data.get('days_stale',0)}d)" if hmm_data.get('stale') else ""
    print(f"  Date: {daily.get('date','?')}  HMM Regime: {_hmm_display}{_hmm_stale_str}  Macro: {daily.get('macro','?')}")
    print(f"  Longs: {[r['ticker'] for r in daily.get('longs',[])[:5]]}")
    print(f"  OOS IC: +{summ.get('oos_ic',0):.3f}  Sharpe: {summ.get('oos_sharpe',0):.3f}")
    print(f"  Backtest: {bt_monthly.get('total_months',0)} months · win rate {bt_monthly.get('win_rate',0):.0f}% vs SPY")
    print(f"  Paper NAV: ${paper_nav.get('final',0):,.2f} ({paper_nav.get('gain',0):+.2f}%) · max DD {paper_nav.get('max_dd',0):.2f}%")
    print(f"  Paper positions: {len([p for p in live.get('positions',[]) if p['side']=='LONG'])} long  {len([p for p in live.get('positions',[]) if p['side']=='SHORT'])} short")
    print(f"  IC days accumulated: {live.get('days_acc', 0)}/21")
    print(f"  v25.1: AR={v251_bt.get('ar',0)*100:+.1f}%  SR={v251_bt.get('sharpe',0):.3f}  MDD={v251_bt.get('mdd',0)*100:.1f}%  Beat {v251_bt.get('beat_years',0)}/8 yrs")
    print(f"  Live regime: {v251_regime.get('regime','?')}  VIX={v251_regime.get('vix',0):.1f}({v251_regime.get('vix_tier','?')})  TQQQ={v251_regime.get('tqqq_wt',0):.0%}")
    print(f"\nOpen in browser:  file://{OUT}")

if __name__ == "__main__":
    main()
