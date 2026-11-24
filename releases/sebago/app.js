const examples = [
  { id:"SBL-014", title:"Lakeshore survey", labels:["Station","Water temperature","Visibility","Sample depth"], values:["North cove","14.6 °C","2.4 m"] },
  { id:"FOR-021", title:"Forest inventory", labels:["Plot","Tree species","Stem diameter","Canopy height"], values:["East transect","Red spruce","22.8 cm"] },
  { id:"FIS-009", title:"Landing tally", labels:["Landing site","Species","Landed weight","Water temperature"], values:["Dock 3","Atlantic herring","84 kg"] }
];
const key = "krummholz.sebago.records.v1";
const $ = id => document.getElementById(id);
let selected = 0, model;
const defaults = () => examples.map(r => ({ values:[...r.values], enabled:[true,true,true] }));
let records = defaults();
try {
  const saved = JSON.parse(localStorage.getItem(key));
  if (Array.isArray(saved) && saved.length === examples.length &&
      saved.every(r => Array.isArray(r?.values) && r.values.length === 3 && r.values.every(x => typeof x === "string" && x.trim().length && x.length <= 120) &&
      Array.isArray(r.enabled) && r.enabled.length === 3 && r.enabled.every(x => typeof x === "boolean"))) records = saved;
} catch { /* Private browsing may disable storage. Inference still works. */ }

function storeRecords() {
  try { localStorage.setItem(key, JSON.stringify(records)); } catch { /* Keep edits for this session. */ }
}
function node(tag, content, className) {
  const element = document.createElement(tag);
  if (content !== undefined) element.textContent = content;
  if (className) element.className = className;
  return element;
}
function stale() {
  $("result").replaceChildren(node("p","Ready to read the selected field.","hint subtle"));
}
function redraw() {
  const r = examples[selected], data = records[selected];
  $("recTitle").textContent = r.title;
  $("recID").textContent = r.id;
  $("editForm").hidden = true;
  $("sourceRows").hidden = false;
  $("edit").hidden = false;
  $("sourceRows").replaceChildren();
  r.labels.slice(0,3).forEach((label,i) => {
    const line = node("div", undefined, "row");
    const toggle = node("button", data.enabled[i] ? "Hide" : "Restore", "rowBtn");
    toggle.setAttribute("aria-label",(data.enabled[i] ? "Hide " : "Restore ") + label);
    toggle.addEventListener("click", () => {
      data.enabled[i] = !data.enabled[i];
      storeRecords(); redraw(); stale();
    });
    line.append(node("span",label),node("b",data.enabled[i] ? data.values[i] : "Hidden", data.enabled[i] ? "" : "off"),toggle);
    $("sourceRows").append(line);
  });
}
function openRecord(index) {
  selected = index;
  $("recordSel").value=String(index);
  $("field").replaceChildren(...examples[selected].labels.map((label,i) => new Option(label,String(i))));
  $("field").value="1";
  redraw(); stale();
}
function edit() {
  $("editForm").hidden = false; $("sourceRows").hidden = true; $("edit").hidden = true; $("inputs").replaceChildren();
  examples[selected].labels.slice(0,3).forEach((name,i) => {
    const label = node("label",name), input = node("input");
    input.name = "field"+i; input.value = records[selected].values[i];
    input.maxLength = 120; input.required = true; input.autocomplete = "off";
    label.append(input); $("inputs").append(label);
  });
  $("inputs").querySelector("input").focus();
}

// A collision-free local codebook translates arbitrary values into the model's three-symbol alphabet.
function valueCode(label, used) {
  let h = 2166136261;
  for (const c of label) h = Math.imul(h ^ c.codePointAt(0),16777619) >>> 0;
  let code = h % 512;
  while (used.has(code)) code = (code + 1) % 512;
  used.add(code);
  return [6 + (code >> 6), 6 + ((code >> 3) & 7), 6 + (code & 7)];
}
function infer(record, query) {
  const fields = record.enabled.flatMap((yes,i) => yes ? [i] : []);
  const usedKeys = new Set([...fields,query]);
  const filler = [5,4,3,2,1,0].filter(k => !usedKeys.has(k));
  const codes = new Set(), prefix=[query], evidence=[];
  for (let i=0;i<3;i++) {
    const present=record.enabled[i], k=present ? i : filler.shift();
    const code=valueCode(present ? record.values[i] : "blank-slot-"+i, codes);
    prefix.push(k,...code);
    if (present) evidence.push({k,code});
  }
  prefix.push(14);
  const output=SebagoModel.generate(prefix,model);
  if (output.length===2 && output[0]===15 && output[1]===16) return {state:"absent"};
  const source=evidence.find(row=>row.k===query && row.code.every((n,i)=>n===output[i]));
  if (!source || output.length!==4 || output[3]!==16) return {state:"review"};
  return {state:"supported", value:record.values[source.k]};
}
function ask() {
  if (!model) return;
  const q=Number($("field").value), record=records[selected], fields=examples[selected].labels;
  const start=performance.now(), answer=infer(record,q), elapsed=performance.now()-start;
  const panel=node("div",undefined,"answer"+(answer.state==="supported"?"":" warn"));
  const status={ supported:"Source match", absent:"Model abstained", review:"Verification needed" }[answer.state];
  const value=answer.state==="supported" ? answer.value : answer.state==="absent" ? "Abstained" : "Check the record";
  const detail=answer.state==="supported" ? "Verified against "+fields[q]+" in "+examples[selected].id+"." :
       answer.state==="review" ? "The generated value could not be verified against the requested field." :
       record.enabled[q] ? "This field is visible, but the model did not return its value." : "The selected field has no visible entry.";
  panel.append(node("span",status,"eyebrow"),node("b",value),node("p",detail));
  $("result").replaceChildren(panel,node("div","On-device inference · "+Math.max(1,Math.round(elapsed))+" ms","hint subtle"));
}
function view(name) {
  const fieldView = name==="Field desk";
  $("desk").hidden=!fieldView; $("model").hidden=fieldView;
  $("fieldTab").toggleAttribute("aria-current",fieldView); $("modelTab").toggleAttribute("aria-current",!fieldView);
  $(fieldView?"fieldTab":"modelTab").setAttribute("aria-current","page");
  $("crumb").textContent=name;
}
$("fieldTab").onclick=()=>view("Field desk"); $("modelTab").onclick=()=>view("Model card");
$("recordSel").replaceChildren(...examples.map((r,i)=>new Option(r.title,String(i))));
$("recordSel").onchange=()=>openRecord(Number($("recordSel").value));
$("field").onchange=stale; $("edit").onclick=edit; $("cancel").onclick=redraw;
$("editForm").onsubmit=event=>{
  event.preventDefault(); const inputs=[...$("inputs").querySelectorAll("input")];
  for(const input of inputs) {
    if (!input.value.trim()) { input.setCustomValidity("Enter a value or hide this field."); input.reportValidity(); input.setCustomValidity(""); return; }
  }
  records[selected].values=inputs.map(el=>el.value.trim()); storeRecords(); redraw(); stale();
};
$("ask").onclick=ask;
openRecord(0);
try {
  const bytes=Uint8Array.from(atob($("weights").textContent.trim()), c=>c.charCodeAt(0));
  model=SebagoModel.read(bytes.buffer);
  $("download").href=URL.createObjectURL(new Blob([bytes],{type:"application/octet-stream"}));
  $("engine").textContent="On device"; ask();
} catch {
  $("engine").textContent="Model unavailable"; $("ask").disabled=true;
  $("result").replaceChildren(node("p","The checkpoint could not be loaded. Reopen the complete Sebago file.","hint"));
}
