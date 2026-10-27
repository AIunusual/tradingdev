"""Offline Bokeh tables over the same evidence used by the static report.

Only the installed Bokeh JavaScript bundles and a fixed initializer are emitted.
Models are created lazily in the browser, so random model identifiers and copies
of per-scope records never enter the saved HTML or its content hash.
"""

from __future__ import annotations

import hashlib
from functools import lru_cache

import bokeh
from bokeh.resources import Resources


@lru_cache(maxsize=1)
def bokeh_resources_html() -> str:
    """Return one self-contained, deterministic set of table/widget resources."""
    return Resources(
        mode="inline",
        minified=True,
        log_level="error",
        components=["bokeh", "bokeh-widgets", "bokeh-tables", "bokeh-api"],
    ).render_js()


def bokeh_provenance() -> dict[str, str]:
    """Identify the installed renderer resources before computing a report ID."""
    return {
        "version": bokeh.__version__,
        "resources_sha256": hashlib.sha256(
            bokeh_resources_html().encode("utf-8")
        ).hexdigest(),
    }


# Public browser API: TradingDevTables.mount(targetElement, spec) -> Promise.
# spec.rows holds unrounded values. spec.columns controls their ordered labels:
# {field, title, kind?: "number"|"integer"|"percent"|"text", visible?: bool,
#  group?: str, width?: int, sortable?: bool}. full_precision_values maps each
# numeric field to backend-formatted strings indexed by original row position.
# The returned setPrecision(bool) changes display without changing source values
# or sorting; integer fields always retain the backend exact text.
# Optional filter_field/event_options ({value,label})
# filter the original rows. Optional navigate_field permits only result anchors.
# Optional column_presets ({label,fields}) provides named column selections.
# The caller owns lazy scope activation and the independently readable static HTML.
BOKEH_TABLE_SCRIPT = r"""
window.TradingDevTables=(()=>{
 'use strict';
 const mounted=new WeakMap();
 const escapeText=value=>String(value).replaceAll('&','&amp;')
  .replaceAll('<','&lt;').replaceAll('>','&gt;')
  .replaceAll('"','&quot;').replaceAll("'",'&#39;');
 function makeFormatter(column,fullValues,showFull){
  const formats={number:'0,0.00',integer:'0,0',percent:'0.00%'};
  const format=formats[column.kind];
  if(!format){
   return new Bokeh.Tables.StringFormatter({null_format:'—',nan_format:'—'});
  }
  const rounded=new Bokeh.Tables.NumberFormatter({
   format,text_align:'right',null_format:'—',nan_format:'—'
  });
  const text=new Bokeh.Tables.StringFormatter({
   text_align:'right',null_format:'—',nan_format:'—'
  });
  const formatter=new Bokeh.Tables.StringFormatter();
  formatter.doFormat=(row,cell,value,columnDef,dataContext)=>{
   const full=fullValues?.[column.field]?.[dataContext.__report_row_index];
   const unsafeInteger=typeof full==='string'&&/^-?\d+$/.test(full)
    &&Number.isSafeInteger(value)===false;
   const display=(showFull()||column.kind==='integer'||unsafeInteger)&&full!==undefined
    ?text.doFormat(row,cell,full,columnDef,dataContext)
    :rounded.doFormat(row,cell,value,columnDef,dataContext);
   return `<span title="${escapeText(full??'—')}">${display}</span>`;
  };
  return formatter;
 }
 function makeNavigationFormatter(rows,navigateField){
  const formatter=new Bokeh.Tables.StringFormatter();
  formatter.doFormat=(row,cell,value,columnDef,dataContext)=>{
   const record=rows[dataContext.__report_row_index];
   const destination=record?.[navigateField];
   if(typeof destination!=='string'||!/^#result-\d+-\d+$/.test(destination)){
    return `${escapeText(value)} · 無可用明細`;
   }
   return `<a href="${escapeText(destination)}"`
    +` aria-label="查看結果 ${escapeText(value)} 的明細"`
    +' style="color:#2367cc;text-decoration:underline;font-weight:600">'
    +`${escapeText(value)} · 查看明細 →</a>`;
  };
  return formatter;
 }
 async function create(target,spec){
  const rows=spec.rows,columns=spec.columns;
  const fullValues=spec.full_precision_values;
  let fullPrecision=false;
  const defaults=columns.map((column,index)=>column.visible!==false?index:-1)
   .filter(index=>index>=0);
  let active=new Set(defaults);
  const makeData=indices=>{
   const data=Object.create(null);
   for(const column of columns){
    data[column.field]=indices.map(index=>rows[index][column.field]??null);
   }
   data.__report_row_index=indices;
   return data;
  };
  const source=new Bokeh.ColumnDataSource({
   data:makeData(rows.map((_,index)=>index))
  });
  const tableColumns=columns.map(column=>new Bokeh.Tables.TableColumn({
   field:column.field,title:column.title,sortable:column.sortable!==false,
   width:column.width??Math.min(290,Math.max(150,column.title.length*12)),
   formatter:spec.navigate_field&&column.field==='sequence'
    ?makeNavigationFormatter(rows,spec.navigate_field)
    :makeFormatter(column,fullValues,()=>fullPrecision)
  }));
  const table=new Bokeh.Tables.DataTable({
   source,columns:tableColumns.filter((_,index)=>active.has(index)),
   height:spec.height??520,sizing_mode:'stretch_width',autosize_mode:'none',
   index_position:null,editable:false,reorderable:false,sortable:true,
   selectable:true,scroll_to_selection:false,row_height:32,
   frozen_columns:Math.min(2,Math.max(0,defaults.length-1))||null
  });
  const setPrecision=value=>{
   const full=value===true;
   if(fullPrecision===full){return;}
   fullPrecision=full;
   source.change.emit();
  };
  const status=new Bokeh.Widgets.Select({title:'事件篩選',value:'all',width:185,
   options:[['all','全部事件'],...(spec.event_options??[])
    .map(option=>[option.value,option.label])],
   visible:Boolean(spec.filter_field)
  });
  const query=new Bokeh.Widgets.TextInput({title:'搜尋所有欄位',
   placeholder:spec.search_placeholder??'時間、價格、委託或成交編號…',
   sizing_mode:'stretch_width'});
  const count=new Bokeh.Widgets.Div({text:'',sizing_mode:'stretch_width'});
  const updateRows=()=>{
   const needle=query.value.trim().toLocaleLowerCase();
   const indices=[];
   rows.forEach((record,index)=>{
    const statusMatches=!spec.filter_field||status.value==='all'
     ||String(record[spec.filter_field]??'')===status.value;
    const queryMatches=!needle||columns.some(column=>{
     const exact=fullValues?.[column.field]?.[index];
     const value=(column.kind==='integer'||column.kind==='number')&&exact!==undefined
      ?exact:record[column.field];
     return String(value??'').toLocaleLowerCase().includes(needle)
      ||String(exact??'').toLocaleLowerCase().includes(needle);
    });
    if(statusMatches&&queryMatches){indices.push(index);}
   });
   source.selected.indices=[];
   source.data=makeData(indices);
   count.text=`顯示 ${indices.length} 筆／全部 ${rows.length} 筆`;
  };
  status.properties.value.change.connect(updateRows);
  query.properties.value_input.change.connect(()=>{
   query.value=query.value_input;
  });
  query.properties.value.change.connect(updateRows);
  const selectorDetails=document.createElement('details');
  selectorDetails.className='bokeh-column-controls';
  const selectorSummary=document.createElement('summary');
  selectorDetails.appendChild(selectorSummary);
  const selectorHost=document.createElement('div');
  selectorDetails.appendChild(selectorHost);
  const groups=new Map();
  columns.forEach((column,index)=>{
   const group=column.group??'顯示欄位';
   if(!groups.has(group)){groups.set(group,[]);}
   groups.get(group).push(index);
  });
  const selectors=[];
  const updateColumns=()=>{
   active=new Set(selectors.flatMap(({widget,indices})=>
    widget.active.map(index=>indices[index])));
   table.frozen_columns=Math.min(2,Math.max(0,active.size-1))||null;
   table.columns=tableColumns.filter((_,index)=>active.has(index));
   selectorSummary.textContent=`勾選顯示欄位（${active.size}／${columns.length}）`;
  };
  const chooserChildren=[];
  for(const [group,indices] of groups){
   const widget=new Bokeh.Widgets.CheckboxGroup({
    labels:indices.map(index=>columns[index].title),inline:true,
    active:indices.map((index,local)=>active.has(index)?local:-1)
     .filter(index=>index>=0),sizing_mode:'stretch_width'
   });
   selectors.push({widget,indices});
   widget.properties.active.change.connect(updateColumns);
   chooserChildren.push(
    new Bokeh.Widgets.Div({text:`<b>${escapeText(group)}</b>`}),widget);
  }
  const selectAll=new Bokeh.Widgets.Button({label:'顯示全部欄位',width:150});
  const selectDefault=new Bokeh.Widgets.Button({label:'恢復預設欄位',width:150});
  const setColumns=selected=>{
   for(const {widget,indices} of selectors){
    widget.active=indices.map((index,local)=>selected.has(index)?local:-1)
     .filter(index=>index>=0);
   }
   updateColumns();
  };
  selectAll.on_event('button_click',()=>setColumns(new Set(columns.map((_,i)=>i))));
  selectDefault.on_event('button_click',()=>setColumns(new Set(defaults)));
  const presetButtons=(spec.column_presets??[]).map(preset=>{
   const button=new Bokeh.Widgets.Button({label:preset.label,width:150});
   const fields=new Set(preset.fields);
   button.on_event('button_click',()=>setColumns(new Set(
    columns.map((column,index)=>fields.has(column.field)?index:-1)
     .filter(index=>index>=0))));
   return button;
  });
  const chooser=new Bokeh.Column({children:[
   new Bokeh.Row({children:[selectAll,selectDefault,...presetButtons]}),
   ...chooserChildren
  ],sizing_mode:'stretch_width'});
  const toolbar=new Bokeh.Column({children:[
   new Bokeh.Row({children:[status,query],sizing_mode:'stretch_width'}),count
  ],sizing_mode:'stretch_width'});
  const toolbarHost=document.createElement('div');
  const tableHost=document.createElement('div');
  target.replaceChildren(toolbarHost,selectorDetails,tableHost);
  if(spec.navigate_field){
   // Selection also changes during keyboard focus. Only a plain cell click is
   // a row-navigation shortcut; anchors retain native keyboard/new-tab behavior.
   tableHost.addEventListener('keydown',event=>{
    if(event.key==='Tab'
     ||event.composedPath().some(node=>node instanceof HTMLAnchorElement)){
     // Let the browser focus/activate links instead of SlickGrid moving cells.
     event.stopPropagation();
    }
   },true);
   tableHost.addEventListener('click',event=>{
    const path=event.composedPath();
    if(event.button!==0||event.ctrlKey||event.metaKey||event.shiftKey||event.altKey
     ||path.some(node=>node instanceof HTMLAnchorElement)
     ||!path.some(node=>node instanceof Element
      &&node.classList.contains('slick-cell'))){
     return;
    }
    // This bubble listener runs after SlickGrid updates the selected source row.
    const selected=source.selected.indices[0];
    if(selected===undefined){return;}
    const rowIndex=source.data.__report_row_index[selected];
    const destination=rows[rowIndex]?.[spec.navigate_field];
    if(typeof destination==='string'&&/^#result-\d+-\d+$/.test(destination)){
     source.selected.indices=[];
     window.location.hash=destination;
    }
   });
  }
  updateRows();
  updateColumns();
  await Bokeh.Plotting.show(toolbar,toolbarHost);
  let chooserReady=false;
  selectorDetails.addEventListener('toggle',async()=>{
   if(selectorDetails.open&&!chooserReady){
    chooserReady=true;
    try{await Bokeh.Plotting.show(chooser,selectorHost);}
    catch(error){
     chooserReady=false;
     selectorHost.textContent='欄位控制載入失敗；請重新展開重試。';
     console.error('Bokeh column controls failed',error);
    }
   }
  });
  await Bokeh.Plotting.show(table,tableHost);
  return {source,table,status,query,selectors,selectAll,selectDefault,
   presetButtons,setPrecision};
 }
 return {mount(target,spec){
  if(!mounted.has(target)){
   const pending=create(target,spec).catch(error=>{
    mounted.delete(target);
    throw error;
   });
   mounted.set(target,pending);
  }
  return mounted.get(target);
 }};
})();
"""
