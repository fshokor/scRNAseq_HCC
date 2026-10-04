import fs from 'node:fs/promises';
import path from 'node:path';
import { Presentation, PresentationFile, FileBlob } from '@oai/artifact-tool';
const dir=path.resolve('tmp/hcc_slide_edit');
const source=path.resolve('tmp/hcc_slide_edit/source-normalized.pptx');
const reference=await PresentationFile.importPptx(await FileBlob.load(source));
for (let i=0;i<reference.slides.items.length;i++) {
 if (await fs.access(path.join(dir,`reference-${i+1}.png`)).then(()=>true).catch(()=>false)) continue;
 const png=await reference.export({slide:reference.slides.items[i],format:'png',scale:1});
 await fs.writeFile(path.join(dir,`reference-${i+1}.png`),new Uint8Array(await png.arrayBuffer()));
}
const p=Presentation.create({slideSize:{width:1280,height:720}});
const s=p.slides.add();s.background.fill='#0B1822';
function text(t,x,y,w,h,size,color='#F4F8FB',bold=false){
 const a=s.shapes.add({geometry:'textbox',position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});
 a.text=t;a.text.style={typeface:'Aptos',fontSize:size,color,bold,autoFit:'none'};return a;
}
text('07 / HCC GENE AND PATHWAY EXAMPLES',60,30,1150,32,17,'#55D0C7',true);
text('Selected genes with lower expression in HCC2',60,78,1150,60,36,'#F4F8FB',true);
text('Pooled HCC2 versus HCC1 comparison. Each gene belongs to the listed GSEA leading edge.',60,146,1150,45,20,'#A9BAC6');
const values=[
 ['Gene','log2FC','Enriched gene set','Biological context'],
 ['APOA2','−12.20','Cholesterol metabolism\nNES −2.28  ·  FDR 9.57 × 10⁻⁹','Lipoprotein transport\nLower contribution of liver-associated transcripts.'],
 ['SERPINA1','−6.45','Complement and coagulation\nNES −2.34  ·  FDR 4.32 × 10⁻¹³','Alpha-1 antitrypsin / protease regulation\nLower contribution of secreted-protein transcripts.'],
 ['FTL','−4.61','Ferroptosis\nNES −2.09  ·  FDR 8.63 × 10⁻⁶','Ferritin light chain / iron storage\nIron-related expression differs between samples.']
];
const table=s.tables.add({rows:4,columns:4,left:60,top:216,width:1160,height:310,columnWidths:[132,112,368,548],values});
table.cells.block({row:0,column:0,rowCount:4,columnCount:4}).assign({fill:'#112634',textStyle:{typeface:'Aptos',fontSize:20,color:'#F4F8FB'},margins:{left:14,right:14,top:15,bottom:15}});
table.cells.block({row:0,column:0,rowCount:1,columnCount:4}).assign({fill:'#245466',textStyle:{typeface:'Aptos',fontSize:20,color:'#F4F8FB',bold:true}});
for(let r=1;r<4;r++){table.getCell(r,0).text.style={typeface:'Aptos',fontSize:22,color:'#55D0C7',bold:true};table.getCell(r,1).text.style={typeface:'Aptos',fontSize:22,color:'#F6A04D',bold:true};}
table.rows[0].height=52;for(let r=1;r<4;r++)table.rows[r].height=86;
text('These are sample-associated differences, not confirmed changes in malignant cells.',60,557,1150,37,22,'#55D0C7',true);
text('One sample per group; composition and possible ambient RNA remain. Negative NES favors HCC1.\nExpression and enrichment do not establish functional activity or drug efficacy.',60,607,1150,54,18,'#A9BAC6');
text('Current DE and GSEA results · GSE166635 · selected examples',60,674,1040,24,14,'#7F9AAF');text('7',1170,673,50,26,16,'#7F9AAF');
s.speakerNotes.textFrame.setText(`Selected pooled HCC2-versus-HCC1 results, not malignant-cell comparisons. APOA2 log2FC -12.198462; SERPINA1 -6.4540467; FTL -4.614934. All meet FDR <0.05 and absolute log2FC >=1. Stored gene FDR values underflow to zero and must not be called literally zero probabilities. KEGG cholesterol metabolism NES -2.28463661861109, adjusted p 9.56994237462538e-09; complement and coagulation cascades NES -2.34291493486101, adjusted p 4.32318970305741e-13; ferroptosis NES -2.08599306895663, adjusted p 8.63458370357047e-06. Each selected gene is verified in the corresponding leading edge. Negative NES is relative enrichment, not functional pathway suppression. APOA2 detection in HCC1 T cells is 93.3%, raising ambient RNA/technical concerns. Only one HCC2 hepatocyte-labelled cell is present. HCC1/HCC2 labels follow Wang, with source-metadata ambiguity. No validated efficacy claim: SERPINA1-IGMESINE is a database association with unresolved mechanism; no FTL drug association in the current snapshot. See docs/interview_preparation/FIVE_PAPER_GENES_AUDIT.md and supporting CSVs. Biological sources: https://www.ncbi.nlm.nih.gov/gene/336/ ; https://medlineplus.gov/genetics/gene/serpina1/ ; https://www.ncbi.nlm.nih.gov/gene/2512 . Reference: https://doi.org/10.1038/s41698-025-00952-3 .`);
await (await PresentationFile.exportPptx(p)).save(path.join(dir,'new-slide.pptx'));
const png=await p.export({slide:s,format:'png',scale:1.5});await fs.writeFile(path.join(dir,'new-slide.png'),new Uint8Array(await png.arrayBuffer()));
console.log('Reference previews and new editable slide exported.');
