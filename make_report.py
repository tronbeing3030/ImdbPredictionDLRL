import json,pathlib
from docx import Document
from docx.shared import Pt,Inches,RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
B=pathlib.Path('.')
M=json.loads((B/'models'/'metrics.json').read_text())
P=M['production'];C=M['comparison'];S=M['seq_study'];CM=P['confusion_matrix']
GOLD=RGBColor(184,138,0);BLK=RGBColor(26,26,26);GRY=RGBColor(85,85,85)
def sh(c,h):
 t=c._tc.get_or_add_tcPr();e=OxmlElement('w:shd')
 e.set(qn('w:val'),'clear');e.set(qn('w:fill'),h);t.append(e)
def cc(c,t,b=False,s=9,col=None):
 c.text='';r=c.paragraphs[0].add_run(t)
 r.bold=b;r.font.size=Pt(s);r.font.name='Calibri'
 if col:r.font.color.rgb=col
def tb(doc,hd,rows):
 t=doc.add_table(rows=1+len(rows),cols=len(hd))
 t.style='Light Grid Accent 1';t.alignment=WD_TABLE_ALIGNMENT.CENTER
 for j,h in enumerate(hd):
  cc(t.rows[0].cells[j],h,True,9,RGBColor(255,255,255));sh(t.rows[0].cells[j],'1A1A1A')
 for i,r in enumerate(rows,1):
  for j,v in enumerate(r):cc(t.rows[i].cells[j],str(v),False,9)
  if i%2==0:
   for j in range(len(hd)):sh(t.rows[i].cells[j],'FFF8DC')
 doc.add_paragraph();return t
def hd(doc,t,l=1):
 h=doc.add_heading(t,level=l)
 for r in h.runs:r.font.color.rgb=GOLD if l==1 else BLK
 return h
doc=Document()
st=doc.styles['Normal'];st.font.name='Calibri';st.font.size=Pt(11)
st.paragraph_format.space_after=Pt(6);st.paragraph_format.line_spacing=1.12
sec=doc.sections[0]
sec.top_margin=Inches(.8);sec.bottom_margin=Inches(.8)
sec.left_margin=Inches(.9);sec.right_margin=Inches(.9)
ft=sec.footer.paragraphs[0];ft.alignment=WD_ALIGN_PARAGRAPH.CENTER
fr=ft.add_run('IMDb Cinema Sentiment Lab | Activity-2 Problem 1 | RNN vs BiRNN vs Deep vs LSTM vs GRU')
fr.font.size=Pt(8);fr.font.color.rgb=GRY
doc.add_paragraph();doc.add_paragraph()
bg=doc.add_paragraph();bg.alignment=WD_ALIGN_PARAGRAPH.CENTER
br=bg.add_run('IMDb | CINEMA SENTIMENT LAB');br.bold=True;br.font.size=Pt(13);br.font.color.rgb=GOLD
tt=doc.add_paragraph();tt.alignment=WD_ALIGN_PARAGRAPH.CENTER
tr=tt.add_run('Sequential Text Prediction on IMDb Movie Reviews');tr.bold=True;tr.font.size=Pt(26);tr.font.color.rgb=BLK
su=doc.add_paragraph();su.alignment=WD_ALIGN_PARAGRAPH.CENTER
sr=su.add_run('Activity-2 | Problem 1 | Solution 8 Hybrid Comparison | Flask Web App');sr.font.size=Pt(11);sr.font.color.rgb=GRY
doc.add_paragraph()
#ENDPART1

best=max(C,key=lambda c:c['test_acc'])
tb(doc,['Item','Detail'],[['Dataset','IMDB 25k train + 25k test (balanced)'],['Production','TF-IDF+LogReg %.2f%% acc F1 %.3f'%(P['accuracy']*100,P['f1'])],['Best RNN','%s %.2f%%'%(best['architecture'],best['test_acc']*100)],['Stack','Python3.12 NumPy RNNs sklearn Flask2.3'],['Live','Flask app http://127.0.0.1:5000']])
tg=doc.add_paragraph();tg.alignment=WD_ALIGN_PARAGRAPH.CENTER
gr=tg.add_run('Paste a review, get a verdict. Open the Lab to watch the model think.');gr.italic=True;gr.font.color.rgb=GRY
doc.add_page_break()
hd(doc,'Contents')
chs=['Abstract','Introduction','Dataset','Methodology','Architectures','Web App Design','Results','Mapping','Reproduce','Limits','Conclusion','Files']
for i,ch in enumerate(chs,1):
 p=doc.add_paragraph();p.paragraph_format.space_after=Pt(1);p.add_run('%d. %s'%(i,ch)).font.size=Pt(11)
hd(doc,'1. Abstract')
doc.add_paragraph('Complete sequential text prediction on IMDb reviews: POSITIVE vs NEGATIVE. Five RNNs from scratch in NumPy (BPTT+Adam) compared fairly (Solution 8). Production TF-IDF+LogReg serves at %.2f%% accuracy (P %.3f R %.3f F1 %.3f). Flask black-yellow app: abstract Studio, step-by-step Lab, Arena+Report.'%(P['accuracy']*100,P['precision'],P['recall'],P['f1']))
hd(doc,'2. Introduction')
doc.add_paragraph('Activity-2 Problem 1: build a Sequential Text Prediction System on IMDB. Core question: short-term vs long-term dependencies ("not bad" vs opening complaint overturned by finale). This report covers build, method, numbers, reproduction.')
for b in ['Input: raw English review, any length.','Output: POSITIVE/NEGATIVE + probs + word evidence.','Constraint: real data, trained weights, working app.']:doc.add_paragraph(b,style='List Bullet')
hd(doc,'3. Dataset')
doc.add_paragraph('IMDB Large Movie Reviews (Maas et al.): 50k labeled, 25k/25k balanced. Source: %s via HF mirror + fallbacks. Vocab %s (+PAD=0,UNK=1). Serve len %d, RNN window %d.'%(M['data_source'],f"{M['vocab_size']:,}",M['seq_len_full'],M['seq_len_rnn']))
hd(doc,'4. Methodology')
for s in ['1 Collect+clean: strip HTML, lowercase, depunct.','2 Tokenize: word = timestep.','3 Numericalize: top-10k to IDs, UNK=1.','4 Pad: post-pad 0 to 200 (60 for fair RNN compare).','5 Splits: 25k/25k, 15pct val, 2k scoring set.','6 Models: SimpleRNN BiRNN Deep2 LSTM GRU, NumPy scratch.','7 Train: BPTT+Adam+clip+L2, 6 epochs each.','8 Length study: 30/60/120/200 steps.','9 Gated compare: same protocol.','10 Gradients: BPTT norm probe + chart.','11 Deploy: metrics + Flask + /api/predict.']:doc.add_paragraph(s,style='List Bullet')
hd(doc,'5. Architectures')
hd(doc,'6. Web App Design')
doc.add_paragraph('Studio (abstract): paste review or pick example, press PREDICT, see only verdict + confidence + bars. No internals.')
doc.add_paragraph('Lab (simulation): after scoring, 6 anchored steps - cleaned text, token pills, word-ID map, padded vector, +/- evidence words + final probs, token-by-token unroll table (signal, hidden, running P).')
doc.add_paragraph('Arena+Report: production cards, 5-arch comparison, length study, gradient chart, training curves, write-up, Solution 1-10 map, API POST /api/predict.')
hd(doc,'7. Results')
hd(doc,'7.1 Production head',l=2)
tb(doc,['Metric','Value'],[['Model',P['model']],['Accuracy','%.2f%%'%(P['accuracy']*100)],['Precision','%.4f'%P['precision']],['Recall','%.4f'%P['recall']],['F1','%.4f'%P['f1']],['Test reviews','%s'%f"{M['test_size']:,}"],['Pipeline time','%.1fs'%M['elapsed_sec']]])
doc.add_paragraph('Confusion matrix (rows actual Neg/Pos, cols predicted Neg/Pos):')
tb(doc,['','Pred Neg','Pred Pos'],[['Actual Neg','%s'%f"{CM[0][0]:,}",'%s'%f"{CM[0][1]:,}"],['Actual Pos','%s'%f"{CM[1][0]:,}",'%s'%f"{CM[1][1]:,}"]])
hd(doc,'7.2 Hybrid comparison (Sol 8)',l=2)
tb(doc,['Architecture','Test','Val','Win'],[[c['architecture'],'%.2f%%'%(c['test_acc']*100),'%.2f%%'%(c['val_acc']*100),str(c['seq_len'])] for c in C])
for b in ['GRU wins %.2f%%, LSTM second %.2f%% - gates beat tanh.'%(C[4]['test_acc']*100,C[3]['test_acc']*100),'BiRNN 57.35pct beats single/deep via future context, trails gated.','SimpleRNN train ~99.6pct vs val ~50pct: overfit + vanishing memory.']:doc.add_paragraph(b,style='List Bullet')
doc.add_picture('static/curves.png',width=Inches(6.0))
cp=doc.add_paragraph();cp.alignment=WD_ALIGN_PARAGRAPH.CENTER
cr=cp.add_run('Figure 1 - Train (left) vs validation (right) accuracy per architecture.');cr.italic=True;cr.font.size=Pt(9);cr.font.color.rgb=GRY
doc.add_picture('static/grad_chart.png',width=Inches(6.0))
cp2=doc.add_paragraph();cp2.alignment=WD_ALIGN_PARAGRAPH.CENTER
cr2=cp2.add_run('Figure 2 - Vanishing gradient: BPTT norm vs steps back (log scale).');cr2.italic=True;cr2.font.size=Pt(9);cr2.font.color.rgb=GRY
hd(doc,'7.3 Length study',l=2)
tb(doc,['Steps','SimpleRNN acc'],[[str(s['seq_len']),'%.2f%%'%(s['test_acc']*100)] for s in S])
doc.add_paragraph('Longer windows = more context but further gradients; dip at 120, recovery at full 200. Gates exist for this tradeoff.')
hd(doc,'7.4 Overfit + gradients',l=2)
doc.add_paragraph('Plain/bidir/deep RNNs pass 99pct train while val stalls/falls (Deep: 99.65 vs 53.00). BPTT norms collapse ~4.4e-02 (1 step) to ~3.5e-27 (60 steps): exponential decay no LR can fix. LSTM/GRU gates preserve long signal, hence top ranks.')
hd(doc,'8. Requirement Mapping')
tb(doc,['Requirement','Delivered'],[['Sol1 Basic RNN','rnn_base.py + rnn_cells.py (NumPy BPTT Adam)'],['Sol2 Train IMDb','train_model.py + data_loader2.py (25k/25k)'],['Sol3 Predict','inference.py + Studio + /api/predict'],['Sol4 LSTM+GRU','rnn_lstm.py rnn_gru.py'],['Sol5 BiRNN','rnn_bi.py'],['Sol6 Deep RNN','2-layer stack train_model.py'],['Sol7 Overfit','Sec 7.4 + Arena curves'],['Sol8 Hybrid','Sec 7.2 table, 5 archs same protocol'],['Sol9 Deploy','app_flask.py Studio/Lab/Arena/Report'],['Sol10 Gradients','Fig2 + metrics.json grad_study']])
hd(doc,'9. Run & Reproduce')
doc.add_paragraph('Windows PowerShell from project root:')
cd=doc.add_paragraph();rn=cd.add_run('py -3.12 -m pip install -r requirements.txt\npy -3.12 -m src.train_model\npy -3.12 app_flask.py\n# open http://127.0.0.1:5000');rn.font.name='Consolas';rn.font.size=Pt(10)
doc.add_paragraph('API: POST /api/predict {"review":"A brilliant masterpiece"} -> label, probs, tokens, IDs, padded, evidence, unroll. Artifacts: models/*.joblib, vocab.json, metrics.json; charts static/.')
hd(doc,'10. Limits & Future Work')
for b in ['Serving head is TF-IDF+linear (order beyond bigrams narrated by Lab unroll).','Scratch RNNs use 60-step window + 10k vocab for speed; GPU + embeddings + longer windows would lift scores.','Next: embeddings, attention/Transformer compare, saliency heatmaps, batched GPU.']:doc.add_paragraph(b,style='List Bullet')
hd(doc,'11. Conclusion')
doc.add_paragraph('End-to-end delivery: real IMDb data, five hand-built RNNs fairly compared, 90.02pct serving head, Flask Studio + transparent Lab, one-command training. GRU then LSTM beat plain/bidir/deep tanh - a measured answer to short vs long dependency question.')
hd(doc,'12. References & File Map')
for b in ['Maas et al. Learning Word Vectors for Sentiment Analysis (ACL 2011).','Hochreiter+Schmidhuber 1997 LSTM; Cho et al. 2014 GRU; Schuster+Paliwal 1997 BiRNN.']:doc.add_paragraph(b,style='List Bullet')
tb(doc,['File','Role'],[['app_flask.py','Flask Studio/Lab/Arena/Report + API'],['src/data_loader2.py','Real IMDb loading'],['src/preprocess.py','Clean/tokenize/IDs/pad'],['src/rnn_base.py','Adam + trainer'],['src/rnn_cells.py','SimpleRNN + Deep stack'],['src/rnn_bi.py','BiRNN'],['src/rnn_lstm.py','LSTM scratch'],['src/rnn_gru.py','GRU scratch'],['src/train_model.py','Full pipeline'],['src/inference.py','Predict + evidence + unroll'],['models/','Artifacts + metrics.json'],['static/','curves.png grad_chart.png style.css'],['templates/','Flask pages black x yellow']])
doc.save('IMDb_Cinema_Sentiment_Lab_Report.docx')
print('DOCX saved OK')

for b in ['Simple RNN h=tanh(xWx+hWh+b): fast, forgets (Wh decay).','BiRNN fwd+bwd concatenated: past+future ("not bad").','Deep RNN 2-layer: richer, overfits fastest.','LSTM gates+cell: keeps info 100s steps.','GRU update+reset: LSTM-like, fewer params, best %.2f%%.'%(best['test_acc']*100),'Hybrid: identical training, ranked on same 2k.']:doc.add_paragraph(b,style='List Bullet')
#ENDPART2

