"""Render the English report to a paginated PDF; does not load any ML data."""
import html
import json
import re
import textwrap
from pathlib import Path
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,Preformatted,KeepTogether
from pypdf import PdfReader

ROOT=Path(__file__).resolve().parent.parent
OUTPUT=ROOT/'output/pdf/REPORT_V2.pdf'


def ascii_text(s):
    replacements={'\u2014':' - ','\u2013':'-','\u2011':'-','\u2019':"'",'\u2018':"'",'\u201c':'"','\u201d':'"',
                  '\u00b7':' | ','\u00d7':' x ','\u2248':'~','\u2192':'->','\u2212':'-','\u00a0':' '}
    for a,b in replacements.items(): s=s.replace(a,b)
    return s


def markup(s):
    s=html.escape(ascii_text(s))
    s=re.sub(r'`([^`]+)`',r'<font name="Courier" size="8.5">\1</font>',s)
    return re.sub(r'\*\*([^*]+)\*\*',r'<b>\1</b>',s)


def page(canvas,doc):
    canvas.saveState(); canvas.setStrokeColor(colors.HexColor('#ccd5df'))
    canvas.line(48,A4[1]-31,A4[0]-48,A4[1]-31)
    canvas.setFont('Helvetica',8); canvas.setFillColor(colors.HexColor('#526171'))
    canvas.drawString(48,A4[1]-23,'DASE7506 MP2 | Context-conditioned reliability')
    canvas.drawRightString(A4[0]-48,24,str(doc.page)); canvas.restoreState()


def main():
    OUTPUT.parent.mkdir(parents=True,exist_ok=True)
    styles=getSampleStyleSheet()
    styles.add(ParagraphStyle(name='BodyReport',fontName='Helvetica',fontSize=10,leading=12.5,spaceAfter=6))
    styles.add(ParagraphStyle(name='TitleReport',fontName='Helvetica-Bold',fontSize=19,leading=23,spaceAfter=12))
    styles.add(ParagraphStyle(name='SectionReport',fontName='Helvetica-Bold',fontSize=12.2,leading=16.4,spaceBefore=9,spaceAfter=6,keepWithNext=True))
    styles.add(ParagraphStyle(name='TableReport',fontName='Helvetica',fontSize=8.7,leading=11.8))
    styles.add(ParagraphStyle(name='CodeReport',fontName='Courier',fontSize=8,leading=10.5,leftIndent=8,spaceBefore=4,spaceAfter=9))
    lines=(ROOT/'report/REPORT_V2.md').read_text(encoding='utf-8').splitlines()
    flow=[]; i=0
    while i<len(lines):
        line=lines[i].strip()
        if not line: i+=1; continue
        if line.startswith('# '):
            flow.append(Paragraph(markup(line[2:]),styles['TitleReport'])); i+=1
        elif line.startswith('## '):
            flow.append(Paragraph(markup(line[3:]),styles['SectionReport'])); i+=1
        elif line.startswith('```'):
            i+=1; code=[]
            while i<len(lines) and not lines[i].startswith('```'):
                raw=ascii_text(lines[i]); code.extend(textwrap.wrap(raw,width=96,break_long_words=False,break_on_hyphens=False) or ['']);i+=1
            flow.append(Preformatted('\n'.join(code),styles['CodeReport']));i+=1
        elif line.startswith('|'):
            table=[]
            while i<len(lines) and lines[i].strip().startswith('|'):
                cells=[c.strip() for c in lines[i].strip().strip('|').split('|')]
                if not all(re.fullmatch(r'[-: ]+',c) for c in cells):
                    table.append([Paragraph(markup(c),styles['TableReport']) for c in cells])
                i+=1
            width=A4[0]-96
            widths=[width*.58,width*.21,width*.21] if len(table[0])==3 else [width*.65,width*.35]
            t=Table(table,colWidths=widths,repeatRows=1,hAlign='LEFT')
            t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e9eff6')),
                ('LINEBELOW',(0,0),(-1,0),.7,colors.HexColor('#7b93ad')),
                ('LINEBELOW',(0,1),(-1,-1),.25,colors.HexColor('#d4dce4')),
                ('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),7),
                ('RIGHTPADDING',(0,0),(-1,-1),7),('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),6)]))
            flow.extend([Spacer(1,4),KeepTogether([t]),Spacer(1,10)])
        else:
            paragraph=[line];i+=1
            while i<len(lines) and lines[i].strip() and not lines[i].startswith(('#','```','|')):
                paragraph.append(lines[i].strip());i+=1
            flow.append(Paragraph(markup(' '.join(paragraph)),styles['BodyReport']))
    doc=SimpleDocTemplate(str(OUTPUT),pagesize=A4,leftMargin=48,rightMargin=48,topMargin=47,bottomMargin=43,
        title='Selecting Reliable LLM Reasoning with Context-Conditioned Preferences',author='DASE7506 MP2 project')
    doc.build(flow,onFirstPage=page,onLaterPages=page)
    reader=PdfReader(OUTPUT); assert len(reader.pages)<=10
    qa=ROOT/'code/runs/submission_audit/pdf';qa.mkdir(parents=True,exist_ok=True)
    text='\n'.join(p.extract_text() or '' for p in reader.pages)
    assert '83.94004282655246' in text and 'AI assistance' in text
    (qa/'extracted_text.txt').write_text(text,encoding='utf-8')
    import pypdfium2 as pdfium
    pdf=pdfium.PdfDocument(str(OUTPUT))
    for idx in range(len(pdf)):
        pdf[idx].render(scale=1.35).to_pil().save(qa/f'page-{idx+1}.png')
    record=dict(page_count=len(reader.pages),limit=10,status='page-count passed',output=str(OUTPUT))
    (qa/'verification.json').write_text(json.dumps(record,indent=2));print(json.dumps(record))


if __name__=='__main__': main()
