from pathlib import Path
import zipfile
from lxml import etree as E
P='http://schemas.openxmlformats.org/presentationml/2006/main'
R='http://schemas.openxmlformats.org/package/2006/relationships'
A='http://schemas.openxmlformats.org/drawingml/2006/main'
C='http://schemas.openxmlformats.org/package/2006/content-types'
def xml(b):return E.fromstring(b)
def dump(r):return E.tostring(r,xml_declaration=True,encoding='UTF-8',standalone=True)
with zipfile.ZipFile('tmp/hcc_slide_edit/source-normalized.pptx') as src,zipfile.ZipFile('tmp/hcc_slide_edit/new-slide.pptx') as new:
 parts={n:src.read(n) for n in src.namelist()}
 root=xml(parts['ppt/presentation.xml']);lst=root.find('{'+P+'}sldIdLst')
 s=E.Element('{'+P+'}sldId',id='264');s.set('{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id','rId15');lst.insert(6,s)
 parts['ppt/presentation.xml']=dump(root)
 root=xml(parts['ppt/_rels/presentation.xml.rels']);E.SubElement(root,'{'+R+'}Relationship',Id='rId15',Type='http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide',Target='slides/slide9.xml');parts['ppt/_rels/presentation.xml.rels']=dump(root)
 parts['ppt/slides/slide9.xml']=new.read('ppt/slides/slide1.xml')
 root=xml(new.read('ppt/slides/_rels/slide1.xml.rels'))
 for r in root:
  if r.get('Type').endswith('/slideLayout'):r.set('Target','../slideLayouts/slideLayout1.xml')
  elif r.get('Type').endswith('/notesSlide'):r.set('Target','../notesSlides/notesSlide9.xml')
  else:raise ValueError(r.attrib)
 parts['ppt/slides/_rels/slide9.xml.rels']=dump(root)
 parts['ppt/notesSlides/notesSlide9.xml']=new.read('ppt/notesSlides/notesSlide1.xml')
 root=xml(new.read('ppt/notesSlides/_rels/notesSlide1.xml.rels'))
 for r in root:
  if r.get('Type').endswith('/slide'):r.set('Target','../slides/slide9.xml')
  elif r.get('Type').endswith('/notesMaster'):r.set('Target','../notesMasters/notesMaster1.xml')
 parts['ppt/notesSlides/_rels/notesSlide9.xml.rels']=dump(root)
 root=xml(parts['[Content_Types].xml'])
 for name,kind in [('slides/slide9.xml','slide'),('notesSlides/notesSlide9.xml','notesSlide')]:E.SubElement(root,'{'+C+'}Override',PartName='/ppt/'+name,ContentType='application/vnd.openxmlformats-officedocument.presentationml.'+kind+'+xml')
 parts['[Content_Types].xml']=dump(root)
 for number in [7,8]:
  name=f'ppt/slides/slide{number}.xml';root=xml(parts[name])
  for t in root.findall('.//{'+A+'}t'):
   if t.text==str(number):t.text=str(number+1)
   elif t.text and t.text.startswith(f'0{number} /'):t.text=f'0{number+1} /'+t.text.split('/',1)[1]
  parts[name]=dump(root)
 root=xml(parts['docProps/app.xml'])
 for x in root.iter():
  if E.QName(x).localname=='Slides':x.text='9'
 parts['docProps/app.xml']=dump(root)
 with zipfile.ZipFile('tmp/hcc_slide_edit/candidate.pptx','w',zipfile.ZIP_DEFLATED) as out:
  for n,b in parts.items():out.writestr(n,b)
print('Inserted editable gene slide after slide 6; spatial and positioning slides renumbered.')
