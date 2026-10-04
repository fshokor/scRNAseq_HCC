from pathlib import Path
import zipfile
from lxml import etree as E
R='http://schemas.openxmlformats.org/package/2006/relationships'
C='http://schemas.openxmlformats.org/package/2006/content-types'
def xml(b):return E.fromstring(b)
def dump(r):return E.tostring(r,xml_declaration=True,encoding='UTF-8',standalone=True)
with zipfile.ZipFile('docs/interview_preparation/presentation/Fatima_Shockor_Interview_Updated_HCC_With_GNN.pptx') as src,zipfile.ZipFile('tmp/hcc_slide_edit/spatial-slide.pptx') as new:
 parts={n:src.read(n) for n in src.namelist()}
 # Original slide7 is logical slide9 after the two HCC insertions.
 parts['ppt/slides/slide7.xml']=new.read('ppt/slides/slide1.xml')
 root=xml(new.read('ppt/slides/_rels/slide1.xml.rels'))
 for r in root:
  if r.get('Type').endswith('/slideLayout'):r.set('Target','../slideLayouts/slideLayout1.xml')
  elif r.get('Type').endswith('/notesSlide'):r.set('Target','../notesSlides/notesSlide11.xml')
  else:raise ValueError(r.attrib)
 parts['ppt/slides/_rels/slide7.xml.rels']=dump(root)
 parts['ppt/notesSlides/notesSlide11.xml']=new.read('ppt/notesSlides/notesSlide1.xml')
 root=xml(new.read('ppt/notesSlides/_rels/notesSlide1.xml.rels'))
 for r in root:
  if r.get('Type').endswith('/slide'):r.set('Target','../slides/slide7.xml')
  elif r.get('Type').endswith('/notesMaster'):r.set('Target','../notesMasters/notesMaster1.xml')
 parts['ppt/notesSlides/_rels/notesSlide11.xml.rels']=dump(root)
 root=xml(parts['[Content_Types].xml']);E.SubElement(root,'{'+C+'}Override',PartName='/ppt/notesSlides/notesSlide11.xml',ContentType='application/vnd.openxmlformats-officedocument.presentationml.notesSlide+xml');parts['[Content_Types].xml']=dump(root)
 with zipfile.ZipFile('tmp/hcc_slide_edit/candidate-spatial.pptx','w',zipfile.ZIP_DEFLATED) as out:
  for n,b in parts.items():out.writestr(n,b)
print('Replaced logical slide9; other slide content preserved.')
