from pathlib import Path
import zipfile,posixpath
from lxml import etree as E
R='http://schemas.openxmlformats.org/package/2006/relationships'
C='http://schemas.openxmlformats.org/package/2006/content-types'
P='http://schemas.openxmlformats.org/presentationml/2006/main'
O='http://schemas.openxmlformats.org/officeDocument/2006/relationships'
def xml(b):return E.fromstring(b)
def dump(r):return E.tostring(r,xml_declaration=True,encoding='UTF-8',standalone=True)
source='docs/interview_preparation/presentation/Fatima_Shockor_Interview_Updated_HCC_Spatial_Case_Study.pptx'
with zipfile.ZipFile(source) as src,zipfile.ZipFile('tmp/hcc_slide_edit/spatial-updated-slides.pptx') as new:
 parts={n:src.read(n) for n in src.namelist()}
 ct=xml(parts['[Content_Types].xml'])
 for idx,dest,notes in [(1,7,12),(2,11,13)]:
  parts[f'ppt/slides/slide{dest}.xml']=new.read(f'ppt/slides/slide{idx}.xml')
  rel=xml(new.read(f'ppt/slides/_rels/slide{idx}.xml.rels'))
  for r in rel:
   typ=r.get('Type')
   if typ.endswith('/slideLayout'):r.set('Target','../slideLayouts/slideLayout1.xml')
   elif typ.endswith('/notesSlide'):r.set('Target',f'../notesSlides/notesSlide{notes}.xml')
   elif typ.endswith('/image'):
    origin=posixpath.normpath(posixpath.join('ppt/slides',r.get('Target'))).lstrip('/')
    name='updated_spatial_'+posixpath.basename(origin)
    parts['ppt/media/'+name]=new.read(origin);r.set('Target','../media/'+name)
   else:raise ValueError(r.attrib)
  parts[f'ppt/slides/_rels/slide{dest}.xml.rels']=dump(rel)
  parts[f'ppt/notesSlides/notesSlide{notes}.xml']=new.read(f'ppt/notesSlides/notesSlide{idx}.xml')
  rel=xml(new.read(f'ppt/notesSlides/_rels/notesSlide{idx}.xml.rels'))
  for r in rel:
   if r.get('Type').endswith('/slide'):r.set('Target',f'../slides/slide{dest}.xml')
   elif r.get('Type').endswith('/notesMaster'):r.set('Target','../notesMasters/notesMaster1.xml')
  parts[f'ppt/notesSlides/_rels/notesSlide{notes}.xml.rels']=dump(rel)
  E.SubElement(ct,'{'+C+'}Override',PartName=f'/ppt/notesSlides/notesSlide{notes}.xml',ContentType='application/vnd.openxmlformats-officedocument.presentationml.notesSlide+xml')
 E.SubElement(ct,'{'+C+'}Override',PartName='/ppt/slides/slide11.xml',ContentType='application/vnd.openxmlformats-officedocument.presentationml.slide+xml')
 if not any(e.get('Extension')=='png' for e in ct):E.SubElement(ct,'{'+C+'}Default',Extension='png',ContentType='image/png')
 parts['[Content_Types].xml']=dump(ct)
 rel=xml(parts['ppt/_rels/presentation.xml.rels'])
 E.SubElement(rel,'{'+R+'}Relationship',Id='rIdSpatialBiology',Type=O+'/slide',Target='slides/slide11.xml')
 lookup={e.get('Id'):e.get('Target') for e in rel}
 parts['ppt/_rels/presentation.xml.rels']=dump(rel)
 pres=xml(parts['ppt/presentation.xml']);ids=pres.find('{'+P+'}sldIdLst')
 index=next(i for i,e in enumerate(ids) if lookup[e.get('{'+O+'}id')]=='slides/slide7.xml')
 item=E.Element('{'+P+'}sldId',id=str(max(int(e.get('id')) for e in ids)+1));item.set('{'+O+'}id','rIdSpatialBiology');ids.insert(index+1,item)
 parts['ppt/presentation.xml']=dump(pres)
 # The closing positioning slide becomes logical slide 11.
 closing=xml(parts['ppt/slides/slide8.xml'])
 for e in closing.iter():
  if e.tag.endswith('}t') and e.text=='10':e.text='11'
  elif e.tag.endswith('}t') and e.text=='10 / POSITIONING':e.text='11 / POSITIONING'
 parts['ppt/slides/slide8.xml']=dump(closing)
 for shape in closing.iter('{'+P+'}sp'):
  if ''.join(shape.itertext())=='11':
   ns={'a':'http://schemas.openxmlformats.org/drawingml/2006/main'}
   shape.find('.//a:xfrm/a:ext',ns).set('cx','457200')
   shape.find('.//a:xfrm/a:off',ns).set('x','10972800')
 parts['ppt/slides/slide8.xml']=dump(closing)
 app=xml(parts['docProps/app.xml'])
 for e in app.iter():
  if e.tag.endswith('}Slides'):e.text='11'
 parts['docProps/app.xml']=dump(app)
 with zipfile.ZipFile('tmp/hcc_slide_edit/candidate-spatial-updated.pptx','w',zipfile.ZIP_DEFLATED) as out:
  for n,b in parts.items():out.writestr(n,b)
print('Updated spatial workflow and inserted biological slide; 11 slides total.')
