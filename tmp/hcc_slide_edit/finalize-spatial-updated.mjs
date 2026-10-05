import path from 'node:path';
import fs from 'node:fs/promises';
import {pathToFileURL} from 'node:url';
import {createHash} from 'node:crypto';
import {PresentationFile,FileBlob} from '@oai/artifact-tool';
const skill='C:/Users/shoko/.codex/plugins/cache/openai-primary-runtime/presentations/26.904.11930/skills/presentations';
process.env.RUNTIME_NODE_MODULES='C:/Users/shoko/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules';
const {finalizePresentation}=await import(pathToFileURL(path.join(skill,'container_tools/artifact_tool_utils.mjs')));
const source=path.resolve('docs/interview_preparation/Fatima_Shockor_Interview_Core_Slides_1_8_repaired.pptx');
const result=await finalizePresentation({workspaceDir:path.resolve('.'),candidatePath:path.resolve('tmp/hcc_slide_edit/candidate-spatial-updated.pptx'),finalPath:path.resolve('docs/interview_preparation/presentation/Fatima_Shockor_Interview_Updated_HCC_Spatial_Results_Final.pptx'),pythonExecutable:'C:/Users/shoko/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe',integrityValidatorPath:path.join(skill,'container_tools/inspect_presentation_package_integrity.py'),layoutValidatorPath:path.join(skill,'container_tools/inspect_presentation_layout_geometry.py'),layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-bullet-geometry','--validate-heading-fit','--require-native-table-slide','7','--require-native-table-slide','8'],requiredNativeTableOwnerSlides:[7,8],explicitTotalSlideCount:11,fontPolicy:{basis:'reference',families:['Aptos','Aptos Display'],referencePath:source,referenceSha256:createHash('sha256').update(await fs.readFile(source)).digest('hex')},verifyArtifactToolImport:true,receiptPath:path.resolve('tmp/hcc_slide_edit/validation-spatial-final.json')});
console.log(JSON.stringify(result));
const p=await PresentationFile.importPptx(await FileBlob.load(path.resolve('docs/interview_preparation/presentation/Fatima_Shockor_Interview_Updated_HCC_Spatial_Results_Final.pptx')));
for(let i=0;i<p.slides.items.length;i++) {const png=await p.export({slide:p.slides.items[i],format:'png',scale:1});await fs.writeFile(path.resolve(`tmp/hcc_slide_edit/final-spatial-updated-${i+1}.png`),new Uint8Array(await png.arrayBuffer()));}
console.log('Final slides rendered.');




